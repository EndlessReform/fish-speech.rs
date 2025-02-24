use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use fish_speech_core::config::{WhichLM, WhichModel};
use fish_speech_core::lm::generate::generate_blocking;
use fish_speech_core::lm::{dual_ar::TokenConfig, BaseModelArgs, DualARTransformer};
use fish_speech_core::text::prompt::PromptEncoder;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokenizers::Tokenizer;

use super::utils::{get_device, get_version, wrap_err, PyRes};

#[pyclass]
pub struct LM {
    model: DualARTransformer,
    device: Device,
    tokenizer: Tokenizer,
    cfg: BaseModelArgs,
}

#[pymethods]
impl LM {
    #[pyo3(signature = (dir, version="1.5", device="cpu", dtype="f32"))]
    #[new]
    fn new(dir: std::path::PathBuf, version: &str, device: &str, dtype: &str) -> PyResult<Self> {
        let model_type = get_version(version)
            .map_err(|_| PyException::new_err(format!("Unsupported model version: {}", version)))?;

        let dtype = match (dtype, device) {
            ("bf16", "cuda") | ("bf16", "metal") => DType::BF16,
            ("f32", _) => DType::F32,
            (d, _) => {
                return Err(PyException::new_err(format!(
                    "Unsupported dtype on device {}: {}",
                    device, d
                )))
            }
        };
        let device = get_device(device)?;
        let vb = match model_type {
            WhichModel::Fish1_2 => {
                VarBuilder::from_pth(dir.join("model.safetensors"), dtype, &device)
            }
            _ => unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[dir.join("model.safetensors")],
                    dtype,
                    &device,
                )
            },
        }
        .map_err(wrap_err)?;
        let cfg = BaseModelArgs::from_file(dir.join("config.json")).map_err(wrap_err)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|_| PyException::new_err("Failed to load tokenizer"))?;

        let lm_type = WhichLM::from_model(model_type);
        let token_config = TokenConfig::new(lm_type.clone(), &tokenizer, &cfg)
            .map_err(|e| PyException::new_err(format!("Failed to create token config: {}", e)))?;
        let model =
            DualARTransformer::load(&vb, &cfg, &token_config, lm_type.clone()).map_err(wrap_err)?;

        Ok(Self {
            model,
            device,
            tokenizer,
            cfg,
        })
    }

    #[pyo3(signature = (input, sysprompt= Some("Speak out the provided text".into()), speaker_prompt=None, temp=0.7, top_p=0.9, top_k=50, repetition_penalty=1.2))]
    fn __call__(
        &mut self,
        input: Bound<'_, PyAny>,
        sysprompt: Option<String>,
        speaker_prompt: Option<numpy::PyReadonlyArray3<u32>>,
        temp: f64,
        top_p: f64,
        top_k: usize,
        repetition_penalty: f32,
    ) -> PyResult<PyObject> {
        self.model.clear_slow_layer_caches();

        let py = input.py();
        let input_vec: Vec<String> = input.extract()?;
        let maybe_speaker_prompt = match speaker_prompt {
            Some(codes) => {
                let codes = codes.as_array();
                let codes_shape = codes.shape().to_vec();
                let codes = codes
                    .to_slice()
                    .ok_or(PyException::new_err("input data is not contiguous"))?;

                let codes =
                    Tensor::from_slice(codes, codes_shape, &self.device).map_err(wrap_err)?;
                let codes = codes.squeeze(0).map_err(wrap_err)?;
                Some(codes)
            }
            None => None,
        };

        let prompt_encoder = PromptEncoder::new(
            &self.tokenizer,
            &self.device,
            self.cfg.num_codebooks,
            self.model.model_type.clone(),
        );

        // TODO: Implement voice encoding logic
        let (num_conditioning_tokens, prompts) = prompt_encoder
            .encode_sequence(input_vec, sysprompt, maybe_speaker_prompt, true)
            .map_err(wrap_err)?;

        let mut outputs: Vec<Tensor> = Vec::with_capacity(prompts.len());
        let sampling_args = fish_speech_core::lm::sampling::SamplingArgs {
            temp,
            top_p,
            top_k,
            repetition_penalty,
        };
        for prompt in prompts {
            let x = py
                .allow_threads(|| {
                    generate_blocking(&mut self.model, &prompt, 1024, &sampling_args, false)
                })
                .w()?;

            outputs.push(x);
            self.model
                .clear_slow_caches_until(num_conditioning_tokens)
                .map_err(wrap_err)?;
        }
        let output = Tensor::cat(&outputs, D::Minus1)
            .map_err(wrap_err)?
            .unsqueeze(0)
            .map_err(wrap_err)?
            .to_vec3::<u32>()
            .map_err(wrap_err)?;

        self.model.clear_slow_layer_caches();
        let codes = numpy::PyArray3::from_vec3(py, &output)?;

        Ok(codes.into_any().unbind())
    }

    fn create_speaker_prompt(&self, input: Vec<Bound<'_, PyDict>>) -> PyResult<PyObject> {
        let prompt_encoder = PromptEncoder::new(
            &self.tokenizer,
            &self.device,
            self.model.cfg.num_codebooks,
            self.model.model_type.clone(),
        );
        if input.is_empty() {
            return Err(PyException::new_err("input is empty"));
        }
        let py = input[0].py();
        let mut prompts: Vec<Tensor> = Vec::with_capacity(input.len());
        for sample in input {
            // Extract "text" as a String
            let text: String = sample
                .get_item("text")?
                .ok_or(PyException::new_err(format!(
                    "Missing 'text' field in sample"
                )))?
                .extract()?;
            let audio: numpy::PyReadonlyArray3<u32> = sample
                .get_item("codes")?
                .ok_or(PyException::new_err(format!(
                    "Missing 'codes' field in sample (encoded audio only)"
                )))?
                .extract()?;
            let codes = audio.as_array();
            let codes_shape = codes.shape().to_vec();
            let codes = codes
                .to_slice()
                .ok_or(PyException::new_err("input data is not contiguous"))?;
            let codes_tensor =
                Tensor::from_slice(&codes, codes_shape, &self.device).map_err(wrap_err)?;
            let codes_tensor = if codes_tensor.rank() == 3 {
                codes_tensor.squeeze(0).map_err(wrap_err)?
            } else {
                codes_tensor
            };
            prompts.push(
                prompt_encoder
                    .encode_conditioning_prompt(&text, &codes_tensor)
                    .map_err(wrap_err)?,
            );
        }
        let prompts = Tensor::cat(&prompts, D::Minus1).map_err(wrap_err)?;
        // move to npy
        let prompts = prompts.unsqueeze(0).map_err(wrap_err)?;
        let prompts = prompts.to_vec3::<u32>().map_err(wrap_err)?;
        let prompts = numpy::PyArray3::from_vec3(py, &prompts)?;

        Ok(prompts.into_any().unbind())
    }
}

impl LM {}
