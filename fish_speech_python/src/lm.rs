use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use fish_speech_core::config::{WhichLM, WhichModel};
use fish_speech_core::lm::generate::generate_blocking;
use fish_speech_core::lm::{dual_ar::TokenConfig, BaseModelArgs, DualARTransformer};
use fish_speech_core::text::prompt::PromptEncoder;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use tokenizers::Tokenizer;

use super::utils::{get_version, wrap_err, PyRes};

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

        // TODO Modularization
        let dtype = match dtype {
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            d => return Err(PyException::new_err(format!("Unsupported dtype: {}", d))),
        };
        // TODO hardware acceleration
        let device = Device::Cpu;
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

    #[pyo3(signature = (input, sysprompt= Some("Speak out the provided text".into())))]
    fn __call__(
        &mut self,
        input: Bound<'_, PyAny>,
        sysprompt: Option<String>,
    ) -> PyResult<PyObject> {
        self.model.clear_slow_layer_caches();

        let py = input.py();
        let input_vec: Vec<String> = input.extract()?;

        let prompt_encoder = PromptEncoder::new(
            &self.tokenizer,
            &self.device,
            self.cfg.num_codebooks,
            self.model.model_type.clone(),
        );

        // TODO: Implement voice encoding logic
        let (num_conditioning_tokens, prompts) = prompt_encoder
            .encode_sequence(input_vec, sysprompt, None, true)
            .map_err(wrap_err)?;

        let mut outputs: Vec<Tensor> = Vec::with_capacity(prompts.len());
        // TODO: Expose sampling arguments
        let sampling_args = fish_speech_core::lm::sampling::SamplingArgs {
            temp: 0.0,
            top_p: 0.85,
            top_k: 128,
            repetition_penalty: 1.0,
        };
        for prompt in prompts {
            let x = py
                .allow_threads(|| generate_blocking(&mut self.model, &prompt, 1024, &sampling_args))
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
}
