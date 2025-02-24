use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use fish_speech_core::codec::{FireflyCodec as FireflyCodecModel, FireflyConfig};
use fish_speech_core::config::{WhichCodec, WhichFishVersion, WhichModel};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use super::utils::{get_device, get_version, wrap_err, PyRes};

#[pyclass]
pub struct FireflyCodec {
    model: FireflyCodecModel,
    device: Device,
    dtype: DType,
}

#[pymethods]
impl FireflyCodec {
    #[pyo3(signature = (dir, version="1.5", device="cpu", dtype="f32"))]
    #[new]
    fn new(dir: std::path::PathBuf, version: &str, device: &str, dtype: &str) -> PyResult<Self> {
        let model_type = get_version(version)
            .map_err(|_| PyException::new_err(format!("Unsupported model version: {}", version)))?;
        let codec_type = WhichCodec::from_model(model_type);

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
            WhichModel::Fish1_2 => VarBuilder::from_pth(
                "firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
                dtype,
                &device,
            ),
            _ => unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[dir.join("firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors")],
                    dtype,
                    &device,
                )
            },
        }
        .map_err(wrap_err)?;

        let fish_version = match codec_type {
            WhichCodec::Fish(version) => version,
            _ => WhichFishVersion::Fish1_5,
        };
        let config = FireflyConfig::get_config_for(fish_version);
        let model = FireflyCodecModel::load(config.clone(), vb, fish_version)
            .map_err(|err| PyException::new_err(format!("Could not load model: error {}", err)))?;
        Ok(Self {
            model,
            device,
            dtype,
        })
    }

    #[getter]
    pub fn sample_rate(&self) -> u32 {
        self.model.cfg.spec_transform.sample_rate as u32
    }

    fn encode(&self, pcm_data: numpy::PyReadonlyArray3<f32>) -> PyResult<PyObject> {
        let py = pcm_data.py();
        let pcm_data = pcm_data.as_array();
        let pcm_shape = pcm_data.shape().to_vec();
        let pcm_data = match pcm_data.to_slice() {
            None => Err(PyException::new_err("Data must be a contiguous array")),
            Some(data) => Ok(data),
        }
        .map_err(wrap_err)?;
        let codes = py
            .allow_threads(|| {
                let pcm_data = candle_core::Tensor::from_slice(pcm_data, pcm_shape, &self.device)?
                    .to_dtype(self.dtype)?;
                let codes = self.model.encode(&pcm_data)?.to_dtype(DType::U32)?;
                codes.to_vec3::<u32>()
            })
            .w()?;
        let codes = numpy::PyArray3::from_vec3(py, &codes)?;
        Ok(codes.into_any().unbind())
    }

    fn decode(&mut self, codes: numpy::PyReadonlyArray3<u32>, py: Python) -> PyResult<PyObject> {
        let codes = codes.as_array();
        let codes_shape = codes.shape().to_vec();
        let codes = match codes.to_slice() {
            None => Err(PyException::new_err("input data is not contiguous")),
            Some(data) => Ok(data),
        }
        .map_err(wrap_err)?;
        let pcm = py
            .allow_threads(|| {
                let codes = candle_core::Tensor::from_slice(codes, codes_shape, &self.device)?;
                let pcm = self
                    .model
                    .decode(&codes)?
                    .to_dtype(candle_core::DType::F32)?;
                pcm.to_vec3::<f32>()
            })
            .w()?;
        let pcm = numpy::PyArray3::from_vec3(py, &pcm)?;
        Ok(pcm.into_any().unbind())
    }
}
