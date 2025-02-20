use anyhow;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use fish_speech_core::codec::{FireflyCodec as FireflyCodecModel, FireflyConfig};
use fish_speech_core::config::{WhichCodec, WhichFishVersion, WhichModel};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

fn wrap_err(err: impl std::error::Error) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
}

trait PyRes<R> {
    #[allow(unused)]
    fn w(self) -> PyResult<R>;
}

/// Copied blindly from rustymimi
impl<R, E: Into<anyhow::Error>> PyRes<R> for Result<R, E> {
    fn w(self) -> PyResult<R> {
        self.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.into().to_string()))
    }
}

#[pyclass(unsendable)]
struct FireflyCodec {
    model: FireflyCodecModel,
    device: Device,
    dtype: DType,
}

#[pymethods]
impl FireflyCodec {
    #[pyo3(signature = (dir, version="1.5", device="cpu", dtype="f32"))]
    #[new]
    fn new(dir: std::path::PathBuf, version: &str, device: &str, dtype: &str) -> PyResult<Self> {
        let model_type = match version {
            "1.5" => WhichModel::Fish1_5,
            "1.4" => WhichModel::Fish1_4,
            "1.2" => WhichModel::Fish1_2,
            v => return Err(PyException::new_err(format!("Unsupported version: {}", v))),
        };
        let codec_type = WhichCodec::from_model(model_type);

        // TODO gate this on feature flag
        let device = match device {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0).map_err(wrap_err)?,
            "metal" => Device::new_metal(0).map_err(wrap_err)?,
            d => return Err(PyException::new_err(format!("Unsupported device: {}", d))),
        };

        let dtype = match dtype {
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            d => return Err(PyException::new_err(format!("Unsupported dtype: {}", d))),
        };

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
                let codes = self.model.encode(&pcm_data)?;
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

#[pymodule]
fn fish_speech(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FireflyCodec>()?;
    Ok(())
}
