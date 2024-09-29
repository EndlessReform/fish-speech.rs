use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;
use fish_speech_core::models::vqgan::config::{FireflyConfig, WhichModel};
use fish_speech_core::models::vqgan::encoder::FireflyEncoder;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

fn wrap_err(err: impl std::error::Error) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
}

#[pyclass(unsendable)]
struct FishSpeechModel {
    model: FireflyEncoder,
}

#[pymethods]
impl FishSpeechModel {
    #[new]
    #[pyo3(signature = (repo_id = None))]
    fn new(repo_id: Option<String>) -> PyResult<Self> {
        let repo_id = repo_id.unwrap_or_else(|| "fishaudio/fish-speech-1.2".to_string());
        let api = hf_hub::api::sync::Api::new().map_err(wrap_err)?;
        let api = api.repo(hf_hub::Repo::with_revision(
            repo_id,
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        let filename = api
            .get("firefly-gan-vq-fsq-4x1024-42hz-generator.pth")
            .map_err(wrap_err)?;
        let config = FireflyConfig::fish_speech_1_2();
        let vb = VarBuilder::from_pth(&filename, DType::F32, &Device::Cpu).map_err(wrap_err)?;
        let model = FireflyEncoder::load(vb, &config, &WhichModel::Fish1_2)
            .map_err(|err| PyException::new_err(format!("Could not load model: error {}", err)))?;
        Ok(Self { model })
    }

    fn forward(&self, py: Python, input: &PyAny) -> PyResult<PyObject> {
        // Extract shape and data from the input
        let shape: Vec<usize> = input.getattr("shape")?.extract()?;
        let flattened_data: Vec<f32> = input.call_method0("flatten")?.extract()?;

        // Create a Candle Tensor from the data
        let tensor = Tensor::from_slice(&flattened_data, Shape::from(shape), &Device::Cpu)
            .map_err(wrap_err)?;

        // Forward pass
        let output = self
            .model
            .encode(&tensor)
            .map_err(wrap_err)?
            .squeeze(0)
            .map_err(wrap_err)?;
        println!("Forward pass done");

        // Convert output Tensor to nested Python list
        let output_data: Vec<Vec<i64>> = output.to_vec2::<i64>().map_err(wrap_err)?;

        // Convert 2D Vec to Python list of lists
        let py_output = output_data
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|val| val.into_py(py))
                    .collect::<Vec<PyObject>>()
                    .into_py(py)
            })
            .collect::<Vec<PyObject>>();

        Ok(py_output.into_py(py))
    }
}

#[pymodule]
fn fish_speech(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FishSpeechModel>()?;
    Ok(())
}
