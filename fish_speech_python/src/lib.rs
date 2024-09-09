use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use fish_speech_core::models::vqgan::FireflyArchitecture;
use hf_hub::{api::sync::Api, Repo};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn encode_mels() -> PyResult<()> {
    let api = Api::new()
        .map_err(|err| PyValueError::new_err(format!("Could not load HF API: {}", err)))?;

    let api = api.repo(Repo::model("fishaudio/fish-speech-1.2".into()));
    let _filename = api
        .get("firefly-gan-vq-fsq-4x1024-42hz-generator.pth")
        .map_err(|err| {
            PyValueError::new_err(format!("Could not download encoder weights: {}", err))
        })?;

    // TODO: Add GPU support
    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(_filename, DType::F32, &device)
        .map_err(|err| PyValueError::new_err(format!("Could not load encoder weights: {}", err)))?;

    let encoder_model = FireflyArchitecture::load(vb).unwrap();

    // Add forward pass after we verify this works
    println!("Model loaded!");

    Ok(())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn fish_speech(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(encode_mels, m)?)?;

    Ok(())
}
