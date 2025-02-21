use fish_speech_core::config::WhichModel;
use pyo3::prelude::*;

pub fn wrap_err(err: impl std::error::Error) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
}

pub trait PyRes<R> {
    #[allow(unused)]
    fn w(self) -> PyResult<R>;
}

/// Copied blindly from rustymimi
impl<R, E: Into<anyhow::Error>> PyRes<R> for Result<R, E> {
    fn w(self) -> PyResult<R> {
        self.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.into().to_string()))
    }
}

pub fn get_version(raw_version: &str) -> Result<WhichModel, anyhow::Error> {
    match raw_version {
        "1.5" => Ok(WhichModel::Fish1_5),
        "1.4" => Ok(WhichModel::Fish1_4),
        "1.2" => Ok(WhichModel::Fish1_2),
        "dual_ar" => Ok(WhichModel::DualAR),
        v => Err(anyhow::anyhow!("Unsupported version: {}", v)),
    }
}
