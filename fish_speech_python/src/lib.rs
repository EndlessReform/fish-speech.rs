mod codec;
mod lm;
mod utils;

use pyo3::prelude::*;

use codec::FireflyCodec;
use lm::LM;

#[pymodule]
fn fish_speech(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FireflyCodec>()?;
    m.add_class::<LM>()?;
    Ok(())
}
