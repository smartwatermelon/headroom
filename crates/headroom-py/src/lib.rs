//! PyO3 bindings for headroom-core. Exposed to Python as `headroom._core`.

use pyo3::prelude::*;

/// Return the identity string from headroom-core.
#[pyfunction]
fn hello() -> &'static str {
    headroom_core::hello()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
