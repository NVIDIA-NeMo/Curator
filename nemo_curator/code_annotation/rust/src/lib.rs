// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Minimal code annotation library for NeMo Curator.
//!
//! This module provides Rust-based annotation functions exposed to Python via PyO3:
//! - `detect_language` - Programming language detection via hyperpolyglot
//! - `basic` - Basic statistics (byte count, UTF-8 validation, line stats, patterns)
//! - `tokenize` - BPE tokenization
//! - `software_metrics` - Code complexity metrics via rust-code-analysis
//! - `opencoder_rs` - Comment line/character fractions

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod annotations;
use annotations::process_annotations;

/// Main entry point: accepts PyArrow Table, returns PyArrow Table
#[pyfunction]
fn annotate(py: Python<'_>, function_dict: &Bound<'_, PyDict>, table: PyObject) -> PyResult<PyObject> {
    process_annotations(py, function_dict, table)
}

#[pymodule]
fn _code_annotation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(annotate, m)?)?;
    Ok(())
}
