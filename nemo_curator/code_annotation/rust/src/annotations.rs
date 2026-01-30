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

//! Minimal annotation functions for code data curation using PyArrow tables.
//!
//! Includes: detect_language, basic, tokenize, software_metrics, opencoder_rs

use bpe_openai;
use hyperpolyglot;
use lazy_regex::Regex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use software_metrics::{get_metrics, has_parsing_errors};
use std::cmp::min;
use std::io::BufReader;
use tiktoken_rs;

mod vendored_loc;

const CODE_COLUMN_NAME: &str = "content";
const LANGUAGE_COL_NAME: &str = "language";
const FILENAME_COLUMN_NAME: &str = "representative_filename";

/// Extract a string column from PyArrow table as Vec<Option<String>>
fn extract_string_column(py: Python<'_>, table: &PyObject, col_name: &str) -> PyResult<Vec<Option<String>>> {
    let column = table.call_method1(py, "column", (col_name,))?;
    let py_list = column.call_method0(py, "to_pylist")?;
    let list: Vec<Option<String>> = py_list.extract(py)?;
    Ok(list)
}

/// Check if a column exists in the table
fn has_column(py: Python<'_>, table: &PyObject, col_name: &str) -> PyResult<bool> {
    let column_names = table.getattr(py, "column_names")?;
    let names: Vec<String> = column_names.extract(py)?;
    Ok(names.contains(&col_name.to_string()))
}

/// Add a string column to PyArrow table
fn add_string_column(
    py: Python<'_>,
    table: PyObject,
    name: &str,
    data: Vec<Option<String>>,
) -> PyResult<PyObject> {
    let pa = py.import_bound("pyarrow")?;
    let array = pa.call_method1("array", (data,))?;
    table.call_method1(py, "append_column", (name, array))
}

/// Add a f32 column to PyArrow table
fn add_f32_column(
    py: Python<'_>,
    table: PyObject,
    name: &str,
    data: Vec<f32>,
) -> PyResult<PyObject> {
    let pa = py.import_bound("pyarrow")?;
    let array = pa.call_method1("array", (data,))?;
    table.call_method1(py, "append_column", (name, array))
}

/// Add an optional f32 column to PyArrow table
fn add_opt_f32_column(
    py: Python<'_>,
    table: PyObject,
    name: &str,
    data: Vec<Option<f32>>,
) -> PyResult<PyObject> {
    let pa = py.import_bound("pyarrow")?;
    let array = pa.call_method1("array", (data,))?;
    table.call_method1(py, "append_column", (name, array))
}

/// Add a bool column to PyArrow table
fn add_bool_column(
    py: Python<'_>,
    table: PyObject,
    name: &str,
    data: Vec<bool>,
) -> PyResult<PyObject> {
    let pa = py.import_bound("pyarrow")?;
    let array = pa.call_method1("array", (data,))?;
    table.call_method1(py, "append_column", (name, array))
}

/// Add an optional u32 column to PyArrow table
fn add_opt_u32_column(
    py: Python<'_>,
    table: PyObject,
    name: &str,
    data: Vec<Option<u32>>,
) -> PyResult<PyObject> {
    let pa = py.import_bound("pyarrow")?;
    let array = pa.call_method1("array", (data,))?;
    table.call_method1(py, "append_column", (name, array))
}

/// Add an optional u64 column to PyArrow table
fn add_opt_u64_column(
    py: Python<'_>,
    table: PyObject,
    name: &str,
    data: Vec<Option<u64>>,
) -> PyResult<PyObject> {
    let pa = py.import_bound("pyarrow")?;
    let array = pa.call_method1("array", (data,))?;
    table.call_method1(py, "append_column", (name, array))
}

/// Main entry point for processing annotations
pub fn process_annotations(
    py: Python<'_>,
    function_dict: &Bound<'_, PyDict>,
    table: PyObject,
) -> PyResult<PyObject> {
    // Check required column exists
    if !has_column(py, &table, CODE_COLUMN_NAME)? {
        return Err(PyValueError::new_err(format!(
            "Table must have '{}' column",
            CODE_COLUMN_NAME
        )));
    }

    let mut current_table = table;

    // Process each requested function
    for (func_name, args) in function_dict.iter() {
        let func_name_str: String = func_name.extract()?;
        let args_dict: &Bound<'_, PyDict> = args.downcast()?;

        current_table = match func_name_str.as_str() {
            "detect_language" => detect_language(py, current_table)?,
            "basic" => basic(py, current_table, args_dict)?,
            "software_metrics" => software_metrics(py, current_table)?,
            "opencoder_rs" => opencoder_rs(py, current_table)?,
            "tokenize" => tokenize(py, current_table, args_dict)?,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown function: {}",
                    func_name_str
                )));
            }
        };
    }

    Ok(current_table)
}

/// Detect programming language using hyperpolyglot
fn detect_language(py: Python<'_>, table: PyObject) -> PyResult<PyObject> {
    let codes = extract_string_column(py, &table, CODE_COLUMN_NAME)?;

    let filenames = if has_column(py, &table, FILENAME_COLUMN_NAME)? {
        extract_string_column(py, &table, FILENAME_COLUMN_NAME)?
    } else {
        vec![None; codes.len()]
    };

    let mut languages: Vec<Option<String>> = Vec::with_capacity(codes.len());
    let mut detectors: Vec<Option<String>> = Vec::with_capacity(codes.len());

    for (code_opt, fname_opt) in codes.iter().zip(filenames.iter()) {
        if let Some(code) = code_opt {
            match language_detect(fname_opt.as_deref(), code) {
                Ok(Some(detection)) => {
                    languages.push(Some(detection.language().to_string()));
                    detectors.push(Some(detection.variant().to_string()));
                }
                _ => {
                    languages.push(None);
                    detectors.push(None);
                }
            }
        } else {
            languages.push(None);
            detectors.push(None);
        }
    }

    let table = add_string_column(py, table, LANGUAGE_COL_NAME, languages)?;
    add_string_column(py, table, "language_detector", detectors)
}

/// Language detection via hyperpolyglot
fn language_detect(
    filename: Option<&str>,
    content: &str,
) -> Result<Option<hyperpolyglot::Detection>, std::io::Error> {
    let candidate = filename
        .and_then(|f| hyperpolyglot::detectors::get_language_from_filename(f));
    if let Some(candidate) = candidate {
        return Ok(Some(hyperpolyglot::Detection::Filename(candidate)));
    }

    let extension = filename.and_then(|f| hyperpolyglot::detectors::get_extension(f));
    let candidates = extension
        .map(|ext| hyperpolyglot::detectors::get_languages_from_extension(ext))
        .unwrap_or_default();

    if candidates.len() == 1 {
        return Ok(Some(hyperpolyglot::Detection::Extension(candidates[0])));
    }

    let mut reader = BufReader::new(content.as_bytes());
    let candidates = hyperpolyglot::filter_candidates(
        candidates,
        hyperpolyglot::detectors::get_languages_from_shebang(&mut reader)?,
    );
    if candidates.len() == 1 {
        return Ok(Some(hyperpolyglot::Detection::Shebang(candidates[0])));
    }

    let content = hyperpolyglot::truncate(content);
    let candidates = if candidates.len() > 1 {
        if let Some(ext) = extension {
            let langs = hyperpolyglot::detectors::get_languages_from_heuristics(&ext, &candidates, &content);
            hyperpolyglot::filter_candidates(candidates, langs)
        } else {
            candidates
        }
    } else {
        candidates
    };

    match candidates.len() {
        1 => Ok(Some(hyperpolyglot::Detection::Heuristics(candidates[0]))),
        _ => Ok(Some(hyperpolyglot::Detection::Classifier(
            hyperpolyglot::detectors::classify(&content, &candidates),
        ))),
    }
}

/// Basic statistics: byte count, UTF-8 validation, line stats, patterns
fn basic(py: Python<'_>, table: PyObject, args: &Bound<'_, PyDict>) -> PyResult<PyObject> {
    let xml_search_len: usize = args
        .get_item("xml_header_search_length")?
        .map(|v| v.extract().unwrap_or(100))
        .unwrap_or(100);

    let codes = extract_string_column(py, &table, CODE_COLUMN_NAME)?;
    let n = codes.len();

    let mut num_bytes: Vec<Option<u32>> = Vec::with_capacity(n);
    let mut valid_utf8: Vec<bool> = Vec::with_capacity(n);
    let mut max_line_length: Vec<Option<u64>> = Vec::with_capacity(n);
    let mut num_lines: Vec<Option<u64>> = Vec::with_capacity(n);
    let mut avg_line_length: Vec<Option<f32>> = Vec::with_capacity(n);
    let mut alpha_percent: Vec<f32> = Vec::with_capacity(n);
    let mut alnum_percent: Vec<f32> = Vec::with_capacity(n);
    let mut contains_xml: Vec<bool> = Vec::with_capacity(n);

    let base64_pattern = Regex::new(r"[a-zA-Z0-9+/\n=]{64,}").unwrap();
    let hex_pattern = Regex::new(r"(?:\b(?:0x|\\x)?[0-9a-fA-F]{2}(?:,|\b\s*)){8,}").unwrap();

    let mut base64_percent: Vec<f32> = Vec::with_capacity(n);
    let mut hex_percent: Vec<f32> = Vec::with_capacity(n);

    for code_opt in &codes {
        match code_opt {
            Some(code) => {
                let len = code.len();
                num_bytes.push(Some(len as u32));
                valid_utf8.push(true);

                let lines: Vec<&str> = code.lines().collect();
                let line_count = lines.len();
                num_lines.push(Some(line_count as u64));

                let max_len = lines.iter().map(|l| l.len()).max().unwrap_or(0);
                max_line_length.push(Some(max_len as u64));
                avg_line_length.push(Some(len as f32 / line_count.max(1) as f32));

                let slen = len as f32;
                alpha_percent.push(code.chars().filter(|c| c.is_alphabetic()).count() as f32 / slen.max(1.0));
                alnum_percent.push(code.chars().filter(|c| c.is_alphanumeric()).count() as f32 / slen.max(1.0));

                base64_percent.push(
                    base64_pattern.find_iter(code).map(|m| m.len()).sum::<usize>() as f32 / slen.max(1.0)
                );
                hex_percent.push(
                    hex_pattern.find_iter(code).map(|m| m.len()).sum::<usize>() as f32 / slen.max(1.0)
                );

                // Find a valid UTF-8 character boundary for slicing
                let mut search_end = min(xml_search_len, len);
                while search_end > 0 && !code.is_char_boundary(search_end) {
                    search_end -= 1;
                }
                contains_xml.push(code[..search_end].contains("<?xml version="));
            }
            None => {
                num_bytes.push(None);
                valid_utf8.push(false);
                max_line_length.push(None);
                num_lines.push(None);
                avg_line_length.push(None);
                alpha_percent.push(0.0);
                alnum_percent.push(0.0);
                base64_percent.push(0.0);
                hex_percent.push(0.0);
                contains_xml.push(false);
            }
        }
    }

    let table = add_opt_u32_column(py, table, "basic_num_bytes", num_bytes)?;
    let table = add_bool_column(py, table, "basic_valid_utf8", valid_utf8)?;
    let table = add_opt_u64_column(py, table, "basic_max_line_length", max_line_length)?;
    let table = add_opt_u64_column(py, table, "basic_num_lines", num_lines)?;
    let table = add_opt_f32_column(py, table, "basic_average_line_length", avg_line_length)?;
    let table = add_f32_column(py, table, "basic_alpha_percent", alpha_percent)?;
    let table = add_f32_column(py, table, "basic_alnum_percent", alnum_percent)?;
    let table = add_f32_column(py, table, "basic_base64_percent", base64_percent)?;
    let table = add_f32_column(py, table, "basic_hex_percent", hex_percent)?;
    add_bool_column(py, table, "basic_contains_xml_header", contains_xml)
}

/// Software metrics: complexity, halstead, LOC metrics
fn software_metrics(py: Python<'_>, table: PyObject) -> PyResult<PyObject> {
    let codes = extract_string_column(py, &table, CODE_COLUMN_NAME)?;
    let languages = extract_string_column(py, &table, LANGUAGE_COL_NAME)?;
    let n = codes.len();

    let mut cyclomatic: Vec<f32> = vec![0.0; n];
    let mut cognitive: Vec<f32> = vec![0.0; n];
    let mut maintainability: Vec<f32> = vec![0.0; n];
    let mut halstead_diff: Vec<f32> = vec![0.0; n];
    let mut comment_lines: Vec<f32> = vec![0.0; n];
    let mut blank_lines: Vec<f32> = vec![0.0; n];
    let mut parsed_ok: Vec<bool> = vec![false; n];

    for (idx, (code_opt, lang_opt)) in codes.iter().zip(languages.iter()).enumerate() {
        if let (Some(code), Some(lang)) = (code_opt, lang_opt) {
            parsed_ok[idx] = !has_parsing_errors(code, lang);
            if let Some(mzmetrics) = get_metrics(lang, code) {
                let m = mzmetrics.metrics;
                cyclomatic[idx] = m.cyclomatic.cyclomatic_average() as f32;
                cognitive[idx] = m.cognitive.cognitive_average() as f32;
                maintainability[idx] = m.mi.mi_visual_studio() as f32;
                halstead_diff[idx] = m.halstead.difficulty() as f32;
                comment_lines[idx] = m.loc.cloc() as f32;
                blank_lines[idx] = m.loc.blank() as f32;
            }
        }
    }

    let table = add_f32_column(py, table, "software_metrics_cyclomatic_complexity", cyclomatic)?;
    let table = add_f32_column(py, table, "software_metrics_cognitive_complexity", cognitive)?;
    let table = add_f32_column(py, table, "software_metrics_maintainability_index", maintainability)?;
    let table = add_f32_column(py, table, "software_metrics_halstead_difficulty", halstead_diff)?;
    let table = add_f32_column(py, table, "software_metrics_comment_lines", comment_lines)?;
    let table = add_f32_column(py, table, "software_metrics_blank_lines", blank_lines)?;
    add_bool_column(py, table, "software_metrics_parsed_ok", parsed_ok)
}

/// OpenCoder Rust: comment line/char fractions
fn opencoder_rs(py: Python<'_>, table: PyObject) -> PyResult<PyObject> {
    let codes = extract_string_column(py, &table, CODE_COLUMN_NAME)?;
    let languages = extract_string_column(py, &table, LANGUAGE_COL_NAME)?;

    let mut comment_lines_frac: Vec<Option<f32>> = Vec::with_capacity(codes.len());
    let mut comment_chars_frac: Vec<Option<f32>> = Vec::with_capacity(codes.len());

    for (code_opt, lang_opt) in codes.iter().zip(languages.iter()) {
        if let (Some(code), Some(lang)) = (code_opt, lang_opt) {
            let count = vendored_loc::comment_frac(code, lang);
            let lines = if count.lines == 0 { 1 } else { count.lines };
            comment_lines_frac.push(Some(count.comment as f32 / lines as f32));

            let total_chars = count.comment_chars + count.code_chars;
            let total = if total_chars == 0 { 1 } else { total_chars };
            comment_chars_frac.push(Some(count.comment_chars as f32 / total as f32));
        } else {
            comment_lines_frac.push(None);
            comment_chars_frac.push(None);
        }
    }

    let table = add_opt_f32_column(py, table, "ors_comment_lines_frac", comment_lines_frac)?;
    add_opt_f32_column(py, table, "ors_comment_chars_frac", comment_chars_frac)
}

/// Tokenize using BPE tokenizer
fn tokenize(py: Python<'_>, table: PyObject, args: &Bound<'_, PyDict>) -> PyResult<PyObject> {
    let tokenizer_name: String = args
        .get_item("tokenizer_name")?
        .map(|v| v.extract().unwrap_or("github_o200k_base".to_string()))
        .unwrap_or("github_o200k_base".to_string());

    let codes = extract_string_column(py, &table, CODE_COLUMN_NAME)?;
    let mut num_tokens: Vec<Option<u64>> = Vec::with_capacity(codes.len());

    if tokenizer_name == "tiktoken_o200k_base" {
        let tok = tiktoken_rs::o200k_base().unwrap();
        for code_opt in &codes {
            match code_opt {
                Some(code) => {
                    num_tokens.push(Some(tok.encode_with_special_tokens(code).len() as u64));
                }
                None => num_tokens.push(None),
            }
        }
    } else {
        // Default to github's o200k_base
        let tok = bpe_openai::o200k_base();
        for code_opt in &codes {
            match code_opt {
                Some(code) => {
                    num_tokens.push(Some(tok.encode(code).len() as u64));
                }
                None => num_tokens.push(None),
            }
        }
    }

    let col_name = format!("num_tokens_{}", tokenizer_name);
    add_opt_u64_column(py, table, &col_name, num_tokens)
}
