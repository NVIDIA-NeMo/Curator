use pyo3::types::PyDict;
use pyo3::{prelude::*, types::IntoPyDict};
use rust_code_analysis::{metrics, FuncSpace, ParserTrait};
use std::path::Path;

use lazy_regex::Regex;

use tree_sitter::{Language, Parser};
use tree_sitter_java::language as java_language;
use tree_sitter_javascript::language as javascript_language;
use tree_sitter_mozcpp::language as cpp_language;
use tree_sitter_python::language as python_language;
use tree_sitter_rust::language as rust_language;
use tree_sitter_typescript::language_typescript as typescript_language;

const LANGUAGE_PYTHON: &str = "Python";
const LANGUAGE_RUST: &str = "Rust";
const LANGUAGE_CPP: &str = "C++";
const LANGUAGE_C: &str = "C";
const LANGUAGE_JAVA: &str = "Java";
const LANGUAGE_JAVASCRIPT: &str = "Javascript";
const LANGUAGE_TYPESCRIPT: &str = "Typescript";

fn get_language_regex(language_name: &str) -> Option<Regex> {
    match language_name {
        LANGUAGE_PYTHON => Some(Regex::new(r"(?si)```(?:python|py|\n)\s*(.*?)```").unwrap()),
        LANGUAGE_CPP => Some(Regex::new(r"(?si)```(?:cpp|c\+\+|\n)\s*(.*?)```").unwrap()),
        LANGUAGE_C => Some(Regex::new(r"(?si)```(?:c\n)\s*(.*?)```").unwrap()),
        LANGUAGE_RUST => Some(Regex::new(r"(?si)```(?:rust|rs\n)\s*(.*?)```").unwrap()),
        LANGUAGE_JAVA => Some(Regex::new(r"(?si)```(?:java\n)\s*(.*?)```").unwrap()),
        LANGUAGE_JAVASCRIPT => Some(Regex::new(r"(?si)```(?:javascript|js\n)\s*(.*?)```").unwrap()),
        LANGUAGE_TYPESCRIPT => Some(Regex::new(r"(?si)```(?:typescript|ts\n)\s*(.*?)```").unwrap()),
        _ => None,
    }
}
pub fn extract_code_block(text: &str, language_name: &str) -> Option<String> {
    if let Some(re) = get_language_regex(language_name) {
        if let Some(captures) = re.captures(text) {
            if let Some(block) = captures.get(1) {
                return Some(block.as_str().to_string());
            }
        }
    } else {
        eprintln!("No regex found");
    }
    None
}

pub fn has_parsing_errors(code: &str, language_name: &str) -> bool {
    match get_ts_grammar(language_name) {
        Some(lang) => {
            let mut parser = Parser::new();
            parser.set_language(&lang).expect("Error loading language");
            let tree = parser.parse(code, None);
            match tree {
                Some(tree) => {
                    let root_node = tree.root_node();
                    // If the root node contains errors, return true
                    root_node.has_error()
                }
                None => true, // If the tree is None, it's a parsing failure
            }
        }
        None => true,
    }
}

pub fn get_ts_grammar(language_name: &str) -> Option<Language> {
    match language_name {
        LANGUAGE_PYTHON => Some(python_language()),
        LANGUAGE_CPP | LANGUAGE_C => Some(cpp_language()),
        LANGUAGE_RUST => Some(rust_language()),
        LANGUAGE_JAVA => Some(java_language()),
        LANGUAGE_JAVASCRIPT => Some(javascript_language()),
        LANGUAGE_TYPESCRIPT => Some(typescript_language()),
        _ => None, // Unsupported file extensions
    }
}
pub fn get_metrics(language_name: &str, source_code: &str) -> Option<FuncSpace> {
    let path = Path::new("foo.c");
    let source_as_vec = source_code.as_bytes().to_vec();
    let metrics = match language_name {
        LANGUAGE_PYTHON => {
            let parser = rust_code_analysis::PythonParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        LANGUAGE_CPP | LANGUAGE_C => {
            let parser = rust_code_analysis::CppParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        LANGUAGE_JAVA => {
            let parser = rust_code_analysis::JavaParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        LANGUAGE_JAVASCRIPT => {
            let parser = rust_code_analysis::JavascriptParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        LANGUAGE_TYPESCRIPT => {
            let parser = rust_code_analysis::TypescriptParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        _ => None, // Unsupported file extensions
    };
    metrics
}

pub fn get_mz_metrics(language_name: &str, source_code: &str) -> Vec<(&'static str, f64)> {
    // The path to a dummy file used to contain the source code
    let mut result: Vec<(&'static str, f64)> = Vec::new();
    if let Some(mzmetrics) = get_metrics(language_name, source_code) {
        let metrics = mzmetrics.metrics;
        result.push(("cyclomatic", metrics.cyclomatic.cyclomatic_average()));
        result.push((
            "cognitive_complexity",
            metrics.cognitive.cognitive_average(),
        ));
        result.push(("exits average", metrics.nexits.exit_average()));
        // Maintainability Index with Visual Studio formula
        result.push(("maintainability_index", metrics.mi.mi_visual_studio()));
        result.push(("halstead_difficulty", metrics.halstead.difficulty()));
        result.push(("# comments", metrics.loc.cloc()));
        result.push(("# comments per space", metrics.loc.cloc_average()));
        result.push(("# blank lines", metrics.loc.blank()));
        result.push(("# blank lines per space", metrics.loc.blank_average()));
        result.push(("# args average", metrics.nargs.nargs_average()));
        result.push(("functions/closures per space", metrics.nom.average()));
        let cda = metrics.npa.total_cda();
        if !cda.is_nan() {
            // Interface/Class data accessibility
            result.push(("total cda", metrics.npa.total_cda()));

            // is_disabled function is private in wmc, let's use it only when
            // CDA/COA is active.
            // Weighted Methods Cyclomatic Complexity Per Class
            result.push(("total_wmc", metrics.wmc.total_wmc()));
        }
        let coa = metrics.npm.total_coa();
        if !coa.is_nan() {
            // Interface/Class operation accessibility
            result.push(("total coa", metrics.npm.total_coa()));
        }
        if has_parsing_errors(source_code, language_name) {
            result.push(("parsed_ok", 0.));
        } else {
            result.push(("parsed_ok", 1.));
        }
    }
    result
}

// pub fn get_mz_metrics(language_name: &str, source_code: &str) -> Vec<(&'static str, f64)> {
#[pyfunction]
#[pyo3(name = "get_all_metrics")]
fn get_mz_metrics_b<'a>(
    py: Python<'a>,
    language_name: &'a str,
    source_code: &'a str,
) -> Bound<'a, PyDict> {
    let metrics = get_mz_metrics(language_name, source_code);
    let dict = PyDict::new_bound(py);
    for (k, v) in &metrics {
        let _ = dict.set_item(k, v.into_py(py));
    }
    dict.into_py_dict_bound(py)
}

#[pymodule]
fn software_metrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_mz_metrics_b, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_block() {
        let expected = "EXTRACT ME!\n".to_string();
        assert_eq!(
            extract_code_block("```python\nEXTRACT ME!\n```", LANGUAGE_PYTHON).unwrap(),
            expected
        );
        assert_eq!(
            extract_code_block("```py\nEXTRACT ME!\n```", LANGUAGE_PYTHON).unwrap(),
            expected
        );
        assert_eq!(
            extract_code_block("```rust\nEXTRACT ME!\n```", LANGUAGE_RUST).unwrap(),
            expected
        );
        assert_eq!(
            extract_code_block("```rs\nEXTRACT ME!\n```", LANGUAGE_RUST).unwrap(),
            expected
        );
        assert_eq!(
            extract_code_block("```c\nEXTRACT ME!\n```", LANGUAGE_C).unwrap(),
            expected
        );
        assert_eq!(
            extract_code_block("```cpp\nEXTRACT ME!\n```", LANGUAGE_CPP).unwrap(),
            expected
        );
        assert_eq!(
            extract_code_block("```c++\nEXTRACT ME!\n```", LANGUAGE_CPP).unwrap(),
            expected
        );
        assert_eq!(
            extract_code_block("```C++\nEXTRACT ME!\n```", LANGUAGE_CPP).unwrap(),
            expected
        );
    }

    #[test]
    fn test_python_parsing() {
        assert!(!has_parsing_errors(
            "print('Hello, World!')\n",
            LANGUAGE_PYTHON
        ));
        assert!(has_parsing_errors(
            "print(Hello, World!')\n",
            LANGUAGE_PYTHON
        ));
    }
    #[test]
    fn test_c_parsing() {
        assert!(!has_parsing_errors(
            "#include <stdio.h>\nint main() {printf(\"Hello, World!\"); return 0;}",
            LANGUAGE_C
        ));
        assert!(has_parsing_errors(
            "#include <stdio.h\nint main() {printf(\"Hello, World!\"); return 0;}",
            LANGUAGE_C
        ));
    }
    #[test]
    fn test_cpp_parsing() {
        assert!(!has_parsing_errors(
            "#include <iostream>\nint main() {std::cout << \"Hello, world!\n\";}",
            LANGUAGE_CPP
        ));
        assert!(has_parsing_errors(
            "#include iostream>\nint main() {std::cout << \"Hello, world!\n\";}",
            LANGUAGE_CPP
        ));
    }
    #[test]
    fn test_java_parsing() {
        assert!(!has_parsing_errors(
            "public class Example {
             public static void main(String[] args) {
                System.out.println(\"Hello World!\");
             }
            }",
            LANGUAGE_JAVA
        ));
        assert!(has_parsing_errors(
            "publicclass Example {
             public static void main(String[] args) {
                System.out.println(\"Hello World!\");
             }
            }",
            LANGUAGE_JAVA
        ));
    }
    #[test]
    fn test_js_parsing() {
        assert!(!has_parsing_errors(
            "console.log('Hello World');",
            LANGUAGE_JAVASCRIPT
        ));
        assert!(has_parsing_errors(
            "console..log('Hello World');",
            LANGUAGE_JAVASCRIPT
        ));
    }
    #[test]
    fn test_rust_parsing() {
        assert!(!has_parsing_errors(
            "function main(): void {console.log(\"Hello, world!\");} main();",
            LANGUAGE_TYPESCRIPT
        ));
        assert!(has_parsing_errors(
            "function main(: void {console.log(\"Hello, world!\");} main();",
            LANGUAGE_TYPESCRIPT
        ));
    }
}
