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

/// This is a modified version of the loc crate:
/// https://docs.rs/loc/latest/src/loc/lib.rs.html#1-743
/// Refactored to work from strings with language defined externally.
use smallvec::*;
use std::cmp::{max, min};

#[derive(Debug, PartialEq, Default, Clone)]
pub struct Count {
    pub code: u32,
    pub comment: u32,
    pub blank: u32,
    pub lines: u32,
    pub code_chars: usize,
    pub comment_chars: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub enum Lang {
    ActionScript,
    Ada,
    Agda,
    AmbientTalk,
    Asp,
    AspNet,
    Assembly,
    Autoconf,
    Awk,
    Batch,
    BourneShell,
    C,
    CCppHeader,
    CMake,
    CSharp,
    CShell,
    Clojure,
    CoffeeScript,
    ColdFusion,
    ColdFusionScript,
    Coq,
    Cpp,
    Css,
    CUDA,
    CUDAHeader,
    D,
    Dart,
    DeviceTree,
    Docker,
    Elixir,
    Elm,
    Erlang,
    Forth,
    FortranLegacy,
    FortranModern,
    FSharp,
    Gherkin,
    Glsl,
    Go,
    Groovy,
    Handlebars,
    Haskell,
    Hex,
    Html,
    INI,
    Idris,
    IntelHex,
    Isabelle,
    Jai,
    Java,
    JavaScript,
    Json,
    Jsx,
    Julia,
    Kotlin,
    Less,
    LinkerScript,
    Lean,
    Lisp,
    Lua,
    Make,
    Makefile,
    Markdown,
    Mustache,
    Nim,
    Nix,
    OCaml,
    ObjectiveC,
    ObjectiveCpp,
    OpenCl,
    Oz,
    Pascal,
    Perl,
    Php,
    Polly,
    PowerShell,
    Prolog,
    Protobuf,
    Puppet,
    PureScript,
    Pyret,
    Python,
    Qcl,
    Qml,
    R,
    Razor,
    ReStructuredText,
    Ruby,
    RubyHtml,
    Rust,
    SaltStack,
    Sass,
    Scala,
    Sml,
    Sql,
    Stylus,
    Swift,
    Tcl,
    Terraform,
    Tex,
    Text,
    Toml,
    TypeScript,
    Tsx,
    UnrealScript,
    VimScript,
    Wolfram,
    XML,
    Yacc,
    Yaml,
    Zig,
    Zsh,
    Haxe,
    Unrecognized,
}
use self::Lang::*;

impl Lang {
    pub fn from_name(name: &str) -> Option<Lang> {
        let lang = match name {
            "ActionScript" => Lang::ActionScript,
            "Ada" => Lang::Ada,
            "Agda" => Lang::Agda,
            "AmbientTalk" => Lang::AmbientTalk,
            "ASP" => Lang::Asp,
            "ASP.NET" => Lang::AspNet,
            "Assembly" => Lang::Assembly,
            "Autoconf" => Lang::Autoconf,
            "Awk" => Lang::Awk,
            "Batch" => Lang::Batch,
            "Bourne Shell" => Lang::BourneShell,
            "C" => Lang::C,
            "C/C++ Header" => Lang::CCppHeader,
            "CMake" => Lang::CMake,
            "C#" => Lang::CSharp,
            "C Shell" => Lang::CShell,
            "Clojure" => Lang::Clojure,
            "CoffeeScript" => Lang::CoffeeScript,
            "ColdFusion" => Lang::ColdFusion,
            "ColdFusionScript" => Lang::ColdFusionScript,
            "Coq" => Lang::Coq,
            "C++" => Lang::Cpp,
            "CSS" => Lang::Css,
            "CUDA" => Lang::CUDA,
            "CUDA Header" => Lang::CUDAHeader,
            "D" => Lang::D,
            "Dart" => Lang::Dart,
            "DeviceTree" => Lang::DeviceTree,
            "Docker" => Lang::Docker,
            "Elixir" => Lang::Elixir,
            "Elm" => Lang::Elm,
            "Erlang" => Lang::Erlang,
            "Forth" => Lang::Forth,
            "FORTRAN Legacy" => Lang::FortranLegacy,
            "FORTRAN Modern" => Lang::FortranModern,
            "F#" => Lang::FSharp,
            "Gherkin" => Lang::Gherkin,
            "GLSL" => Lang::Glsl,
            "Go" => Lang::Go,
            "Groovy" => Lang::Groovy,
            "Handlebars" => Lang::Handlebars,
            "Haskell" => Lang::Haskell,
            "Hex" => Lang::Hex,
            "HTML" => Lang::Html,
            "INI" => Lang::INI,
            "Idris" => Lang::Idris,
            "Intel Hex" => Lang::IntelHex,
            "Isabelle" => Lang::Isabelle,
            "Jai" => Lang::Jai,
            "Java" => Lang::Java,
            "JavaScript" => Lang::JavaScript,
            "JSON" => Lang::Json,
            "Jsx" => Lang::Jsx,
            "Julia" => Lang::Julia,
            "Kotlin" => Lang::Kotlin,
            "Less" => Lang::Less,
            "LinkerScript" => Lang::LinkerScript,
            "Lean" => Lang::Lean,
            "Lisp" => Lang::Lisp,
            "Lua" => Lang::Lua,
            "Make" => Lang::Make,
            "Makefile" => Lang::Makefile,
            "Markdown" => Lang::Markdown,
            "Mustache" => Lang::Mustache,
            "Nim" => Lang::Nim,
            "Nix" => Lang::Nix,
            "OCaml" => Lang::OCaml,
            "Objective-C" => Lang::ObjectiveC,
            "Objective-C++" => Lang::ObjectiveCpp,
            "OpenCL" => Lang::OpenCl,
            "Oz" => Lang::Oz,
            "Pascal" => Lang::Pascal,
            "Perl" => Lang::Perl,
            "PHP" => Lang::Php,
            "Polly" => Lang::Polly,
            "PowerShell" => Lang::PowerShell,
            "Prolog" => Lang::Prolog,
            "Protobuf" => Lang::Protobuf,
            "Puppet" => Lang::Puppet,
            "PureScript" => Lang::PureScript,
            "Pyret" => Lang::Pyret,
            "Python" => Lang::Python,
            "Qcl" => Lang::Qcl,
            "Qml" => Lang::Qml,
            "R" => Lang::R,
            "Razor" => Lang::Razor,
            "reStructuredText" => Lang::ReStructuredText,
            "Ruby" => Lang::Ruby,
            "RubyHtml" => Lang::RubyHtml,
            "Rust" => Lang::Rust,
            "SaltStack" => Lang::SaltStack,
            "Sass" => Lang::Sass,
            "Scala" => Lang::Scala,
            "SML" => Lang::Sml,
            "SQL" => Lang::Sql,
            "Stylus" => Lang::Stylus,
            "Swift" => Lang::Swift,
            "Tcl" => Lang::Tcl,
            "Terraform" => Lang::Terraform,
            "TeX" => Lang::Tex,
            "Plain Text" => Lang::Text,
            "Toml" => Lang::Toml,
            "TypeScript" => Lang::TypeScript,
            "Typescript JSX" => Lang::Tsx,
            "UnrealScript" => Lang::UnrealScript,
            "VimL" => Lang::VimScript,
            "Wolfram" => Lang::Wolfram,
            "XML" => Lang::XML,
            "Yacc" => Lang::Yacc,
            "YAML" => Lang::Yaml,
            "Zig" => Lang::Zig,
            "Z Shell" => Lang::Zsh,
            "Haxe" => Lang::Haxe,
            _ => Lang::Unrecognized,
        };
        match lang {
            Lang::Unrecognized => None,
            _ => Some(lang),
        }
    }
}

pub fn counter_config_for_lang<'a>(
    lang: Lang,
) -> (SmallVec<[&'a str; 3]>, SmallVec<[(&'a str, &'a str); 3]>) {
    let c_style = (smallvec!["//"], smallvec![("/*", "*/")]);
    let html_style = (smallvec![], smallvec![("<!--", "-->")]);
    let ml_style = (smallvec![], smallvec![("(*", "*)")]);
    let no_comments = (smallvec![], smallvec![]);
    let prolog_style = (smallvec!["%"], smallvec![("/*", "*/")]);
    let sh_style = (smallvec!["#"], smallvec![]);

    let ctuple = match lang {
        Ada => (smallvec!["--"], smallvec![]),
        Batch => (smallvec!["REM"], smallvec![]),
        Erlang | Tex => (smallvec!["%"], smallvec![]),
        FortranModern => (smallvec!["!"], smallvec![]),
        INI => (smallvec![";"], smallvec![]),
        Protobuf | Zig => (smallvec!["//"], smallvec![]),
        VimScript => (smallvec!["\""], smallvec![]),
        Terraform => (smallvec!["#"], smallvec![("/*", "*/")]),
        Nix => (smallvec!["#"], smallvec![("/*", "*/")]),
        Assembly => (smallvec!["#"], smallvec![("/*", "*/")]),
        CMake => (smallvec!["#"], smallvec![("#[[", "]]")]),
        CoffeeScript => (smallvec!["#"], smallvec![("###", "###")]),
        D => (smallvec!["//"], smallvec![("/*", "*/")]),
        Docker => (smallvec!["#"], smallvec![]),
        Forth => (smallvec!["\\"], smallvec![("(", ")")]),
        FSharp => (smallvec!["//"], smallvec![("(*", "*)")]),
        Julia => (smallvec!["#"], smallvec![("#=", "=#")]),
        Lisp => (smallvec![";"], smallvec![("#|", "|#")]),
        Lean => (smallvec!["--"], smallvec![("/-", "-/")]),
        Lua => (smallvec!["--"], smallvec![("--[[", "]]")]),
        Perl => (smallvec!["#"], smallvec![("=pod", "=cut")]),
        Puppet => (smallvec!["#"], smallvec![]),
        Pyret => (smallvec!["#"], smallvec![("#|", "|#")]),
        Python => (
            smallvec!["#"],
            smallvec![("'''", "'''"), ("\"\"\"", "\"\"\"")],
        ),
        Ruby => (smallvec!["#"], smallvec![("=begin", "=end")]),
        Sql => (smallvec!["--"], smallvec![("/*", "*/")]),
        Haskell | Idris | Agda | PureScript | Elm => (smallvec!["--"], smallvec![("{-", "-}")]),
        ColdFusion => (smallvec![], smallvec![("<!---", "--->")]),
        Mustache => (smallvec![], smallvec![("{{!", "}}")]),
        Asp => (smallvec!["'", "REM"], smallvec![]),
        AspNet => (smallvec![], smallvec![("<!--", "-->"), ("<%--", "-->")]),
        Autoconf => (smallvec!["#", "dnl"], smallvec![]),
        Clojure => (smallvec![";", "#"], smallvec![]),
        FortranLegacy => (smallvec!["c", "C", "!", "*"], smallvec![]),
        Handlebars => (smallvec![], smallvec![("<!--", "-->"), ("{{!", "}}")]),
        Php => (smallvec!["#", "//"], smallvec![("/*", "*/")]),
        PowerShell => (smallvec!["#"], smallvec![("<#", "#>")]),
        Isabelle => {
            (
                smallvec!["--"],
                smallvec![
                    ("{*", "*}"),
                    ("(*", "*)"),
                    ("‹", "›"),
                    ("\\<open>", "\\<close>"),
                ],
            )
        }
        Razor => (smallvec![], smallvec![("<!--", "-->"), ("@*", "*@")]),
        Pascal => (smallvec!["//", "(*"], smallvec![("{", "}")]),
        Text | Markdown | Json | IntelHex | Hex | ReStructuredText => no_comments,
        Oz | Prolog => prolog_style,
        Coq | Sml | Wolfram | OCaml => ml_style,
        Html | Polly | RubyHtml | XML => html_style,
        BourneShell | Make | Awk | CShell | Gherkin | Makefile | Nim | R | SaltStack | Tcl
        | Toml | Yaml | Zsh | Elixir => sh_style,
        AmbientTalk | C | CCppHeader | Rust | Yacc | ActionScript | ColdFusionScript | Css
        | Cpp | CUDA | CUDAHeader | CSharp | Dart | DeviceTree | Glsl | Go | Jai | Java
        | JavaScript | Jsx | Kotlin | Less | LinkerScript | ObjectiveC | ObjectiveCpp | OpenCl
        | Qcl | Sass | Scala | Swift | TypeScript | Tsx | UnrealScript | Stylus | Qml | Haxe
        | Groovy => c_style,
        Unrecognized => unreachable!(),
    };

    ctuple
}

pub fn comment_frac(code: &str, lang: &str) -> Count {
    match Lang::from_name(lang) {
        None => Count::default(),
        Some(lang) => {
            let (singles, multis) = counter_config_for_lang(lang);
            let mut c = Count::default();
            let mut multi_stack: Vec<(&str, &str)> = vec![];

            'line: for line in code.lines() {
                c.lines += 1;

                let line = line.trim_start();
                let line_char_len = line.chars().count();
                if line.is_empty() {
                    c.blank += 1;
                    continue;
                };

                if multi_stack.is_empty() {
                    for single_start in singles.iter() {
                        if line.starts_with(single_start) {
                            if multis.iter().any(|(m_start, _)| line.starts_with(m_start)) {
                                break;
                            }

                            c.comment += 1;
                            c.comment_chars += line_char_len;
                            continue 'line;
                        }
                    }

                    if multis.is_empty() {
                        c.code += 1;
                        c.code_chars += line_char_len;
                        continue 'line;
                    }
                }

                if multi_stack.is_empty()
                    && !multis
                        .iter()
                        .any(|(start, end)| line.contains(start) || line.contains(end))
                {
                    c.code += 1;
                    c.code_chars += line_char_len;
                    continue 'line;
                }

                let mut pos = 0;
                let mut found_code = 0;
                let line_len = line.len();
                let contains_utf8 = (0..line_len).any(|i| !line.is_char_boundary(i));

                'outer: while pos < line_len {
                    for multi in multis.iter() {
                        let (start, end) = multi;
                        let start_len = start.len();
                        let end_len = end.len();

                        if contains_utf8 {
                            for i in pos..pos + min(max(start_len, end_len) + 1, line_len - pos) {
                                if !line.is_char_boundary(i) {
                                    pos += 1;
                                    continue 'outer;
                                }
                            }
                        }

                        if pos + start_len <= line_len
                            && &line[pos..pos + start_len] == *start
                            && (start != end || multi_stack.is_empty())
                        {
                            pos += start_len;
                            multi_stack.push(*multi);
                            continue;
                        }

                        if !multi_stack.is_empty() {
                            let end_len = multi_stack.last().expect("stack last").1.len();
                            let end_str = multi_stack.last().expect("stack last").1;
                            if pos + end_len <= line_len && &line[pos..pos + end_len] == end_str {
                                let _ = multi_stack.pop();
                                pos += end_len;
                            }
                        } else if multi_stack.is_empty()
                            && pos < line_len
                            && !&line[pos..pos + 1]
                                .chars()
                                .next()
                                .expect("whitespace check")
                                .is_whitespace()
                        {
                            found_code += 1;
                        }
                    }
                    pos += 1;
                }

                if found_code >= multis.len() {
                    c.code += 1;
                    c.code_chars += line_char_len;
                } else {
                    c.comment += 1;
                    c.comment_chars += line_char_len;
                }
            }

            c
        }
    }
}
