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

//! Proc-macro crate for code_annotation.
//! Provides #[derive(FromHashMap)] and #[register] macros.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields, ItemFn};

/// This macro automatically adds a FromHashMap method to a struct
/// to initialize a registered struct from a hashmap<argname, argvalue>.
#[proc_macro_derive(FromHashMap)]
pub fn from_hashmap_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = if let Data::Struct(data) = &input.data {
        match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("Only structs with named fields are supported"),
        }
    } else {
        panic!("FromHashMap can only be derived for structs");
    };

    let field_init = fields.iter().map(|field| {
        let field_name = &field.ident;
        let field_type = &field.ty;

        quote! {
            #field_name: map.get(stringify!(#field_name))
                .and_then(|v| v.extract::<#field_type>().ok())?
        }
    });

    let expanded = quote! {
        impl #name {
            pub fn from_hashmap(map: &std::collections::HashMap<String, Bound<'_, PyAny>>) -> Option<Self> {
                Some(#name {
                    #(#field_init),*
                })
            }
        }
    };

    TokenStream::from(expanded)
}

/// This macro allows to decide which functions to call at runtime given
/// a hashmap of <function_name, function_args>.
#[proc_macro_attribute]
pub fn register(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_name_str = fn_name.to_string();
    let register_fn_name = format_ident!("register_{}", fn_name);

    let arg_types: Vec<_> = input.sig.inputs.iter().collect();

    let arg_struct_type = match &arg_types[0] {
        syn::FnArg::Typed(arg) => &arg.ty,
        _ => panic!("Expected a typed first argument."),
    };

    let output = quote! {
        #input

        #[ctor::ctor]
        fn #register_fn_name() {
            FN_REGISTRY.lock().unwrap().insert(
                #fn_name_str.to_string(),
                Box::new(|args: HashMap<String, Bound<'_, PyAny>>, df_ref: &mut DataFrame| -> PyResult<()> {
                    let parsed_args = <#arg_struct_type>::from_hashmap(&args).ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "Failed to parse arguments for {}. Expected a valid {}. Received: {:?}",
                            #fn_name_str,
                            stringify!(#arg_struct_type),
                            args,
                        ))
                    })?;

                    #fn_name(parsed_args, df_ref)
                }),
            );
            INPUT_TYPE_REGISTRY.lock().unwrap().insert(
                #fn_name_str.to_string(),
                Box::new(|args: &HashMap<String, Bound<'_, PyAny>>| {
                    <#arg_struct_type>::from_hashmap(args).map(|parsed| Box::new(parsed) as Box<dyn Any>)
                        .ok_or_else(||
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "Failed to parse arguments for {}. Expected a valid {}. Received: {:?}",
                            #fn_name_str,
                            stringify!(#arg_struct_type),
                            args,
                        )))
                }),
            );
        }
    };

    output.into()
}
