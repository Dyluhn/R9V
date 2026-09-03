//! Internal proc macros for `r9v-config` (Spec 12 §2).
//!
//! `#[section]` generates strongly typed section metadata from annotated
//! structs. `#[setting]` is the field helper consumed by `#[section]`; used
//! alone it passes its item through unchanged.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{punctuated::Punctuated, spanned::Spanned, Data, DeriveInput, Expr, Lit, Meta, Token};

/// Field helper consumed by `#[section]`.
///
/// When expanded on its own (never, in practice) it passes the input through
/// unchanged so annotation placement stays valid during incremental edits.
#[proc_macro_attribute]
pub fn setting(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

struct SettingMeta {
    kind: Option<String>,
    doc: String,
    default: String,
    range_or_enum: String,
    unit: String,
    mutable: syn::Ident,
    interacts: Vec<String>,
    since: u32,
    renamed_from: String,
}

fn lit_str(expr: &Expr) -> Option<String> {
    if let Expr::Lit(l) = expr {
        if let Lit::Str(s) = &l.lit {
            return Some(s.value());
        }
    }
    None
}

fn str_array(expr: &Expr) -> Option<Vec<String>> {
    if let Expr::Array(a) = expr {
        let mut out = Vec::new();
        for e in &a.elems {
            out.push(lit_str(e)?);
        }
        return Some(out);
    }
    None
}

fn parse_setting_meta(
    field_span: Span,
    metas: Punctuated<Meta, Token![,]>,
) -> syn::Result<SettingMeta> {
    let mut doc: Option<String> = None;
    let mut kind: Option<String> = None;
    let mut default: Option<String> = None;
    let mut range: Option<String> = None;
    let mut values: Option<Vec<String>> = None;
    let mut unit = String::new();
    let mut mutable: Option<syn::Ident> = None;
    let mut interacts: Vec<String> = Vec::new();
    let mut since: u32 = 1;
    let mut renamed_from = String::new();
    for m in metas {
        match m {
            Meta::NameValue(nv) => {
                let key = nv
                    .path
                    .get_ident()
                    .map(|i| i.to_string())
                    .unwrap_or_default();
                match key.as_str() {
                    "kind" => {
                        kind =
                            Some(lit_str(&nv.value).ok_or_else(|| {
                                syn::Error::new(nv.span(), "kind must be a string")
                            })?)
                    }
                    "doc" => {
                        doc =
                            Some(lit_str(&nv.value).ok_or_else(|| {
                                syn::Error::new(nv.span(), "doc must be a string")
                            })?)
                    }
                    "default" => {
                        default = Some(lit_str(&nv.value).ok_or_else(|| {
                            syn::Error::new(nv.span(), "default must be a string")
                        })?)
                    }
                    "range" => {
                        range =
                            Some(lit_str(&nv.value).ok_or_else(|| {
                                syn::Error::new(nv.span(), "range must be a string")
                            })?)
                    }
                    "values" | "enum_values" | "enm" => {
                        values = Some(str_array(&nv.value).ok_or_else(|| {
                            syn::Error::new(nv.span(), "values must be an array of strings")
                        })?);
                    }
                    "unit" => {
                        unit = lit_str(&nv.value)
                            .ok_or_else(|| syn::Error::new(nv.span(), "unit must be a string"))?
                    }
                    "mutable" => {
                        let s = match &nv.value {
                            Expr::Path(p) => p.path.get_ident().cloned(),
                            Expr::Lit(l) => {
                                if let Lit::Str(s) = &l.lit {
                                    Some(syn::Ident::new(&s.value(), Span::call_site()))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        let id = s.ok_or_else(|| {
                            syn::Error::new(nv.span(), "mutable must be Runtime, Reload or Load")
                        })?;
                        let name = id.to_string();
                        if !matches!(name.as_str(), "Runtime" | "Reload" | "Load") {
                            return Err(syn::Error::new(
                                id.span(),
                                "mutable must be Runtime, Reload or Load",
                            ));
                        }
                        mutable = Some(id);
                    }
                    "interacts" => {
                        interacts = str_array(&nv.value).ok_or_else(|| {
                            syn::Error::new(nv.span(), "interacts must be an array of strings")
                        })?;
                    }
                    "since" => {
                        if let Expr::Lit(l) = &nv.value {
                            if let Lit::Int(i) = &l.lit {
                                since = i.base10_parse::<u32>()?;
                            } else {
                                return Err(syn::Error::new(nv.span(), "since must be an integer"));
                            }
                        } else {
                            return Err(syn::Error::new(nv.span(), "since must be an integer"));
                        }
                    }
                    "renamed_from" => {
                        renamed_from = lit_str(&nv.value).ok_or_else(|| {
                            syn::Error::new(nv.span(), "renamed_from must be a string")
                        })?;
                    }
                    other => {
                        return Err(syn::Error::new(
                            field_span,
                            format!("unknown setting key `{other}`"),
                        ))
                    }
                }
            }
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "setting entries must be key = value",
                ))
            }
        }
    }
    let doc = doc.ok_or_else(|| syn::Error::new(field_span, "setting is missing `doc`"))?;
    let default =
        default.ok_or_else(|| syn::Error::new(field_span, "setting is missing `default`"))?;
    let mutable =
        mutable.ok_or_else(|| syn::Error::new(field_span, "setting is missing `mutable`"))?;
    if range.is_some() && values.is_some() {
        return Err(syn::Error::new(
            field_span,
            "setting must pass at most one of `range`, `values`",
        ));
    }
    if doc.trim().is_empty() {
        return Err(syn::Error::new(field_span, "setting doc must not be empty"));
    }
    if default.trim().is_empty() {
        return Err(syn::Error::new(
            field_span,
            "setting default must not be empty",
        ));
    }
    if let Some(range) = &range {
        let Some((low, high)) = range.split_once("..=") else {
            return Err(syn::Error::new(
                field_span,
                "range must use numeric `low..=high` syntax",
            ));
        };
        let (Ok(low), Ok(high)) = (low.parse::<f64>(), high.parse::<f64>()) else {
            return Err(syn::Error::new(field_span, "range bounds must be numeric"));
        };
        if !low.is_finite() || !high.is_finite() || low > high {
            return Err(syn::Error::new(
                field_span,
                "range bounds must be finite and ordered",
            ));
        }
    }
    if let Some(values) = &values {
        if values.is_empty() || values.iter().any(|value| value.is_empty()) {
            return Err(syn::Error::new(
                field_span,
                "values must contain non-empty members",
            ));
        }
        if values
            .iter()
            .enumerate()
            .any(|(index, value)| values[..index].contains(value))
        {
            return Err(syn::Error::new(
                field_span,
                "values must not contain duplicates",
            ));
        }
    }
    let range_or_enum = range
        .or_else(|| values.map(|v| v.join("|")))
        .unwrap_or_default();
    Ok(SettingMeta {
        kind,
        doc,
        default,
        range_or_enum,
        unit,
        mutable,
        interacts,
        since,
        renamed_from,
    })
}

/// Section attribute: `#[section("name")]` or `#[section("name", doc = "...")]`.
///
/// Generates `impl` metadata (`SECTION`, `SETTINGS`, `setting(key)`) from the
/// annotated struct's `#[setting(...)]` fields. The struct itself is emitted
/// unchanged apart from stripping the consumed `#[setting]` helpers.
#[proc_macro_attribute]
pub fn section(attr: TokenStream, item: TokenStream) -> TokenStream {
    let out = section_impl(attr, item);
    match out {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error().into(),
    }
}

fn section_impl(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let parsed: Punctuated<Expr, Token![,]> =
        syn::parse::Parser::parse2(Punctuated::parse_terminated, attr.into())?;
    let mut section_name: Option<String> = None;
    let mut section_doc = String::new();
    for e in parsed {
        match e {
            Expr::Lit(l) => {
                if let Lit::Str(s) = &l.lit {
                    if section_name.is_some() {
                        return Err(syn::Error::new(s.span(), "duplicate section name"));
                    }
                    section_name = Some(s.value());
                } else {
                    return Err(syn::Error::new(
                        Span::call_site(),
                        "section name must be a string literal",
                    ));
                }
            }
            Expr::Assign(a) => {
                let key = match *a.left {
                    Expr::Path(ref p) => p
                        .path
                        .get_ident()
                        .map(|i| i.to_string())
                        .unwrap_or_default(),
                    _ => {
                        return Err(syn::Error::new(
                            a.span(),
                            "unknown section key (expected `doc`)",
                        ))
                    }
                };
                let val = match *a.right {
                    Expr::Lit(ref l) => {
                        if let Lit::Str(s) = &l.lit {
                            s.value()
                        } else {
                            return Err(syn::Error::new(a.span(), "doc must be a string"));
                        }
                    }
                    _ => return Err(syn::Error::new(a.span(), "doc must be a string")),
                };
                if key == "doc" {
                    section_doc = val;
                } else if key == "name" {
                    section_name = Some(val);
                } else {
                    return Err(syn::Error::new(
                        a.span(),
                        "unknown section key (expected `doc`)",
                    ));
                }
            }
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "section takes a name literal and optional doc = \"...\"",
                ))
            }
        }
    }
    let section_name = section_name
        .ok_or_else(|| syn::Error::new(Span::call_site(), "section is missing its name"))?;
    if section_name.is_empty()
        || section_name.split('.').any(|part| {
            part.is_empty() || !part.chars().all(|ch| ch.is_ascii_lowercase() || ch == '_')
        })
    {
        return Err(syn::Error::new(
            Span::call_site(),
            "section name must contain lowercase snake-case segments",
        ));
    }

    let mut input: DeriveInput = syn::parse(item)?;
    let fields = match &mut input.data {
        Data::Struct(s) => match &mut s.fields {
            syn::Fields::Named(f) => f,
            _ => {
                return Err(syn::Error::new(
                    input.ident.span(),
                    "section structs need named fields",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new(
                input.ident.span(),
                "section applies to structs",
            ))
        }
    };

    let mut specs = Vec::new();
    for f in fields.named.iter_mut() {
        let fname = f.ident.clone().expect("named field").to_string();
        let idx = f.attrs.iter().position(|a| a.path().is_ident("setting"));
        let Some(i) = idx else {
            return Err(syn::Error::new(
                f.ident.span(),
                format!("field `{fname}` is missing `#[setting(...)]`"),
            ));
        };
        let attr = f.attrs.remove(i);
        let metas: Punctuated<Meta, Token![,]> =
            attr.parse_args_with(Punctuated::parse_terminated)?;
        let sm = parse_setting_meta(f.ident.span(), metas)?;
        let key = format!("{section_name}.{fname}");
        let ty = &f.ty;
        let derived_type = quote!(#ty).to_string().replace(' ', "");
        let type_name = sm.kind.unwrap_or(derived_type);
        let mut_id = sm.mutable;
        let doc = sm.doc;
        let default = sm.default;
        let range_or_enum = sm.range_or_enum;
        let unit = sm.unit;
        let interacts = sm.interacts;
        let since = sm.since;
        let renamed_from = sm.renamed_from;
        specs.push(quote! {
            ::r9v_config::SettingSpec {
                key: #key,
                type_name: #type_name,
                doc: #doc,
                default: #default,
                range_or_enum: #range_or_enum,
                unit: #unit,
                mutability: ::r9v_config::Mutability::#mut_id,
                interacts: &[#(#interacts),*],
                since: #since,
                renamed_from: #renamed_from,
            }
        });
    }

    let ident = &input.ident;
    let n = specs.len();
    let expanded = quote! {
        #input

        impl #ident {
            /// Section name (Spec 12 §3, e.g. `"load"`).
            pub const SECTION: &'static str = #section_name;
            /// Section doc string declared on `#[section]`.
            pub const SECTION_DOC: &'static str = #section_doc;
            /// Strongly typed per-setting metadata, in declaration order.
            pub const SETTINGS: &'static [::r9v_config::SettingSpec] = &[
                #(#specs),*
            ];

            /// Look up one setting by full `section.key`.
            pub fn setting(key: &str) -> Option<&'static ::r9v_config::SettingSpec> {
                Self::SETTINGS.iter().find(|s| s.key == key)
            }

            /// Number of settings in this section.
            pub const fn len() -> usize { #n }
        }
    };
    Ok(expanded.into())
}
