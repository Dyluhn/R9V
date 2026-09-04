// SPDX-License-Identifier: Apache-2.0
//! Sandboxed Jinja subset for `tokenizer.chat_template` (Spec 10 §3.1).
//!
//! Implements the llama.cpp-compatible feature set required by reference
//! model chat templates: `{{ }}`, `{% %}`, `{# #}` (with `-` whitespace
//! control), `if/elif/else`, `for` (+`else`, `break`, `continue`,
//! `loop.*`), `set` (incl. block form and `namespace` attribute sets),
//! `macro` (+`call`/`caller`), `filter` blocks, `{% generation %}`
//! no-ops, tests (`is defined/none/...`), filters (`join`, `tojson`,
//! `selectattr`, string ops, ...), string/list/dict methods, slices,
//! ternary, `and/or/not`, `in`, `~` concat, `range()`, `namespace()`,
//! `raise_exception()`.
//!
//! Sandboxed by construction: there is no `include`/`import`/`extends`,
//! no filesystem or network access, and no clock (`strftime_now` is
//! rejected). Fail closed on anything outside the subset with
//! [`LoaderError::TemplateUnsupported`](crate::LoaderError); malformed
//! templates fail with byte offsets
//! ([`LoaderError::TemplateParse`](crate::LoaderError)); runtime failures
//! name the construct
//! ([`LoaderError::TemplateRender`](crate::LoaderError)).
//!
//! Resource bounds: [`MAX_TEMPLATE_BYTES`] source, [`MAX_OUTPUT_BYTES`]
//! output, [`MAX_LOOP_ITERS`] total loop iterations, [`MAX_DEPTH`]
//! evaluation depth, [`MAX_STEPS`] interpreter steps.

use std::collections::BTreeMap;

use crate::error::LoaderError;

/// Maximum template source bytes.
pub const MAX_TEMPLATE_BYTES: usize = 1 << 20;
/// Maximum rendered output bytes.
pub const MAX_OUTPUT_BYTES: usize = 1 << 22;
/// Maximum total loop iterations per render.
pub const MAX_LOOP_ITERS: usize = 100_000;
/// Maximum expression/call nesting depth.
pub const MAX_DEPTH: usize = 64;
/// Maximum interpreter steps per render.
pub const MAX_STEPS: usize = 1_000_000;
/// Maximum live expression evaluations on the interpreter stack. This is
/// separate from [`MAX_DEPTH`]: flat left-associative chains (`a+a+…`,
/// `x|f|f|…`) recurse once per term at eval time while barely nesting the
/// parser, so they need their own bound against thread-stack overflow.
/// Sized (with margin) for 2MiB test-thread stacks given multi-KiB eval
/// frames; legitimate chat templates nest a handful deep.
pub const MAX_EVAL_DEPTH: usize = 128;

/// A chat message rendered into template context (Spec 10 §3.1).
#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    /// `user` / `assistant` / `system` / `tool` / ...
    pub role: String,
    /// Plain-text content (`content` may also be typed parts).
    pub content: String,
    /// Tool calls attached to an assistant message.
    pub tool_calls: Vec<ToolCall>,
    /// Optional reasoning text some templates consult.
    pub reasoning_content: Option<String>,
}

/// One tool call attached to a message.
#[derive(Debug, Clone, Default)]
pub struct ToolCall {
    /// Tool name.
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: String,
    /// Call id, when the template uses one.
    pub id: Option<String>,
}

/// Full render context (Spec 10 §3.1): messages plus the flags and
/// overrides the serving layer exposes to the template.
#[derive(Debug, Clone, Default)]
pub struct ChatContext {
    /// Conversation history.
    pub messages: Vec<ChatMessage>,
    /// `add_generation_prompt`.
    pub add_generation_prompt: bool,
    /// `bos_token` text.
    pub bos_token: Option<String>,
    /// `eos_token` text.
    pub eos_token: Option<String>,
    /// Tools JSON value (`tools`).
    pub tools: Option<TemplateValue>,
    /// `tool_choice` string or value.
    pub tool_choice: Option<TemplateValue>,
    /// `enable_thinking` / `reasoning_effort` passthrough.
    pub enable_thinking: Option<bool>,
    /// Extra `chat_template_kwargs` (sorted for determinism).
    pub extra: BTreeMap<String, TemplateValue>,
}

/// Template value domain (mirrors minja `value_*` minus functions, which
/// live in the interpreter environment instead).
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateValue {
    /// Undefined name (falsy; `is defined` is false).
    Undefined,
    /// `none` / `null` / `nil`.
    None,
    /// Boolean.
    Bool(bool),
    /// Integer.
    Int(i64),
    /// Float.
    Float(f64),
    /// String.
    Str(String),
    /// HTML-safe string (markupsafe `Markup` equivalent): renders like
    /// `Str`, but `+` concatenation escapes the non-safe side instead of
    /// concatenating raw. Only produced by filters that return markup
    /// (`tojson`); plain string operations (`~`, slicing, `|string`)
    /// decay it back to `Str`, mirroring CPython `str` semantics.
    SafeStr(String),
    /// Sequence.
    List(Vec<TemplateValue>),
    /// Mapping (key order = insertion order).
    Dict(Vec<(String, TemplateValue)>),
}

impl TemplateValue {
    /// minja truthiness: undefined/none/false/0/""/[]/{} are falsy.
    pub fn is_truthy(&self) -> bool {
        match self {
            TemplateValue::Undefined | TemplateValue::None => false,
            TemplateValue::Bool(b) => *b,
            TemplateValue::Int(i) => *i != 0,
            TemplateValue::Float(f) => *f != 0.0,
            TemplateValue::Str(s) | TemplateValue::SafeStr(s) => !s.is_empty(),
            TemplateValue::List(l) => !l.is_empty(),
            TemplateValue::Dict(d) => !d.is_empty(),
        }
    }

    /// `is defined` (mirrors minja: defined = not undefined).
    pub fn is_defined(&self) -> bool {
        !matches!(self, TemplateValue::Undefined)
    }

    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            TemplateValue::Undefined => "undefined",
            TemplateValue::None => "none",
            TemplateValue::Bool(_) => "bool",
            TemplateValue::Int(_) => "int",
            TemplateValue::Float(_) => "float",
            TemplateValue::Str(_) | TemplateValue::SafeStr(_) => "string",
            TemplateValue::List(_) => "array",
            TemplateValue::Dict(_) => "object",
        }
    }

    /// Lookup by dict key (objects only).
    pub(crate) fn get_key(&self, key: &str) -> TemplateValue {
        match self {
            TemplateValue::Dict(entries) => entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap_or(TemplateValue::Undefined),
            _ => TemplateValue::Undefined,
        }
    }

    /// Full JSON encoding (verbatim minja `value_to_json_internal`).
    pub(crate) fn write_json(
        &self,
        out: &mut String,
        level: usize,
        indent: i64,
        item_sep: &str,
        key_sep: &str,
        sort_keys: bool,
    ) {
        let pad = |level: usize| -> String {
            if indent > 0 {
                " ".repeat(level * indent as usize)
            } else {
                String::new()
            }
        };
        let newline = if indent >= 0 { "\n" } else { "" };
        match self {
            TemplateValue::Undefined => out.push_str("null"),
            TemplateValue::None => out.push_str("null"),
            TemplateValue::Bool(true) => out.push_str("true"),
            TemplateValue::Bool(false) => out.push_str("false"),
            TemplateValue::Int(i) => out.push_str(&i.to_string()),
            TemplateValue::Float(f) => {
                out.push_str(&crate::template::builtins::format_float_json(*f));
            }
            TemplateValue::Str(s) | TemplateValue::SafeStr(s) => write_json_string(out, s),
            TemplateValue::List(items) => {
                out.push('[');
                if !items.is_empty() {
                    out.push_str(newline);
                    for (i, item) in items.iter().enumerate() {
                        out.push_str(&pad(level));
                        if indent > 0 {
                            out.push_str(&" ".repeat(indent as usize));
                        }
                        item.write_json(out, level + 1, indent, item_sep, key_sep, sort_keys);
                        if i + 1 < items.len() {
                            out.push_str(item_sep);
                        }
                        out.push_str(newline);
                    }
                    out.push_str(&pad(level));
                }
                out.push(']');
            }
            TemplateValue::Dict(entries) => {
                out.push('{');
                if !entries.is_empty() {
                    out.push_str(newline);
                    // Key order: insertion order, unless `sort_keys` (the
                    // Jinja2 `json.dumps_kwargs` policy default) is set.
                    let mut order: Vec<usize> = (0..entries.len()).collect();
                    if sort_keys {
                        order.sort_by(|&a, &b| entries[a].0.cmp(&entries[b].0));
                    }
                    for (i, &idx) in order.iter().enumerate() {
                        let (k, v) = &entries[idx];
                        out.push_str(&pad(level));
                        if indent > 0 {
                            out.push_str(&" ".repeat(indent as usize));
                        }
                        write_json_string(out, k);
                        out.push_str(key_sep);
                        v.write_json(out, level + 1, indent, item_sep, key_sep, sort_keys);
                        if i + 1 < entries.len() {
                            out.push_str(item_sep);
                        }
                        out.push_str(newline);
                    }
                    out.push_str(&pad(level));
                }
                out.push('}');
            }
        }
    }
}

pub(crate) mod builtins;
pub(crate) mod interp;
pub(crate) mod lexer;
pub(crate) mod parser;

pub use interp::render as render_template;

/// Renders `source` with plain variables (test/support entry point).
pub fn render_vars(
    source: &str,
    vars: BTreeMap<String, TemplateValue>,
) -> Result<String, LoaderError> {
    interp::render_vars(source, vars)
}

/// JSON string quoting (verbatim minja escapes: `\"` `\\` `\b` `\f`
/// `\n` `\r` `\t`, `\u00XX` for other controls).
pub(crate) fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
