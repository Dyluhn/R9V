// SPDX-License-Identifier: Apache-2.0
//! Template value operations: operators, tests, filters, and methods
//! (mirrors minja `runtime.cpp` / `value.cpp`).

use std::cmp::Ordering;

use crate::error::LoaderError;
use crate::template::TemplateValue;

/// `as_string` coercion (mirrors minja): strings raw, ints plain, floats
/// trimmed to one decimal minimum, bools `True`/`False`.
pub(crate) fn as_string(value: &TemplateValue) -> Result<String, LoaderError> {
    match value {
        TemplateValue::Str(s) | TemplateValue::SafeStr(s) => Ok(s.clone()),
        TemplateValue::Int(i) => Ok(i.to_string()),
        TemplateValue::Float(f) => Ok(format_float(*f)),
        TemplateValue::Bool(true) => Ok("True".to_owned()),
        TemplateValue::Bool(false) => Ok("False".to_owned()),
        other => Err(render_error(format!(
            "{} is not a string value",
            other.type_name()
        ))),
    }
}

/// True for both string representations (plain and HTML-safe).
fn is_string_like(value: &TemplateValue) -> bool {
    matches!(value, TemplateValue::Str(_) | TemplateValue::SafeStr(_))
}

/// Strict string operand for `+` beside a safe value (mirrors CPython,
/// which raises `TypeError` for non-string operands of `Markup.__add__`).
fn string_operand(value: &TemplateValue) -> Result<String, LoaderError> {
    match value {
        TemplateValue::Str(s) | TemplateValue::SafeStr(s) => Ok(s.clone()),
        other => Err(render_error(format!(
            "cannot concatenate {} with a safe string",
            other.type_name()
        ))),
    }
}

/// Float display for `as_string` (verbatim `std::to_string` + trim:
/// `%.6f` with trailing zeros removed, one decimal kept).
pub(crate) fn format_float(value: f64) -> String {
    let mut out = format!("{value:.6}");
    while out.ends_with('0') {
        out.pop();
    }
    if out.ends_with('.') {
        out.push('0');
    }
    out
}

/// Float display for JSON output (verbatim C++ `ostream << double`, i.e.
/// `%.6g`: 6 significant digits, `%e` form for exponents < -4 or >= 6).
pub(crate) fn format_float_json(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0".to_owned()
        } else {
            "0".to_owned()
        };
    }
    if !value.is_finite() {
        return "null".to_owned();
    }
    let abs = value.abs();
    let exp = abs.log10().floor() as i32;
    if (-4..6).contains(&exp) {
        let decimals = (6 - 1 - exp).max(0) as usize;
        let mut out = format!("{value:.decimals$}");
        if out.contains('.') {
            while out.ends_with('0') {
                out.pop();
            }
            if out.ends_with('.') {
                out.pop();
            }
        }
        out
    } else {
        let mantissa = format!("{abs:.5e}");
        // Rust formats as `1.23457e6`; C++ as `1.23457e+06`.
        let epos = mantissa.find('e').unwrap_or(mantissa.len());
        let (m, e) = mantissa.split_at(epos);
        let mut m = m.to_owned();
        if m.contains('.') {
            while m.ends_with('0') {
                m.pop();
            }
            if m.ends_with('.') {
                m.pop();
            }
        }
        let exp: i32 = e[1..].parse().unwrap_or(0);
        let sign = if value.is_sign_negative() { "-" } else { "" };
        format!("{sign}{m}e{exp:+03}")
    }
}

pub(crate) fn render_error(detail: String) -> LoaderError {
    LoaderError::TemplateRender { detail }
}

pub(crate) fn unsupported(detail: String) -> LoaderError {
    LoaderError::TemplateUnsupported { detail }
}

/// Numeric value as f64 when both sides are numeric.
fn as_number(value: &TemplateValue) -> Option<f64> {
    match value {
        TemplateValue::Int(i) => Some(*i as f64),
        TemplateValue::Float(f) => Some(*f),
        // minja bools are numeric (`is_numeric` true) with val 0/1.
        TemplateValue::Bool(true) => Some(1.0),
        TemplateValue::Bool(false) => Some(0.0),
        _ => None,
    }
}

/// `==` (mirrors minja `equivalent`: numerics cross-compare, containers
/// compare deeply, undefined only equals undefined).
pub(crate) fn values_equal(left: &TemplateValue, right: &TemplateValue) -> bool {
    if let (Some(a), Some(b)) = (as_number(left), as_number(right)) {
        // Bool vs number: minja compares val_int/val_flt pairs; treat via
        // the numeric values (bool true == 1).
        return a == b;
    }
    match (left, right) {
        (TemplateValue::Undefined, TemplateValue::Undefined) => true,
        (TemplateValue::None, TemplateValue::None) => true,
        (TemplateValue::Str(a), TemplateValue::Str(b))
        | (TemplateValue::Str(a), TemplateValue::SafeStr(b))
        | (TemplateValue::SafeStr(a), TemplateValue::Str(b))
        | (TemplateValue::SafeStr(a), TemplateValue::SafeStr(b)) => a == b,
        (TemplateValue::List(a), TemplateValue::List(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| values_equal(x, y))
        }
        (TemplateValue::Dict(a), TemplateValue::Dict(b)) => {
            a.len() == b.len()
                && a.iter().all(|(k, v)| {
                    b.iter()
                        .find(|(k2, _)| k == k2)
                        .map(|(_, v2)| values_equal(v, v2))
                        .unwrap_or(false)
                })
        }
        _ => false,
    }
}

/// Total ordering for `sort`/`min`/`max` (mirrors minja `value_compare`:
/// numbers numerically, strings lexically, bools as 0/1; mixed kinds
/// compare by a fixed kind rank so sorting never fails).
pub(crate) fn compare_values(left: &TemplateValue, right: &TemplateValue) -> Ordering {
    if let (Some(a), Some(b)) = (as_number(left), as_number(right)) {
        return a.partial_cmp(&b).unwrap_or(Ordering::Equal);
    }
    match (left, right) {
        (
            TemplateValue::Str(a) | TemplateValue::SafeStr(a),
            TemplateValue::Str(b) | TemplateValue::SafeStr(b),
        ) => a.cmp(b),
        (TemplateValue::Bool(a), TemplateValue::Bool(b)) => a.cmp(b),
        _ => kind_rank(left).cmp(&kind_rank(right)),
    }
}

fn kind_rank(value: &TemplateValue) -> u8 {
    match value {
        TemplateValue::Undefined => 0,
        TemplateValue::None => 1,
        TemplateValue::Bool(_) => 2,
        TemplateValue::Int(_) => 3,
        TemplateValue::Float(_) => 4,
        TemplateValue::Str(_) | TemplateValue::SafeStr(_) => 5,
        TemplateValue::List(_) => 6,
        TemplateValue::Dict(_) => 7,
    }
}

/// `in` membership (mirrors minja `test_is_in`): substring, array
/// membership, dict key. `x in undefined` is false.
pub(crate) fn value_in(left: &TemplateValue, right: &TemplateValue) -> Result<bool, LoaderError> {
    if matches!(right, TemplateValue::Undefined) {
        return Ok(false);
    }
    match right {
        TemplateValue::Str(s) | TemplateValue::SafeStr(s) => {
            let needle = as_string(left)?;
            Ok(s.contains(needle.as_str()))
        }
        TemplateValue::List(items) => Ok(items.iter().any(|item| values_equal(item, left))),
        TemplateValue::Dict(entries) => {
            let needle = as_string(left)?;
            Ok(entries.iter().any(|(k, _)| k == &needle))
        }
        _ => Err(render_error(format!(
            "cannot perform 'in' on {}",
            right.type_name()
        ))),
    }
}

/// Binary operators (mirrors `binary_expression::execute_impl`, including
/// the null-concat workaround and short-circuit `and`/`or` handled by the
/// caller with already-evaluated sides mirrored here as values).
pub(crate) fn apply_binary(
    op: &str,
    left: &TemplateValue,
    right: &TemplateValue,
) -> Result<TemplateValue, LoaderError> {
    if op == "==" {
        return Ok(TemplateValue::Bool(values_equal(left, right)));
    }
    if op == "!=" {
        return Ok(TemplateValue::Bool(!values_equal(left, right)));
    }
    // Undefined / null handling.
    let left_null = matches!(left, TemplateValue::Undefined | TemplateValue::None);
    let right_null = matches!(right, TemplateValue::Undefined | TemplateValue::None);
    if matches!(left, TemplateValue::Undefined) || matches!(right, TemplateValue::Undefined) {
        if matches!(right, TemplateValue::Undefined) && (op == "in" || op == "not in") {
            return Ok(TemplateValue::Bool(op == "not in"));
        }
        if op == "+" || op == "~" {
            if let Some(concat) = concat_with_null(left, right) {
                return Ok(TemplateValue::Str(concat));
            }
        }
        return Err(render_error(format!(
            "cannot perform operation {op} on undefined values"
        )));
    }
    if left_null || right_null {
        if op == "+" || op == "~" {
            if let Some(concat) = concat_with_null(left, right) {
                return Ok(TemplateValue::Str(concat));
            }
        }
        return Err(render_error(
            "cannot perform operation on null values".to_owned(),
        ));
    }
    // Numeric operations.
    if let (Some(a), Some(b)) = (as_number(left), as_number(right)) {
        // (The old third clause — bool beside float — was subsumed: it
        // required a float side, which already sets this flag.)
        let is_float =
            matches!(left, TemplateValue::Float(_)) || matches!(right, TemplateValue::Float(_));
        return match op {
            "+" | "-" | "*" => {
                let result = if op == "+" {
                    a + b
                } else if op == "-" {
                    a - b
                } else {
                    a * b
                };
                Ok(if is_float {
                    TemplateValue::Float(result)
                } else {
                    TemplateValue::Int(result as i64)
                })
            }
            "/" => Ok(TemplateValue::Float(a / b)),
            "%" => {
                let result = a % b;
                Ok(if is_float {
                    TemplateValue::Float(result)
                } else {
                    TemplateValue::Int(result as i64)
                })
            }
            "<" => Ok(TemplateValue::Bool(a < b)),
            ">" => Ok(TemplateValue::Bool(a > b)),
            ">=" => Ok(TemplateValue::Bool(a >= b)),
            "<=" => Ok(TemplateValue::Bool(a <= b)),
            _ => Err(render_error(format!("unknown operator {op:?}"))),
        };
    }
    // Array operations.
    if let (TemplateValue::List(a), TemplateValue::List(b)) = (left, right) {
        if op == "+" {
            let mut out = a.clone();
            out.extend(b.iter().cloned());
            return Ok(TemplateValue::List(out));
        }
        if op == "in" || op == "not in" {
            let member = value_in(left, right)?;
            return Ok(TemplateValue::Bool(if op == "in" {
                member
            } else {
                !member
            }));
        }
        return Err(render_error(format!("unknown operator {op:?}")));
    }
    // `in` for arrays / objects / strings is handled here for all types.
    if op == "in" || op == "not in" {
        // String membership needs both sides strings; dict/array handled
        // by value_in.
        if let (
            TemplateValue::Str(_) | TemplateValue::SafeStr(_),
            TemplateValue::Str(_) | TemplateValue::SafeStr(_),
        ) = (left, right)
        {
            let member = value_in(left, right)?;
            return Ok(TemplateValue::Bool(if op == "in" {
                member
            } else {
                !member
            }));
        }
        if matches!(right, TemplateValue::List(_) | TemplateValue::Dict(_)) {
            let member = value_in(left, right)?;
            return Ok(TemplateValue::Bool(if op == "in" {
                member
            } else {
                !member
            }));
        }
        return Err(render_error(format!(
            "cannot perform {op:?} between {} and {}",
            left.type_name(),
            right.type_name()
        )));
    }
    // String concat with `~` / `+` when either side is a string.
    if (is_string_like(left) || is_string_like(right)) && (op == "~" || op == "+") {
        // `~` stringifies and concatenates raw (safety decays, mirroring
        // CPython `str()` on the operands).
        if op == "~" {
            let mut out = as_string(left)?;
            out.push_str(&as_string(right)?);
            return Ok(TemplateValue::Str(out));
        }
        // `+` follows markupsafe: when a safe side is present, the other
        // side is HTML-escaped and the result stays safe; two plain
        // strings concatenate raw. A safe side beside a non-string fails
        // closed (CPython raises `TypeError` there).
        let left_safe = matches!(left, TemplateValue::SafeStr(_));
        let right_safe = matches!(right, TemplateValue::SafeStr(_));
        if !left_safe && !right_safe {
            let mut out = as_string(left)?;
            out.push_str(&as_string(right)?);
            return Ok(TemplateValue::Str(out));
        }
        let left_str = string_operand(left)?;
        let right_str = string_operand(right)?;
        let mut out = if left_safe {
            left_str
        } else {
            html_escape(&left_str)
        };
        if right_safe {
            out.push_str(&right_str);
        } else {
            out.push_str(&html_escape(&right_str));
        }
        return Ok(TemplateValue::SafeStr(out));
    }
    // Python-style string repetition (`str.__mul__` returns a plain `str`,
    // so safety decays here too).
    if op == "*" {
        if let (TemplateValue::Str(s) | TemplateValue::SafeStr(s), TemplateValue::Int(n)) =
            (left, right)
        {
            return Ok(TemplateValue::Str(s.repeat((*n).max(0) as usize)));
        }
        if let (TemplateValue::Int(n), TemplateValue::Str(s) | TemplateValue::SafeStr(s)) =
            (left, right)
        {
            return Ok(TemplateValue::Str(s.repeat((*n).max(0) as usize)));
        }
    }
    Err(render_error(format!(
        "unknown operator {op:?} between {} and {}",
        left.type_name(),
        right.type_name()
    )))
}

/// Null-concat workaround: none/undefined beside a string concatenates as
/// empty (mirrors minja).
fn concat_with_null(left: &TemplateValue, right: &TemplateValue) -> Option<String> {
    let left_is_null = matches!(left, TemplateValue::Undefined | TemplateValue::None);
    let right_is_null = matches!(right, TemplateValue::Undefined | TemplateValue::None);
    let left_is_str = matches!(left, TemplateValue::Str(_));
    let right_is_str = matches!(right, TemplateValue::Str(_));
    if (left_is_null && right_is_str) || (right_is_null && left_is_str) {
        let mut out = if left_is_null {
            String::new()
        } else {
            as_string(left).unwrap_or_default()
        };
        out.push_str(if right_is_null {
            ""
        } else {
            match right {
                TemplateValue::Str(s) => s,
                _ => return None,
            }
        });
        Some(out)
    } else {
        None
    }
}

/// `is` tests (mirror minja `test_is_*`).
pub(crate) fn apply_test(
    name: &str,
    value: &TemplateValue,
    args: &[TemplateValue],
    negated: bool,
) -> Result<TemplateValue, LoaderError> {
    let result = match name {
        "defined" => value.is_defined(),
        "undefined" => matches!(value, TemplateValue::Undefined),
        "none" => matches!(value, TemplateValue::None),
        "true" => matches!(value, TemplateValue::Bool(true)),
        "false" => matches!(value, TemplateValue::Bool(false)),
        "boolean" => matches!(value, TemplateValue::Bool(_)),
        "integer" => matches!(value, TemplateValue::Int(_)),
        "float" => matches!(value, TemplateValue::Float(_)),
        "number" => matches!(value, TemplateValue::Int(_) | TemplateValue::Float(_)),
        "string" => is_string_like(value),
        "mapping" => matches!(value, TemplateValue::Dict(_)),
        "iterable" | "sequence" => matches!(
            value,
            TemplateValue::List(_) | TemplateValue::Str(_) | TemplateValue::Undefined
        ),
        "odd" => int_test(value, |i| i % 2 != 0)?,
        "even" => int_test(value, |i| i % 2 == 0)?,
        "divisibleby" => {
            let divisor = args
                .first()
                .ok_or_else(|| render_error("divisibleby needs an argument".to_owned()))?;
            let (a, b) = (as_number(value), as_number(divisor));
            match (a, b) {
                (Some(a), Some(b)) => a % b == 0.0,
                _ => {
                    return Err(render_error(
                        "divisibleby needs numeric operands".to_owned(),
                    ));
                }
            }
        }
        "lower" => match value {
            TemplateValue::Str(s) => s.bytes().all(|b| !b.is_ascii_uppercase()),
            _ => {
                return Err(render_error("lower test needs a string".to_owned()));
            }
        },
        "upper" => match value {
            TemplateValue::Str(s) => s.bytes().all(|b| !b.is_ascii_lowercase()),
            _ => {
                return Err(render_error("upper test needs a string".to_owned()));
            }
        },
        "eq" | "equalto" => match args.first() {
            Some(other) => values_equal(value, other),
            None => return Err(render_error(format!("{name} test needs an argument"))),
        },
        "ne" => match args.first() {
            Some(other) => !values_equal(value, other),
            None => return Err(render_error("ne test needs an argument".to_owned())),
        },
        "lt" | "lessthan" => compare_test(value, args, Ordering::Less)?,
        "le" => compare_test(value, args, Ordering::Less)?,
        "gt" | "greaterthan" => compare_test(value, args, Ordering::Greater)?,
        "ge" => compare_test(value, args, Ordering::Greater)?,
        "in" => match args.first() {
            Some(other) => value_in(value, other)?,
            None => return Err(render_error("in test needs an argument".to_owned())),
        },
        "sameas" => match args.first() {
            // `is sameas` compares identity for our value domain via
            // strict equality (mirrors minja pointer compare for the
            // literal cases templates use: true/false/none).
            Some(other) => values_equal(value, other),
            None => return Err(render_error("sameas test needs an argument".to_owned())),
        },
        "callable" | "escaped" | "filter" | "test" => {
            // Rarely used in chat templates; fail closed rather than guess.
            return Err(unsupported(format!("test '{name}' is outside the subset")));
        }
        _ => {
            return Err(render_error(format!("unknown test '{name}'")));
        }
    };
    Ok(TemplateValue::Bool(if negated { !result } else { result }))
}

fn int_test(value: &TemplateValue, pred: impl Fn(i64) -> bool) -> Result<bool, LoaderError> {
    match value {
        TemplateValue::Int(i) => Ok(pred(*i)),
        _ => Err(render_error("integer test needs an integer".to_owned())),
    }
}

fn compare_test(
    value: &TemplateValue,
    args: &[TemplateValue],
    order: Ordering,
) -> Result<bool, LoaderError> {
    let other = args
        .first()
        .ok_or_else(|| render_error("comparison test needs an argument".to_owned()))?;
    match order {
        Ordering::Less => Ok(compare_values(value, other) == Ordering::Less),
        Ordering::Greater => Ok(compare_values(value, other) == Ordering::Greater),
        Ordering::Equal => Ok(values_equal(value, other)),
    }
}

/// Positional + keyword call arguments.
#[derive(Debug, Clone, Default)]
pub(crate) struct CallArgs {
    /// Positional values (spreads flattened by the caller).
    pub positional: Vec<TemplateValue>,
    /// Keyword values in order.
    pub keywords: Vec<(String, TemplateValue)>,
}

impl CallArgs {
    /// Positional-or-keyword lookup (mirrors `get_kwarg_or_pos`).
    pub(crate) fn kwarg_or_pos(&self, name: &str, index: usize) -> TemplateValue {
        if let Some((_, v)) = self.keywords.iter().find(|(k, _)| k == name) {
            return v.clone();
        }
        self.positional
            .get(index)
            .cloned()
            .unwrap_or(TemplateValue::Undefined)
    }

    /// Keyword-only lookup.
    pub(crate) fn kwarg(&self, name: &str) -> TemplateValue {
        self.keywords
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
            .unwrap_or(TemplateValue::Undefined)
    }

    /// First positional argument (undefined when absent).
    pub(crate) fn positional_first(&self) -> TemplateValue {
        self.positional
            .first()
            .cloned()
            .unwrap_or(TemplateValue::Undefined)
    }
}

/// Applies a named filter/method to `input` (mirrors the per-type builtin
/// tables plus the `count`/`d`/`e`/`trim` aliases).
#[allow(clippy::too_many_lines)]
pub(crate) fn apply_filter(
    name: &str,
    input: &TemplateValue,
    args: &CallArgs,
) -> Result<TemplateValue, LoaderError> {
    // Aliases (mirror minja).
    let name = match name {
        "count" => "length",
        "d" => "default",
        "e" => "escape",
        "trim" => "strip",
        _ => name,
    };
    // `default` works on every type.
    if name == "default" {
        return Ok(filter_default(input, args));
    }
    // Undefined receiver: type-appropriate empties (mirror the undefined
    // builtin table exactly).
    if matches!(input, TemplateValue::Undefined) {
        return Ok(filter_on_undefined(name));
    }
    // None receiver: the none builtin table.
    if matches!(input, TemplateValue::None) {
        return Ok(filter_on_none(name));
    }
    match input {
        TemplateValue::Str(_) | TemplateValue::SafeStr(_) => filter_on_string(name, input, args),
        TemplateValue::List(_) => filter_on_array(name, input, args),
        TemplateValue::Dict(_) => filter_on_object(name, input, args),
        TemplateValue::Int(_) | TemplateValue::Float(_) => filter_on_number(name, input, args),
        TemplateValue::Bool(_) => filter_on_bool(name, input, args),
        TemplateValue::Undefined | TemplateValue::None => {
            unreachable!("handled above")
        }
    }
}

/// `default(value, default, boolean=false)`.
fn filter_default(input: &TemplateValue, args: &CallArgs) -> TemplateValue {
    let fallback = args
        .positional
        .first()
        .cloned()
        .unwrap_or(TemplateValue::Undefined);
    let check_bool = args.kwarg_or_pos("boolean", 1).is_truthy();
    let missing = if check_bool {
        !input.is_truthy()
    } else {
        matches!(input, TemplateValue::Undefined | TemplateValue::None)
    };
    if missing {
        fallback
    } else {
        input.clone()
    }
}

/// Undefined-receiver table (mirror minja exactly; unknown names error).
fn filter_on_undefined(name: &str) -> TemplateValue {
    match name {
        "capitalize" | "join" | "lower" | "replace" | "safe" | "string" | "strip" | "title"
        | "upper" | "escape" => TemplateValue::Str(String::new()),
        "items" | "list" | "map" | "reject" | "rejectattr" | "select" | "selectattr"
        | "reverse" | "sort" | "unique" => TemplateValue::List(Vec::new()),
        "length" | "sum" | "wordcount" => TemplateValue::Int(0),
        "first" | "last" | "max" | "min" => TemplateValue::Undefined,
        _ => TemplateValue::Undefined,
    }
}

/// None-receiver table (mirror minja exactly).
fn filter_on_none(name: &str) -> TemplateValue {
    match name {
        "tojson" => TemplateValue::SafeStr("null".to_owned()),
        "string" | "safe" => TemplateValue::Str(String::new()),
        "items" | "map" | "reject" | "rejectattr" | "select" | "selectattr" | "unique" => {
            TemplateValue::List(Vec::new())
        }
        _ => TemplateValue::Undefined,
    }
}

/// ASCII case mapping (mirror minja byte-based ctype ops).
fn ascii_upper(s: &str) -> String {
    s.bytes().map(|b| b.to_ascii_uppercase() as char).collect()
}

/// ASCII case mapping (mirror minja byte-based ctype ops).
fn ascii_lower(s: &str) -> String {
    s.bytes().map(|b| b.to_ascii_lowercase() as char).collect()
}

/// String filters/methods.
#[allow(clippy::too_many_lines)]
fn filter_on_string(
    name: &str,
    input: &TemplateValue,
    args: &CallArgs,
) -> Result<TemplateValue, LoaderError> {
    let (s, was_safe) = match input {
        TemplateValue::Str(s) => (s, false),
        TemplateValue::SafeStr(s) => (s, true),
        _ => unreachable!("caller guarantees strings"),
    };
    // Content transforms preserve the receiver's safety (mirrors markupsafe,
    // whose `upper`/`strip`/`replace`/etc. return `Markup`); conversions to
    // other types and `split` items decay to plain values.
    let wrap = |out: String| {
        if was_safe {
            TemplateValue::SafeStr(out)
        } else {
            TemplateValue::Str(out)
        }
    };
    match name {
        "upper" => Ok(wrap(ascii_upper(s))),
        "lower" => Ok(wrap(ascii_lower(s))),
        "capitalize" => {
            let mut out = ascii_lower(s);
            if let Some(first) = out.get_mut(0..1) {
                first.make_ascii_uppercase();
            }
            Ok(wrap(out))
        }
        "title" => {
            let mut out = String::with_capacity(s.len());
            let mut new_word = true;
            for b in s.bytes() {
                if b.is_ascii_alphabetic() {
                    out.push(if new_word {
                        b.to_ascii_uppercase() as char
                    } else {
                        b.to_ascii_lowercase() as char
                    });
                    new_word = false;
                } else {
                    out.push(b as char);
                    new_word = true;
                }
            }
            Ok(wrap(out))
        }
        "strip" | "lstrip" | "rstrip" => {
            let chars = match args.kwarg_or_pos("chars", 0) {
                TemplateValue::Undefined => None,
                TemplateValue::Str(c) => Some(c),
                _ => {
                    return Err(render_error(format!(
                        "{name}() chars argument must be a string"
                    )));
                }
            };
            let strip_set = |c: char| match &chars {
                None => c.is_whitespace(),
                Some(set) => set.contains(c),
            };
            let left = name != "rstrip";
            let right = name != "lstrip";
            let mut start = 0;
            let mut end = s.len();
            if left {
                while start < end && s[start..].chars().next().map(strip_set).unwrap_or(false) {
                    start += s[start..].chars().next().unwrap_or(' ').len_utf8();
                }
            }
            if right {
                while end > start && s[..end].chars().next_back().map(strip_set).unwrap_or(false) {
                    end -= s[..end].chars().next_back().unwrap_or(' ').len_utf8();
                }
            }
            Ok(wrap(s[start..end].to_owned()))
        }
        "length" => Ok(TemplateValue::Int(s.chars().count() as i64)),
        "startswith" => {
            let prefix = as_string(&args.positional_first())?;
            Ok(TemplateValue::Bool(s.starts_with(prefix.as_str())))
        }
        "endswith" => {
            let suffix = as_string(&args.positional_first())?;
            Ok(TemplateValue::Bool(s.ends_with(suffix.as_str())))
        }
        "split" | "rsplit" => {
            let delim = if args.positional.is_empty() {
                " ".to_owned()
            } else {
                as_string(&args.positional[0])?
            };
            if delim.is_empty() {
                return Err(render_error("empty separator".to_owned()));
            }
            let maxsplit = args
                .positional
                .get(1)
                .map(as_int_arg)
                .transpose()?
                .unwrap_or(-1);
            let mut parts: Vec<TemplateValue> = Vec::new();
            let mut rest = s.as_str();
            let mut remaining = maxsplit;
            if name == "split" {
                while remaining != 0 {
                    match rest.find(delim.as_str()) {
                        Some(pos) => {
                            parts.push(TemplateValue::Str(rest[..pos].to_owned()));
                            rest = &rest[pos + delim.len()..];
                            if remaining > 0 {
                                remaining -= 1;
                            }
                        }
                        None => break,
                    }
                }
                parts.push(TemplateValue::Str(rest.to_owned()));
            } else {
                while remaining != 0 {
                    match rest.rfind(delim.as_str()) {
                        Some(pos) => {
                            parts.push(TemplateValue::Str(rest[pos + delim.len()..].to_owned()));
                            rest = &rest[..pos];
                            if remaining > 0 {
                                remaining -= 1;
                            }
                        }
                        None => break,
                    }
                }
                parts.push(TemplateValue::Str(rest.to_owned()));
                parts.reverse();
            }
            Ok(TemplateValue::List(parts))
        }
        "replace" => {
            if args.positional.len() < 2 {
                return Err(render_error("replace() needs old and new".to_owned()));
            }
            let old = as_string(&args.positional[0])?;
            let new = as_string(&args.positional[1])?;
            let count = args
                .positional
                .get(2)
                .map(as_int_arg)
                .transpose()?
                .unwrap_or(-1);
            if count < 0 {
                Ok(wrap(s.replace(old.as_str(), new.as_str())))
            } else {
                let mut out = s.clone();
                for _ in 0..count {
                    match out.find(old.as_str()) {
                        Some(pos) => {
                            out.replace_range(pos..pos + old.len(), &new);
                        }
                        None => break,
                    }
                }
                Ok(wrap(out))
            }
        }
        "format" => {
            // Only `{}` placeholders (mirror minja).
            let mut out = String::new();
            let mut arg_idx = 0;
            let mut chars = s.chars().peekable();
            while let Some(c) = chars.next() {
                if c == '{' {
                    match chars.next() {
                        Some('}') => {
                            let value = args
                                .positional
                                .get(arg_idx)
                                .cloned()
                                .unwrap_or(TemplateValue::Undefined);
                            out.push_str(&as_string(&value)?);
                            arg_idx += 1;
                        }
                        _ => {
                            return Err(unsupported(
                                "format() only supports simple '{}' placeholders".to_owned(),
                            ));
                        }
                    }
                } else {
                    out.push(c);
                }
            }
            Ok(wrap(out))
        }
        "indent" => {
            let width = args.kwarg_or_pos("width", 0);
            let first = args.kwarg_or_pos("first", 1).is_truthy();
            let blank = args.kwarg_or_pos("blank", 2).is_truthy();
            let pad = match &width {
                TemplateValue::Undefined => "    ".to_owned(),
                TemplateValue::Int(n) => " ".repeat((*n).max(0) as usize),
                TemplateValue::Str(p) => p.clone(),
                _ => {
                    return Err(render_error(
                        "indent() width must be int or string".to_owned(),
                    ));
                }
            };
            let mut out = String::new();
            for (i, line) in s.split('\n').enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                if (i == 0 && first) || (i > 0 && (!line.is_empty() || blank)) {
                    out.push_str(&pad);
                }
                out.push_str(line);
            }
            Ok(wrap(out))
        }
        "join" => Err(render_error(
            "string join builtin not implemented".to_owned(),
        )),
        "tojson" => Ok(TemplateValue::SafeStr(filter_tojson(input, args)?)),
        "string" => Ok(TemplateValue::Str(filter_tojson(input, args)?)),
        "safe" => Ok(TemplateValue::SafeStr(s.clone())),
        "escape" => Ok(TemplateValue::SafeStr(html_escape(s))),
        "int" => filter_int(input, args),
        "float" => filter_float(input, args),
        "abs" => Err(render_error("abs() needs a number".to_owned())),
        "wordcount" => {
            let count = s.split_whitespace().count() as i64;
            Ok(TemplateValue::Int(count))
        }
        "truncate" => Err(unsupported("truncate is outside the subset".to_owned())),
        "unique" => Err(render_error(
            "array unique builtin not implemented".to_owned(),
        )),
        "list" | "first" | "last" | "map" | "max" | "min" | "reject" | "rejectattr" | "reverse"
        | "select" | "selectattr" | "slice" | "sort" | "sum" => {
            Err(render_error(format!("{name}() needs an array")))
        }
        "get" | "keys" | "values" | "items" | "dictsort" => {
            Err(render_error(format!("{name}() needs an object")))
        }
        _ => Err(render_error(format!("unknown filter '{name}' for string"))),
    }
}

/// Minimal HTML escaping for the `escape`/`e` filter.
fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&#34;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

fn as_int_arg(value: &TemplateValue) -> Result<i64, LoaderError> {
    match value {
        TemplateValue::Int(i) => Ok(*i),
        TemplateValue::Float(f) => Ok(*f as i64),
        TemplateValue::Bool(true) => Ok(1),
        TemplateValue::Bool(false) => Ok(0),
        _ => Err(render_error("integer argument expected".to_owned())),
    }
}

/// `tojson(value, ensure_ascii, indent, separators, sort_keys)` (minja arg
/// positions; the Jinja2 `json.dumps_kwargs` policy default `sort_keys=true`
/// applies unless explicitly passed false).
pub(crate) fn filter_tojson(input: &TemplateValue, args: &CallArgs) -> Result<String, LoaderError> {
    if args.positional.len() + args.keywords.len() > 4 {
        return Err(render_error(
            "tojson() takes at most 4 arguments".to_owned(),
        ));
    }
    let ensure_ascii = args.kwarg_or_pos("ensure_ascii", 1).is_truthy();
    let indent = match &args.kwarg_or_pos("indent", 2) {
        TemplateValue::Undefined | TemplateValue::None => -1,
        TemplateValue::Int(n) => *n,
        _ => return Err(render_error("tojson indent must be an integer".to_owned())),
    };
    let default_item: &str = if indent < 0 { ", " } else { "," };
    let mut item_sep = default_item.to_owned();
    let mut key_sep = ": ".to_owned();
    match &args.kwarg_or_pos("separators", 3) {
        TemplateValue::Undefined | TemplateValue::None => {}
        TemplateValue::List(items) => {
            if let Some(first) = items.first() {
                item_sep = as_string(first)?;
            }
            if let Some(second) = items.get(1) {
                key_sep = as_string(second)?;
            }
        }
        _ => {
            return Err(render_error(
                "tojson separators must be an array".to_owned(),
            ))
        }
    }
    let sort_keys = match &args.kwarg_or_pos("sort_keys", 4) {
        TemplateValue::Undefined | TemplateValue::None => true,
        other => other.is_truthy(),
    };
    let mut json = String::new();
    input.write_json(&mut json, 0, indent, &item_sep, &key_sep, sort_keys);
    if ensure_ascii {
        json = ascii_escape_json(&json);
    }
    Ok(json)
}

/// Escapes non-ASCII chars as `\uXXXX` outside string quoting structure.
/// Operates on the already-quoted JSON: only bare (non-escaped) chars
/// above 0x7F are rewritten.
fn ascii_escape_json(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    let mut in_string = false;
    let mut escaped = false;
    for c in json.chars() {
        if in_string {
            if escaped {
                out.push(c);
                escaped = false;
            } else if c == '\\' {
                out.push(c);
                escaped = true;
            } else if c == '"' {
                out.push(c);
                in_string = false;
            } else if (c as u32) > 0x7F {
                if (c as u32) > 0xFFFF {
                    let v = c as u32 - 0x1_0000;
                    out.push_str(&format!(
                        "\\u{:04x}\\u{:04x}",
                        0xD800 + (v >> 10),
                        0xDC00 + (v & 0x3FF)
                    ));
                } else {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
            } else {
                out.push(c);
            }
        } else {
            out.push(c);
            if c == '"' {
                in_string = true;
            }
        }
    }
    out
}

/// `int(value, default=0, base=10)`.
fn filter_int(input: &TemplateValue, args: &CallArgs) -> Result<TemplateValue, LoaderError> {
    let default = args.kwarg_or_pos("default", 0);
    let default = if matches!(default, TemplateValue::Undefined) {
        TemplateValue::Int(0)
    } else {
        default
    };
    match input {
        TemplateValue::Int(i) => Ok(TemplateValue::Int(*i)),
        TemplateValue::Float(f) => Ok(TemplateValue::Int(*f as i64)),
        TemplateValue::Bool(true) => Ok(TemplateValue::Int(1)),
        TemplateValue::Bool(false) => Ok(TemplateValue::Int(0)),
        TemplateValue::Str(s) => {
            let trimmed = s.trim();
            if let Ok(i) = trimmed.parse::<i64>() {
                Ok(TemplateValue::Int(i))
            } else if let Ok(f) = trimmed.parse::<f64>() {
                Ok(TemplateValue::Int(f as i64))
            } else {
                Ok(default)
            }
        }
        _ => Ok(default),
    }
}

/// Array filters/methods (mirror the array builtin table).
#[allow(clippy::too_many_lines)]
pub(crate) fn filter_on_array(
    name: &str,
    input: &TemplateValue,
    args: &CallArgs,
) -> Result<TemplateValue, LoaderError> {
    let TemplateValue::List(items) = input else {
        unreachable!("caller guarantees arrays")
    };
    match name {
        "list" => Ok(TemplateValue::List(items.clone())),
        "first" => Ok(items.first().cloned().unwrap_or(TemplateValue::Undefined)),
        "last" => Ok(items.last().cloned().unwrap_or(TemplateValue::Undefined)),
        "length" => Ok(TemplateValue::Int(items.len() as i64)),
        "join" => {
            let delim = match args.kwarg_or_pos("d", 0) {
                TemplateValue::Undefined => String::new(),
                other => as_string(&other)?,
            };
            let attribute = args.kwarg("attribute");
            let attr_is_int = matches!(attribute, TemplateValue::Int(_));
            if !matches!(attribute, TemplateValue::Undefined)
                && !matches!(attribute, TemplateValue::Str(_))
                && !attr_is_int
            {
                return Err(render_error(
                    "join() attribute must be string or integer".to_owned(),
                ));
            }
            let mut out = String::new();
            for (i, item) in items.iter().enumerate() {
                let mut value = item.clone();
                if !matches!(attribute, TemplateValue::Undefined) {
                    if attr_is_int {
                        let TemplateValue::Int(n) = attribute else {
                            unreachable!("checked above")
                        };
                        value = index_array(&value, n)?;
                    } else {
                        let TemplateValue::Str(key) = &attribute else {
                            unreachable!("checked above")
                        };
                        value = value.get_key(key);
                    }
                }
                match &value {
                    TemplateValue::Str(_) | TemplateValue::Int(_) | TemplateValue::Float(_) => {}
                    _ => {
                        return Err(render_error(
                            "join() can only join arrays of strings or numerics".to_owned(),
                        ));
                    }
                }
                out.push_str(&as_string(&value)?);
                if i + 1 < items.len() {
                    out.push_str(&delim);
                }
            }
            Ok(TemplateValue::Str(out))
        }
        "map" => {
            // Only `attribute=` mapping (mirror minja; filter-mapping
            // raises not-implemented there too).
            let attribute = args.kwarg("attribute");
            let attr_is_int = matches!(attribute, TemplateValue::Int(_));
            if !matches!(attribute, TemplateValue::Str(_)) && !attr_is_int {
                return Err(render_error(
                    "map: attribute must be string or integer".to_owned(),
                ));
            }
            let default = args.kwarg("default");
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let value = if attr_is_int {
                    let TemplateValue::Int(n) = attribute else {
                        unreachable!("checked above")
                    };
                    match item {
                        TemplateValue::List(_) => index_array_or(item, n, &default),
                        _ => default.clone(),
                    }
                } else {
                    let TemplateValue::Str(key) = &attribute else {
                        unreachable!("checked above")
                    };
                    match item {
                        TemplateValue::Dict(_) => {
                            let found = item.get_key(key);
                            if matches!(found, TemplateValue::Undefined) {
                                default.clone()
                            } else {
                                found
                            }
                        }
                        _ => default.clone(),
                    }
                };
                out.push(value);
            }
            Ok(TemplateValue::List(out))
        }
        "select" | "reject" => {
            if args.positional.is_empty() || args.positional.len() > 2 {
                return Err(render_error(format!("{name}() takes 1 or 2 arguments")));
            }
            let test_name = match args.positional.first() {
                Some(TemplateValue::Str(t)) => t.clone(),
                _ => {
                    return Err(render_error(format!("{name}() needs a test name")));
                }
            };
            let test_arg = args
                .positional
                .get(1)
                .cloned()
                .unwrap_or(TemplateValue::Undefined);
            let mut out = Vec::new();
            for item in items {
                let selected = eval_select_test(&test_name, item, &test_arg)?;
                if selected != (name == "reject") {
                    out.push(item.clone());
                }
            }
            Ok(TemplateValue::List(out))
        }
        "selectattr" | "rejectattr" => {
            if args.positional.is_empty() || args.positional.len() > 3 {
                return Err(render_error(format!("{name}() takes 1 to 3 arguments")));
            }
            let attribute = match args.positional.first() {
                Some(TemplateValue::Str(a)) => a.clone(),
                _ => {
                    return Err(render_error(format!("{name}() needs an attribute name")));
                }
            };
            let reject = name == "rejectattr";
            let mut out = Vec::new();
            if args.positional.len() == 1 {
                for item in items {
                    let TemplateValue::Dict(_) = item else {
                        return Err(render_error(format!("{name}: item is not an object")));
                    };
                    let selected = item.get_key(&attribute).is_truthy();
                    if selected != reject {
                        out.push(item.clone());
                    }
                }
                return Ok(TemplateValue::List(out));
            }
            // Two args: `selectattr(test, value)` applies the test to the
            // item itself. Three args: `selectattr(attr, test, value)`
            // applies it to the item's attribute (mirror minja arg
            // positions exactly).
            let (on_attr, test_name, test_value) = match args.positional.len() {
                2 => match &args.positional[0] {
                    TemplateValue::Str(t) => (false, t.clone(), args.positional[1].clone()),
                    _ => {
                        return Err(render_error(format!("{name}: test name must be a string")));
                    }
                },
                _ => match &args.positional[1] {
                    TemplateValue::Str(t) => (true, t.clone(), args.positional[2].clone()),
                    _ => {
                        return Err(render_error(format!("{name}: test name must be a string")));
                    }
                },
            };
            for item in items {
                let subject = if on_attr {
                    let TemplateValue::Dict(_) = item else {
                        return Err(render_error(format!("{name}: item is not an object")));
                    };
                    item.get_key(&attribute)
                } else {
                    item.clone()
                };
                let selected = eval_select_test(&test_name, &subject, &test_value)?;
                if selected != reject {
                    out.push(item.clone());
                }
            }
            Ok(TemplateValue::List(out))
        }
        "sort" => {
            let reverse = args.kwarg_or_pos("reverse", 0).is_truthy();
            let attribute = args.kwarg_or_pos("attribute", 2);
            let mut sorted = items.clone();
            sorted.sort_by(|a, b| {
                let (mut x, mut y) = (a.clone(), b.clone());
                if !matches!(attribute, TemplateValue::Undefined) {
                    if let TemplateValue::Int(n) = attribute {
                        if let TemplateValue::List(_) = a {
                            x = index_array_or(a, n, &TemplateValue::Undefined);
                        }
                        if let TemplateValue::List(_) = b {
                            y = index_array_or(b, n, &TemplateValue::Undefined);
                        }
                    } else if let TemplateValue::Str(key) = &attribute {
                        if let TemplateValue::Dict(_) = a {
                            x = a.get_key(key);
                        }
                        if let TemplateValue::Dict(_) = b {
                            y = b.get_key(key);
                        }
                    }
                }
                let ord = compare_values(&x, &y);
                if reverse {
                    ord.reverse()
                } else {
                    ord
                }
            });
            Ok(TemplateValue::List(sorted))
        }
        "reverse" => {
            let mut out = items.clone();
            out.reverse();
            Ok(TemplateValue::List(out))
        }
        "min" => items
            .iter()
            .min_by(|a, b| compare_values(a, b))
            .cloned()
            .map(Ok)
            .unwrap_or(Ok(TemplateValue::Undefined)),
        "max" => items
            .iter()
            .max_by(|a, b| compare_values(a, b))
            .cloned()
            .map(Ok)
            .unwrap_or(Ok(TemplateValue::Undefined)),
        // No `sum` on arrays in minja (only the undefined table has one);
        // fail closed like the reference instead of guessing.
        "sum" => Err(render_error("unknown filter 'sum' for array".to_owned())),
        "slice" => {
            // `slice(start, stop, step)` positional (mirror the member
            // translation; kwargs of the same names also accepted).
            let start = args.kwarg_or_pos("start", 0);
            let stop = args.kwarg_or_pos("stop", 1);
            let step = args.kwarg_or_pos("step", 2);
            Ok(slice_list(
                items,
                opt_int(&start)?,
                opt_int(&stop)?,
                opt_int(&step)?,
            ))
        }
        "append" => {
            let value = args.positional_first();
            let mut out = items.clone();
            out.push(value);
            Ok(TemplateValue::List(out))
        }
        "pop" => {
            let mut out = items.clone();
            out.pop();
            Ok(TemplateValue::List(out))
        }
        "tojson" => Ok(TemplateValue::SafeStr(filter_tojson(input, args)?)),
        "string" => Ok(TemplateValue::Str(filter_tojson(input, args)?)),
        "safe" => Ok(TemplateValue::SafeStr(filter_tojson(input, args)?)),
        "int" => filter_int(input, args),
        "float" => filter_float(input, args),
        "upper" | "lower" | "strip" | "title" | "capitalize" | "replace" | "startswith"
        | "endswith" | "split" | "rsplit" | "format" | "indent" | "wordcount" => {
            Err(render_error(format!("{name}() needs a string")))
        }
        "get" | "keys" | "values" | "items" | "dictsort" => {
            Err(render_error(format!("{name}() needs an object")))
        }
        "escape" => Err(render_error("escape() needs a string".to_owned())),
        "truncate" | "unique" => Err(unsupported(format!("{name} is outside the subset"))),
        _ => Err(render_error(format!("unknown filter '{name}' for array"))),
    }
}

/// Array index with negative wrap (mirror minja `at`).
fn index_array(value: &TemplateValue, index: i64) -> Result<TemplateValue, LoaderError> {
    match value {
        TemplateValue::List(items) => {
            let mut i = index;
            if i < 0 {
                i += items.len() as i64;
            }
            if i < 0 || i >= items.len() as i64 {
                return Err(render_error("array index out of range".to_owned()));
            }
            Ok(items[i as usize].clone())
        }
        _ => Err(render_error("not an array".to_owned())),
    }
}

/// Array index with default (mirror minja `at(v, default)`).
fn index_array_or(value: &TemplateValue, index: i64, default: &TemplateValue) -> TemplateValue {
    index_array(value, index).unwrap_or_else(|_| default.clone())
}

/// Optional int argument (undefined → None).
fn opt_int(value: &TemplateValue) -> Result<Option<i64>, LoaderError> {
    opt_int_or_none(value)
}

/// Optional int argument (undefined → None), shared with the interpreter.
pub(crate) fn opt_int_or_none(value: &TemplateValue) -> Result<Option<i64>, LoaderError> {
    match value {
        TemplateValue::Undefined => Ok(None),
        TemplateValue::Int(i) => Ok(Some(*i)),
        TemplateValue::Float(f) => Ok(Some(*f as i64)),
        _ => Err(render_error("integer argument expected".to_owned())),
    }
}

/// Python-style slicing with step (verbatim port of minja `slice<T>`).
fn slice_indexed(len: i64, start: i64, stop: i64, step: i64) -> Vec<i64> {
    let direction = if step > 0 { 1 } else { -1 };
    let (start_val, stop_val) = if direction >= 0 {
        (
            if start < 0 {
                (len + start).max(0)
            } else {
                start.min(len)
            },
            if stop < 0 {
                (len + stop).max(0)
            } else {
                stop.min(len)
            },
        )
    } else {
        (
            if start < 0 {
                (len + start).max(0)
            } else {
                start.min(len - 1)
            },
            if stop < -1 {
                (len + stop).max(-1)
            } else {
                stop.min(len - 1)
            },
        )
    };
    let mut out = Vec::new();
    let mut i = start_val;
    while direction * i < direction * stop_val {
        if i >= 0 && i < len {
            out.push(i);
        }
        i += step;
    }
    out
}

/// Python-style list slicing with step (mirror minja `slice<T>`).
pub(crate) fn slice_list(
    items: &[TemplateValue],
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> TemplateValue {
    let len = items.len() as i64;
    let step = step.unwrap_or(1);
    if step == 0 {
        return TemplateValue::List(Vec::new());
    }
    // The member translation always passes all three; the `slice` filter
    // allows 1-3 args (missing stop/step default per the builtin).
    let (start, stop) = match (start, stop) {
        (Some(a), Some(b)) => (a, b),
        (Some(a), None) => (0, a),
        (None, _) => (0, len),
    };
    TemplateValue::List(
        slice_indexed(len, start, stop, step)
            .into_iter()
            .map(|i| items[i as usize].clone())
            .collect(),
    )
}

/// Byte-based string slicing (mirror minja slicing `std::string`).
pub(crate) fn slice_str(
    s: &str,
    start: &TemplateValue,
    stop: &TemplateValue,
    step: i64,
) -> Result<String, LoaderError> {
    let bytes = s.as_bytes();
    let len = bytes.len() as i64;
    let start = opt_int(start)?.unwrap_or(0);
    let stop = opt_int(stop)?.unwrap_or(if step < 0 { -1 } else { len });
    let mut out = Vec::new();
    for i in slice_indexed(len, start, stop, step) {
        out.push(bytes[i as usize]);
    }
    // Byte slices can split UTF-8 (mirror minja keeping raw bytes);
    // lossy conversion is the closest total approximation.
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// `select`/`selectattr` test evaluation against the known test names.
fn eval_select_test(
    test_name: &str,
    item: &TemplateValue,
    test_value: &TemplateValue,
) -> Result<bool, LoaderError> {
    let args = if matches!(test_value, TemplateValue::Undefined) {
        Vec::new()
    } else {
        vec![test_value.clone()]
    };
    match apply_test(test_name, item, &args, false)? {
        TemplateValue::Bool(b) => Ok(b),
        _ => Err(render_error(format!(
            "selectattr: unknown test '{test_name}'"
        ))),
    }
}

/// Object filters/methods (mirror the object builtin table; `default` is
/// notably absent there, matching the reference comment about gpt-oss).
pub(crate) fn filter_on_object(
    name: &str,
    input: &TemplateValue,
    args: &CallArgs,
) -> Result<TemplateValue, LoaderError> {
    let TemplateValue::Dict(entries) = input else {
        unreachable!("caller guarantees objects")
    };
    match name {
        "get" => {
            if args.positional.is_empty() || args.positional.len() > 2 {
                return Err(render_error(
                    "get() needs a key and optional default".to_owned(),
                ));
            }
            let TemplateValue::Str(key) = &args.positional[0] else {
                return Err(render_error(
                    "get: second argument must be a string (key)".to_owned(),
                ));
            };
            let default = args
                .positional
                .get(1)
                .cloned()
                .unwrap_or(TemplateValue::None);
            Ok(entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap_or(default))
        }
        "keys" => Ok(TemplateValue::List(
            entries
                .iter()
                .map(|(k, _)| TemplateValue::Str(k.clone()))
                .collect(),
        )),
        "values" => Ok(TemplateValue::List(
            entries.iter().map(|(_, v)| v.clone()).collect(),
        )),
        "items" => Ok(TemplateValue::List(
            entries
                .iter()
                .map(|(k, v)| TemplateValue::List(vec![TemplateValue::Str(k.clone()), v.clone()]))
                .collect(),
        )),
        "length" => Ok(TemplateValue::Int(entries.len() as i64)),
        "dictsort" => {
            let by_value =
                matches!(args.kwarg_or_pos("by", 1), TemplateValue::Str(ref s) if s == "value");
            let reverse = args.kwarg_or_pos("reverse", 2).is_truthy();
            let mut sorted = entries.clone();
            sorted.sort_by(|a, b| {
                let ord = if by_value {
                    compare_values(&a.1, &b.1)
                } else {
                    a.0.cmp(&b.0)
                };
                if reverse {
                    ord.reverse()
                } else {
                    ord
                }
            });
            Ok(TemplateValue::Dict(sorted))
        }
        "tojson" => Ok(TemplateValue::SafeStr(filter_tojson(input, args)?)),
        "string" => Ok(TemplateValue::Str(filter_tojson(input, args)?)),
        "join" => Err(render_error("object join not implemented".to_owned())),
        "default" => Err(render_error(
            "unknown filter 'default' for object".to_owned(),
        )),
        _ => Err(render_error(format!("unknown filter '{name}' for object"))),
    }
}

/// Number filters (`int`/`float`/`abs`/`default`/`safe`/`string`/`tojson`).
pub(crate) fn filter_on_number(
    name: &str,
    input: &TemplateValue,
    args: &CallArgs,
) -> Result<TemplateValue, LoaderError> {
    match name {
        "int" => filter_int(input, args),
        "float" => filter_float(input, args),
        "abs" => match input {
            TemplateValue::Int(i) => Ok(TemplateValue::Int(i.abs())),
            TemplateValue::Float(f) => Ok(TemplateValue::Float(f.abs())),
            _ => unreachable!("caller guarantees numbers"),
        },
        "string" => Ok(TemplateValue::Str(filter_tojson(input, args)?)),
        "safe" | "tojson" => Ok(TemplateValue::SafeStr(filter_tojson(input, args)?)),
        _ => Err(render_error(format!("unknown filter '{name}' for number"))),
    }
}

/// Bool filters (`default`/`int`/`float`/`safe`/`string`/`tojson`).
pub(crate) fn filter_on_bool(
    name: &str,
    input: &TemplateValue,
    args: &CallArgs,
) -> Result<TemplateValue, LoaderError> {
    match name {
        "int" => filter_int(input, args),
        "float" => filter_float(input, args),
        "string" => Ok(TemplateValue::Str(match input {
            TemplateValue::Bool(true) => "True".to_owned(),
            _ => "False".to_owned(),
        })),
        "safe" => Ok(TemplateValue::SafeStr(match input {
            TemplateValue::Bool(true) => "True".to_owned(),
            _ => "False".to_owned(),
        })),
        "tojson" => Ok(TemplateValue::SafeStr(filter_tojson(input, args)?)),
        _ => Err(render_error(format!("unknown filter '{name}' for bool"))),
    }
}

/// `float(value, default=0.0)`.
fn filter_float(input: &TemplateValue, args: &CallArgs) -> Result<TemplateValue, LoaderError> {
    let default = args.kwarg_or_pos("default", 0);
    let default = if matches!(default, TemplateValue::Undefined) {
        TemplateValue::Float(0.0)
    } else {
        default
    };
    match input {
        TemplateValue::Float(f) => Ok(TemplateValue::Float(*f)),
        TemplateValue::Int(i) => Ok(TemplateValue::Float(*i as f64)),
        TemplateValue::Bool(true) => Ok(TemplateValue::Float(1.0)),
        TemplateValue::Bool(false) => Ok(TemplateValue::Float(0.0)),
        TemplateValue::Str(s) => match s.trim().parse::<f64>() {
            Ok(f) => Ok(TemplateValue::Float(f)),
            Err(_) => Ok(default),
        },
        _ => Ok(default),
    }
}
