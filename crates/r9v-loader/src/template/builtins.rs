// SPDX-License-Identifier: Apache-2.0
//! Template value operations: operators, tests, filters, and methods
//! (mirrors minja `runtime.cpp` / `value.cpp`).

use std::cmp::Ordering;

use crate::error::LoaderError;
use crate::template::{TemplateValue, MAX_LOOP_ITERS, MAX_OUTPUT_BYTES};

/// Refusal for an intermediate string that would exceed the rendered-output
/// budget. All string growth is capped by [`MAX_OUTPUT_BYTES`] (the 4 MiB
/// rendered-output contract): no new ceiling is introduced for strings.
pub(crate) fn limit_str_bytes(what: &'static str, got: usize) -> LoaderError {
    LoaderError::Limit {
        what,
        limit: MAX_OUTPUT_BYTES,
        got,
    }
}

/// Refusal for a materialized list that would exceed the loop-iteration
/// budget. Lists feed `for` loops directly, so [`MAX_LOOP_ITERS`] is the
/// proportional ceiling: no new ceiling is introduced for lists.
pub(crate) fn limit_list_len(what: &'static str, got: usize) -> LoaderError {
    LoaderError::Limit {
        what,
        limit: MAX_LOOP_ITERS,
        got,
    }
}

/// Preflight `current + add` string bytes with checked arithmetic: hostile
/// lengths must saturate into a refusal, never wrap or panic. Returns the
/// total on success so callers can pre-reserve exactly.
pub(crate) fn ensure_str_add(
    current: usize,
    add: usize,
    what: &'static str,
) -> Result<usize, LoaderError> {
    match current.checked_add(add) {
        Some(total) if total <= MAX_OUTPUT_BYTES => Ok(total),
        _ => Err(limit_str_bytes(what, current.saturating_add(add))),
    }
}

/// Preflight `current + add` list length with checked arithmetic.
pub(crate) fn ensure_list_add(
    current: usize,
    add: usize,
    what: &'static str,
) -> Result<usize, LoaderError> {
    match current.checked_add(add) {
        Some(total) if total <= MAX_LOOP_ITERS => Ok(total),
        _ => Err(limit_list_len(what, current.saturating_add(add))),
    }
}

/// Refuses when a computed byte size exceeds the string budget (for sizes
/// derived by multiplication, where the sum form above does not apply).
pub(crate) fn check_str_total(total: u64, what: &'static str) -> Result<usize, LoaderError> {
    if total <= MAX_OUTPUT_BYTES as u64 {
        Ok(total as usize)
    } else {
        Err(limit_str_bytes(what, total.min(usize::MAX as u64) as usize))
    }
}

/// Fallible vector reserve: attacker-controlled growth maps allocation
/// failure to a typed refusal instead of aborting the process.
pub(crate) fn try_reserve_list<T>(
    vec: &mut Vec<T>,
    additional: usize,
    what: &'static str,
) -> Result<(), LoaderError> {
    ensure_list_add(vec.len(), additional, what)?;
    vec.try_reserve(additional)
        .map_err(|_| limit_list_len(what, vec.len().saturating_add(additional)))
}

/// Refuses an already-sized string result that exceeds the output budget
/// (for same-length transforms and shrinks, where preflight-by-sum does
/// not apply but the uniform intermediate-string cap still does).
pub(crate) fn check_str_result(len: usize, what: &'static str) -> Result<(), LoaderError> {
    if len <= MAX_OUTPUT_BYTES {
        Ok(())
    } else {
        Err(limit_str_bytes(what, len))
    }
}

/// Fallible string reserve against the output budget.
pub(crate) fn try_reserve_str(
    buf: &mut String,
    additional: usize,
    what: &'static str,
) -> Result<(), LoaderError> {
    ensure_str_add(buf.len(), additional, what)?;
    buf.try_reserve(additional)
        .map_err(|_| limit_str_bytes(what, buf.len().saturating_add(additional)))
}

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

/// True for exact-integer operands (ints and bools-as-0/1).
fn is_int_like(value: &TemplateValue) -> bool {
    matches!(value, TemplateValue::Int(_) | TemplateValue::Bool(_))
}

/// Exact integer value of an int-like operand (checked by [`is_int_like`]).
fn int_value(value: &TemplateValue) -> i64 {
    match value {
        TemplateValue::Int(i) => *i,
        TemplateValue::Bool(true) => 1,
        TemplateValue::Bool(false) => 0,
        _ => 0,
    }
}

/// Checked integer binary operator: no wrap, no panic on `i64::MIN`, typed
/// refusal on overflow and on division/remainder by zero.
fn int_binary(op: &str, a: i64, b: i64) -> Result<TemplateValue, LoaderError> {
    let overflow = || render_error(format!("integer {op} overflow"));
    match op {
        "+" => Ok(TemplateValue::Int(a.checked_add(b).ok_or_else(overflow)?)),
        "-" => Ok(TemplateValue::Int(a.checked_sub(b).ok_or_else(overflow)?)),
        "*" => Ok(TemplateValue::Int(a.checked_mul(b).ok_or_else(overflow)?)),
        "/" => {
            if b == 0 {
                return Err(render_error("integer division by zero".to_owned()));
            }
            if a == i64::MIN && b == -1 {
                return Err(render_error("integer division overflow".to_owned()));
            }
            // `/` always yields a float (mirrors the numeric path above).
            Ok(TemplateValue::Float(a as f64 / b as f64))
        }
        "%" => {
            if b == 0 {
                return Err(render_error("integer modulo by zero".to_owned()));
            }
            Ok(TemplateValue::Int(a.checked_rem(b).ok_or_else(overflow)?))
        }
        "<" => Ok(TemplateValue::Bool(a < b)),
        ">" => Ok(TemplateValue::Bool(a > b)),
        ">=" => Ok(TemplateValue::Bool(a >= b)),
        "<=" => Ok(TemplateValue::Bool(a <= b)),
        _ => Err(render_error(format!("unknown operator {op:?}"))),
    }
}

/// Charges one equality-comparison unit against [`MAX_LOOP_ITERS`]: hostile
/// wide or nested values trip the budget with a typed refusal instead of
/// spinning, and over-budget work is an error, never a silent `false`.
fn charge_eq_work(budget: &mut usize) -> Result<(), LoaderError> {
    *budget = budget.checked_add(1).ok_or(LoaderError::Limit {
        what: "template equality comparisons",
        limit: MAX_LOOP_ITERS,
        got: usize::MAX,
    })?;
    if *budget > MAX_LOOP_ITERS {
        return Err(LoaderError::Limit {
            what: "template equality comparisons",
            limit: MAX_LOOP_ITERS,
            got: *budget,
        });
    }
    Ok(())
}

/// `==` (mirrors minja `equivalent`: numerics cross-compare, containers
/// compare deeply, undefined only equals undefined). Iterative over an
/// explicit heap stack, never recursive: loop-accumulated values nest deeper
/// than entry validation allows, and the old recursion overflowed the thread
/// stack on them. Every popped pair, every container breadth preflight, and
/// every dict key-scan step charges the shared [`MAX_LOOP_ITERS`] budget, so
/// hostile wide inputs fail fast with [`LoaderError::Limit`].
/// DECISION(A2.9): equality breadth reuses `MAX_LOOP_ITERS` (the iteration
/// budget every list already feeds; rejected a new `MAX_EQ_*` ceiling, which
/// the mandate forbids, and rejected unbounded comparison, which lets a wide
/// nested value spin a single `==`). Quick outcomes stay quick: length
/// mismatches return `false` before any charge, and `in` keeps incremental
/// charges so an early hit still succeeds. Spec 10 §3.1 is silent here.
pub(crate) fn values_equal(
    left: &TemplateValue,
    right: &TemplateValue,
) -> Result<bool, LoaderError> {
    let mut budget = 0usize;
    values_equal_in(left, right, &mut budget)
}

/// Budget-sharing `==` used by `in`, tests, and select filters so one
/// operator call (however many elements it visits) trips a single budget.
fn values_equal_in(
    left: &TemplateValue,
    right: &TemplateValue,
    budget: &mut usize,
) -> Result<bool, LoaderError> {
    let mut stack: Vec<(&TemplateValue, &TemplateValue)> = vec![(left, right)];
    while let Some((l, r)) = stack.pop() {
        charge_eq_work(budget)?;
        if let (Some(a), Some(b)) = (as_number(l), as_number(r)) {
            // Bool vs number: minja compares val_int/val_flt pairs; treat via
            // the numeric values (bool true == 1).
            if a != b {
                return Ok(false);
            }
            continue;
        }
        match (l, r) {
            (TemplateValue::Undefined, TemplateValue::Undefined) => {}
            (TemplateValue::None, TemplateValue::None) => {}
            (TemplateValue::Str(a), TemplateValue::Str(b))
            | (TemplateValue::Str(a), TemplateValue::SafeStr(b))
            | (TemplateValue::SafeStr(a), TemplateValue::Str(b))
            | (TemplateValue::SafeStr(a), TemplateValue::SafeStr(b)) => {
                if a != b {
                    return Ok(false);
                }
            }
            (TemplateValue::List(a), TemplateValue::List(b)) => {
                if a.len() != b.len() {
                    return Ok(false);
                }
                // A single container wider than the iteration budget refuses
                // before its pairs are pushed, instead of materializing a
                // hostile-length work stack first.
                ensure_list_add(0, a.len(), "template equality breadth")?;
                stack.extend(a.iter().zip(b.iter()));
            }
            (TemplateValue::Dict(a), TemplateValue::Dict(b)) => {
                if a.len() != b.len() {
                    return Ok(false);
                }
                ensure_list_add(0, a.len(), "template equality breadth")?;
                for (k, v) in a {
                    // Linear key scan with a per-entry charge: a hostile
                    // wide object trips the budget mid-scan instead of
                    // spending quadratic work first. A missing key still
                    // returns `false` (charged for the work actually done).
                    let mut found = None;
                    for (k2, v2) in b.iter() {
                        charge_eq_work(budget)?;
                        if k == k2 {
                            found = Some(v2);
                            break;
                        }
                    }
                    match found {
                        Some(v2) => stack.push((v, v2)),
                        None => return Ok(false),
                    }
                }
            }
            _ => return Ok(false),
        }
    }
    Ok(true)
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
/// membership, dict key. `x in undefined` is false. Array membership shares
/// one [`MAX_LOOP_ITERS`] budget across every element comparison (see
/// [`values_equal`]), so a hostile wide list fails fast instead of spending
/// an unbounded number of individually cheap comparisons.
pub(crate) fn value_in(left: &TemplateValue, right: &TemplateValue) -> Result<bool, LoaderError> {
    let mut budget = 0usize;
    value_in_in(left, right, &mut budget)
}

/// Budget-sharing `in` used by operators, tests, and select filters.
fn value_in_in(
    left: &TemplateValue,
    right: &TemplateValue,
    budget: &mut usize,
) -> Result<bool, LoaderError> {
    if matches!(right, TemplateValue::Undefined) {
        return Ok(false);
    }
    match right {
        TemplateValue::Str(s) | TemplateValue::SafeStr(s) => {
            let needle = as_string(left)?;
            Ok(s.contains(needle.as_str()))
        }
        TemplateValue::List(items) => {
            for item in items {
                // Per-element charge first: a hostile list with no match
                // trips the budget on breadth alone, while an early hit
                // still succeeds without visiting the rest.
                charge_eq_work(budget)?;
                if values_equal_in(item, left, budget)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
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
        return Ok(TemplateValue::Bool(values_equal(left, right)?));
    }
    if op == "!=" {
        return Ok(TemplateValue::Bool(!values_equal(left, right)?));
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
    // Exact integer arithmetic for int/bool operands (bools count as 0/1,
    // mirroring minja numerics): checked ops refuse overflow, division by
    // zero, and the `i64::MIN / -1` case with a typed error instead of
    // panicking, wrapping, or losing precision through f64.
    // DECISION(A2.9): integer overflow and division/remainder by zero fail
    // with `TemplateRender` (an operation error, not a resource breach);
    // rejected saturating to `i64::MAX` via `as` casts (silently wrong) and
    // float `inf` for integer division by zero (unactionable). Spec 10 §3.1
    // names the feature set, not edge semantics.
    if is_int_like(left) && is_int_like(right) {
        return int_binary(op, int_value(left), int_value(right));
    }
    // Numeric operations with at least one float side (floats cannot wrap
    // or panic; `%`/`/` by zero yield NaN/inf like the reference).
    if let (Some(a), Some(b)) = (as_number(left), as_number(right)) {
        return match op {
            "+" => Ok(TemplateValue::Float(a + b)),
            "-" => Ok(TemplateValue::Float(a - b)),
            "*" => Ok(TemplateValue::Float(a * b)),
            "/" => Ok(TemplateValue::Float(a / b)),
            "%" => Ok(TemplateValue::Float(a % b)),
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
            // Preflight the combined length: hostile lists must not
            // materialize past the loop-iteration budget.
            let total = ensure_list_add(a.len(), b.len(), "template array concat length")?;
            let mut out = Vec::new();
            try_reserve_list(&mut out, total, "template array concat length")?;
            out.extend(a.iter().cloned());
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
            let left_str = as_string(left)?;
            let right_str = as_string(right)?;
            return concat_str(&left_str, &right_str, false);
        }
        // `+` follows markupsafe: when a safe side is present, the other
        // side is HTML-escaped and the result stays safe; two plain
        // strings concatenate raw. A safe side beside a non-string fails
        // closed (CPython raises `TypeError` there).
        let left_safe = matches!(left, TemplateValue::SafeStr(_));
        let right_safe = matches!(right, TemplateValue::SafeStr(_));
        if !left_safe && !right_safe {
            let left_str = as_string(left)?;
            let right_str = as_string(right)?;
            return concat_str(&left_str, &right_str, false);
        }
        let left_str = string_operand(left)?;
        let right_str = string_operand(right)?;
        let left_escaped = if left_safe {
            left_str
        } else {
            html_escape(&left_str)?
        };
        let right_escaped = if right_safe {
            right_str
        } else {
            html_escape(&right_str)?
        };
        return concat_str(&left_escaped, &right_escaped, true);
    }
    // Python-style string repetition (`str.__mul__` returns a plain `str`,
    // so safety decays here too). Both operand orders preflight the product
    // against the output budget before allocating anything.
    // DECISION(A2.9): repetition past [`MAX_OUTPUT_BYTES`] fails with
    // `Limit`; rejected allocating first and relying on the output check
    // (an exabyte `repeat` aborts before any check runs). Spec 10 §3.1 is
    // silent on resource behavior.
    if op == "*" {
        if let (TemplateValue::Str(s) | TemplateValue::SafeStr(s), TemplateValue::Int(n)) =
            (left, right)
        {
            return repeat_str(s, *n);
        }
        if let (TemplateValue::Int(n), TemplateValue::Str(s) | TemplateValue::SafeStr(s)) =
            (left, right)
        {
            return repeat_str(s, *n);
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

/// `is` tests (mirror minja `test_is_*`). Equality, ordering-equality, and
/// membership arms share one [`MAX_LOOP_ITERS`] budget per test call (see
/// [`values_equal`]).
pub(crate) fn apply_test(
    name: &str,
    value: &TemplateValue,
    args: &[TemplateValue],
    negated: bool,
) -> Result<TemplateValue, LoaderError> {
    let mut budget = 0usize;
    apply_test_in(name, value, args, negated, &mut budget)
}

/// Budget-sharing `is` test used by select filters so one filter call trips
/// a single budget no matter how many items it visits.
fn apply_test_in(
    name: &str,
    value: &TemplateValue,
    args: &[TemplateValue],
    negated: bool,
    budget: &mut usize,
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
            Some(other) => values_equal_in(value, other, budget)?,
            None => return Err(render_error(format!("{name} test needs an argument"))),
        },
        "ne" => match args.first() {
            Some(other) => !values_equal_in(value, other, budget)?,
            None => return Err(render_error("ne test needs an argument".to_owned())),
        },
        "lt" | "lessthan" => compare_test(value, args, Ordering::Less, budget)?,
        "le" => compare_test(value, args, Ordering::Less, budget)?,
        "gt" | "greaterthan" => compare_test(value, args, Ordering::Greater, budget)?,
        "ge" => compare_test(value, args, Ordering::Greater, budget)?,
        "in" => match args.first() {
            Some(other) => value_in_in(value, other, budget)?,
            None => return Err(render_error("in test needs an argument".to_owned())),
        },
        "sameas" => match args.first() {
            // `is sameas` compares identity for our value domain via
            // strict equality (mirrors minja pointer compare for the
            // literal cases templates use: true/false/none).
            Some(other) => values_equal_in(value, other, budget)?,
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
    budget: &mut usize,
) -> Result<bool, LoaderError> {
    let other = args
        .first()
        .ok_or_else(|| render_error("comparison test needs an argument".to_owned()))?;
    match order {
        Ordering::Less => Ok(compare_values(value, other) == Ordering::Less),
        Ordering::Greater => Ok(compare_values(value, other) == Ordering::Greater),
        Ordering::Equal => Ok(values_equal_in(value, other, budget)?),
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
        // Same-length transforms: the result is as long as the input, so a
        // result past the output budget is refused rather than materialized.
        // DECISION(A2.9): every intermediate string is capped by
        // `MAX_OUTPUT_BYTES`, even non-amplifying transforms of large
        // inputs (only `|length`-style uses on multi-MiB inputs change
        // outcome, from late output refusal to early filter refusal — both
        // are errors). Rejected capping only amplifying ops (inconsistent:
        // `split` halves and `upper` copies would disagree on the same
        // input). Spec 10 §3.1 is silent on intermediate budgets.
        "upper" => {
            check_str_result(s.len(), "template upper bytes")?;
            Ok(wrap(ascii_upper(s)))
        }
        "lower" => {
            check_str_result(s.len(), "template lower bytes")?;
            Ok(wrap(ascii_lower(s)))
        }
        "capitalize" => {
            check_str_result(s.len(), "template capitalize bytes")?;
            let mut out = ascii_lower(s);
            if let Some(first) = out.get_mut(0..1) {
                first.make_ascii_uppercase();
            }
            Ok(wrap(out))
        }
        "title" => {
            check_str_result(s.len(), "template title bytes")?;
            let mut out = String::new();
            try_reserve_str(&mut out, s.len(), "template title bytes")?;
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
            check_str_result(end - start, "template strip bytes")?;
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
            // Part count is capped by the loop-iteration budget and each part
            // by the output budget: `maxsplit` is bounded by occurrences
            // (each consumes input), so the loops below cannot spin past
            // the input size, and the caps refuse before materializing more.
            let mut parts: Vec<TemplateValue> = Vec::new();
            let mut rest = s.as_str();
            let mut remaining = maxsplit;
            if name == "split" {
                while remaining != 0 {
                    match rest.find(delim.as_str()) {
                        Some(pos) => {
                            push_split_part(&mut parts, &rest[..pos])?;
                            rest = &rest[pos + delim.len()..];
                            if remaining > 0 {
                                remaining -= 1;
                            }
                        }
                        None => break,
                    }
                }
                push_split_part(&mut parts, rest)?;
            } else {
                while remaining != 0 {
                    match rest.rfind(delim.as_str()) {
                        Some(pos) => {
                            push_split_part(&mut parts, &rest[pos + delim.len()..])?;
                            rest = &rest[..pos];
                            if remaining > 0 {
                                remaining -= 1;
                            }
                        }
                        None => break,
                    }
                }
                push_split_part(&mut parts, rest)?;
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
            Ok(wrap(filter_replace(s, &old, &new, count)?))
        }
        "format" => {
            // Only `{}` placeholders (mirror minja). Every push is
            // preflighted: hostile placeholder/argument counts fail before
            // the result passes the output budget.
            const WHAT: &str = "template format bytes";
            let mut out = String::new();
            let mut arg_idx = 0usize;
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
                            let text = as_string(&value)?;
                            ensure_str_add(out.len(), text.len(), WHAT)?;
                            out.push_str(&text);
                            arg_idx = arg_idx.checked_add(1).ok_or_else(|| {
                                render_error("format() too many placeholders".to_owned())
                            })?;
                        }
                        _ => {
                            return Err(unsupported(
                                "format() only supports simple '{}' placeholders".to_owned(),
                            ));
                        }
                    }
                } else {
                    ensure_str_add(out.len(), c.len_utf8(), WHAT)?;
                    out.push(c);
                }
            }
            Ok(wrap(out))
        }
        "indent" => {
            let width = args.kwarg_or_pos("width", 0);
            let first = args.kwarg_or_pos("first", 1).is_truthy();
            let blank = args.kwarg_or_pos("blank", 2).is_truthy();
            // Both the pad width and the per-line growth are preflighted:
            // a hostile width (or a hostile pad string) fails before the
            // first pad is built, and line growth is bounded exactly.
            const WHAT: &str = "template indent bytes";
            let pad = match &width {
                TemplateValue::Undefined => "    ".to_owned(),
                TemplateValue::Int(n) => {
                    if *n <= 0 {
                        String::new()
                    } else {
                        check_str_total(*n as u64, WHAT)?;
                        " ".repeat(*n as usize)
                    }
                }
                TemplateValue::Str(p) => {
                    check_str_result(p.len(), WHAT)?;
                    p.clone()
                }
                _ => {
                    return Err(render_error(
                        "indent() width must be int or string".to_owned(),
                    ));
                }
            };
            // Exact size: subject plus one pad per indented line. Line
            // count is input-bounded (split on one subject).
            let mut pad_lines = 0u64;
            let mut first_line = true;
            for line in s.split('\n') {
                if (first_line && first) || (!first_line && (!line.is_empty() || blank)) {
                    pad_lines += 1;
                }
                first_line = false;
            }
            let total = (s.len() as u64).saturating_add(pad_lines.saturating_mul(pad.len() as u64));
            let total = check_str_total(total, WHAT)?;
            let mut out = String::new();
            try_reserve_str(&mut out, total, WHAT)?;
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
        "safe" => {
            check_str_result(s.len(), "template safe bytes")?;
            Ok(TemplateValue::SafeStr(s.clone()))
        }
        "escape" => Ok(TemplateValue::SafeStr(html_escape(s)?)),
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

/// Concatenates two string pieces with a preflight against the output
/// budget. `safe` selects the `SafeStr` (markupsafe) result form.
fn concat_str(left: &str, right: &str, safe: bool) -> Result<TemplateValue, LoaderError> {
    let total = ensure_str_add(left.len(), right.len(), "template concat bytes")?;
    let mut out = String::new();
    try_reserve_str(&mut out, total, "template concat bytes")?;
    out.push_str(left);
    out.push_str(right);
    if safe {
        Ok(TemplateValue::SafeStr(out))
    } else {
        Ok(TemplateValue::Str(out))
    }
}

/// String repetition with a preflight of `s.len() * max(n, 0)` against the
/// output budget: enormous counts fail before any allocation.
fn repeat_str(s: &str, n: i64) -> Result<TemplateValue, LoaderError> {
    if n <= 0 || s.is_empty() {
        return Ok(TemplateValue::Str(String::new()));
    }
    let total = (s.len() as u64).saturating_mul(n as u64);
    let total = check_str_total(total, "template string repeat bytes")?;
    let mut out = String::new();
    try_reserve_str(&mut out, total, "template string repeat bytes")?;
    out.push_str(&s.repeat(n as usize));
    Ok(TemplateValue::Str(out))
}

/// Minimal HTML escaping for the `escape`/`e` filter, capped at the output
/// budget: each entity is preflighted before it is pushed.
fn html_escape(s: &str) -> Result<String, LoaderError> {
    const WHAT: &str = "template escape bytes";
    let mut out = String::new();
    try_reserve_str(&mut out, s.len().min(MAX_OUTPUT_BYTES), WHAT)?;
    for c in s.chars() {
        let entity = match c {
            '&' => "&amp;",
            '<' => "&lt;",
            '>' => "&gt;",
            '"' => "&#34;",
            '\'' => "&#39;",
            _ => {
                ensure_str_add(out.len(), c.len_utf8(), WHAT)?;
                out.push(c);
                continue;
            }
        };
        ensure_str_add(out.len(), entity.len(), WHAT)?;
        out.push_str(entity);
    }
    Ok(out)
}

/// Bounded `replace(old, new, count)`: single-pass left-to-right
/// non-overlapping replacement. For `count < 0` the result matches
/// `str::replace` exactly; otherwise at most `count` replacements run. The
/// exact result size is preflighted with checked arithmetic before anything
/// is built, so huge counts fail fast.
/// Empty needles insert `new` at Unicode scalar boundaries with pinned
/// Jinja/minja semantics: `count == 0` leaves the subject unchanged,
/// `count < 0` inserts at every boundary (`"abc"` → `"xaxbxcx"`), and a
/// finite `count` inserts at the first `count` boundaries (`"abc"` with
/// `count=2` → `"xaxbc"`). The boundary count is input-bounded (chars + 1)
/// and clamps huge counts before the size preflight, so neither spins.
/// DECISION(A2.9): empty-needle replacement keeps pinned Jinja/minja
/// boundary semantics under the same output cap (rejected erroring on empty
/// needles, and rejected the old front-loaded `new.repeat(count) + subject`
/// form: it disagrees with the reference on every finite count); result
/// sizes past `MAX_OUTPUT_BYTES` fail with `Limit`. Spec 10 §3.1 is silent
/// here.
fn filter_replace(s: &str, old: &str, new: &str, count: i64) -> Result<String, LoaderError> {
    const WHAT: &str = "template replace bytes";
    if count == 0 {
        check_str_result(s.len(), WHAT)?;
        return Ok(s.to_owned());
    }
    if old.is_empty() {
        if new.is_empty() {
            // Inserting empties changes nothing (verified against
            // `str::replace`: `"abc".replace("", "") == "abc"`).
            check_str_result(s.len(), WHAT)?;
            return Ok(s.to_owned());
        }
        // Boundary insertion (both the finite-count and the unlimited
        // forms): the count clamps to the input-bounded boundary total
        // before the exact size preflight, so hostile counts refuse or
        // succeed on the true result size instead of spinning.
        let boundaries = s.chars().count().saturating_add(1) as u64;
        let insertions = if count < 0 {
            boundaries
        } else {
            (count as u64).min(boundaries)
        };
        let total = (s.len() as u64).saturating_add(insertions.saturating_mul(new.len() as u64));
        let total = check_str_total(total, WHAT)?;
        let mut out = String::new();
        try_reserve_str(&mut out, total, WHAT)?;
        let mut done = 0u64;
        for ch in s.chars() {
            if done < insertions {
                out.push_str(new);
                done += 1;
            }
            out.push(ch);
        }
        if done < insertions {
            out.push_str(new);
        }
        return Ok(out);
    }
    // Non-empty needle: one bounded scan counts occurrences, the
    // replacements clamp to `count`, the exact size is preflighted, and a
    // single pass builds the result (no quadratic re-scan).
    let occurrences = s.match_indices(old).count();
    let reps = if count < 0 {
        occurrences
    } else {
        (count as usize).min(occurrences)
    };
    let total = if new.len() >= old.len() {
        (s.len() as u64)
            .saturating_add((reps as u64).saturating_mul((new.len() - old.len()) as u64))
    } else {
        // Each replacement consumes `old.len()` subject bytes, so the total
        // shrink cannot exceed the subject length: saturating is exact here.
        (s.len() as u64)
            .saturating_sub((reps as u64).saturating_mul((old.len() - new.len()) as u64))
    };
    let total = check_str_total(total, WHAT)?;
    let mut out = String::new();
    try_reserve_str(&mut out, total, WHAT)?;
    let mut rest = s;
    let mut done = 0usize;
    while done < reps {
        match rest.find(old) {
            Some(pos) => {
                out.push_str(&rest[..pos]);
                out.push_str(new);
                rest = &rest[pos + old.len()..];
                done += 1;
            }
            None => break,
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// Pushes one `split`/`rsplit` piece with count and size caps.
fn push_split_part(parts: &mut Vec<TemplateValue>, piece: &str) -> Result<(), LoaderError> {
    ensure_list_add(parts.len(), 1, "template split parts")?;
    check_str_result(piece.len(), "template split bytes")?;
    parts.push(TemplateValue::Str(piece.to_owned()));
    Ok(())
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
    input.write_json(&mut json, 0, indent, &item_sep, &key_sep, sort_keys)?;
    // The encoded document is itself an intermediate string: cap it like
    // every other one (a hostile indent or value fails here, not later).
    check_str_result(json.len(), "template tojson bytes")?;
    if ensure_ascii {
        json = ascii_escape_json(&json)?;
    }
    Ok(json)
}

/// Escapes non-ASCII chars as `\uXXXX` outside string quoting structure.
/// Operates on the already-quoted JSON: only bare (non-escaped) chars
/// above 0x7F are rewritten. Expansion is preflighted per escape so a
/// hostile document fails at the output budget instead of trebling it.
fn ascii_escape_json(json: &str) -> Result<String, LoaderError> {
    const WHAT: &str = "template tojson bytes";
    let mut out = String::new();
    try_reserve_str(&mut out, json.len().min(MAX_OUTPUT_BYTES), WHAT)?;
    let mut in_string = false;
    let mut escaped = false;
    for c in json.chars() {
        if in_string {
            if escaped {
                ensure_str_add(out.len(), c.len_utf8(), WHAT)?;
                out.push(c);
                escaped = false;
            } else if c == '\\' {
                ensure_str_add(out.len(), 1, WHAT)?;
                out.push(c);
                escaped = true;
            } else if c == '"' {
                ensure_str_add(out.len(), 1, WHAT)?;
                out.push(c);
                in_string = false;
            } else if (c as u32) > 0x7F {
                // Longest expansion is 12 bytes (a surrogate pair).
                ensure_str_add(out.len(), 12, WHAT)?;
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
                ensure_str_add(out.len(), c.len_utf8(), WHAT)?;
                out.push(c);
            }
        } else {
            ensure_str_add(out.len(), c.len_utf8(), WHAT)?;
            out.push(c);
            if c == '"' {
                in_string = true;
            }
        }
    }
    Ok(out)
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
        "list" => Ok(TemplateValue::List(clone_list(
            items,
            "template list length",
        )?)),
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
            // Every piece and delimiter is preflighted before it lands: a
            // hostile delimiter or element fails before the result passes
            // the output budget.
            const WHAT: &str = "template join bytes";
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
                let text = as_string(&value)?;
                ensure_str_add(out.len(), text.len(), WHAT)?;
                out.push_str(&text);
                if i + 1 < items.len() {
                    ensure_str_add(out.len(), delim.len(), WHAT)?;
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
            let mut out = Vec::new();
            try_reserve_list(&mut out, items.len(), "template map length")?;
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
            // One equality budget for the whole filter call: every item's
            // test charges it, so a hostile list trips it mid-scan.
            let mut eq_budget = 0usize;
            for item in items {
                let selected = eval_select_test(&test_name, item, &test_arg, &mut eq_budget)?;
                if selected != (name == "reject") {
                    ensure_list_add(out.len(), 1, "template select length")?;
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
            // One equality budget for the whole filter call (see above).
            let mut eq_budget = 0usize;
            if args.positional.len() == 1 {
                for item in items {
                    let TemplateValue::Dict(_) = item else {
                        return Err(render_error(format!("{name}: item is not an object")));
                    };
                    let selected = item.get_key(&attribute).is_truthy();
                    if selected != reject {
                        ensure_list_add(out.len(), 1, "template select length")?;
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
                let selected = eval_select_test(&test_name, &subject, &test_value, &mut eq_budget)?;
                if selected != reject {
                    ensure_list_add(out.len(), 1, "template select length")?;
                    out.push(item.clone());
                }
            }
            Ok(TemplateValue::List(out))
        }
        "sort" => {
            let reverse = args.kwarg_or_pos("reverse", 0).is_truthy();
            let attribute = args.kwarg_or_pos("attribute", 2);
            let mut sorted = clone_list(items, "template sort length")?;
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
            let mut out = clone_list(items, "template reverse length")?;
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
            slice_list(items, opt_int(&start)?, opt_int(&stop)?, opt_int(&step)?)
        }
        "append" => {
            let value = args.positional_first();
            ensure_list_add(items.len(), 1, "template append length")?;
            let mut out = clone_list(items, "template append length")?;
            out.push(value);
            Ok(TemplateValue::List(out))
        }
        "pop" => {
            let mut out = clone_list(items, "template pop length")?;
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

/// Capped list clone with a fallible reserve: attacker-controlled lists
/// refuse past the loop-iteration budget instead of over-allocating.
fn clone_list(
    items: &[TemplateValue],
    what: &'static str,
) -> Result<Vec<TemplateValue>, LoaderError> {
    let mut out = Vec::new();
    try_reserve_list(&mut out, items.len(), what)?;
    out.extend(items.iter().cloned());
    Ok(out)
}

/// Array index with negative wrap (mirror minja `at`). The wrap-around
/// addition is checked: a hostile index near `i64::MAX` saturates into the
/// out-of-range error instead of wrapping or panicking.
fn index_array(value: &TemplateValue, index: i64) -> Result<TemplateValue, LoaderError> {
    match value {
        TemplateValue::List(items) => {
            let mut i = index;
            if i < 0 {
                i = i.saturating_add(items.len() as i64);
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

/// Python-style slicing with step (verbatim port of minja `slice<T>`, with
/// checked index math). `len` is a real container length (small); the
/// clamping additions cannot overflow it, and the stepping addition is
/// checked so a hostile step (e.g. `i64::MIN`) ends the walk instead of
/// wrapping or panicking.
fn slice_indexed(len: i64, start: i64, stop: i64, step: i64) -> Vec<i64> {
    let direction: i64 = if step > 0 { 1 } else { -1 };
    let (start_val, stop_val) = if direction >= 0 {
        (
            if start < 0 {
                len.saturating_add(start).max(0)
            } else {
                start.min(len)
            },
            if stop < 0 {
                len.saturating_add(stop).max(0)
            } else {
                stop.min(len)
            },
        )
    } else {
        (
            if start < 0 {
                len.saturating_add(start).max(0)
            } else {
                start.min(len.saturating_sub(1))
            },
            if stop < -1 {
                len.saturating_add(stop).max(-1)
            } else {
                stop.min(len.saturating_sub(1))
            },
        )
    };
    let mut out = Vec::new();
    let mut i = start_val;
    while direction.saturating_mul(i) < direction.saturating_mul(stop_val) {
        if i >= 0 && i < len {
            out.push(i);
        }
        match i.checked_add(step) {
            Some(next) => i = next,
            // A hostile step that would carry the cursor out of range ends
            // the walk: at most one more element could have been visited.
            None => break,
        }
    }
    out
}

/// Python-style list slicing with step (mirror minja `slice<T>`). The result
/// is capped by the loop-iteration budget like every other materialized list.
pub(crate) fn slice_list(
    items: &[TemplateValue],
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Result<TemplateValue, LoaderError> {
    let len = items.len() as i64;
    let step = step.unwrap_or(1);
    if step == 0 {
        return Ok(TemplateValue::List(Vec::new()));
    }
    // The member translation always passes all three; the `slice` filter
    // allows 1-3 args (missing stop/step default per the builtin).
    let (start, stop) = match (start, stop) {
        (Some(a), Some(b)) => (a, b),
        (Some(a), None) => (0, a),
        (None, _) => (0, len),
    };
    let indices = slice_indexed(len, start, stop, step);
    let mut out = Vec::new();
    try_reserve_list(&mut out, indices.len(), "template slice length")?;
    for i in indices {
        out.push(items[i as usize].clone());
    }
    Ok(TemplateValue::List(out))
}

/// Byte-based string slicing (mirror minja slicing `std::string`). The walk
/// is output-proportioned (indices plus pushed bytes), so the result cap is
/// enforced after the build: shrinks of large inputs still succeed.
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
    let out = String::from_utf8_lossy(&out).into_owned();
    check_str_result(out.len(), "template slice bytes")?;
    Ok(out)
}

/// `select`/`selectattr` test evaluation against the known test names.
/// One filter call shares a single budget across all its items (see
/// [`values_equal`]), so a hostile wide list fails fast instead of spending
/// an unbounded number of per-item tests.
fn eval_select_test(
    test_name: &str,
    item: &TemplateValue,
    test_value: &TemplateValue,
    budget: &mut usize,
) -> Result<bool, LoaderError> {
    let args = if matches!(test_value, TemplateValue::Undefined) {
        Vec::new()
    } else {
        vec![test_value.clone()]
    };
    match apply_test_in(test_name, item, &args, false, budget)? {
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
        "keys" => {
            ensure_list_add(0, entries.len(), "template keys length")?;
            Ok(TemplateValue::List(
                entries
                    .iter()
                    .map(|(k, _)| TemplateValue::Str(k.clone()))
                    .collect(),
            ))
        }
        "values" => {
            ensure_list_add(0, entries.len(), "template values length")?;
            Ok(TemplateValue::List(
                entries.iter().map(|(_, v)| v.clone()).collect(),
            ))
        }
        "items" => {
            ensure_list_add(0, entries.len(), "template items length")?;
            Ok(TemplateValue::List(
                entries
                    .iter()
                    .map(|(k, v)| {
                        TemplateValue::List(vec![TemplateValue::Str(k.clone()), v.clone()])
                    })
                    .collect(),
            ))
        }
        "length" => Ok(TemplateValue::Int(entries.len() as i64)),
        "dictsort" => {
            let by_value =
                matches!(args.kwarg_or_pos("by", 1), TemplateValue::Str(ref s) if s == "value");
            let reverse = args.kwarg_or_pos("reverse", 2).is_truthy();
            ensure_list_add(0, entries.len(), "template dictsort length")?;
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
            // `i64::MIN.abs()` overflows: refuse with a typed error rather
            // than panicking (debug) or wrapping (release).
            TemplateValue::Int(i) => {
                Ok(TemplateValue::Int(i.checked_abs().ok_or_else(|| {
                    render_error("abs() integer overflow".to_owned())
                })?))
            }
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
