//! A2.9 hostile resource and arithmetic hardening: every template and
//! metadata value is hostile. Enormous values fail fast with typed errors —
//! never a panic, wrap, spin, or huge allocation — while small cases keep
//! reference behavior. Every hostile render runs under `catch_unwind` to
//! prove no panic path survives. No case here allocates more than a few MiB
//! (the output-contract scale); the enormous values all refuse before any
//! backing allocation.

use r9v_loader::{render_chat_template, TemplateValue, ToolCall};
use r9v_loader::{render_template_vars as render_vars, ChatContext, ChatMessage, LoaderError};
use std::collections::BTreeMap;
use std::panic::AssertUnwindSafe;

fn vars(pairs: &[(&str, TemplateValue)]) -> BTreeMap<String, TemplateValue> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect()
}

/// Renders under `catch_unwind`: `Ok` carries the render outcome, `Err` a
/// panic report (a test failure by itself).
fn hostile(
    source: &str,
    pairs: &[(&str, TemplateValue)],
) -> Result<Result<String, LoaderError>, String> {
    match std::panic::catch_unwind(AssertUnwindSafe(|| render_vars(source, vars(pairs)))) {
        Ok(outcome) => Ok(outcome),
        Err(_) => Err("render panicked".to_owned()),
    }
}

fn assert_err(source: &str, pairs: &[(&str, TemplateValue)]) -> LoaderError {
    match hostile(source, pairs) {
        Ok(Err(e)) => e,
        Ok(Ok(out)) => panic!("{source:?} rendered unexpectedly: {out:?}"),
        Err(panicked) => panic!("{source:?} {panicked}"),
    }
}

fn assert_limit(source: &str, pairs: &[(&str, TemplateValue)]) {
    match assert_err(source, pairs) {
        LoaderError::Limit { .. } => {}
        other => panic!("{source:?} refused with {other:?}, expected Limit"),
    }
}

fn assert_render_err(source: &str, pairs: &[(&str, TemplateValue)]) {
    match assert_err(source, pairs) {
        LoaderError::TemplateRender { .. } => {}
        other => panic!("{source:?} refused with {other:?}, expected TemplateRender"),
    }
}

fn assert_ok(source: &str, pairs: &[(&str, TemplateValue)], expected: &str) {
    match hostile(source, pairs) {
        Ok(Ok(out)) => assert_eq!(out, expected, "{source:?}"),
        Ok(Err(e)) => panic!("{source:?} failed unexpectedly: {e}"),
        Err(panicked) => panic!("{source:?} {panicked}"),
    }
}

fn int_pairs() -> Vec<(String, TemplateValue)> {
    vec![
        ("big".to_owned(), TemplateValue::Int(i64::MAX)),
        ("min".to_owned(), TemplateValue::Int(i64::MIN)),
    ]
}

fn int_refs(pairs: &[(String, TemplateValue)]) -> Vec<(&str, TemplateValue)> {
    pairs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect()
}

/// `range()` cardinality is preflighted in 128-bit math: hostile bounds
/// refuse before one element materializes, exact small ranges are unchanged.
#[test]
fn range_cardinality_refused_before_allocation() {
    let ints = int_pairs();
    let iv = int_refs(&ints);
    assert_limit("{{ range(9223372036854775807) }}", &[]);
    assert_limit("{{ range(0, 9223372036854775807) }}", &[]);
    assert_limit(
        "{{ range(-9223372036854775808, 9223372036854775807) }}",
        &[],
    );
    assert_limit("{{ range(0, -9223372036854775808, -1) }}", &[]);
    assert_limit("{{ range(big) }}", &iv);
    assert_limit("{{ range(0, big) }}", &iv);
    assert_limit("{{ range(min, big) }}", &iv);
    assert_limit("{{ range(200000) }}", &[]);
    // Huge steps that visit almost nothing still succeed exactly.
    assert_ok(
        "{{ range(5, 0, -9223372036854775808)|join(',') }}",
        &[],
        "5",
    );
    assert_ok("{{ range(3)|join(',') }}", &[], "0,1,2");
    assert_ok("{{ range(2, 8, 3)|join(',') }}", &[], "2,5");
    assert_ok("{{ range(5, 0, -2)|join(',') }}", &[], "5,3,1");
    assert_ok("{{ range(0)|length }}", &[], "0");
}

/// String repetition preflights `len * count` in both operand orders.
#[test]
fn string_repeat_both_orders_refuse_huge() {
    let ints = int_pairs();
    let iv = int_refs(&ints);
    assert_limit("{{ 'x' * 9223372036854775807 }}", &[]);
    assert_limit("{{ 9223372036854775807 * 'x' }}", &[]);
    assert_limit("{{ 'ab' * big }}", &iv);
    assert_limit("{{ big * 'ab' }}", &iv);
    assert_limit("{{ 'x' * 5000000 }}", &[]);
    assert_ok("{{ 'ab' * 3 }}", &[], "ababab");
    assert_ok("{{ 3 * 'ab' }}", &[], "ababab");
    assert_ok("{{ 'ab' * 0 }}", &[], "");
    assert_ok("{{ 'ab' * -5 }}", &[], "");
    assert_ok("{{ '' * 9223372036854775807 }}", &[], "");
}

/// Concatenation preflights the combined length for `~` and `+`.
#[test]
fn concat_refuses_past_budget() {
    let three = TemplateValue::Str("y".repeat(3 << 20));
    let two = TemplateValue::Str("z".repeat(2 << 20));
    let pairs = [("a", three), ("b", two)];
    assert_limit("{{ a ~ b }}", &pairs);
    assert_limit("{{ a + b }}", &pairs);
    assert_limit("{{ b ~ a }}", &pairs);
    assert_ok("{{ 'a' ~ 'b' }}", &[], "ab");
    assert_ok("{{ 'a' + 'b' }}", &[], "ab");
    assert_ok("{{ 1 ~ 'b' }}", &[], "1b");
}

/// `indent()` preflights both the pad width and the per-line growth.
#[test]
fn indent_width_and_line_growth_capped() {
    assert_limit("{{ 'a'|indent(width=9223372036854775807) }}", &[]);
    assert_limit("{{ 'a\\nb'|indent(width=5000000) }}", &[]);
    assert_ok("{{ 'a\\nb'|indent(width=4) }}", &[], "a\n    b");
    assert_ok(
        "{{ 'a\\n\\nb'|indent(width=2, blank=true) }}",
        &[],
        "a\n  \n  b",
    );
    assert_ok("{{ 'a\\nb'|indent(width=0) }}", &[], "a\nb");
}

/// `replace()` keeps exact semantics (including empty needles) while huge
/// counts clamp to actual replacements before the size preflight.
#[test]
fn replace_empty_needle_and_huge_counts() {
    assert_ok("{{ 'abc'|replace('', 'x') }}", &[], "xaxbxcx");
    // Pinned Jinja/minja boundary semantics: a finite count inserts at the
    // first `count` boundaries, not front-loaded (Python
    // `"abc".replace("", "x", 2) == "xaxbc"`).
    assert_ok("{{ 'abc'|replace('', 'x', 2) }}", &[], "xaxbc");
    assert_ok("{{ 'abc'|replace('', 'x', 0) }}", &[], "abc");
    assert_ok("{{ 'abc'|replace('', '') }}", &[], "abc");
    assert_ok("{{ ''|replace('', 'x') }}", &[], "x");
    assert_ok("{{ ''|replace('', 'x', 2) }}", &[], "x");
    assert_ok("{{ 'aaa'|replace('a', 'bb') }}", &[], "bbbbbb");
    assert_ok("{{ 'abc'|replace('b', 'x', 0) }}", &[], "abc");
    // A uselessly huge count that changes one occurrence still succeeds.
    assert_ok(
        "{{ 'abc'|replace('b', 'XYZ', 9223372036854775807) }}",
        &[],
        "aXYZc",
    );
    // A uselessly huge count clamped to one real replacement succeeds.
    assert_ok(
        "{{ 'a'|replace('a', 'bb', 9223372036854775807) }}",
        &[],
        "bb",
    );
    // A uselessly huge empty-needle count clamps to the input-bounded
    // boundary total (2 for `"a"`) and succeeds on the true result size.
    assert_ok(
        "{{ 'a'|replace('', 'x', 9223372036854775807) }}",
        &[],
        "xax",
    );
    assert_ok(
        "{{ 'a'|replace('', 'xxxxxxxxxx', 500000) }}",
        &[],
        "xxxxxxxxxxaxxxxxxxxxx",
    );
    // Empty-needle growth that truly passes the budget refuses up front,
    // both unlimited and finite-count (3 MiB of subjects each gain ~6 MiB
    // of insertions).
    let triple = TemplateValue::Str("a".repeat(3 << 20));
    assert_limit("{{ y|replace('', 'bb') }}", &[("y", triple.clone())]);
    assert_limit("{{ y|replace('', 'bb', 3000000) }}", &[("y", triple)]);
    // Doubling 3 MiB of matches would pass the budget: refused up front.
    let triple = TemplateValue::Str("a".repeat(3 << 20));
    assert_limit("{{ y|replace('a', 'bb') }}", &[("y", triple)]);
}

/// Empty-needle `replace()` inserts at Unicode scalar boundaries exactly,
/// including multibyte text: byte length never stands in for boundary
/// count, and zero/empty cases are unchanged.
#[test]
fn replace_empty_needle_unicode_boundaries() {
    assert_ok("{{ 'héllo'|replace('', 'x') }}", &[], "xhxéxlxlxox");
    assert_ok("{{ 'héllo'|replace('', 'x', 2) }}", &[], "xhxéllo");
    assert_ok("{{ 'héllo'|replace('', 'x', 0) }}", &[], "héllo");
    assert_ok("{{ 'a💯b'|replace('', '-') }}", &[], "-a-💯-b-");
    assert_ok("{{ 'a💯b'|replace('', '-', 1) }}", &[], "-a💯b");
    assert_ok("{{ '💯'|replace('💯', 'ab') }}", &[], "ab");
}

/// `format()` preflights every push, so hostile placeholder/argument sizes
/// fail at the output budget.
#[test]
fn format_growth_capped() {
    let big = TemplateValue::Str("q".repeat(5 << 20));
    assert_limit("{{ '{}'|format(x) }}", &[("x", big)]);
    assert_ok("{{ '{}-{}'|format(1, 2) }}", &[], "1-2");
    assert_ok("{{ 'a{}c'|format('b') }}", &[], "abc");
}

/// `join()` preflights pieces and delimiters; `split()` caps part count and
/// size; slices with hostile steps visit at most one element exactly.
#[test]
fn join_split_slice_chains_capped() {
    let huge = TemplateValue::Str("d".repeat(5 << 20));
    assert_limit("{{ ['a', 'b']|join(h) }}", &[("h", huge)]);
    assert_ok("{{ ['a', 'b']|join(', ') }}", &[], "a, b");
    // 400 KiB of pairs would materialize 200001 parts: refused by count.
    assert_limit("{{ ('a,' * 200000)|split(',') }}", &[]);
    assert_ok("{{ 'a,b'|split(',')|length }}", &[], "2");
    assert_ok(
        "{{ [1, 2, 3][0:10:9223372036854775807]|join(',') }}",
        &[],
        "1",
    );
    assert_ok("{{ 'hello'[0:10:9223372036854775807] }}", &[], "h");
    // A hugely negative step visits the start element, then the checked
    // cursor ends the walk (matches `range(4, -1, -huge) == [4]`).
    assert_ok("{{ 'hello'[4::-9223372036854775808] }}", &[], "o");
    assert_ok("{{ [3, 1, 2]|sort|join(',') }}", &[], "1,2,3");
}

/// `tojson()` preflights pads, separators, strings, and the document, and
/// refuses nesting past the depth budget.
#[test]
fn tojson_indent_recursion_output_capped() {
    assert_limit("{{ {'a': 1}|tojson(indent=9223372036854775807) }}", &[]);
    // A 1M indent on a nested document trips cumulatively past the budget.
    assert_limit("{{ {'a': {'b': {'c': 1}}}|tojson(indent=1000000) }}", &[]);
    let big = TemplateValue::Str("w".repeat(5 << 20));
    assert_limit("{{ x|tojson }}", &[("x", big)]);
    // 200-deep nesting refuses (entry validation reuses MAX_DEPTH).
    let mut deep = TemplateValue::Int(0);
    for _ in 0..200 {
        deep = TemplateValue::List(vec![deep]);
    }
    assert_limit("{{ x|tojson }}", &[("x", deep)]);
    assert_ok("{{ {'b': 2, 'a': 1}|tojson }}", &[], r#"{"a": 1, "b": 2}"#);
    assert_ok("{{ [1, 'a']|tojson }}", &[], r#"[1, "a"]"#);
}

/// Integer arithmetic is exact and checked: overflow, division/remainder by
/// zero, `i64::MIN` negation/absorption, and `MIN / -1` refuse typed.
#[test]
fn integer_arithmetic_edges_refuse() {
    let ints = int_pairs();
    let iv = int_refs(&ints);
    assert_render_err("{{ 9223372036854775807 + 1 }}", &[]);
    assert_render_err("{{ -9223372036854775807 - 2 }}", &[]);
    assert_render_err("{{ 3037000500 * 3037000500 }}", &[]);
    assert_render_err("{{ big + 1 }}", &iv);
    assert_render_err("{{ min - 1 }}", &iv);
    assert_render_err("{{ min * 2 }}", &iv);
    // `-` before a name never parses (the lexer folds signs only into
    // numbers, and only `not` builds a unary node), so negation overflow is
    // refused at parse; the evaluator's checked negation is defense in
    // depth for that unreachable arm.
    assert!(!assert_err("{{ -min }}", &iv).to_string().is_empty());
    assert_render_err("{{ min|abs }}", &iv);
    assert_render_err("{{ 1 / 0 }}", &[]);
    assert_render_err("{{ 1 % 0 }}", &[]);
    assert_render_err("{{ min / -1 }}", &iv);
    assert_render_err("{{ big % 0 }}", &iv);
    assert_ok("{{ 7 / 2 }}", &[], "3.5");
    assert_ok("{{ 7 % 3 }}", &[], "1");
    assert_ok("{{ 6 * 7 }}", &[], "42");
    assert_ok("{{ -5 }}", &[], "-5");
    assert_ok("{{ true + true }}", &[], "2");
    assert_ok("{{ (0 - 7) % 3 }}", &[], "-1");
}

/// Loop, step, and depth budgets trip on hostile iteration, unguarded
/// materialization, and macro recursion alike.
#[test]
fn loop_step_depth_counters_trip() {
    assert_limit("{% for i in range(200000) %}{{ i }}{% endfor %}", &[]);
    let big: Vec<TemplateValue> = (0..200_000).map(|_| TemplateValue::Int(1)).collect();
    assert_limit(
        "{% for x in xs %}{% endfor %}",
        &[("xs", TemplateValue::List(big))],
    );
    assert_limit("{% macro m() %}{{ m() }}{% endmacro %}{{ m() }}", &[]);
    assert_limit(
        "{% for x in xs if x %}{{ x }}{% endfor %}",
        &[(
            "xs",
            TemplateValue::List((0..200_000).map(|_| TemplateValue::Int(1)).collect()),
        )],
    );
}

/// Output appends and macro/caller buffers preflight: bombs refuse inside
/// the macro, not after the output check.
#[test]
fn output_appends_and_macro_buffers_capped() {
    let big = TemplateValue::Str("z".repeat(5 << 20));
    assert_limit("{{ x }}", &[("x", big)]);
    assert_limit(
        "{% macro m() %}{{ 'y' * 5000000 }}{% endmacro %}{{ m() }}",
        &[],
    );
    assert_limit(
        "{{ x|upper }}",
        &[("x", TemplateValue::Str("z".repeat(5 << 20)))],
    );
    assert_ok(
        "{% macro m(n) %}hi {{ n }}{% endmacro %}{{ m(1) }}{{ m(2) }}",
        &[],
        "hi 1hi 2",
    );
}

/// Hostile nesting inside the source budget fails closed (parser and
/// evaluator depth budgets) without touching the thread stack.
#[test]
fn hostile_nesting_depths_fail_closed() {
    let nested = format!(
        "{}{}{}",
        "{% if true %}".repeat(500),
        "1",
        "{% endif %}".repeat(500)
    );
    assert_limit(&nested, &[]);
    let parens = format!("{{{{ {}1{} }}}}", "(".repeat(500), ")".repeat(500));
    assert!(!assert_err(&parens, &[]).to_string().is_empty());
    let nots = format!("{{{{ {}true }}}}", "not ".repeat(500));
    assert!(!assert_err(&nots, &[]).to_string().is_empty());
    let adds = format!("{{{{ {} }}}}", vec!["1"; 5000].join("+"));
    assert_limit(&adds, &[]);
    let filters = format!("{{{{ x{} }}}}", "|upper".repeat(5000));
    let x = TemplateValue::Str("a".to_owned());
    assert_limit(&filters, &[("x", x)]);
    assert_ok("{{ (1 + 2) * 3 }}", &[], "9");
}

/// Values nested past entry validation by repeated wrapping (`[x]`
/// accumulation; loop bodies run in their own scope so the wrapping is
/// spelled out) still compare and interpolate safely: only `tojson` refuses
/// by its documented nesting cap.
#[test]
fn loop_built_deep_values_render_safely() {
    let build = format!("{{% set x = 0 %}}{}", "{% set x = [x] %}".repeat(200));
    assert_ok(&format!("{build}{{{{ x == x }}}}"), &[], "True");
    assert_ok(&format!("{build}{{{{ x }}}}"), &[], "0");
    assert_limit(&format!("{build}{{{{ x|tojson }}}}"), &[]);
}

/// Entry values deeper than the nesting budget fail fast; the message list
/// itself is capped like every materialized list.
#[test]
fn hostile_entry_values_refused() {
    let mut deep = TemplateValue::Int(0);
    for _ in 0..200 {
        deep = TemplateValue::List(vec![deep]);
    }
    assert_limit("{{ x }}", &[("x", deep)]);
    let messages: Vec<ChatMessage> = (0..150_000)
        .map(|i| ChatMessage {
            role: "user".to_owned(),
            content: format!("m{i}"),
            tool_calls: Vec::new(),
            reasoning_content: None,
        })
        .collect();
    let ctx = ChatContext {
        messages,
        ..ChatContext::default()
    };
    match std::panic::catch_unwind(AssertUnwindSafe(|| render_chat_template("{{ 1 }}", &ctx))) {
        Ok(Err(LoaderError::Limit { .. })) => {}
        Ok(Ok(_)) => panic!("150k messages rendered unexpectedly"),
        Ok(Err(other)) => panic!("150k messages refused with {other:?}, expected Limit"),
        Err(_) => panic!("150k messages panicked"),
    }
}

/// Equality and membership charge hostile wide inputs against the loop
/// budget and fail fast with `Limit` (never a panic or a silent `false`),
/// while small cases keep exact reference outcomes.
#[test]
fn equality_breadth_fails_fast() {
    let wide: Vec<TemplateValue> = (0..200_000).map(TemplateValue::Int).collect();
    let a = TemplateValue::List(wide.clone());
    let b = TemplateValue::List(wide);
    let pairs = [("a", a.clone()), ("b", b)];
    // Two 200k lists refuse on breadth before visiting elements.
    assert_limit("{{ a == b }}", &pairs);
    assert_limit("{{ a != b }}", &pairs);
    // A miss over a hostile list trips the shared budget mid-scan.
    assert_limit("{{ -1 in a }}", &[("a", a.clone())]);
    // A whole-filter `select` shares one budget across its items.
    assert_limit("{{ a|select('eq', -1)|length }}", &[("a", a.clone())]);
    // An early hit still succeeds without visiting the rest.
    assert_ok("{{ 1 in a }}", &[("a", a)], "True");
    // Wide objects trip the budget mid-scan instead of spending quadratic
    // key-search work first.
    let wide_obj = || {
        TemplateValue::Dict(
            (0..2000)
                .map(|i| (format!("k{i}"), TemplateValue::Int(i)))
                .collect(),
        )
    };
    assert_limit("{{ a == b }}", &[("a", wide_obj()), ("b", wide_obj())]);
    // Small cases keep exact outcomes through every equality caller.
    assert_ok(
        "{{ [1, [2, {'a': 1}]] == [1, [2, {'a': 1}]] }}",
        &[],
        "True",
    );
    assert_ok(
        "{{ [1, [2, {'a': 1}]] == [1, [2, {'a': 2}]] }}",
        &[],
        "False",
    );
    assert_ok("{{ [1, 2] == [1] }}", &[], "False");
    assert_ok("{{ [1, 2] != [1] }}", &[], "True");
    assert_ok("{{ 2 in [1, 2, 3] }}", &[], "True");
    assert_ok("{{ 9 in [1, 2, 3] }}", &[], "False");
    assert_ok("{{ 1 is eq(1) }}", &[], "True");
    assert_ok("{{ [1] is eq([2]) }}", &[], "False");
    assert_ok("{{ [1] is ne([2]) }}", &[], "True");
    assert_ok("{{ 2 is in([1, 2, 3]) }}", &[], "True");
    assert_ok("{{ {'a': 1} == {'a': 1} }}", &[], "True");
    assert_ok("{{ {'a': 1} == {'b': 1} }}", &[], "False");
}

/// Renders a full chat context under `catch_unwind`.
fn chat_hostile(source: &str, ctx: &ChatContext) -> Result<Result<String, LoaderError>, String> {
    match std::panic::catch_unwind(AssertUnwindSafe(|| render_chat_template(source, ctx))) {
        Ok(outcome) => Ok(outcome),
        Err(_) => Err("render panicked".to_owned()),
    }
}

fn assert_chat_limit(source: &str, ctx: &ChatContext) {
    match chat_hostile(source, ctx) {
        Ok(Err(LoaderError::Limit { .. })) => {}
        Ok(Err(other)) => panic!("{source:?} refused with {other:?}, expected Limit"),
        Ok(Ok(out)) => panic!("{source:?} rendered unexpectedly: {out:?}"),
        Err(panicked) => panic!("{source:?} {panicked}"),
    }
}

fn assert_chat_ok(source: &str, ctx: &ChatContext, expected: &str) {
    match chat_hostile(source, ctx) {
        Ok(Ok(out)) => assert_eq!(out, expected, "{source:?}"),
        Ok(Err(e)) => panic!("{source:?} failed unexpectedly: {e}"),
        Err(panicked) => panic!("{source:?} {panicked}"),
    }
}

fn user_message(content: String) -> ChatMessage {
    ChatMessage {
        role: "user".to_owned(),
        content,
        tool_calls: Vec::new(),
        reasoning_content: None,
    }
}

/// The aggregate context (roles, contents, tool-call names/arguments/ids,
/// reasoning, tools, tool choice, extras, nested values) is preflighted
/// against the byte/node/depth budgets before `message_value` or any clone:
/// unused hostile messages refuse instead of cloning gigabytes. Small
/// legitimate contexts render exactly.
#[test]
fn message_context_preflight_before_clone() {
    // A 5 MiB content the template never consults still refuses.
    let ctx = ChatContext {
        messages: vec![user_message("q".repeat(5 << 20))],
        ..ChatContext::default()
    };
    assert_chat_limit("{{ 1 }}", &ctx);
    // Tool-call names/arguments/ids count even when unused.
    let ctx = ChatContext {
        messages: vec![ChatMessage {
            role: "assistant".to_owned(),
            content: String::new(),
            tool_calls: vec![ToolCall {
                name: "f".to_owned(),
                arguments: "a".repeat(5 << 20),
                id: None,
            }],
            reasoning_content: None,
        }],
        ..ChatContext::default()
    };
    assert_chat_limit("{{ 1 }}", &ctx);
    // Reasoning text counts too.
    let ctx = ChatContext {
        messages: vec![ChatMessage {
            reasoning_content: Some("r".repeat(5 << 20)),
            ..user_message(String::new())
        }],
        ..ChatContext::default()
    };
    assert_chat_limit("{{ 1 }}", &ctx);
    // Many medium messages trip the aggregate even when each fits alone.
    let ctx = ChatContext {
        messages: (0..10)
            .map(|_| user_message("m".repeat(600 << 10)))
            .collect(),
        ..ChatContext::default()
    };
    assert_chat_limit("{{ 1 }}", &ctx);
    // Nested tools/tool_choice/extras count bytes, nodes, and depth.
    let big = TemplateValue::Str("w".repeat(5 << 20));
    let ctx = ChatContext {
        tools: Some(TemplateValue::List(vec![big.clone()])),
        ..ChatContext::default()
    };
    assert_chat_limit("{{ 1 }}", &ctx);
    let mut deep = TemplateValue::Int(0);
    for _ in 0..200 {
        deep = TemplateValue::List(vec![deep]);
    }
    let ctx = ChatContext {
        tool_choice: Some(deep.clone()),
        ..ChatContext::default()
    };
    assert_chat_limit("{{ 1 }}", &ctx);
    let mut extra = std::collections::BTreeMap::new();
    extra.insert("k".to_owned(), big);
    extra.insert("deep".to_owned(), deep);
    let ctx = ChatContext {
        extra,
        ..ChatContext::default()
    };
    assert_chat_limit("{{ 1 }}", &ctx);
    // A checked-arithmetic hostile: node counts near the budget refuse
    // instead of wrapping or cloning (120k one-key tool objects).
    let ctx = ChatContext {
        tools: Some(TemplateValue::List(
            (0..120_000)
                .map(|i| TemplateValue::Dict(vec![(format!("k{i}"), TemplateValue::Int(i))]))
                .collect(),
        )),
        ..ChatContext::default()
    };
    assert_chat_limit("{{ 1 }}", &ctx);
    // Legitimate reference-scale contexts are untouched: messages, a tool
    // call, tools, tool choice, and extras all render exactly.
    let mut extra = std::collections::BTreeMap::new();
    extra.insert("mode".to_owned(), TemplateValue::Str("fast".to_owned()));
    let ctx = ChatContext {
        messages: vec![
            user_message("hi".to_owned()),
            ChatMessage {
                role: "assistant".to_owned(),
                content: "hello".to_owned(),
                tool_calls: vec![ToolCall {
                    name: "f".to_owned(),
                    arguments: "{}".to_owned(),
                    id: Some("0".to_owned()),
                }],
                reasoning_content: None,
            },
        ],
        add_generation_prompt: true,
        tools: Some(TemplateValue::List(vec![TemplateValue::Dict(vec![(
            "name".to_owned(),
            TemplateValue::Str("f".to_owned()),
        )])])),
        tool_choice: Some(TemplateValue::Str("auto".to_owned())),
        extra,
        ..ChatContext::default()
    };
    assert_chat_ok(
        "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}",
        &ctx,
        "user:hi;assistant:hello;",
    );
    assert_chat_ok(
        "{{ tools|length }}-{{ tool_choice }}-{{ mode }}",
        &ctx,
        "1-auto-fast",
    );
}
