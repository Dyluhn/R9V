// SPDX-License-Identifier: Apache-2.0
//! Template interpreter: scopes, statement execution, expression
//! evaluation, and the `render` entry point (mirrors minja
//! `runtime.cpp`).

use std::collections::{BTreeMap, HashMap};

use crate::error::LoaderError;
use crate::template::builtins::{
    apply_binary, apply_filter, apply_test, as_string, ensure_list_add, filter_tojson, slice_list,
    slice_str, try_reserve_list, unsupported, CallArgs,
};
use crate::template::lexer::lex;
use crate::template::parser::{parse, Arg, AssignTarget, Expr, Param, Stmt};
use crate::template::{
    ChatContext, ChatMessage, TemplateValue, MAX_DEPTH, MAX_EVAL_DEPTH, MAX_LOOP_ITERS,
    MAX_OUTPUT_BYTES, MAX_STEPS, MAX_TEMPLATE_BYTES,
};

/// Renders `source` with `ctx` (Spec 10 §3.1).
pub fn render(source: &str, ctx: &ChatContext) -> Result<String, LoaderError> {
    if source.len() > MAX_TEMPLATE_BYTES {
        return Err(LoaderError::Limit {
            what: "chat template bytes",
            limit: MAX_TEMPLATE_BYTES,
            got: source.len(),
        });
    }
    let tokens = lex(source)?;
    let program = parse(&tokens)?;
    let mut interp = Interp::new();
    interp.push_scope();
    interp.define_globals(ctx)?;
    let mut out = String::new();
    let flow = interp.exec_body(&program.body, &mut out)?;
    debug_assert!(matches!(flow, Flow::Next));
    if out.len() > MAX_OUTPUT_BYTES {
        return Err(LoaderError::Limit {
            what: "rendered chat template bytes",
            limit: MAX_OUTPUT_BYTES,
            got: out.len(),
        });
    }
    Ok(out)
}

/// Renders `source` in a bare variable scope.
pub(crate) fn render_vars(
    source: &str,
    vars: BTreeMap<String, TemplateValue>,
) -> Result<String, LoaderError> {
    if source.len() > MAX_TEMPLATE_BYTES {
        return Err(LoaderError::Limit {
            what: "chat template bytes",
            limit: MAX_TEMPLATE_BYTES,
            got: source.len(),
        });
    }
    let tokens = lex(source)?;
    let program = parse(&tokens)?;
    let mut interp = Interp::new();
    interp.push_scope();
    for (k, v) in vars {
        Interp::check_value_depth(&v)?;
        interp.set(&k, v);
    }
    let mut out = String::new();
    let _ = interp.exec_body(&program.body, &mut out)?;
    Ok(out)
}

/// Statement flow control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    /// Continue normally.
    Next,
    /// `{% break %}`.
    Break,
    /// `{% continue %}`.
    Continue,
}

/// A callable value in the interpreter.
#[derive(Debug, Clone)]
enum Func {
    /// `{% macro %}` with its definition scope depth.
    Macro {
        /// Parameters.
        params: Vec<Param>,
        /// Body statements.
        body: Vec<Stmt>,
        /// Scope depth where defined (for closure-ish lookup; dynamic
        /// chain is used at call time like the reference).
        _depth: usize,
    },
    /// Bound builtin method (`receiver.name`).
    Method {
        /// Receiver value.
        receiver: TemplateValue,
        /// Method name.
        name: String,
    },
    /// Global function (`range`, `namespace`, `tojson`, ...).
    Global(String),
    /// The `caller` block inside `{% call %}`.
    Caller {
        /// Caller parameters.
        params: Vec<Param>,
        /// Caller body.
        body: Vec<Stmt>,
        /// Snapshot depth (executed in the call scope).
        _depth: usize,
    },
}

/// Resolved expression result: plain value or callable.
#[derive(Debug, Clone)]
enum Resolved {
    /// Plain value.
    Val(TemplateValue),
    /// Callable.
    Func(Func),
}

/// A user macro definition: parameters plus body.
type MacroDef = (Vec<Param>, Vec<Stmt>);

struct Interp {
    scopes: Vec<BTreeMap<String, TemplateValue>>,
    macros: Vec<HashMap<String, MacroDef>>,
    steps: usize,
    loop_iters: usize,
    depth: usize,
    /// Live `eval` frames (bounds flat-chain recursion; see
    /// [`MAX_EVAL_DEPTH`](crate::template::MAX_EVAL_DEPTH)).
    eval_depth: usize,
    /// True while the innermost `loop` binding is the for-loop object
    /// (which carries no builtins, mirroring minja).
    loop_is_bare: bool,
    /// `{% call %}` block stack for `caller`.
    caller_stack: Vec<Func>,
    /// True while the running loop iterates an object (single-name
    /// targets bind keys, mirroring minja).
    iter_object: bool,
}

impl Interp {
    fn new() -> Self {
        Self {
            scopes: Vec::new(),
            macros: Vec::new(),
            steps: 0,
            loop_iters: 0,
            depth: 0,
            eval_depth: 0,
            loop_is_bare: false,
            caller_stack: Vec::new(),
            iter_object: false,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
        self.macros.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.macros.pop();
    }

    fn set(&mut self, name: &str, value: TemplateValue) {
        if name == "loop" {
            // A user `{% set loop = ... %}` replaces the loop object.
            self.loop_is_bare = false;
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_owned(), value);
        }
    }

    /// Binds the for-loop object (bare: no builtins).
    fn set_loop(&mut self, value: TemplateValue) {
        self.loop_is_bare = true;
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert("loop".to_owned(), value);
        }
    }

    fn get(&self, name: &str) -> TemplateValue {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return value.clone();
            }
        }
        TemplateValue::Undefined
    }

    fn get_macro(&self, name: &str) -> Option<MacroDef> {
        for table in self.macros.iter().rev() {
            if let Some(found) = table.get(name) {
                return Some(found.clone());
            }
        }
        None
    }

    fn set_macro(&mut self, name: &str, params: Vec<Param>, body: Vec<Stmt>) {
        if let Some(table) = self.macros.last_mut() {
            table.insert(name.to_owned(), (params, body));
        }
    }

    fn step(&mut self) -> Result<(), LoaderError> {
        // Checked increment: hostile templates must trip the budget, never
        // wrap the counter (the bound makes overflow unreachable, but the
        // increment itself is still checked so no build mode can wrap it).
        self.steps = self.steps.checked_add(1).ok_or(LoaderError::Limit {
            what: "template interpreter steps",
            limit: MAX_STEPS,
            got: usize::MAX,
        })?;
        if self.steps > MAX_STEPS {
            return Err(LoaderError::Limit {
                what: "template interpreter steps",
                limit: MAX_STEPS,
                got: self.steps,
            });
        }
        Ok(())
    }

    fn enter(&mut self) -> Result<(), LoaderError> {
        self.depth = self.depth.checked_add(1).ok_or(LoaderError::Limit {
            what: "template evaluation depth",
            limit: MAX_DEPTH,
            got: usize::MAX,
        })?;
        if self.depth > MAX_DEPTH {
            return Err(LoaderError::Limit {
                what: "template evaluation depth",
                limit: MAX_DEPTH,
                got: self.depth,
            });
        }
        Ok(())
    }

    /// Charges one loop iteration against [`MAX_LOOP_ITERS`] with a checked
    /// increment. Both the filter pass and the run pass of `for` loops
    /// charge here, so unguarded materialization cannot bypass the budget.
    fn charge_loop_iter(&mut self) -> Result<(), LoaderError> {
        self.loop_iters = self.loop_iters.checked_add(1).ok_or(LoaderError::Limit {
            what: "template loop iterations",
            limit: MAX_LOOP_ITERS,
            got: usize::MAX,
        })?;
        if self.loop_iters > MAX_LOOP_ITERS {
            return Err(LoaderError::Limit {
                what: "template loop iterations",
                limit: MAX_LOOP_ITERS,
                got: self.loop_iters,
            });
        }
        Ok(())
    }

    fn exit(&mut self) {
        self.depth -= 1;
    }

    /// Preflights the aggregate render context before `message_value` or any
    /// clone materializes it: message roles, contents, names, tool-call
    /// arguments (and ids), reasoning text, tokens, tools, tool choice,
    /// extras, and nested values are summed without cloning, and the totals
    /// are checked against the existing budgets — bytes against
    /// [`MAX_OUTPUT_BYTES`], node counts against [`MAX_LOOP_ITERS`], nesting
    /// against [`MAX_DEPTH`]. An unused hostile message carrying gigabytes
    /// refuses here instead of after the clone. There is no new ceiling.
    /// DECISION(A2.9): only the aggregate is bounded (rejected per-field
    /// caps: legitimate reference templates vary across fields, and only
    /// the total clone cost matters); quick outcomes stay quick (small
    /// contexts add a few integer ops). Spec 10 §3.1 is silent on value
    /// shapes.
    fn check_context_budgets(ctx: &ChatContext) -> Result<(), LoaderError> {
        fn add_bytes(total: usize, add: usize, what: &'static str) -> Result<usize, LoaderError> {
            match total.checked_add(add) {
                Some(sum) if sum <= MAX_OUTPUT_BYTES => Ok(sum),
                _ => Err(LoaderError::Limit {
                    what,
                    limit: MAX_OUTPUT_BYTES,
                    got: total.saturating_add(add),
                }),
            }
        }
        fn add_nodes(total: usize, add: usize, what: &'static str) -> Result<usize, LoaderError> {
            match total.checked_add(add) {
                Some(sum) if sum <= MAX_LOOP_ITERS => Ok(sum),
                _ => Err(LoaderError::Limit {
                    what,
                    limit: MAX_LOOP_ITERS,
                    got: total.saturating_add(add),
                }),
            }
        }
        /// Adds one nested value's aggregate bytes, nodes, and depth without
        /// cloning it. Each container pre-charges its element count, so a
        /// hostile wide value refuses before its children are pushed.
        fn walk_value(
            value: &TemplateValue,
            mut bytes: usize,
            mut nodes: usize,
        ) -> Result<(usize, usize), LoaderError> {
            let mut stack: Vec<(&TemplateValue, usize)> = vec![(value, 0)];
            nodes = add_nodes(nodes, 1, "template context values")?;
            while let Some((v, depth)) = stack.pop() {
                if depth > MAX_DEPTH {
                    return Err(LoaderError::Limit {
                        what: "template value nesting depth",
                        limit: MAX_DEPTH,
                        got: depth,
                    });
                }
                match v {
                    TemplateValue::Str(s) | TemplateValue::SafeStr(s) => {
                        bytes = add_bytes(bytes, s.len(), "template context bytes")?;
                    }
                    TemplateValue::List(items) => {
                        nodes = add_nodes(nodes, items.len(), "template context values")?;
                        stack.extend(items.iter().map(|item| (item, depth + 1)));
                    }
                    TemplateValue::Dict(entries) => {
                        nodes = add_nodes(nodes, entries.len(), "template context values")?;
                        for (k, child) in entries {
                            bytes = add_bytes(bytes, k.len(), "template context bytes")?;
                            stack.push((child, depth + 1));
                        }
                    }
                    _ => {}
                }
            }
            Ok((bytes, nodes))
        }

        let mut bytes = 0usize;
        let mut nodes = 0usize;
        // The message list feeds `for` loops directly: cap its
        // materialization like every other list.
        nodes = add_nodes(nodes, ctx.messages.len(), "template messages length")?;
        for message in &ctx.messages {
            nodes = add_nodes(
                nodes,
                message.tool_calls.len(),
                "template tool calls length",
            )?;
            bytes = add_bytes(bytes, message.role.len(), "template context bytes")?;
            bytes = add_bytes(bytes, message.content.len(), "template context bytes")?;
            if let Some(reasoning) = &message.reasoning_content {
                bytes = add_bytes(bytes, reasoning.len(), "template context bytes")?;
            }
            for call in &message.tool_calls {
                bytes = add_bytes(bytes, call.name.len(), "template context bytes")?;
                bytes = add_bytes(bytes, call.arguments.len(), "template context bytes")?;
                if let Some(id) = &call.id {
                    bytes = add_bytes(bytes, id.len(), "template context bytes")?;
                }
            }
        }
        if let Some(token) = &ctx.bos_token {
            bytes = add_bytes(bytes, token.len(), "template context bytes")?;
        }
        if let Some(token) = &ctx.eos_token {
            bytes = add_bytes(bytes, token.len(), "template context bytes")?;
        }
        if let Some(tools) = &ctx.tools {
            (bytes, nodes) = walk_value(tools, bytes, nodes)?;
        }
        if let Some(choice) = &ctx.tool_choice {
            (bytes, nodes) = walk_value(choice, bytes, nodes)?;
        }
        for (k, v) in &ctx.extra {
            bytes = add_bytes(bytes, k.len(), "template context bytes")?;
            (bytes, nodes) = walk_value(v, bytes, nodes)?;
        }
        Ok(())
    }

    /// Validates one entry-point value's nesting with an explicit heap stack
    /// (never recursion): hostile values nest deeper than the thread stack.
    /// `write_json` recurses and relies on this bound; interpolation and
    /// equality are iterative and additionally safe for loop-deepened
    /// values. Reuses [`MAX_DEPTH`]; there is no new ceiling.
    /// DECISION(A2.9): entry values deeper than `MAX_DEPTH` fail with
    /// `Limit` (rejected trusting serving-layer shapes: tools and kwargs
    /// are request data). Loop-built values are handled by making the
    /// remaining consumers iterative or depth-capped, not by re-checking
    /// every assignment. Spec 10 §3.1 is silent on value shapes.
    fn check_value_depth(value: &TemplateValue) -> Result<(), LoaderError> {
        let mut stack: Vec<(&TemplateValue, usize)> = vec![(value, 0)];
        while let Some((v, depth)) = stack.pop() {
            if depth > MAX_DEPTH {
                return Err(LoaderError::Limit {
                    what: "template value nesting depth",
                    limit: MAX_DEPTH,
                    got: depth,
                });
            }
            match v {
                TemplateValue::List(items) => {
                    stack.extend(items.iter().map(|i| (i, depth + 1)));
                }
                TemplateValue::Dict(entries) => {
                    stack.extend(entries.iter().map(|(_, i)| (i, depth + 1)));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Defines the serving globals (Spec 10 §3.1).
    fn define_globals(&mut self, ctx: &ChatContext) -> Result<(), LoaderError> {
        // Aggregate preflight before `message_value` or any clone: an
        // unused hostile message must refuse here, not after cloning
        // gigabytes of strings no template consults.
        Self::check_context_budgets(ctx)?;
        let messages: Vec<TemplateValue> = ctx.messages.iter().map(message_value).collect();
        self.set("messages", TemplateValue::List(messages));
        self.set(
            "add_generation_prompt",
            TemplateValue::Bool(ctx.add_generation_prompt),
        );
        self.set(
            "bos_token",
            ctx.bos_token
                .clone()
                .map(TemplateValue::Str)
                .unwrap_or(TemplateValue::None),
        );
        self.set(
            "eos_token",
            ctx.eos_token
                .clone()
                .map(TemplateValue::Str)
                .unwrap_or(TemplateValue::None),
        );
        // DECISION(A2.9): absent tools/tool_choice render as `none` (not
        // undefined) so `tools is none` works like the reference server
        // context, which always defines these keys. Rejected undefined
        // (would break `is defined` checks templates rely on). Spec 10
        // §3.1 names the keys but not their absent values.
        self.set("tools", ctx.tools.clone().unwrap_or(TemplateValue::None));
        self.set(
            "tool_choice",
            ctx.tool_choice.clone().unwrap_or(TemplateValue::None),
        );
        self.set(
            "enable_thinking",
            ctx.enable_thinking
                .map(TemplateValue::Bool)
                .unwrap_or(TemplateValue::None),
        );
        for (k, v) in &ctx.extra {
            self.set(k, v.clone());
        }
        Ok(())
    }

    /// Executes a statement list, appending rendered text to `out`.
    fn exec_body(&mut self, body: &[Stmt], out: &mut String) -> Result<Flow, LoaderError> {
        for stmt in body {
            self.step()?;
            match self.exec_stmt(stmt, out)? {
                Flow::Next => {}
                flow => return Ok(flow),
            }
            if out.len() > MAX_OUTPUT_BYTES {
                return Err(LoaderError::Limit {
                    what: "rendered chat template bytes",
                    limit: MAX_OUTPUT_BYTES,
                    got: out.len(),
                });
            }
        }
        Ok(Flow::Next)
    }

    fn exec_stmt(&mut self, stmt: &Stmt, out: &mut String) -> Result<Flow, LoaderError> {
        match stmt {
            Stmt::Text(text) => {
                // Literal text is source-bounded, but the running output
                // still preflights so a hostile template fails here rather
                // than past the budget.
                crate::template::builtins::ensure_str_add(
                    out.len(),
                    text.len(),
                    "rendered chat template bytes",
                )?;
                out.push_str(text);
                Ok(Flow::Next)
            }
            Stmt::Expr(expr) => {
                let value = self.eval_as_value(expr)?;
                append_interpolation(&value, out)?;
                Ok(Flow::Next)
            }
            Stmt::If { branches, orelse } => {
                for (test, body) in branches {
                    if self.eval_as_value(test)?.is_truthy() {
                        return self.exec_body(body, out);
                    }
                }
                self.exec_body(orelse, out)
            }
            Stmt::For {
                targets,
                iter,
                guard,
                body,
                orelse,
            } => self.exec_for(targets, iter, guard, body, orelse, out),
            Stmt::Set {
                targets,
                value,
                body,
            } => {
                if let Some(expr) = value {
                    let val = self.eval_as_value(expr)?;
                    self.assign(targets, val)?;
                } else {
                    let mut buf = String::new();
                    let _ = self.exec_body(body, &mut buf)?;
                    let val = TemplateValue::Str(buf);
                    self.assign(targets, val)?;
                }
                Ok(Flow::Next)
            }
            Stmt::Macro { name, params, body } => {
                self.set_macro(name, params.clone(), body.clone());
                Ok(Flow::Next)
            }
            Stmt::Call {
                callee,
                args,
                caller_params,
                body,
            } => {
                let (params, macro_body) =
                    self.get_macro(callee)
                        .ok_or_else(|| LoaderError::TemplateRender {
                            detail: format!("unknown macro '{callee}'"),
                        })?;
                let call_args = self.eval_args(args)?;
                // The caller block becomes a zero-arg-ish callable bound
                // as `caller` inside the macro scope.
                let caller = Func::Caller {
                    params: caller_params.clone(),
                    body: body.clone(),
                    _depth: self.scopes.len(),
                };
                let returned = self.invoke_macro(&params, &macro_body, call_args, Some(caller))?;
                append_interpolation(&returned, out)?;
                Ok(Flow::Next)
            }
            Stmt::FilterBlock { name, args, body } => {
                let mut buf = String::new();
                let _ = self.exec_body(body, &mut buf)?;
                let call_args = self.eval_args(args)?;
                let value = apply_filter(name, &TemplateValue::Str(buf), &call_args)?;
                append_interpolation(&value, out)?;
                Ok(Flow::Next)
            }
            Stmt::Break => Ok(Flow::Break),
            Stmt::Continue => Ok(Flow::Continue),
        }
    }

    /// Assigns `set` targets (names, tuples, namespace attributes).
    fn assign(
        &mut self,
        targets: &[AssignTarget],
        value: TemplateValue,
    ) -> Result<(), LoaderError> {
        if targets.len() == 1 {
            return self.assign_one(&targets[0], value);
        }
        let TemplateValue::List(items) = value else {
            return Err(LoaderError::TemplateRender {
                detail: "cannot unpack non-iterable type in set".to_owned(),
            });
        };
        if items.len() != targets.len() {
            return Err(LoaderError::TemplateRender {
                detail: format!(
                    "too {} items to unpack in set",
                    if targets.len() > items.len() {
                        "few"
                    } else {
                        "many"
                    }
                ),
            });
        }
        for (target, item) in targets.iter().zip(items) {
            self.assign_one(target, item)?;
        }
        Ok(())
    }

    fn assign_one(
        &mut self,
        target: &AssignTarget,
        value: TemplateValue,
    ) -> Result<(), LoaderError> {
        match target {
            AssignTarget::Name(name) => {
                self.set(name, value);
                Ok(())
            }
            AssignTarget::Attr { obj, attr } => {
                // DECISION(A2.9): namespace attribute mutation updates the
                // declaring scope; rejected assigning into the innermost scope
                // (shadows and loses mutations on scope exit) and rejected
                // global mutation of undeclared names. Spec 10 §3.1 silent
                // on namespace scoping.
                for scope in self.scopes.iter_mut().rev() {
                    if let Some(target_val) = scope.get_mut(obj) {
                        let TemplateValue::Dict(entries) = target_val else {
                            return Err(LoaderError::TemplateRender {
                                detail: format!("cannot set attribute on non-namespace '{obj}'"),
                            });
                        };
                        match entries.iter_mut().find(|(k, _)| k == attr) {
                            Some((_, v)) => *v = value,
                            None => {
                                ensure_list_add(entries.len(), 1, "template namespace attributes")?;
                                entries.push((attr.clone(), value));
                            }
                        }
                        return Ok(());
                    }
                }
                Err(LoaderError::TemplateRender {
                    detail: format!("cannot set attribute on non-namespace '{obj}'"),
                })
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn exec_for(
        &mut self,
        targets: &[String],
        iter: &Expr,
        guard: &Option<Expr>,
        body: &[Stmt],
        orelse: &[Stmt],
        out: &mut String,
    ) -> Result<Flow, LoaderError> {
        let iter_value = self.eval_as_value(iter)?;
        // Undefined iterates as empty (mirror minja).
        let items: Vec<TemplateValue> = match &iter_value {
            TemplateValue::Undefined => Vec::new(),
            TemplateValue::List(items) => {
                ensure_list_add(0, items.len(), "template loop list length")?;
                items.clone()
            }
            TemplateValue::Dict(entries) => {
                // Pair expansion clones every entry: cap it before
                // materializing, like every other list.
                ensure_list_add(0, entries.len(), "template loop object length")?;
                entries
                    .iter()
                    .map(|(k, v)| {
                        TemplateValue::List(vec![TemplateValue::Str(k.clone()), v.clone()])
                    })
                    .collect()
            }
            _ => {
                return Err(LoaderError::TemplateRender {
                    detail: format!(
                        "expected iterable or object type in for loop: got {}",
                        iter_value.type_name()
                    ),
                });
            }
        };
        // Filter pass with the guard (`for x in y if c`). The pass charges
        // loop iterations like the run pass: an unguarded 10M-item hostile
        // list must trip the budget during filtering, not after cloning it
        // all. Without a guard the pass is skipped (nothing to test).
        let iter_is_object = matches!(iter_value, TemplateValue::Dict(_));
        let saved_iter_object = std::mem::replace(&mut self.iter_object, iter_is_object);
        let filtered: Vec<TemplateValue> = if guard.is_none() {
            items
        } else {
            self.push_scope();
            let mut filtered: Vec<TemplateValue> = Vec::new();
            for item in &items {
                // Errors abort the render, so scope cleanup on this path is
                // unnecessary; the success path below pops.
                self.charge_loop_iter()?;
                self.bind_target(targets, item.clone())?;
                if let Some(test) = guard {
                    if !self.eval_as_value(test)?.is_truthy() {
                        continue;
                    }
                }
                filtered.push(item.clone());
            }
            // Drop the filter scope; the run scope binds per iteration below.
            self.pop_scope();
            filtered
        };

        self.push_scope();
        let mut iterated = false;
        for (i, item) in filtered.iter().enumerate() {
            self.charge_loop_iter()?;
            self.bind_loop_object(&filtered, i);
            self.bind_target(targets, item.clone())?;
            match self.exec_body(body, out)? {
                Flow::Next => {}
                Flow::Continue => {}
                Flow::Break => break,
            }
            iterated = true;
        }
        self.iter_object = saved_iter_object;
        self.pop_scope();
        if !iterated {
            let flow = self.exec_body(orelse, out)?;
            if !matches!(flow, Flow::Next) {
                return Ok(flow);
            }
        }
        Ok(Flow::Next)
    }

    /// Binds loop/unpack targets (1 name, or 2 names over a pair).
    fn bind_target(&mut self, targets: &[String], value: TemplateValue) -> Result<(), LoaderError> {
        if targets.len() == 1 {
            // Single name over a dict iterates keys (mirror minja: pairs
            // are built for objects, then `[0]` is bound).
            if let TemplateValue::List(pair) = &value {
                if pair.len() == 2 && self.iter_object {
                    self.set(&targets[0], pair[0].clone());
                    return Ok(());
                }
            }
            self.set(&targets[0], value);
            return Ok(());
        }
        let TemplateValue::List(pair) = value else {
            return Err(LoaderError::TemplateRender {
                detail: "cannot unpack non-iterable type".to_owned(),
            });
        };
        if pair.len() != targets.len() {
            return Err(LoaderError::TemplateRender {
                detail: format!(
                    "too {} items to unpack",
                    if targets.len() > pair.len() {
                        "few"
                    } else {
                        "many"
                    }
                ),
            });
        }
        for (name, item) in targets.iter().zip(pair) {
            self.set(name, item);
        }
        Ok(())
    }

    /// Binds the for-loop `loop` object for iteration `i` of `total`.
    fn bind_loop_object(&mut self, filtered: &[TemplateValue], i: usize) {
        let total = filtered.len();
        self.set_loop(TemplateValue::Dict(vec![
            ("index".to_owned(), TemplateValue::Int(i as i64 + 1)),
            ("index0".to_owned(), TemplateValue::Int(i as i64)),
            (
                "revindex".to_owned(),
                TemplateValue::Int(total as i64 - i as i64),
            ),
            (
                "revindex0".to_owned(),
                TemplateValue::Int(total as i64 - i as i64 - 1),
            ),
            ("first".to_owned(), TemplateValue::Bool(i == 0)),
            ("last".to_owned(), TemplateValue::Bool(i + 1 == total)),
            ("length".to_owned(), TemplateValue::Int(total as i64)),
            (
                "previtem".to_owned(),
                if i > 0 {
                    filtered[i - 1].clone()
                } else {
                    TemplateValue::Undefined
                },
            ),
            (
                "nextitem".to_owned(),
                if i + 1 < total {
                    filtered[i + 1].clone()
                } else {
                    TemplateValue::Undefined
                },
            ),
        ]));
    }

    /// Evaluates an expression to a plain value (callables made by bare
    /// macro names evaluate to their rendered... no: using a macro as a
    /// value is an error, mirroring minja's function type).
    fn eval_as_value(&mut self, expr: &Expr) -> Result<TemplateValue, LoaderError> {
        match self.eval(expr)? {
            Resolved::Val(v) => Ok(v),
            Resolved::Func(_) => Err(LoaderError::TemplateRender {
                detail: "macro used as a value".to_owned(),
            }),
        }
    }

    /// Evaluates call arguments (spreads flattened).
    fn eval_args(&mut self, args: &[Arg]) -> Result<CallArgs, LoaderError> {
        let mut out = CallArgs::default();
        for arg in args {
            match arg {
                Arg::Pos(expr) => out.positional.push(self.eval_as_value(expr)?),
                Arg::Kw(name, expr) => {
                    out.keywords.push((name.clone(), self.eval_as_value(expr)?));
                }
                Arg::Spread(expr) => {
                    let value = self.eval_as_value(expr)?;
                    match value {
                        TemplateValue::List(items) => {
                            // A hostile list must not blow up the argument
                            // vector past the loop-iteration budget.
                            ensure_list_add(
                                out.positional.len(),
                                items.len(),
                                "template call arguments length",
                            )?;
                            out.positional.extend(items);
                        }
                        TemplateValue::Dict(entries) => {
                            ensure_list_add(
                                out.keywords.len(),
                                entries.len(),
                                "template call arguments length",
                            )?;
                            for (k, v) in entries {
                                out.keywords.push((k, v));
                            }
                        }
                        _ => {
                            return Err(LoaderError::TemplateRender {
                                detail: "cannot spread non-iterable".to_owned(),
                            });
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn eval(&mut self, expr: &Expr) -> Result<Resolved, LoaderError> {
        self.step()?;
        // Bound flat-chain recursion (`a+a+…` nests one frame per term).
        // Errors abort the render, so a leaked count on the error path is
        // harmless; every success path below decrements.
        self.eval_depth = self.eval_depth.checked_add(1).ok_or(LoaderError::Limit {
            what: "template evaluation depth",
            limit: MAX_EVAL_DEPTH,
            got: usize::MAX,
        })?;
        if self.eval_depth > MAX_EVAL_DEPTH {
            return Err(LoaderError::Limit {
                what: "template evaluation depth",
                limit: MAX_EVAL_DEPTH,
                got: self.eval_depth,
            });
        }
        let result = self.eval_inner(expr);
        self.eval_depth -= 1;
        result
    }

    #[allow(clippy::too_many_lines)]
    fn eval_inner(&mut self, expr: &Expr) -> Result<Resolved, LoaderError> {
        match expr {
            Expr::Lit(value) => Ok(Resolved::Val(value.clone())),
            Expr::Name(name) => {
                if let Some((params, body)) = self.get_macro(name) {
                    return Ok(Resolved::Func(Func::Macro {
                        params,
                        body,
                        _depth: self.scopes.len(),
                    }));
                }
                match name.as_str() {
                    // Globals (mirrors minja `global_builtins`; anything
                    // else resolves through scopes, defaulting to
                    // undefined, so unknown callees fail at call time).
                    "range" | "namespace" | "tojson" | "raise_exception" | "strftime_now" => {
                        Ok(Resolved::Func(Func::Global(name.clone())))
                    }
                    _ => Ok(Resolved::Val(self.get(name))),
                }
            }
            Expr::Attr(object, attr) => self.eval_attr(object, attr),
            // Heavier arms live in helpers below so their locals stay out
            // of `eval_inner`'s frame, which is live at every level of a
            // hostile flat chain (129 deep) in debug builds.
            Expr::Index(object, key) => self.eval_index_expr(object, key),
            Expr::Slice(object, start, stop, step) => {
                self.eval_slice_expr(object, start, stop, step)
            }
            Expr::Call(callee, args) => self.eval_call_expr(callee, args),
            Expr::Filter(operand, name, args) => self.eval_filter_expr(operand, name, args),
            Expr::Test(operand, name, args, negated) => {
                self.eval_test_expr(operand, name, args, *negated)
            }
            Expr::BinOp(op, left, right) if op == "and" || op == "or" => {
                let left_value = self.eval_as_value(left)?;
                if op == "and" {
                    if left_value.is_truthy() {
                        Ok(Resolved::Val(self.eval_as_value(right)?))
                    } else {
                        Ok(Resolved::Val(left_value))
                    }
                } else if left_value.is_truthy() {
                    Ok(Resolved::Val(left_value))
                } else {
                    Ok(Resolved::Val(self.eval_as_value(right)?))
                }
            }
            Expr::BinOp(op, left, right) => {
                let left_value = self.eval_as_value(left)?;
                let right_value = self.eval_as_value(right)?;
                Ok(Resolved::Val(apply_binary(op, &left_value, &right_value)?))
            }
            Expr::UnOp(op, operand) => {
                let value = self.eval_as_value(operand)?;
                match op.as_str() {
                    "not" => Ok(Resolved::Val(TemplateValue::Bool(!value.is_truthy()))),
                    // `-i64::MIN` overflows: checked negation refuses with a
                    // typed error instead of panicking (debug) or wrapping
                    // (release).
                    "-" => match value {
                        TemplateValue::Int(i) => Ok(Resolved::Val(TemplateValue::Int(
                            i.checked_neg().ok_or_else(|| LoaderError::TemplateRender {
                                detail: "unary - integer overflow".to_owned(),
                            })?,
                        ))),
                        TemplateValue::Float(f) => Ok(Resolved::Val(TemplateValue::Float(-f))),
                        _ => Err(LoaderError::TemplateRender {
                            detail: "unary - operator requires numeric operand".to_owned(),
                        }),
                    },
                    _ => Err(LoaderError::TemplateRender {
                        detail: format!("unknown unary operator {op:?}"),
                    }),
                }
            }
            Expr::Ternary(value, test, orelse) => {
                if self.eval_as_value(test)?.is_truthy() {
                    Ok(Resolved::Val(self.eval_as_value(value)?))
                } else {
                    Ok(Resolved::Val(self.eval_as_value(orelse)?))
                }
            }
            Expr::Select(_, _) => Err(LoaderError::TemplateRender {
                detail: "select guard outside a for loop".to_owned(),
            }),
            Expr::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.eval_as_value(item)?);
                }
                Ok(Resolved::Val(TemplateValue::List(out)))
            }
            Expr::Tuple(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.eval_as_value(item)?);
                }
                Ok(Resolved::Val(TemplateValue::List(out)))
            }
            Expr::Dict(entries) => {
                let mut out = Vec::with_capacity(entries.len());
                for (k, v) in entries {
                    let key = as_string(&self.eval_as_value(k)?)?;
                    out.push((key, self.eval_as_value(v)?));
                }
                Ok(Resolved::Val(TemplateValue::Dict(out)))
            }
        }
    }

    /// Outlined `Expr::Index` arm (see the note at the call site).
    fn eval_index_expr(&mut self, object: &Expr, key: &Expr) -> Result<Resolved, LoaderError> {
        let obj = self.eval_as_value(object)?;
        let key_value = self.eval_as_value(key)?;
        self.eval_index(&obj, &key_value)
    }

    /// Outlined `Expr::Slice` arm (see the note at the call site).
    fn eval_slice_expr(
        &mut self,
        object: &Expr,
        start: &Option<Box<Expr>>,
        stop: &Option<Box<Expr>>,
        step: &Option<Box<Expr>>,
    ) -> Result<Resolved, LoaderError> {
        let obj = self.eval_as_value(object)?;
        let start_v = match start {
            Some(e) => self.eval_as_value(e)?,
            None => TemplateValue::Undefined,
        };
        let stop_v = match stop {
            Some(e) => self.eval_as_value(e)?,
            None => TemplateValue::Undefined,
        };
        let step_v = match step {
            Some(e) => self.eval_as_value(e)?,
            None => TemplateValue::Undefined,
        };
        Ok(Resolved::Val(self.slice(&obj, &start_v, &stop_v, &step_v)?))
    }

    /// Outlined `Expr::Call` arm (see the note at the call site).
    fn eval_call_expr(&mut self, callee: &Expr, args: &[Arg]) -> Result<Resolved, LoaderError> {
        let target = self.eval(callee)?;
        let call_args = self.eval_args(args)?;
        Ok(Resolved::Val(self.invoke(target, call_args)?))
    }

    /// Outlined `Expr::Filter` arm (see the note at the call site).
    fn eval_filter_expr(
        &mut self,
        operand: &Expr,
        name: &str,
        args: &[Arg],
    ) -> Result<Resolved, LoaderError> {
        let input = self.eval_as_value(operand)?;
        let call_args = self.eval_args(args)?;
        Ok(Resolved::Val(apply_filter(name, &input, &call_args)?))
    }

    /// Outlined `Expr::Test` arm (see the note at the call site).
    fn eval_test_expr(
        &mut self,
        operand: &Expr,
        name: &str,
        args: &[Expr],
        negated: bool,
    ) -> Result<Resolved, LoaderError> {
        let input = self.eval_as_value(operand)?;
        let mut evaluated = Vec::with_capacity(args.len());
        for arg in args {
            evaluated.push(self.eval_as_value(arg)?);
        }
        Ok(Resolved::Val(apply_test(
            name, &input, &evaluated, negated,
        )?))
    }

    /// Static `obj.prop`: builtin first (as a bound method value), then
    /// key (mirror minja). The `loop` object carries no builtins in the
    /// reference, so it always uses key lookup while it is loop-bound.
    fn eval_attr(&mut self, object: &Expr, prop: &str) -> Result<Resolved, LoaderError> {
        if let Expr::Name(name) = object {
            if name == "caller" {
                return self.eval_caller();
            }
            if name == "loop" && self.loop_is_bare {
                let obj = self.eval_as_value(object)?;
                return Ok(Resolved::Val(
                    self.plain_attr(&obj, &TemplateValue::Str(prop.to_owned()))?,
                ));
            }
        }
        let obj = self.eval_as_value(object)?;
        if builtin_exists(&obj, prop) {
            return Ok(Resolved::Func(Func::Method {
                receiver: obj,
                name: prop.to_owned(),
            }));
        }
        Ok(Resolved::Val(
            self.plain_attr(&obj, &TemplateValue::Str(prop.to_owned()))?,
        ))
    }

    /// Resolves bare `caller` (the `{% call %}` block) from the stack.
    fn eval_caller(&mut self) -> Result<Resolved, LoaderError> {
        match self.caller_stack.last().cloned() {
            Some(caller) => Ok(Resolved::Func(caller)),
            None => Ok(Resolved::Val(TemplateValue::Undefined)),
        }
    }

    /// Computed `obj[expr]`: key first for objects (then builtin), index
    /// or builtin for arrays/strings (mirror minja).
    fn eval_index(
        &mut self,
        object: &TemplateValue,
        property: &TemplateValue,
    ) -> Result<Resolved, LoaderError> {
        if matches!(property, TemplateValue::Undefined) {
            return Ok(Resolved::Val(TemplateValue::Undefined));
        }
        match object {
            TemplateValue::Dict(_) => {
                let key = attr_key(property)?;
                let found = object.get_key(&key);
                if found.is_defined() {
                    return Ok(Resolved::Val(found));
                }
                if builtin_exists(object, &key) {
                    return Ok(Resolved::Func(Func::Method {
                        receiver: object.clone(),
                        name: key,
                    }));
                }
                Ok(Resolved::Val(TemplateValue::Undefined))
            }
            TemplateValue::List(items) => match property {
                TemplateValue::Int(i) => {
                    // Checked wrap-around: a hostile index must saturate
                    // into `undefined`, never wrap or panic.
                    let mut index = *i;
                    if index < 0 {
                        index = index.saturating_add(items.len() as i64);
                    }
                    if index < 0 || index >= items.len() as i64 {
                        return Ok(Resolved::Val(TemplateValue::Undefined));
                    }
                    Ok(Resolved::Val(items[index as usize].clone()))
                }
                TemplateValue::Str(key) => {
                    if builtin_exists(object, key) {
                        return Ok(Resolved::Func(Func::Method {
                            receiver: object.clone(),
                            name: key.clone(),
                        }));
                    }
                    Err(LoaderError::TemplateRender {
                        detail: format!("unknown array builtin '{key}'"),
                    })
                }
                _ => Err(LoaderError::TemplateRender {
                    detail: "cannot access property with non-string/non-number".to_owned(),
                }),
            },
            TemplateValue::Str(s) => match property {
                TemplateValue::Int(i) => {
                    // Byte indexing, no negative wrap (mirror minja).
                    if *i < 0 || *i >= s.len() as i64 {
                        return Ok(Resolved::Val(TemplateValue::Undefined));
                    }
                    let byte = s.as_bytes()[*i as usize];
                    Ok(Resolved::Val(TemplateValue::Str(
                        (byte as char).to_string(),
                    )))
                }
                TemplateValue::Str(key) => {
                    if builtin_exists(object, key) {
                        return Ok(Resolved::Func(Func::Method {
                            receiver: object.clone(),
                            name: key.clone(),
                        }));
                    }
                    Err(LoaderError::TemplateRender {
                        detail: format!("unknown string builtin '{key}'"),
                    })
                }
                _ => Err(LoaderError::TemplateRender {
                    detail: "cannot access property with non-string/non-number".to_owned(),
                }),
            },
            TemplateValue::Undefined => Ok(Resolved::Val(TemplateValue::Undefined)),
            _ => {
                let TemplateValue::Str(key) = property else {
                    return Err(LoaderError::TemplateRender {
                        detail: "cannot access property with non-string".to_owned(),
                    });
                };
                if builtin_exists(object, key) {
                    return Ok(Resolved::Func(Func::Method {
                        receiver: object.clone(),
                        name: key.clone(),
                    }));
                }
                Err(LoaderError::TemplateRender {
                    detail: format!("unknown builtin '{key}' for {}", object.type_name()),
                })
            }
        }
    }

    /// Plain key/index access without builtin lookup.
    fn plain_attr(
        &mut self,
        object: &TemplateValue,
        property: &TemplateValue,
    ) -> Result<TemplateValue, LoaderError> {
        match object {
            TemplateValue::Dict(_) => {
                let key = attr_key(property)?;
                Ok(object.get_key(&key))
            }
            TemplateValue::List(items) => match property {
                TemplateValue::Int(i) => {
                    let mut index = *i;
                    if index < 0 {
                        index = index.saturating_add(items.len() as i64);
                    }
                    if index < 0 || index >= items.len() as i64 {
                        return Ok(TemplateValue::Undefined);
                    }
                    Ok(items[index as usize].clone())
                }
                _ => Ok(TemplateValue::Undefined),
            },
            TemplateValue::Undefined => Ok(TemplateValue::Undefined),
            _ => Ok(TemplateValue::Undefined),
        }
    }

    /// Slicing `obj[a:b:c]` (mirror minja member translation to
    /// `slice(start, stop, step)` with negative-step defaults).
    fn slice(
        &mut self,
        object: &TemplateValue,
        start: &TemplateValue,
        stop: &TemplateValue,
        step: &TemplateValue,
    ) -> Result<TemplateValue, LoaderError> {
        let step = opt_index(step)?.unwrap_or(1);
        if step == 0 {
            return Err(LoaderError::TemplateRender {
                detail: "slice step cannot be zero".to_owned(),
            });
        }
        match object {
            TemplateValue::List(items) => {
                let len = items.len() as i64;
                let (default_start, default_stop) = if step < 0 {
                    (len.saturating_sub(1), -1)
                } else {
                    (0, len)
                };
                let start = opt_index(start)?.unwrap_or(default_start);
                let stop = opt_index(stop)?.unwrap_or(default_stop);
                Ok(slice_list(items, Some(start), Some(stop), Some(step))?)
            }
            // Slicing a safe string preserves safety (mirrors markupsafe
            // `Markup.__getitem__` re-wrapping the slice).
            TemplateValue::Str(s) => Ok(TemplateValue::Str(slice_str(s, start, stop, step)?)),
            TemplateValue::SafeStr(s) => {
                Ok(TemplateValue::SafeStr(slice_str(s, start, stop, step)?))
            }
            _ => Err(LoaderError::TemplateRender {
                detail: "slice needs an array or string".to_owned(),
            }),
        }
    }

    /// Invokes a callable value.
    fn invoke(&mut self, target: Resolved, args: CallArgs) -> Result<TemplateValue, LoaderError> {
        match target {
            Resolved::Val(other) => Err(LoaderError::TemplateRender {
                detail: format!("callee is not a function: got {}", other.type_name()),
            }),
            Resolved::Func(func) => self.invoke_func(func, args),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn invoke_func(&mut self, func: Func, args: CallArgs) -> Result<TemplateValue, LoaderError> {
        match func {
            Func::Global(name) => self.invoke_global(&name, args),
            Func::Method { receiver, name } => {
                let mut full = CallArgs {
                    positional: vec![receiver],
                    keywords: Vec::new(),
                };
                full.positional.extend(args.positional);
                full.keywords.extend(args.keywords);
                self.invoke_method(&name, full)
            }
            Func::Macro { params, body, .. } => self.invoke_macro(&params, &body, args, None),
            Func::Caller { params, body, .. } => {
                // `caller(...)`: bind params, render the body, return text.
                self.enter()?;
                self.push_scope();
                bind_params(&params, &args, self)?;
                let mut buf = String::new();
                let _ = self.exec_body(&body, &mut buf)?;
                self.pop_scope();
                self.exit();
                Ok(TemplateValue::Str(buf))
            }
        }
    }

    /// Invokes a bound builtin method given full args (receiver first).
    fn invoke_method(&mut self, name: &str, args: CallArgs) -> Result<TemplateValue, LoaderError> {
        let receiver = args
            .positional
            .first()
            .cloned()
            .unwrap_or(TemplateValue::Undefined);
        let rest = CallArgs {
            positional: args.positional.into_iter().skip(1).collect(),
            keywords: args.keywords,
        };
        apply_filter(name, &receiver, &rest)
    }

    /// Invokes a macro with an optional `caller` binding.
    fn invoke_macro(
        &mut self,
        params: &[Param],
        body: &[Stmt],
        args: CallArgs,
        caller: Option<Func>,
    ) -> Result<TemplateValue, LoaderError> {
        self.enter()?;
        self.push_scope();
        bind_params(params, &args, self)?;
        let has_caller = caller.is_some();
        if let Some(caller) = caller {
            self.caller_stack.push(caller);
        }
        let mut buf = String::new();
        let _ = self.exec_body(body, &mut buf)?;
        if has_caller {
            self.caller_stack.pop();
        }
        self.pop_scope();
        self.exit();
        Ok(TemplateValue::Str(buf))
    }

    /// Global functions: `range`, `namespace`, `tojson`,
    /// `raise_exception`. Everything else fails closed.
    fn invoke_global(&mut self, name: &str, args: CallArgs) -> Result<TemplateValue, LoaderError> {
        match name {
            "range" => {
                if args.positional.is_empty() || args.positional.len() > 3 {
                    return Err(LoaderError::TemplateRender {
                        detail: "range() takes 1 to 3 arguments".to_owned(),
                    });
                }
                let ints: Result<Vec<i64>, _> = args.positional.iter().map(as_int_value).collect();
                let ints = ints?;
                let (start, stop, step) = match ints.len() {
                    1 => (0, ints[0], 1),
                    2 => (ints[0], ints[1], 1),
                    _ => (ints[0], ints[1], ints[2]),
                };
                if step == 0 {
                    return Err(LoaderError::TemplateRender {
                        detail: "range() step argument must not be zero".to_owned(),
                    });
                }
                // Cardinality preflight with overflow-proof arithmetic: a
                // hostile range fails before materializing a single element.
                // DECISION(A2.9): ranges longer than `MAX_LOOP_ITERS` fail
                // with `Limit` at construction (rejected materializing up to
                // the loop trip: `range(0, 2**62)` would allocate first).
                // The stepping addition stays checked so hostile extremes
                // cannot wrap the cursor on any build mode.
                let count = range_len(start, stop, step);
                let count = match count {
                    Some(n) if n <= MAX_LOOP_ITERS as i128 => n as usize,
                    _ => {
                        return Err(LoaderError::Limit {
                            what: "template range length",
                            limit: MAX_LOOP_ITERS,
                            got: count
                                .map(|n| n.min(usize::MAX as i128) as usize)
                                .unwrap_or(usize::MAX),
                        });
                    }
                };
                let mut out = Vec::new();
                try_reserve_list(&mut out, count, "template range length")?;
                let mut i = start;
                for k in 0..count {
                    out.push(TemplateValue::Int(i));
                    if k + 1 < count {
                        i = i
                            .checked_add(step)
                            .ok_or_else(|| LoaderError::TemplateRender {
                                detail: "range() overflow".to_owned(),
                            })?;
                    }
                }
                Ok(TemplateValue::List(out))
            }
            "namespace" => {
                for (k, _) in &args.keywords {
                    let _ = k;
                }
                if !args.positional.is_empty() {
                    return Err(LoaderError::TemplateRender {
                        detail: "namespace() arguments must be kwargs".to_owned(),
                    });
                }
                Ok(TemplateValue::Dict(args.keywords))
            }
            "tojson" => {
                // Global form: the input is positional 0, options follow
                // (mirror minja `tojson(args)` with `get_pos(0)` input).
                let input = args.positional_first();
                let rest = CallArgs {
                    positional: args.positional.into_iter().skip(1).collect(),
                    keywords: args.keywords,
                };
                Ok(TemplateValue::Str(filter_tojson(&input, &rest)?))
            }
            "raise_exception" => {
                let message = match args.positional_first() {
                    TemplateValue::Str(s) => s,
                    _ => {
                        return Err(LoaderError::TemplateRender {
                            detail: "raise_exception() needs a string".to_owned(),
                        });
                    }
                };
                Err(LoaderError::TemplateRender {
                    detail: format!("Jinja Exception: {message}"),
                })
            }
            "strftime_now" => Err(unsupported(
                "strftime_now uses wall-clock time and is rejected for determinism".to_owned(),
            )),
            _ => Err(unsupported(format!(
                "function '{name}' is outside the subset"
            ))),
        }
    }
}

fn as_int_value(value: &TemplateValue) -> Result<i64, LoaderError> {
    match value {
        TemplateValue::Int(i) => Ok(*i),
        _ => Err(LoaderError::TemplateRender {
            detail: "range() needs integer arguments".to_owned(),
        }),
    }
}

/// Exact `range()` cardinality in 128-bit math (never overflows: the inputs
/// are `i64`, so every difference and quotient fits). `None` is unreachable
/// for valid steps and kept as a fail-closed shape.
fn range_len(start: i64, stop: i64, step: i64) -> Option<i128> {
    debug_assert!(step != 0);
    let (start, stop, step) = (start as i128, stop as i128, step as i128);
    if step > 0 {
        if start >= stop {
            return Some(0);
        }
        Some((stop - start + step - 1) / step)
    } else {
        if start <= stop {
            return Some(0);
        }
        Some((start - stop - step - 1) / (-step))
    }
}

/// Binds macro/call parameters (positional, keyword, defaults; missing
/// without default is an error, mirroring minja `bind_parameters`).
fn bind_params(params: &[Param], args: &CallArgs, interp: &mut Interp) -> Result<(), LoaderError> {
    let mut positional = args.positional.iter();
    for param in params {
        if let Some((_, v)) = args.keywords.iter().find(|(k, _)| k == &param.name) {
            interp.set(&param.name, v.clone());
        } else if let Some(v) = positional.next() {
            interp.set(&param.name, v.clone());
        } else if let Some(default) = &param.default {
            let value = interp.eval_as_value(default)?;
            interp.set(&param.name, value);
        } else {
            return Err(LoaderError::TemplateRender {
                detail: format!("missing argument '{}'", param.name),
            });
        }
    }
    // Arity is strict (mirror CPython `TypeError`): leftover positionals
    // and unknown keywords fail closed. The subset has no `*args` form, so
    // any surplus is always an error.
    if positional.next().is_some() {
        return Err(LoaderError::TemplateRender {
            detail: "too many positional arguments".to_owned(),
        });
    }
    for (name, _) in &args.keywords {
        if !params.iter().any(|p| &p.name == name) {
            return Err(LoaderError::TemplateRender {
                detail: format!("unknown keyword argument '{name}'"),
            });
        }
    }
    Ok(())
}

/// Optional slice index (undefined → None).
fn opt_index(value: &TemplateValue) -> Result<Option<i64>, LoaderError> {
    crate::template::builtins::opt_int_or_none(value)
}

/// Object key coercion (mirrors `as_string` on the property).
fn attr_key(property: &TemplateValue) -> Result<String, LoaderError> {
    as_string(property)
}

/// Appends interpolation output (mirrors `gather_string_parts_recursive`:
/// strings raw, int/float/bool via `as_string`, arrays flattened,
/// none/undefined/dicts dropped). Every piece is preflighted against the
/// output budget before it lands, and flattening is iterative (explicit heap
/// stack, never recursion): loop-accumulated values nest deeper than entry
/// validation allows, and recursion there would overflow the thread stack.
fn append_interpolation(value: &TemplateValue, out: &mut String) -> Result<(), LoaderError> {
    use crate::template::builtins::{ensure_str_add, format_float};
    const WHAT: &str = "rendered chat template bytes";
    let mut stack: Vec<&TemplateValue> = vec![value];
    while let Some(next) = stack.pop() {
        match next {
            TemplateValue::Str(s) | TemplateValue::SafeStr(s) => {
                ensure_str_add(out.len(), s.len(), WHAT)?;
                out.push_str(s);
            }
            TemplateValue::Int(i) => {
                let text = i.to_string();
                ensure_str_add(out.len(), text.len(), WHAT)?;
                out.push_str(&text);
            }
            TemplateValue::Float(f) => {
                let text = format_float(*f);
                ensure_str_add(out.len(), text.len(), WHAT)?;
                out.push_str(&text);
            }
            TemplateValue::Bool(true) => {
                ensure_str_add(out.len(), 4, WHAT)?;
                out.push_str("True");
            }
            TemplateValue::Bool(false) => {
                ensure_str_add(out.len(), 5, WHAT)?;
                out.push_str("False");
            }
            TemplateValue::List(items) => {
                for item in items.iter().rev() {
                    stack.push(item);
                }
            }
            TemplateValue::Undefined | TemplateValue::None | TemplateValue::Dict(_) => {}
        }
    }
    Ok(())
}

/// Converts a chat message to a template object (Spec 10 §3.1).
fn message_value(message: &ChatMessage) -> TemplateValue {
    let mut entries = vec![
        ("role".to_owned(), TemplateValue::Str(message.role.clone())),
        (
            "content".to_owned(),
            TemplateValue::Str(message.content.clone()),
        ),
    ];
    if !message.tool_calls.is_empty() {
        entries.push((
            "tool_calls".to_owned(),
            TemplateValue::List(
                message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        TemplateValue::Dict(vec![
                            ("type".to_owned(), TemplateValue::Str("function".to_owned())),
                            (
                                "function".to_owned(),
                                TemplateValue::Dict({
                                    let mut f = vec![
                                        ("name".to_owned(), TemplateValue::Str(call.name.clone())),
                                        (
                                            "arguments".to_owned(),
                                            TemplateValue::Str(call.arguments.clone()),
                                        ),
                                    ];
                                    if let Some(id) = &call.id {
                                        f.push(("id".to_owned(), TemplateValue::Str(id.clone())));
                                    }
                                    f
                                }),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ));
    }
    if let Some(reasoning) = &message.reasoning_content {
        entries.push((
            "reasoning_content".to_owned(),
            TemplateValue::Str(reasoning.clone()),
        ));
    }
    TemplateValue::Dict(entries)
}

/// Whether `name` is a builtin for `object`'s type.
fn builtin_exists(object: &TemplateValue, name: &str) -> bool {
    const STRING_METHODS: &[&str] = &[
        "upper",
        "lower",
        "strip",
        "rstrip",
        "lstrip",
        "title",
        "capitalize",
        "length",
        "startswith",
        "endswith",
        "split",
        "rsplit",
        "replace",
        "format",
        "indent",
        "tojson",
        "string",
        "safe",
        "escape",
        "int",
        "float",
        "wordcount",
        "default",
    ];
    const ARRAY_METHODS: &[&str] = &[
        "list",
        "first",
        "last",
        "length",
        "join",
        "map",
        "select",
        "reject",
        "selectattr",
        "rejectattr",
        "sort",
        "reverse",
        "min",
        "max",
        "slice",
        "append",
        "pop",
        "tojson",
        "string",
        "safe",
        "int",
        "float",
        "default",
    ];
    const OBJECT_METHODS: &[&str] = &[
        "get", "keys", "values", "items", "length", "dictsort", "tojson", "string", "join",
    ];
    const NUMBER_METHODS: &[&str] = &["int", "float", "abs", "string", "safe", "tojson", "default"];
    const BOOL_METHODS: &[&str] = &["int", "float", "string", "safe", "tojson", "default"];
    match object {
        TemplateValue::Str(_) | TemplateValue::SafeStr(_) => STRING_METHODS.contains(&name),
        TemplateValue::List(_) => ARRAY_METHODS.contains(&name),
        TemplateValue::Dict(_) => OBJECT_METHODS.contains(&name),
        TemplateValue::Int(_) | TemplateValue::Float(_) => NUMBER_METHODS.contains(&name),
        TemplateValue::Bool(_) => BOOL_METHODS.contains(&name),
        TemplateValue::Undefined | TemplateValue::None => false,
    }
}
