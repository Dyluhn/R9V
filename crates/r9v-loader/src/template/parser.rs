// SPDX-License-Identifier: Apache-2.0
//! Template parser: tokens to AST (mirrors minja `parser.cpp`).

use crate::error::LoaderError;
use crate::template::lexer::{Token, TokenKind};
use crate::template::TemplateValue;
use crate::template::MAX_DEPTH;

/// A parsed template program.
#[derive(Debug, Clone)]
pub(crate) struct Program {
    /// Top-level statements.
    pub body: Vec<Stmt>,
}

/// Statements (mirror minja statement types).
#[derive(Debug, Clone)]
pub(crate) enum Stmt {
    /// Raw text.
    Text(String),
    /// `{{ expr }}`.
    Expr(Expr),
    /// `{% if %}` chain.
    If {
        /// (test, body) branches; first is `if`, rest are `elif`.
        branches: Vec<(Expr, Vec<Stmt>)>,
        /// `else` body.
        orelse: Vec<Stmt>,
    },
    /// `{% for %}`.
    For {
        /// Loop targets (1 or 2 names).
        targets: Vec<String>,
        /// Iterable expression.
        iter: Expr,
        /// Optional `if` guard (`for x in y if c`).
        guard: Option<Expr>,
        /// Body.
        body: Vec<Stmt>,
        /// `{% else %}` body (empty iteration).
        orelse: Vec<Stmt>,
    },
    /// `{% set name = expr %}` / tuple unpack / block set.
    Set {
        /// Assignment targets.
        targets: Vec<AssignTarget>,
        /// Value (`None` for block form with body).
        value: Option<Expr>,
        /// Block body for `{% set x %}...{% endset %}`.
        body: Vec<Stmt>,
    },
    /// `{% macro name(args) %}`.
    Macro {
        /// Macro name.
        name: String,
        /// Parameters with optional defaults.
        params: Vec<Param>,
        /// Body.
        body: Vec<Stmt>,
    },
    /// `{% call [(args)] name(args) %}`.
    Call {
        /// Callee macro name.
        callee: String,
        /// Call arguments.
        args: Vec<Arg>,
        /// `caller` parameters.
        caller_params: Vec<Param>,
        /// Caller body.
        body: Vec<Stmt>,
    },
    /// `{% filter name %}` block.
    FilterBlock {
        /// Filter name.
        name: String,
        /// Filter arguments.
        args: Vec<Arg>,
        /// Body.
        body: Vec<Stmt>,
    },
    /// `{% break %}`.
    Break,
    /// `{% continue %}`.
    Continue,
}

/// Assignment target for `set`.
#[derive(Debug, Clone)]
pub(crate) enum AssignTarget {
    /// Plain name.
    Name(String),
    /// `obj.attr` (namespace attribute sets).
    Attr {
        /// Object expression (a name).
        obj: String,
        /// Attribute name.
        attr: String,
    },
}

/// Macro / caller parameter.
#[derive(Debug, Clone)]
pub(crate) struct Param {
    /// Parameter name.
    pub name: String,
    /// Default value expression.
    pub default: Option<Expr>,
}

/// Call argument.
#[derive(Debug, Clone)]
pub(crate) enum Arg {
    /// Positional.
    Pos(Expr),
    /// Keyword (`name=expr`).
    Kw(String, Expr),
    /// Spread (`*expr`).
    Spread(Expr),
}

/// Expressions (mirror minja expression types).
#[derive(Debug, Clone)]
pub(crate) enum Expr {
    /// Literal.
    Lit(TemplateValue),
    /// Variable name.
    Name(String),
    /// `obj.attr`.
    Attr(Box<Expr>, String),
    /// `obj[key]`.
    Index(Box<Expr>, Box<Expr>),
    /// `obj[a:b:c]` (any part may be absent).
    Slice(
        Box<Expr>,
        Option<Box<Expr>>,
        Option<Box<Expr>>,
        Option<Box<Expr>>,
    ),
    /// `callee(args)`.
    Call(Box<Expr>, Vec<Arg>),
    /// `operand | filter(args)`.
    Filter(Box<Expr>, String, Vec<Arg>),
    /// `operand is [not] test [(args)]`.
    Test(Box<Expr>, String, Vec<Expr>, bool),
    /// `left op right`.
    BinOp(String, Box<Expr>, Box<Expr>),
    /// `not x` / `-x`.
    UnOp(String, Box<Expr>),
    /// `a if c else b`.
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `a if b` with no `else`: a select guard, only valid as a
    /// `for`-loop iterable (mirrors minja `select_expression`).
    Select(Box<Expr>, Box<Expr>),
    /// `[items]`.
    List(Vec<Expr>),
    /// `(items)` tuple.
    Tuple(Vec<Expr>),
    /// `{k: v}`.
    Dict(Vec<(Expr, Expr)>),
}

/// Parses lexed tokens into a program.
pub(crate) fn parse(tokens: &[Token]) -> Result<Program, LoaderError> {
    let mut parser = Parser {
        tokens,
        pos: 0,
        depth: 0,
    };
    let body = parser.parse_body(&[])?;
    Ok(Program { body })
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    /// Current nesting depth (blocks via `parse_body`, expressions via
    /// `parse_expr`, sharing one budget). Bounds the parser call stack so
    /// hostile nesting inside the source-size limit fails closed instead of
    /// overflowing the thread stack.
    depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&'a Token> {
        self.tokens.get(self.pos)
    }

    fn peek_kind(&self, offset: usize) -> Option<TokenKind> {
        self.tokens.get(self.pos + offset).map(|t| t.kind)
    }

    fn next(&mut self) -> Result<&'a Token, LoaderError> {
        let token = self
            .tokens
            .get(self.pos)
            .ok_or_else(|| LoaderError::TemplateParse {
                offset: self.tokens.last().map(|t| t.pos).unwrap_or(0),
                detail: "unexpected end of template".to_owned(),
            })?;
        self.pos += 1;
        Ok(token)
    }

    fn expect(&mut self, kind: TokenKind, what: &str) -> Result<&'a Token, LoaderError> {
        let token = self.next()?;
        if token.kind != kind {
            return Err(LoaderError::TemplateParse {
                offset: token.pos,
                detail: format!("expected {what}, got {}", describe(token)),
            });
        }
        Ok(token)
    }

    /// True when positioned at `{% name` (statement opener + identifier).
    fn at_stmt(&self, names: &[&str]) -> bool {
        match (self.tokens.get(self.pos), self.tokens.get(self.pos + 1)) {
            (Some(open), Some(name))
                if open.kind == TokenKind::OpenStmt && name.kind == TokenKind::Ident =>
            {
                names.contains(&name.text.as_str())
            }
            _ => false,
        }
    }

    fn parse_body(&mut self, ends: &[&str]) -> Result<Vec<Stmt>, LoaderError> {
        self.enter_nest()?;
        let result = self.parse_body_inner(ends);
        self.exit_nest();
        result
    }

    /// Enters one nesting level, failing closed past the shared budget.
    fn enter_nest(&mut self) -> Result<(), LoaderError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(LoaderError::Limit {
                what: "template nesting depth",
                limit: MAX_DEPTH,
                got: self.depth,
            });
        }
        Ok(())
    }

    /// Leaves one nesting level (the parser is single-shot: an error aborts
    /// the whole parse, so a leaked count on the error path is harmless).
    fn exit_nest(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    fn parse_body_inner(&mut self, ends: &[&str]) -> Result<Vec<Stmt>, LoaderError> {
        let mut body = Vec::new();
        loop {
            // Comments are discarded (mirror minja).
            while self.peek_kind(0) == Some(TokenKind::Comment) {
                self.pos += 1;
            }
            match self.peek() {
                None => break,
                Some(token) => {
                    if token.kind == TokenKind::OpenStmt && self.at_stmt(ends) {
                        break;
                    }
                }
            }
            body.push(self.parse_any()?);
        }
        Ok(body)
    }

    fn parse_any(&mut self) -> Result<Stmt, LoaderError> {
        // Comments are discarded (mirror minja).
        while self.peek_kind(0) == Some(TokenKind::Comment) {
            self.pos += 1;
        }
        let token = self.peek().ok_or_else(|| LoaderError::TemplateParse {
            offset: 0,
            detail: "unexpected end of template".to_owned(),
        })?;
        match token.kind {
            TokenKind::Text => {
                let text = token.text.clone();
                self.pos += 1;
                Ok(Stmt::Text(text))
            }
            TokenKind::OpenExpr => {
                self.pos += 1;
                let expr = self.parse_expr()?;
                self.expect(TokenKind::CloseExpr, "'}}'")?;
                Ok(Stmt::Expr(expr))
            }
            TokenKind::OpenStmt => self.parse_statement(),
            _ => Err(LoaderError::TemplateParse {
                offset: token.pos,
                detail: format!("unexpected {}", describe(token)),
            }),
        }
    }

    fn parse_statement(&mut self) -> Result<Stmt, LoaderError> {
        let open = self.expect(TokenKind::OpenStmt, "'{%'")?;
        let _ = open;
        let name_token = self.next()?;
        if name_token.kind != TokenKind::Ident {
            return Err(LoaderError::TemplateParse {
                offset: name_token.pos,
                detail: "expected statement name".to_owned(),
            });
        }
        let name = name_token.text.clone();
        match name.as_str() {
            "set" => self.parse_set(),
            "if" => self.parse_if(),
            "for" => self.parse_for(),
            "macro" => self.parse_macro(),
            "call" => self.parse_call(),
            "filter" => self.parse_filter_block(),
            "break" => {
                self.expect(TokenKind::CloseStmt, "'%}'")?;
                Ok(Stmt::Break)
            }
            "continue" => {
                self.expect(TokenKind::CloseStmt, "'%}'")?;
                Ok(Stmt::Continue)
            }
            "generation" | "endgeneration" => {
                // Transformers-specific no-op blocks (mirror minja).
                self.expect(TokenKind::CloseStmt, "'%}'")?;
                Ok(Stmt::Text(String::new()))
            }
            _ => Err(LoaderError::TemplateUnsupported {
                detail: format!("statement '{{% {name} %}}' is outside the sandboxed subset"),
            }),
        }
    }

    fn parse_set(&mut self) -> Result<Stmt, LoaderError> {
        let targets = self.parse_assign_targets()?;
        if self.peek_kind(0) == Some(TokenKind::Equals) {
            self.pos += 1;
            let value = self.parse_expr_seq()?;
            self.expect(TokenKind::CloseStmt, "'%}'")?;
            Ok(Stmt::Set {
                targets,
                value: Some(value),
                body: Vec::new(),
            })
        } else {
            self.expect(TokenKind::CloseStmt, "'%}'")?;
            let body = self.parse_body(&["endset"])?;
            self.expect_stmt_name("endset")?;
            Ok(Stmt::Set {
                targets,
                value: None,
                body,
            })
        }
    }

    fn parse_assign_targets(&mut self) -> Result<Vec<AssignTarget>, LoaderError> {
        let mut targets = vec![self.parse_assign_target()?];
        while self.peek_kind(0) == Some(TokenKind::Comma) {
            self.pos += 1;
            targets.push(self.parse_assign_target()?);
        }
        Ok(targets)
    }

    fn parse_assign_target(&mut self) -> Result<AssignTarget, LoaderError> {
        let token = self.next()?;
        if token.kind != TokenKind::Ident {
            return Err(LoaderError::TemplateParse {
                offset: token.pos,
                detail: "expected assignment target".to_owned(),
            });
        }
        let first = token.text.clone();
        if self.peek_kind(0) == Some(TokenKind::Dot) {
            self.pos += 1;
            let attr = self.next()?;
            if attr.kind != TokenKind::Ident {
                return Err(LoaderError::TemplateParse {
                    offset: attr.pos,
                    detail: "expected attribute name".to_owned(),
                });
            }
            Ok(AssignTarget::Attr {
                obj: first,
                attr: attr.text.clone(),
            })
        } else {
            Ok(AssignTarget::Name(first))
        }
    }

    fn parse_if(&mut self) -> Result<Stmt, LoaderError> {
        let mut branches = Vec::new();
        let test = self.parse_expr()?;
        self.expect(TokenKind::CloseStmt, "'%}'")?;
        let body = self.parse_body(&["elif", "else", "endif"])?;
        branches.push((test, body));
        let mut orelse = Vec::new();
        loop {
            if self.at_stmt(&["elif"]) {
                self.pos += 2;
                let test = self.parse_expr()?;
                self.expect(TokenKind::CloseStmt, "'%}'")?;
                let body = self.parse_body(&["elif", "else", "endif"])?;
                branches.push((test, body));
            } else if self.at_stmt(&["else"]) {
                self.pos += 2;
                self.expect(TokenKind::CloseStmt, "'%}'")?;
                orelse = self.parse_body(&["endif"])?;
                break;
            } else {
                break;
            }
        }
        self.expect_stmt_name("endif")?;
        Ok(Stmt::If { branches, orelse })
    }

    fn parse_for(&mut self) -> Result<Stmt, LoaderError> {
        let targets = self.parse_primary_seq()?;
        let mut names = Vec::new();
        for target in &targets {
            match target {
                Expr::Name(n) => names.push(n.clone()),
                _ => {
                    return Err(LoaderError::TemplateParse {
                        offset: self.pos,
                        detail: "loop variables must be identifiers".to_owned(),
                    });
                }
            }
        }
        if names.is_empty() || names.len() > 2 {
            return Err(LoaderError::TemplateParse {
                offset: self.pos,
                detail: "for loop takes 1 or 2 variables".to_owned(),
            });
        }
        let in_token = self.next()?;
        if in_token.kind != TokenKind::Ident || in_token.text != "in" {
            return Err(LoaderError::TemplateParse {
                offset: in_token.pos,
                detail: "expected 'in'".to_owned(),
            });
        }
        // The iterable may carry a select guard (`for x in y if c` parses
        // as a select expression, mirroring minja).
        let iter = self.parse_expr()?;
        let (iter, guard) = match iter {
            Expr::Select(inner, test) => (*inner, Some(*test)),
            other => (other, None),
        };
        self.expect(TokenKind::CloseStmt, "'%}'")?;
        let body = self.parse_body(&["else", "endfor"])?;
        let mut orelse = Vec::new();
        if self.at_stmt(&["else"]) {
            self.pos += 2;
            self.expect(TokenKind::CloseStmt, "'%}'")?;
            orelse = self.parse_body(&["endfor"])?;
        }
        self.expect_stmt_name("endfor")?;
        Ok(Stmt::For {
            targets: names,
            iter,
            guard,
            body,
            orelse,
        })
    }

    fn parse_macro(&mut self) -> Result<Stmt, LoaderError> {
        let name_token = self.next()?;
        if name_token.kind != TokenKind::Ident {
            return Err(LoaderError::TemplateParse {
                offset: name_token.pos,
                detail: "expected macro name".to_owned(),
            });
        }
        let name = name_token.text.clone();
        let params = self.parse_params()?;
        self.expect(TokenKind::CloseStmt, "'%}'")?;
        let body = self.parse_body(&["endmacro"])?;
        self.expect_stmt_name("endmacro")?;
        Ok(Stmt::Macro { name, params, body })
    }

    fn parse_call(&mut self) -> Result<Stmt, LoaderError> {
        let mut caller_params = Vec::new();
        if self.peek_kind(0) == Some(TokenKind::OpenParen) {
            caller_params = self.parse_params()?;
        }
        let callee_token = self.next()?;
        if callee_token.kind != TokenKind::Ident {
            return Err(LoaderError::TemplateParse {
                offset: callee_token.pos,
                detail: "expected macro name".to_owned(),
            });
        }
        let callee = callee_token.text.clone();
        let args = self.parse_call_args()?;
        self.expect(TokenKind::CloseStmt, "'%}'")?;
        let body = self.parse_body(&["endcall"])?;
        self.expect_stmt_name("endcall")?;
        Ok(Stmt::Call {
            callee,
            args,
            caller_params,
            body,
        })
    }

    fn parse_filter_block(&mut self) -> Result<Stmt, LoaderError> {
        let name_token = self.next()?;
        if name_token.kind != TokenKind::Ident {
            return Err(LoaderError::TemplateParse {
                offset: name_token.pos,
                detail: "expected filter name".to_owned(),
            });
        }
        let name = name_token.text.clone();
        let mut args = Vec::new();
        if self.peek_kind(0) == Some(TokenKind::OpenParen) {
            args = self.parse_call_args_inner()?;
        }
        self.expect(TokenKind::CloseStmt, "'%}'")?;
        let body = self.parse_body(&["endfilter"])?;
        self.expect_stmt_name("endfilter")?;
        Ok(Stmt::FilterBlock { name, args, body })
    }

    fn expect_stmt_name(&mut self, name: &str) -> Result<(), LoaderError> {
        self.expect(TokenKind::OpenStmt, "'{%'")?;
        let token = self.next()?;
        if token.kind != TokenKind::Ident || token.text != name {
            return Err(LoaderError::TemplateParse {
                offset: token.pos,
                detail: format!("expected '{{% {name} %}}'"),
            });
        }
        self.expect(TokenKind::CloseStmt, "'%}'")?;
        Ok(())
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, LoaderError> {
        self.expect(TokenKind::OpenParen, "'('")?;
        let mut params = Vec::new();
        while self.peek_kind(0) != Some(TokenKind::CloseParen) {
            let token = self.next()?;
            if token.kind != TokenKind::Ident {
                return Err(LoaderError::TemplateParse {
                    offset: token.pos,
                    detail: "expected parameter name".to_owned(),
                });
            }
            let name = token.text.clone();
            let mut default = None;
            if self.peek_kind(0) == Some(TokenKind::Equals) {
                self.pos += 1;
                default = Some(self.parse_expr()?);
            }
            params.push(Param { name, default });
            if self.peek_kind(0) == Some(TokenKind::Comma) {
                self.pos += 1;
                if self.peek_kind(0) == Some(TokenKind::CloseParen) {
                    break;
                }
            } else {
                break;
            }
        }
        self.expect(TokenKind::CloseParen, "')'")?;
        Ok(params)
    }

    fn parse_call_args(&mut self) -> Result<Vec<Arg>, LoaderError> {
        if self.peek_kind(0) != Some(TokenKind::OpenParen) {
            return Ok(Vec::new());
        }
        self.parse_call_args_inner()
    }

    fn parse_call_args_inner(&mut self) -> Result<Vec<Arg>, LoaderError> {
        self.expect(TokenKind::OpenParen, "'('")?;
        let mut args = Vec::new();
        while self.peek_kind(0) != Some(TokenKind::CloseParen) {
            if self.peek_kind(0) == Some(TokenKind::Multiplicative)
                && self.tokens.get(self.pos).map(|t| t.text.as_str()) == Some("*")
            {
                self.pos += 1;
                args.push(Arg::Spread(self.parse_expr()?));
            } else {
                let expr = self.parse_expr()?;
                // Keyword argument: a bare name followed by `=`.
                if let Expr::Name(name) = &expr {
                    if self.peek_kind(0) == Some(TokenKind::Equals) {
                        self.pos += 1;
                        let value = self.parse_expr()?;
                        args.push(Arg::Kw(name.clone(), value));
                    } else {
                        args.push(Arg::Pos(expr));
                    }
                } else {
                    args.push(Arg::Pos(expr));
                }
            }
            if self.peek_kind(0) == Some(TokenKind::Comma) {
                self.pos += 1;
                if self.peek_kind(0) == Some(TokenKind::CloseParen) {
                    break;
                }
            } else {
                break;
            }
        }
        self.expect(TokenKind::CloseParen, "')'")?;
        Ok(args)
    }

    // Expression grammar (precedence mirrors minja):
    // ternary < or < and < not < comparison < additive < multiplicative
    // < test < filter < call/member < primary.

    fn parse_expr(&mut self) -> Result<Expr, LoaderError> {
        self.enter_nest()?;
        let result = self.parse_ternary();
        self.exit_nest();
        result
    }

    fn parse_expr_seq(&mut self) -> Result<Expr, LoaderError> {
        let first = self.parse_expr()?;
        if self.peek_kind(0) != Some(TokenKind::Comma) {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.peek_kind(0) == Some(TokenKind::Comma) {
            self.pos += 1;
            items.push(self.parse_expr()?);
        }
        Ok(Expr::Tuple(items))
    }

    fn parse_primary_seq(&mut self) -> Result<Vec<Expr>, LoaderError> {
        let first = self.parse_primary()?;
        let mut items = vec![first];
        while self.peek_kind(0) == Some(TokenKind::Comma) {
            self.pos += 1;
            items.push(self.parse_primary()?);
        }
        Ok(items)
    }

    fn parse_ternary(&mut self) -> Result<Expr, LoaderError> {
        // Guarded directly: `else`-chains recurse here without passing
        // through `parse_expr`.
        self.enter_nest()?;
        let result = self.parse_ternary_inner();
        self.exit_nest();
        result
    }

    fn parse_ternary_inner(&mut self) -> Result<Expr, LoaderError> {
        let value = self.parse_or()?;
        if self.peek_kind(0) == Some(TokenKind::Ident)
            && self.tokens.get(self.pos).map(|t| t.text.as_str()) == Some("if")
        {
            self.pos += 1;
            let test = self.parse_or()?;
            if self.is_keyword("else") {
                self.pos += 1;
                let orelse = self.parse_ternary()?;
                return Ok(Expr::Ternary(
                    Box::new(value),
                    Box::new(test),
                    Box::new(orelse),
                ));
            }
            // No `else`: a select guard (only valid as a for-iterable;
            // the interpreter rejects it anywhere else, mirroring minja).
            return Ok(Expr::Select(Box::new(value), Box::new(test)));
        }
        Ok(value)
    }

    fn parse_or(&mut self) -> Result<Expr, LoaderError> {
        let mut left = self.parse_and()?;
        while self.is_keyword("or") {
            self.pos += 1;
            let right = self.parse_and()?;
            left = Expr::BinOp("or".to_owned(), Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, LoaderError> {
        let mut left = self.parse_not()?;
        while self.is_keyword("and") {
            self.pos += 1;
            let right = self.parse_not()?;
            left = Expr::BinOp("and".to_owned(), Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, LoaderError> {
        // Guarded directly: `not not …` recurses here without passing
        // through `parse_expr`.
        self.enter_nest()?;
        let result = self.parse_not_inner();
        self.exit_nest();
        result
    }

    fn parse_not_inner(&mut self) -> Result<Expr, LoaderError> {
        if self.is_keyword("not") {
            // A leading `not` negates the comparison below, so `not a in
            // b` parses as `not (a in b)` (mirror minja).
            self.next()?;
            let operand = self.parse_not()?;
            return Ok(Expr::UnOp("not".to_owned(), Box::new(operand)));
        }
        self.parse_comparison()
    }

    fn is_keyword(&self, word: &str) -> bool {
        matches!(self.tokens.get(self.pos), Some(t) if t.kind == TokenKind::Ident && t.text == word)
    }

    fn parse_comparison(&mut self) -> Result<Expr, LoaderError> {
        let mut left = self.parse_additive()?;
        loop {
            let op = if self.is_keyword("in") {
                self.pos += 1;
                "in".to_owned()
            } else if self.is_keyword("not")
                && matches!(self.tokens.get(self.pos + 1), Some(t) if t.kind == TokenKind::Ident && t.text == "in")
            {
                self.pos += 2;
                "not in".to_owned()
            } else if self.peek_kind(0) == Some(TokenKind::Comparison) {
                self.next()?.text.clone()
            } else {
                break;
            };
            let right = self.parse_additive()?;
            left = Expr::BinOp(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, LoaderError> {
        let mut left = self.parse_multiplicative()?;
        while self.peek_kind(0) == Some(TokenKind::Additive) {
            let op = self.next()?.text.clone();
            let right = self.parse_multiplicative()?;
            left = Expr::BinOp(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, LoaderError> {
        let mut left = self.parse_test()?;
        while self.peek_kind(0) == Some(TokenKind::Multiplicative) {
            let token = self.next()?;
            let right = self.parse_test()?;
            left = Expr::BinOp(token.text.clone(), Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_test(&mut self) -> Result<Expr, LoaderError> {
        let operand = self.parse_filter()?;
        if !self.is_keyword("is") {
            return Ok(operand);
        }
        self.pos += 1;
        let negated = if self.is_keyword("not") {
            self.pos += 1;
            true
        } else {
            false
        };
        let name_token = self.next()?;
        if name_token.kind != TokenKind::Ident {
            return Err(LoaderError::TemplateParse {
                offset: name_token.pos,
                detail: "expected test name".to_owned(),
            });
        }
        let name = name_token.text.clone();
        let mut args = Vec::new();
        if self.peek_kind(0) == Some(TokenKind::OpenParen) {
            self.pos += 1;
            while self.peek_kind(0) != Some(TokenKind::CloseParen) {
                args.push(self.parse_expr()?);
                if self.peek_kind(0) == Some(TokenKind::Comma) {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            self.expect(TokenKind::CloseParen, "')'")?;
        }
        Ok(Expr::Test(Box::new(operand), name, args, negated))
    }

    fn parse_filter(&mut self) -> Result<Expr, LoaderError> {
        let mut operand = self.parse_member()?;
        while self.peek_kind(0) == Some(TokenKind::Pipe) {
            self.pos += 1;
            let name_token = self.next()?;
            if name_token.kind != TokenKind::Ident {
                return Err(LoaderError::TemplateParse {
                    offset: name_token.pos,
                    detail: "expected filter name".to_owned(),
                });
            }
            let name = name_token.text.clone();
            let mut args = Vec::new();
            if self.peek_kind(0) == Some(TokenKind::OpenParen) {
                args = self.parse_call_args_inner()?;
            }
            operand = Expr::Filter(Box::new(operand), name, args);
        }
        Ok(operand)
    }

    fn parse_member(&mut self) -> Result<Expr, LoaderError> {
        let mut object = self.parse_primary()?;
        loop {
            if self.peek_kind(0) == Some(TokenKind::OpenParen) {
                let args = self.parse_call_args_inner()?;
                object = Expr::Call(Box::new(object), args);
            } else if self.peek_kind(0) == Some(TokenKind::Dot) {
                self.pos += 1;
                let attr = self.next()?;
                if attr.kind != TokenKind::Ident {
                    return Err(LoaderError::TemplateParse {
                        offset: attr.pos,
                        detail: "expected attribute name".to_owned(),
                    });
                }
                object = Expr::Attr(Box::new(object), attr.text.clone());
                if self.peek_kind(0) == Some(TokenKind::OpenParen) {
                    let args = self.parse_call_args_inner()?;
                    object = Expr::Call(Box::new(object), args);
                }
            } else if self.peek_kind(0) == Some(TokenKind::OpenBracket) {
                self.pos += 1;
                object = self.parse_subscript(object)?;
            } else {
                break;
            }
        }
        Ok(object)
    }

    fn parse_subscript(&mut self, object: Expr) -> Result<Expr, LoaderError> {
        // Slice form `[a:b:c]` (any part may be empty); otherwise an index.
        let mut parts: Vec<Option<Expr>> = Vec::new();
        let mut is_slice = false;
        loop {
            if self.peek_kind(0) == Some(TokenKind::CloseBracket) {
                break;
            }
            if self.peek_kind(0) == Some(TokenKind::Colon) {
                self.pos += 1;
                parts.push(None);
                is_slice = true;
                continue;
            }
            parts.push(Some(self.parse_expr()?));
            if self.peek_kind(0) == Some(TokenKind::Colon) {
                self.pos += 1;
                is_slice = true;
            } else {
                break;
            }
        }
        if is_slice {
            if parts.len() > 3 {
                return Err(LoaderError::TemplateParse {
                    offset: self.pos,
                    detail: "slice takes at most 3 parts".to_owned(),
                });
            }
            while parts.len() < 3 {
                parts.push(None);
            }
            let mut owned = parts.into_iter().map(|p| p.map(Box::new));
            let start = owned.next().flatten();
            let stop = owned.next().flatten();
            let step = owned.next().flatten();
            self.expect(TokenKind::CloseBracket, "']'")?;
            return Ok(Expr::Slice(Box::new(object), start, stop, step));
        }
        if parts.is_empty() {
            // `[]`: blank index, evaluates to undefined (mirror minja
            // `blank_expression`).
            self.expect(TokenKind::CloseBracket, "']'")?;
            return Ok(Expr::Index(
                Box::new(object),
                Box::new(Expr::Lit(TemplateValue::Undefined)),
            ));
        }
        if parts.len() != 1 || parts[0].is_none() {
            return Err(LoaderError::TemplateParse {
                offset: self.pos,
                detail: "expected index expression".to_owned(),
            });
        }
        let index = parts
            .pop()
            .flatten()
            .unwrap_or(Expr::Lit(TemplateValue::None));
        self.expect(TokenKind::CloseBracket, "']'")?;
        Ok(Expr::Index(Box::new(object), Box::new(index)))
    }

    fn parse_primary(&mut self) -> Result<Expr, LoaderError> {
        let token = self.next()?;
        match token.kind {
            TokenKind::Ident => {
                let word = token.text.clone();
                match word.as_str() {
                    "true" | "True" => Ok(Expr::Lit(TemplateValue::Bool(true))),
                    "false" | "False" => Ok(Expr::Lit(TemplateValue::Bool(false))),
                    "none" | "None" | "null" | "nil" => Ok(Expr::Lit(TemplateValue::None)),
                    _ => Ok(Expr::Name(word)),
                }
            }
            TokenKind::Int => {
                // Folded signed literals may carry a leading `+` (mirror
                // minja lexing `+1` as one literal); Rust rejects `+1`.
                let text = token.text.strip_prefix('+').unwrap_or(&token.text);
                let value = text
                    .parse::<i64>()
                    .map_err(|_| LoaderError::TemplateParse {
                        offset: token.pos,
                        detail: format!("bad integer {}", token.text),
                    })?;
                Ok(Expr::Lit(TemplateValue::Int(value)))
            }
            TokenKind::Float => {
                let text = token.text.strip_prefix('+').unwrap_or(&token.text);
                let value = text
                    .parse::<f64>()
                    .map_err(|_| LoaderError::TemplateParse {
                        offset: token.pos,
                        detail: format!("bad float {}", token.text),
                    })?;
                Ok(Expr::Lit(TemplateValue::Float(value)))
            }
            TokenKind::Str => {
                // Adjacent string literals concatenate (mirror minja).
                let mut text = token.text.clone();
                while self.peek_kind(0) == Some(TokenKind::Str) {
                    text.push_str(&self.next()?.text);
                }
                Ok(Expr::Lit(TemplateValue::Str(text)))
            }
            TokenKind::OpenParen => {
                if self.peek_kind(0) == Some(TokenKind::CloseParen) {
                    self.pos += 1;
                    return Ok(Expr::Tuple(Vec::new()));
                }
                let first = self.parse_expr()?;
                if self.peek_kind(0) == Some(TokenKind::Comma) {
                    let mut items = vec![first];
                    while self.peek_kind(0) == Some(TokenKind::Comma) {
                        self.pos += 1;
                        if self.peek_kind(0) == Some(TokenKind::CloseParen) {
                            break;
                        }
                        items.push(self.parse_expr()?);
                    }
                    self.expect(TokenKind::CloseParen, "')'")?;
                    return Ok(Expr::Tuple(items));
                }
                self.expect(TokenKind::CloseParen, "')'")?;
                Ok(first)
            }
            TokenKind::OpenBracket => {
                let mut items = Vec::new();
                while self.peek_kind(0) != Some(TokenKind::CloseBracket) {
                    items.push(self.parse_expr()?);
                    if self.peek_kind(0) == Some(TokenKind::Comma) {
                        self.pos += 1;
                        if self.peek_kind(0) == Some(TokenKind::CloseBracket) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                self.expect(TokenKind::CloseBracket, "']'")?;
                Ok(Expr::List(items))
            }
            TokenKind::OpenBrace => {
                let mut entries = Vec::new();
                while self.peek_kind(0) != Some(TokenKind::CloseBrace) {
                    let key = self.parse_expr()?;
                    self.expect(TokenKind::Colon, "':'")?;
                    let value = self.parse_expr()?;
                    entries.push((key, value));
                    if self.peek_kind(0) == Some(TokenKind::Comma) {
                        self.pos += 1;
                        if self.peek_kind(0) == Some(TokenKind::CloseBrace) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                self.expect(TokenKind::CloseBrace, "'}'")?;
                Ok(Expr::Dict(entries))
            }
            _ => Err(LoaderError::TemplateParse {
                offset: token.pos,
                detail: format!("unexpected {}", describe(token)),
            }),
        }
    }
}

fn describe(token: &Token) -> String {
    format!("{:?} {:?}", token.kind, token.text)
}
