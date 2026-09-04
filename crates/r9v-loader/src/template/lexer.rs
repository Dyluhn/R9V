// SPDX-License-Identifier: Apache-2.0
//! Template lexer: source text to tokens (mirrors minja `lexer.cpp`).
//!
//! Default chat configuration: `lstrip_blocks` and `trim_blocks` are ON,
//! `\r\n`/`\r` normalize to `\n`, and one trailing newline is stripped.

use crate::error::LoaderError;

/// One lexed token with its byte offset (for error reports).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Token {
    /// Token kind.
    pub kind: TokenKind,
    /// Token text (escape-processed for strings/numbers; raw otherwise).
    pub text: String,
    /// Byte offset in the normalized source.
    pub pos: usize,
}

/// Token kinds (mirror minja `token::`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenKind {
    /// Raw template text.
    Text,
    /// `{{` (or `{{-`).
    OpenExpr,
    /// `}}` (or `-}}`).
    CloseExpr,
    /// `{%` (or `{%-`).
    OpenStmt,
    /// `%}` (or `-%}`).
    CloseStmt,
    /// Identifier or keyword.
    Ident,
    /// Integer literal.
    Int,
    /// Float literal.
    Float,
    /// String literal (escape-processed).
    Str,
    /// `+` `-` `~`.
    Additive,
    /// `*` `/` `%`.
    Multiplicative,
    /// `<` `>` `<=` `>=` `==` `!=`.
    Comparison,
    /// `=`.
    Equals,
    /// `(`.
    OpenParen,
    /// `)`.
    CloseParen,
    /// `[`.
    OpenBracket,
    /// `]`.
    CloseBracket,
    /// `.`.
    Dot,
    /// `,`.
    Comma,
    /// `:`.
    Colon,
    /// `|`.
    Pipe,
    /// `{`.
    OpenBrace,
    /// `}`.
    CloseBrace,
    /// Unary `-`/`+` not followed by a number (the parser rejects these,
    /// mirroring the reference).
    Unary,
    /// `{# ... #}` comment (discarded by the parser; kept for positions).
    Comment,
}

/// Lexes a full template source into tokens.
pub(crate) fn lex(source: &str) -> Result<Vec<Token>, LoaderError> {
    // Normalize line endings and strip one trailing newline (mirror minja).
    let mut src = source.replace("\r\n", "\n").replace('\r', "\n");
    if src.ends_with('\n') {
        src.pop();
    }
    let source = src.as_str();
    let bytes = source.as_bytes();
    let mut tokens: Vec<Token> = Vec::new();
    let mut pos = 0;
    // `{`/`}` depth inside `{{ }}` so dict literals don't close early.
    let mut curly_depth: usize = 0;

    while pos < bytes.len() {
        let rest = &source[pos..];
        if rest.starts_with("{#") {
            // Jinja2 comment whitespace control: `{#-` rstrips preceding
            // text; `-#}` skips all following whitespace; a plain `#}`
            // drops one following `\n` when trim_blocks is on.
            let dashed_open = rest.starts_with("{#-");
            if dashed_open {
                rstrip_text(&mut tokens);
            }
            let content_start = pos + if dashed_open { 3 } else { 2 };
            // Non-greedy closer scan: at each `#`, a directly preceding `-`
            // (inside the content region, so the `{#-` opener dash never
            // counts) makes it `-#}`; otherwise `#}` closes plainly.
            let mut cursor = content_start;
            let (closer_at, dashed_close) = loop {
                let rel = source[cursor..]
                    .find('#')
                    .ok_or_else(|| LoaderError::TemplateParse {
                        offset: pos,
                        detail: "missing end of comment tag".to_owned(),
                    })?;
                let hash_at = cursor + rel;
                if !source[hash_at..].starts_with("#}") {
                    cursor = hash_at + 1;
                    continue;
                }
                if hash_at > content_start && bytes[hash_at - 1] == b'-' {
                    break (hash_at - 1, true);
                }
                break (hash_at, false);
            };
            tokens.push(Token {
                kind: TokenKind::Comment,
                text: source[content_start..closer_at].to_owned(),
                pos,
            });
            pos = closer_at + if dashed_close { 3 } else { 2 };
            if dashed_close {
                pos = skip_ws(source, pos);
            } else if pos < bytes.len() && bytes[pos] == b'\n' {
                // trim_blocks (always on for chat templates).
                pos += 1;
            }
        } else if rest.starts_with("{{-") || rest.starts_with("{{") {
            let dashed = rest.starts_with("{{-");
            // `{%-`/`{{-` rstrips the preceding text token (before the
            // opener is pushed).
            if dashed {
                rstrip_text(&mut tokens);
            }
            tokens.push(Token {
                kind: TokenKind::OpenExpr,
                text: String::new(),
                pos,
            });
            pos += if dashed { 3 } else { 2 };
            curly_depth = 0;
            pos = lex_tag(source, pos, true, &mut tokens, &mut curly_depth)?;
        } else if rest.starts_with("{%-") || rest.starts_with("{%") {
            let dashed = rest.starts_with("{%-");
            if dashed {
                rstrip_text(&mut tokens);
            }
            tokens.push(Token {
                kind: TokenKind::OpenStmt,
                text: String::new(),
                pos,
            });
            pos += if dashed { 3 } else { 2 };
            pos = lex_tag(source, pos, false, &mut tokens, &mut curly_depth)?;
        } else {
            // Text until the next tag, with block whitespace rules.
            let text_start = pos;
            while pos < bytes.len() {
                let rest = &source[pos..];
                if rest.starts_with("{#") || rest.starts_with("{{") || rest.starts_with("{%") {
                    break;
                }
                pos += utf8_len_at(bytes, pos);
            }
            // lstrip_blocks (Jinja2 rule): before `{%` or `{#` (never
            // `{{`), drop the whitespace after the last newline — or the
            // whole chunk when it is all whitespace at a line start.
            let mut text_end = pos;
            if pos < bytes.len() {
                let next2 = &source[pos..(pos + 2).min(bytes.len())];
                if next2.starts_with("{%") || next2.starts_with("{#") {
                    let chunk = &source[text_start..pos];
                    let line_starting = text_start == 0 || bytes[text_start - 1] == b'\n';
                    let keep_from = chunk.rfind('\n').map(|i| i + 1).unwrap_or(0);
                    if keep_from > 0 || line_starting {
                        let tail = &chunk[keep_from..];
                        // `\s+` fullmatch needs a non-empty tail; an empty
                        // tail keeps `text_end` unchanged either way.
                        if !tail.is_empty() && tail.chars().all(|c| c.is_whitespace()) {
                            text_end = text_start + keep_from;
                        }
                    }
                }
            }
            let mut text = source[text_start..text_end].to_owned();
            // trim_blocks: text after `%}`/`#}`/`-}}` drops one `\n`.
            if follows_block_close(source, text_start) && text.starts_with('\n') {
                text.remove(0);
            }
            // `-%}`/`-}}`/`-#}` skips all following whitespace (`\s*`).
            if follows_dash_close(source, text_start) {
                text = text
                    .trim_start_matches(|c: char| c.is_whitespace())
                    .to_owned();
            }
            if !text.is_empty() {
                tokens.push(Token {
                    kind: TokenKind::Text,
                    text,
                    pos: text_start,
                });
            }
        }
    }
    Ok(tokens)
}

/// Removes trailing whitespace from the last text token (mirrors Python
/// `str.rstrip()`, which strips Unicode whitespace).
fn rstrip_text(tokens: &mut Vec<Token>) {
    if let Some(last) = tokens.last_mut() {
        if last.kind == TokenKind::Text {
            let trimmed = last.text.trim_end_matches(|c: char| c.is_whitespace());
            last.text = trimmed.to_owned();
            if last.text.is_empty() {
                tokens.pop();
            }
        }
    }
}

/// Byte offset after skipping Unicode whitespace at `pos`.
fn skip_ws(source: &str, mut pos: usize) -> usize {
    let bytes = source.as_bytes();
    while pos < bytes.len() {
        let len = utf8_len_at(bytes, pos);
        let c = source[pos..].chars().next().unwrap_or('\0');
        if !c.is_whitespace() {
            break;
        }
        pos += len;
    }
    pos
}

/// True when `text_start` immediately follows `%}`, `#}`, `-}}`, `-%}` or
/// `-#}` (mirror `([#%-]})`).
fn follows_block_close(source: &str, text_start: usize) -> bool {
    let bytes = source.as_bytes();
    if text_start < 2 {
        return false;
    }
    let c1 = bytes[text_start - 2] as char;
    let c2 = bytes[text_start - 1] as char;
    c2 == '}' && (c1 == '#' || c1 == '%' || c1 == '-')
}

/// True when `text_start` immediately follows `-%}`, `-}}` or `-#}`.
fn follows_dash_close(source: &str, text_start: usize) -> bool {
    let bytes = source.as_bytes();
    if text_start < 3 {
        return false;
    }
    let c0 = bytes[text_start - 3] as char;
    let c1 = bytes[text_start - 2] as char;
    let c2 = bytes[text_start - 1] as char;
    c0 == '-' && (c1 == '%' || c1 == '}' || c1 == '#') && c2 == '}'
}

/// UTF-8 sequence length at `pos` (templates are `&str`, so always valid;
/// invalid bytes consume 1 defensively).
fn utf8_len_at(bytes: &[u8], pos: usize) -> usize {
    let b = bytes[pos];
    if b < 0x80 {
        1
    } else if b >= 0xF0 {
        4
    } else if b >= 0xE0 {
        3
    } else if b >= 0xC0 {
        2
    } else {
        1
    }
}

/// Lexes one tag body; returns the offset after its closer.
#[allow(clippy::too_many_lines)]
fn lex_tag(
    source: &str,
    mut pos: usize,
    is_expr: bool,
    tokens: &mut Vec<Token>,
    curly_depth: &mut usize,
) -> Result<usize, LoaderError> {
    let bytes = source.as_bytes();
    loop {
        while pos < bytes.len() && matches!(bytes[pos], b' ' | b'\t' | b'\n' | b'\r') {
            pos += 1;
        }
        if pos >= bytes.len() {
            return Err(LoaderError::TemplateParse {
                offset: pos,
                detail: "unterminated tag".to_owned(),
            });
        }
        // Closers (`-}}`/`- %}` first so the dash binds to the tag).
        if source[pos..].starts_with("-}}") || source[pos..].starts_with("-%}") {
            let expr_close = source[pos..].starts_with("-}}");
            if expr_close != is_expr {
                return Err(LoaderError::TemplateParse {
                    offset: pos,
                    detail: "mismatched tag closer".to_owned(),
                });
            }
            // A `-}}` inside a dict literal still closes (mirror minja:
            // only the non-dash `}}` checks depth).
            tokens.push(Token {
                kind: if is_expr {
                    TokenKind::CloseExpr
                } else {
                    TokenKind::CloseStmt
                },
                text: String::new(),
                pos,
            });
            pos += 3;
            // `-}}`/`-%}` lstrips the following text (handled when the
            // text is lexed via `follows_dash_close`).
            return Ok(pos);
        }
        if source[pos..].starts_with("}}") || source[pos..].starts_with("%}") {
            let expr_close = source[pos..].starts_with("}}");
            if expr_close != is_expr {
                return Err(LoaderError::TemplateParse {
                    offset: pos,
                    detail: "mismatched tag closer".to_owned(),
                });
            }
            // Inside a dict literal, `}}` does not end the expression.
            if expr_close && *curly_depth > 0 {
                *curly_depth -= 1;
                tokens.push(Token {
                    kind: TokenKind::CloseBrace,
                    text: "}".to_owned(),
                    pos,
                });
                pos += 1;
                continue;
            }
            tokens.push(Token {
                kind: if is_expr {
                    TokenKind::CloseExpr
                } else {
                    TokenKind::CloseStmt
                },
                text: String::new(),
                pos,
            });
            return Ok(pos + 2);
        }
        let token_start = pos;
        let b = bytes[pos];
        // Unary `-`/`+` folding (mirror minja): binary after an operand,
        // else folded into a following number or a unary token.
        if (b == b'-' || b == b'+') && !prev_is_operand(tokens) {
            pos += 1;
            if pos < bytes.len() && bytes[pos].is_ascii_digit() {
                let (text, kind, end) = lex_number(source, pos)?;
                let mut signed = String::from(b as char);
                signed.push_str(&text);
                tokens.push(Token {
                    kind,
                    text: signed,
                    pos: token_start,
                });
                pos = end;
                continue;
            }
            tokens.push(Token {
                kind: TokenKind::Unary,
                text: source[token_start..pos].to_owned(),
                pos: token_start,
            });
            continue;
        }
        if b.is_ascii_alphabetic() || b == b'_' {
            let mut end = pos + 1;
            while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
                end += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Ident,
                text: source[pos..end].to_owned(),
                pos: token_start,
            });
            pos = end;
        } else if b.is_ascii_digit() {
            let (text, kind, end) = lex_number(source, pos)?;
            tokens.push(Token {
                kind,
                text,
                pos: token_start,
            });
            pos = end;
        } else if b == b'\'' || b == b'"' {
            let (text, end) = lex_string(source, pos)?;
            tokens.push(Token {
                kind: TokenKind::Str,
                text,
                pos: token_start,
            });
            pos = end;
        } else {
            let two = if pos + 2 <= bytes.len() {
                &source[pos..pos + 2]
            } else {
                ""
            };
            let (kind, len) = match two {
                "<=" | ">=" | "==" | "!=" => (TokenKind::Comparison, 2),
                _ => match b {
                    b'+' | b'-' | b'~' => (TokenKind::Additive, 1),
                    b'*' | b'/' | b'%' => (TokenKind::Multiplicative, 1),
                    b'<' | b'>' => (TokenKind::Comparison, 1),
                    b'=' => (TokenKind::Equals, 1),
                    b'(' => (TokenKind::OpenParen, 1),
                    b')' => (TokenKind::CloseParen, 1),
                    b'[' => (TokenKind::OpenBracket, 1),
                    b']' => (TokenKind::CloseBracket, 1),
                    b'.' => (TokenKind::Dot, 1),
                    b',' => (TokenKind::Comma, 1),
                    b':' => (TokenKind::Colon, 1),
                    b'|' => (TokenKind::Pipe, 1),
                    b'{' => (TokenKind::OpenBrace, 1),
                    b'}' => (TokenKind::CloseBrace, 1),
                    _ => {
                        return Err(LoaderError::TemplateParse {
                            offset: token_start,
                            detail: format!("unexpected character {b:?} in tag"),
                        });
                    }
                },
            };
            if kind == TokenKind::OpenBrace && is_expr {
                *curly_depth += 1;
            } else if kind == TokenKind::CloseBrace && is_expr && *curly_depth > 0 {
                *curly_depth -= 1;
            }
            tokens.push(Token {
                kind,
                text: source[pos..pos + len].to_owned(),
                pos: token_start,
            });
            pos += len;
        }
    }
}

/// True when the last tag token ends an operand (identifier, number,
/// string, `)` or `]`), making a following `-`/`+` binary (mirror minja).
fn prev_is_operand(tokens: &[Token]) -> bool {
    matches!(
        tokens.last().map(|t| t.kind),
        Some(
            TokenKind::Ident
                | TokenKind::Int
                | TokenKind::Float
                | TokenKind::Str
                | TokenKind::CloseParen
                | TokenKind::CloseBracket
        )
    )
}

/// Lexes an int or float literal at `pos`: digits with an optional
/// `.fraction` (only when a digit follows the dot); no exponents, and a
/// trailing `.` stays separate (mirror minja `consume_numeric`).
fn lex_number(source: &str, pos: usize) -> Result<(String, TokenKind, usize), LoaderError> {
    let bytes = source.as_bytes();
    let mut end = pos;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    let mut kind = TokenKind::Int;
    if end < bytes.len()
        && bytes[end] == b'.'
        && end + 1 < bytes.len()
        && bytes[end + 1].is_ascii_digit()
    {
        kind = TokenKind::Float;
        end += 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }
    Ok((source[pos..end].to_owned(), kind, end))
}

/// Lexes a quoted string at `pos` with the strict minja escape map
/// (`n t r b f v \ ' "`); anything else fails closed like the reference.
fn lex_string(source: &str, pos: usize) -> Result<(String, usize), LoaderError> {
    let quote = source.as_bytes()[pos];
    let mut out = String::new();
    let mut i = pos + 1;
    let bytes = source.as_bytes();
    loop {
        if i >= bytes.len() {
            return Err(LoaderError::TemplateParse {
                offset: pos,
                detail: "unterminated string literal".to_owned(),
            });
        }
        let b = bytes[i];
        if b == quote {
            return Ok((out, i + 1));
        }
        if b == b'\\' {
            i += 1;
            if i >= bytes.len() {
                return Err(LoaderError::TemplateParse {
                    offset: pos,
                    detail: "unexpected end of input after escape character".to_owned(),
                });
            }
            match bytes[i] {
                b'n' => out.push('\n'),
                b't' => out.push('\t'),
                b'r' => out.push('\r'),
                b'b' => out.push('\u{08}'),
                b'f' => out.push('\u{0C}'),
                b'v' => out.push('\u{0B}'),
                b'\'' => out.push('\''),
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                other => {
                    return Err(LoaderError::TemplateParse {
                        offset: pos,
                        detail: format!("unknown escape character \\{}", other as char),
                    });
                }
            }
            i += 1;
        } else {
            let len = utf8_len_at(bytes, i);
            out.push_str(&source[i..i + len]);
            i += len;
        }
    }
}
