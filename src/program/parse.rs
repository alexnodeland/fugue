//! The text front end: a lexer and a recursive-descent parser for the
//! `prob!`-subset syntax (the grammar is in the module docs).
//!
//! The parser enforces the format's nesting cap *while* it builds, measured
//! exactly as the JSON form nests (see `Program::nesting_depth`): a statement
//! knows its depth, an expression node its height, and a node that would sit
//! deeper than [`MAX_NESTING`] is refused as soon as it is built. So every
//! tree the parser builds is shallow — including those built by loops
//! (operator chains, runs of prefix or postfix operators), whose recursive
//! `Drop` would otherwise be what overflows — and a separate counter bounds
//! the parser's own recursion (parentheses, brackets, blocks), so hostile
//! input fails with an error instead of a stack overflow. Binary operators
//! are parsed with an explicit operator stack and prefix operators with a
//! loop, so a nesting level costs the parser a few frames whatever the
//! operators: programs at the cap parse on a 1 MiB stack (wasm's default)
//! even in a debug build.

use super::ast::{is_ident, Addr, BinOp, DistCall, Expr, Program, Stmt, MAX_NESTING, RESERVED};
use super::ProgramError;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    /// An integer literal's magnitude (a leading `-` is a separate token; the
    /// parser folds it, which is how `-9223372036854775808` fits an `i64`).
    Int(u64),
    Float(f64),
    Str(String),
    Sym(&'static str),
    Eof,
}

/// Longest first, so `==` is never read as `=` `=`. There is deliberately no
/// `<-` token: `a<-1` is a comparison with a negative number, as in Rust, and
/// the sampling arrow is recognized as `<` `-` only right after `let name`.
const SYMS: &[&str] = &[
    "::", "..", "==", "!=", "<=", ">=", "&&", "||", "(", ")", "{", "}", "[", "]", ",", ";", "+",
    "-", "*", "/", "=", ".", "!", "<", ">",
];

/// The parser's recursion budget (parentheses, brackets, blocks). In any
/// program the text printer writes, each recursion also descends at least one
/// JSON level, so no program within [`MAX_NESTING`] needs more.
const MAX_RECURSION: usize = MAX_NESTING;

/// An expression and its height in JSON levels (a leaf is 1).
type Parsed = (Expr, usize);

struct Parser {
    toks: Vec<(Tok, usize)>,
    pos: usize,
    line_starts: Vec<usize>,
    /// Current recursion depth, at most [`MAX_RECURSION`].
    rec: usize,
    /// JSON depth of the statement being parsed.
    stmt_depth: usize,
    /// JSON depth of the root of the expression being parsed.
    expr_depth: usize,
}

fn line_of(line_starts: &[usize], byte: usize) -> usize {
    match line_starts.binary_search(&byte) {
        Ok(l) => l + 1,
        Err(l) => l,
    }
}

fn syntax(line: usize, message: impl Into<String>) -> ProgramError {
    ProgramError::Syntax {
        line,
        message: message.into(),
    }
}

fn lex(src: &str, line_starts: &[usize]) -> Result<Vec<(Tok, usize)>, ProgramError> {
    let bytes = src.as_bytes();
    let err = |at: usize, msg: String| syntax(line_of(line_starts, at), msg);
    let mut toks = Vec::new();
    let mut i = 0;
    'outer: while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'"' {
            let start = i;
            let mut out = String::new();
            let mut j = i + 1;
            loop {
                match bytes.get(j) {
                    None => return Err(err(start, "unterminated string literal".to_string())),
                    Some(b'"') => {
                        j += 1;
                        break;
                    }
                    Some(b'\\') => {
                        let unescaped = match bytes.get(j + 1) {
                            Some(b'\\') => '\\',
                            Some(b'"') => '"',
                            Some(b'\'') => '\'',
                            Some(b'n') => '\n',
                            Some(b'r') => '\r',
                            Some(b't') => '\t',
                            Some(b'0') => '\0',
                            _ => {
                                let shown = src[j..].chars().take(2).collect::<String>();
                                return Err(err(
                                    j,
                                    format!(
                                        "unsupported escape `{shown}` in a string literal \
                                         (use \\\\, \\\", \\', \\n, \\r, \\t or \\0)"
                                    ),
                                ));
                            }
                        };
                        out.push(unescaped);
                        j += 2;
                    }
                    Some(_) => {
                        let ch = src[j..].chars().next().expect("j is a char boundary");
                        out.push(ch);
                        j += ch.len_utf8();
                    }
                }
            }
            toks.push((Tok::Str(out), start));
            i = j;
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            let mut is_float = false;
            while i < bytes.len() {
                let d = bytes[i];
                if d.is_ascii_digit() {
                    i += 1;
                } else if d == b'.'
                    && !is_float
                    && bytes.get(i + 1).is_some_and(|b| b.is_ascii_digit())
                {
                    // A '.' is a decimal point only when a digit follows:
                    // `0..n` must lex as Int(0) `..` n.
                    is_float = true;
                    i += 1;
                } else if d == b'e' || d == b'E' {
                    let sign = matches!(bytes.get(i + 1), Some(b'+' | b'-'));
                    let first = i + 1 + usize::from(sign);
                    if !bytes.get(first).is_some_and(|b| b.is_ascii_digit()) {
                        break;
                    }
                    is_float = true;
                    i = first;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    break;
                } else {
                    break;
                }
            }
            let text = &src[start..i];
            let tok = if is_float {
                match text.parse::<f64>() {
                    Ok(v) if v.is_finite() => Tok::Float(v),
                    _ => {
                        return Err(err(
                            start,
                            format!("number literal `{text}` is out of range"),
                        ))
                    }
                }
            } else {
                match text.parse::<u64>() {
                    Ok(v) => Tok::Int(v),
                    Err(_) => {
                        return Err(err(
                            start,
                            format!("integer literal `{text}` is out of range"),
                        ))
                    }
                }
            };
            toks.push((tok, start));
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            toks.push((Tok::Ident(src[start..i].to_string()), start));
            continue;
        }
        for s in SYMS {
            if src[i..].starts_with(s) {
                toks.push((Tok::Sym(s), i));
                i += s.len();
                continue 'outer;
            }
        }
        let ch = src[i..].chars().next().expect("i is a char boundary");
        if ch.is_whitespace() {
            i += ch.len_utf8();
            continue;
        }
        return Err(err(i, format!("unexpected character `{ch}`")));
    }
    toks.push((Tok::Eof, src.len()));
    Ok(toks)
}

/// Parse the text syntax into a [`Program`] (see [`Program::parse`]).
pub(crate) fn parse_program(src: &str) -> Result<Program, ProgramError> {
    let line_starts: Vec<usize> = std::iter::once(0)
        .chain(src.match_indices('\n').map(|(p, _)| p + 1))
        .collect();
    let toks = lex(src, &line_starts)?;
    // The program object is JSON level 1, its `body` array level 2, each
    // top-level statement level 3; `ret` sits at level 2.
    let mut p = Parser {
        toks,
        pos: 0,
        line_starts,
        rec: 0,
        stmt_depth: 3,
        expr_depth: 2,
    };
    let mut body = Vec::new();
    loop {
        if p.eat_kw("pure") {
            p.expect_sym("(")?;
            let ret = p.expr_at(2)?;
            p.expect_sym(")")?;
            p.eat_sym(";"); // optional trailing semicolon
            if p.peek() != &Tok::Eof {
                return Err(p.err("end of input after `pure(..)`"));
            }
            let program = Program::new(body, ret);
            program.validate()?;
            return Ok(program);
        }
        if p.peek() == &Tok::Eof {
            return Err(syntax(
                p.line(),
                "a model must end with `pure(<expr>)`".to_string(),
            ));
        }
        body.push(p.parse_stmt()?);
    }
}

impl Parser {
    fn line(&self) -> usize {
        line_of(&self.line_starts, self.toks[self.pos].1)
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].0
    }

    fn peek_at(&self, k: usize) -> &Tok {
        &self.toks[(self.pos + k).min(self.toks.len() - 1)].0
    }

    fn next(&mut self) -> Tok {
        let t = self.toks[self.pos].0.clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn err(&self, what: &str) -> ProgramError {
        let found = match self.peek() {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Int(v) => format!("`{v}`"),
            Tok::Float(v) => format!("`{v:?}`"),
            Tok::Str(s) => format!("{s:?}"),
            Tok::Sym(s) => format!("`{s}`"),
            Tok::Eof => "end of input".to_string(),
        };
        syntax(self.line(), format!("expected {what}, found {found}"))
    }

    fn eat_sym(&mut self, s: &'static str) -> bool {
        if self.peek() == &Tok::Sym(s) {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect_sym(&mut self, s: &'static str) -> Result<(), ProgramError> {
        if self.eat_sym(s) {
            Ok(())
        } else {
            Err(self.err(&format!("`{s}`")))
        }
    }

    fn at_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == kw)
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.at_kw(kw) {
            self.next();
            true
        } else {
            false
        }
    }

    /// A name for a variable, distribution or function: an identifier that
    /// is not a reserved word.
    fn expect_name(&mut self, what: &str) -> Result<String, ProgramError> {
        match self.peek() {
            Tok::Ident(s) if is_ident(s) => {
                let s = s.clone();
                self.next();
                Ok(s)
            }
            _ => Err(self.err(what)),
        }
    }

    // -- nesting bookkeeping -------------------------------------------------

    fn too_deep(&self) -> ProgramError {
        syntax(
            self.line(),
            format!("nested too deeply: the program format allows at most {MAX_NESTING} levels"),
        )
    }

    /// Enter one level of the parser's own recursion.
    fn enter(&mut self) -> Result<(), ProgramError> {
        self.rec += 1;
        if self.rec > MAX_RECURSION {
            return Err(syntax(
                self.line(),
                format!(
                    "nested too deeply (more than {MAX_RECURSION} levels of parentheses, \
                     brackets and blocks)"
                ),
            ));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.rec -= 1;
    }

    /// A built expression node of height `h`: refused if it already reaches
    /// below the nesting cap from where its expression's root sits.
    fn node(&self, e: Expr, h: usize) -> Result<Parsed, ProgramError> {
        if self.expr_depth + h - 1 > MAX_NESTING {
            return Err(self.too_deep());
        }
        Ok((e, h))
    }

    /// Parse an expression whose root sits at JSON depth `depth`.
    fn expr_at(&mut self, depth: usize) -> Result<Expr, ProgramError> {
        let outer = std::mem::replace(&mut self.expr_depth, depth);
        let parsed = self.parse_expr();
        self.expr_depth = outer;
        parsed.map(|(e, _)| e)
    }

    /// Step into a block: its array sits one JSON level below the statement
    /// that owns it, its statements two.
    fn open_block(&mut self) -> Result<(), ProgramError> {
        if self.stmt_depth + 1 > MAX_NESTING {
            return Err(self.too_deep());
        }
        self.enter()?;
        self.stmt_depth += 2;
        Ok(())
    }

    fn close_block(&mut self) {
        self.stmt_depth -= 2;
        self.leave();
    }

    // -- statements ----------------------------------------------------------

    fn parse_stmt(&mut self) -> Result<Stmt, ProgramError> {
        if self.stmt_depth > MAX_NESTING {
            return Err(self.too_deep());
        }
        if self.at_kw("let") {
            return self.parse_let();
        }
        if self.at_kw("observe") {
            return self.parse_observe();
        }
        if self.eat_kw("factor") {
            self.expect_sym("(")?;
            let logw = self.expr_at(self.stmt_depth + 1)?;
            self.expect_sym(")")?;
            self.expect_sym(";")?;
            return Ok(Stmt::Factor { logw });
        }
        if self.at_kw("for") {
            return self.parse_for();
        }
        if self.at_kw("if") {
            return self.parse_if();
        }
        if self.eat_kw("break") {
            self.expect_sym(";")?;
            return Ok(Stmt::Break {});
        }
        if self.at_kw("pure") {
            return Err(syntax(
                self.line(),
                "`pure(..)` ends the program; it cannot appear inside a block".to_string(),
            ));
        }
        if let Tok::Ident(name) = self.peek() {
            if is_ident(name) && self.peek_at(1) == &Tok::Sym("=") {
                let var = name.clone();
                self.next();
                self.next();
                let value = self.expr_at(self.stmt_depth + 1)?;
                self.expect_sym(";")?;
                return Ok(Stmt::Assign { var, value });
            }
        }
        Err(self.err(
            "a statement (`let`, `observe`, `factor`, `for`, `if`, `break`, an assignment, or `pure`)",
        ))
    }

    fn parse_let(&mut self) -> Result<Stmt, ProgramError> {
        self.next(); // `let`
        self.eat_kw("mut"); // every variable is assignable; `mut` is accepted sugar
        let var = self.expect_name("a variable name")?;
        if self.eat_sym("<") {
            // The sampling arrow `<-`.
            if !self.eat_sym("-") {
                return Err(self.err("`<-`"));
            }
            if !self.eat_kw("sample") {
                return Err(self.err("`sample`"));
            }
            self.expect_sym("(")?;
            let addr = self.parse_addr()?;
            self.expect_sym(",")?;
            let dist = self.parse_dist()?;
            self.expect_sym(")")?;
            self.expect_sym(";")?;
            return Ok(Stmt::Sample { var, addr, dist });
        }
        if !self.eat_sym("=") {
            return Err(self.err("`<-` or `=`"));
        }
        let value = self.expr_at(self.stmt_depth + 1)?;
        self.expect_sym(";")?;
        Ok(Stmt::Let { var, value })
    }

    fn parse_observe(&mut self) -> Result<Stmt, ProgramError> {
        self.next(); // `observe`
        self.expect_sym("(")?;
        let addr = self.parse_addr()?;
        self.expect_sym(",")?;
        let dist = self.parse_dist()?;
        self.expect_sym(",")?;
        let value = self.expr_at(self.stmt_depth + 1)?;
        self.expect_sym(")")?;
        self.expect_sym(";")?;
        Ok(Stmt::Observe { addr, dist, value })
    }

    fn parse_for(&mut self) -> Result<Stmt, ProgramError> {
        self.next(); // `for`
        let var = self.expect_name("a loop variable name")?;
        if !self.eat_kw("in") {
            return Err(self.err("`in`"));
        }
        let start = self.expr_at(self.stmt_depth + 1)?;
        self.expect_sym("..")?;
        let end = self.expr_at(self.stmt_depth + 1)?;
        let body = self.parse_block()?;
        Ok(Stmt::For {
            var,
            start,
            end,
            body,
        })
    }

    fn parse_if(&mut self) -> Result<Stmt, ProgramError> {
        self.next(); // `if`
        let cond = self.expr_at(self.stmt_depth + 1)?;
        let then_branch = self.parse_block()?;
        let else_branch = if self.eat_kw("else") {
            if self.at_kw("if") {
                // `else if` nests an `If` as the else branch's only statement.
                self.open_block()?;
                let nested = self.parse_stmt();
                self.close_block();
                vec![nested?]
            } else {
                self.parse_block()?
            }
        } else {
            Vec::new()
        };
        Ok(Stmt::If {
            cond,
            then_branch,
            else_branch,
        })
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ProgramError> {
        self.expect_sym("{")?;
        self.open_block()?;
        let mut stmts = Vec::new();
        while self.peek() != &Tok::Sym("}") {
            if self.peek() == &Tok::Eof {
                return Err(self.err("`}`"));
            }
            stmts.push(self.parse_stmt()?);
        }
        self.next();
        self.close_block();
        Ok(stmts)
    }

    fn parse_addr(&mut self) -> Result<Addr, ProgramError> {
        if !self.eat_kw("addr") {
            return Err(self.err("`addr!(..)`"));
        }
        self.expect_sym("!")?;
        self.expect_sym("(")?;
        let name = match self.peek() {
            Tok::Str(s) => {
                let s = s.clone();
                self.next();
                s
            }
            _ => return Err(self.err("a string literal address name")),
        };
        let index = if self.eat_sym(",") {
            // The addr object is one level below the statement, its index two.
            Some(self.expr_at(self.stmt_depth + 2)?)
        } else {
            None
        };
        self.expect_sym(")")?;
        Ok(Addr { name, index })
    }

    fn parse_dist(&mut self) -> Result<DistCall, ProgramError> {
        let name = self.expect_name("a distribution name")?;
        if self.eat_sym("::") {
            let line = self.line();
            let method = self.expect_name("`new`")?;
            if method != "new" {
                return Err(syntax(
                    line,
                    format!("unknown distribution constructor `{name}::{method}` (only `::new` is accepted)"),
                ));
            }
        }
        // The dist object is one level below the statement, its (always
        // written) `args` array two, the arguments three.
        if self.stmt_depth + 2 > MAX_NESTING {
            return Err(self.too_deep());
        }
        self.expect_sym("(")?;
        let outer = std::mem::replace(&mut self.expr_depth, self.stmt_depth + 3);
        let args = self.parse_list(")");
        self.expr_depth = outer;
        let args = args?.into_iter().map(|(e, _)| e).collect();
        // Accepted, ignored Rust-ism: a trailing `.unwrap()`.
        if self.peek() == &Tok::Sym(".")
            && matches!(self.peek_at(1), Tok::Ident(s) if s == "unwrap")
        {
            self.next();
            self.next();
            self.expect_sym("(")?;
            self.expect_sym(")")?;
        }
        Ok(DistCall { name, args })
    }

    // -- expressions ---------------------------------------------------------

    /// Comma-separated expressions up to `close` (the opening bracket has been
    /// consumed); a trailing comma is allowed.
    fn parse_list(&mut self, close: &'static str) -> Result<Vec<Parsed>, ProgramError> {
        let mut items = Vec::new();
        while !self.eat_sym(close) {
            items.push(self.parse_expr()?);
            if !self.eat_sym(",") {
                self.expect_sym(close)?;
                break;
            }
        }
        Ok(items)
    }

    fn binary_op(&self) -> Option<BinOp> {
        Some(match self.peek() {
            Tok::Sym("||") => BinOp::Or,
            Tok::Sym("&&") => BinOp::And,
            Tok::Sym("==") => BinOp::Eq,
            Tok::Sym("!=") => BinOp::Ne,
            Tok::Sym("<") => BinOp::Lt,
            Tok::Sym("<=") => BinOp::Le,
            Tok::Sym(">") => BinOp::Gt,
            Tok::Sym(">=") => BinOp::Ge,
            Tok::Sym("+") => BinOp::Add,
            Tok::Sym("-") => BinOp::Sub,
            Tok::Sym("*") => BinOp::Mul,
            Tok::Sym("/") => BinOp::Div,
            _ => return None,
        })
    }

    /// Binary operators by precedence climbing over an explicit operator
    /// stack: left-associative, with comparisons refusing to chain.
    fn parse_expr(&mut self) -> Result<Parsed, ProgramError> {
        let mut operands = vec![self.parse_operand()?];
        let mut ops: Vec<BinOp> = Vec::new();
        while let Some(op) = self.binary_op() {
            let line = self.line();
            self.next();
            while let Some(&top) = ops.last() {
                if top.precedence() < op.precedence() {
                    break;
                }
                if top.precedence() == 3 && op.precedence() == 3 {
                    return Err(syntax(
                        line,
                        "comparison operators cannot be chained; add parentheses".to_string(),
                    ));
                }
                ops.pop();
                self.reduce(&mut operands, top)?;
            }
            ops.push(op);
            operands.push(self.parse_operand()?);
        }
        while let Some(op) = ops.pop() {
            self.reduce(&mut operands, op)?;
        }
        Ok(operands.pop().expect("one operand remains"))
    }

    fn reduce(&self, operands: &mut Vec<Parsed>, op: BinOp) -> Result<(), ProgramError> {
        let (rhs, rh) = operands.pop().expect("a right operand");
        let (lhs, lh) = operands.pop().expect("a left operand");
        operands.push(self.node(Expr::binary(op, lhs, rhs), 1 + lh.max(rh))?);
        Ok(())
    }

    /// Whether the next tokens are `-` and a number literal to fold into a
    /// negative literal: not when a postfix operator follows (`-x.len()` is
    /// `-(x.len())`, and so is `-2.0.len()`).
    fn negative_literal_ahead(&self) -> bool {
        self.peek() == &Tok::Sym("-")
            && matches!(self.peek_at(1), Tok::Int(_) | Tok::Float(_))
            && !matches!(self.peek_at(2), Tok::Sym("[") | Tok::Sym("."))
    }

    /// Prefix operators, a primary, postfix operators. Postfix binds tighter
    /// than prefix, as in Rust. Both runs are loops, not recursion.
    fn parse_operand(&mut self) -> Result<Parsed, ProgramError> {
        let mut prefix: Vec<bool> = Vec::new(); // `true` for `-`, `false` for `!`
        loop {
            match self.peek() {
                Tok::Sym("-") if !self.negative_literal_ahead() => prefix.push(true),
                Tok::Sym("!") => prefix.push(false),
                _ => break,
            }
            self.next();
        }
        let (mut e, mut h) = self.parse_primary()?;
        loop {
            if self.eat_sym("[") {
                self.enter()?;
                let index = self.parse_expr();
                self.leave();
                let (index, ih) = index?;
                self.expect_sym("]")?;
                let indexed = Expr::Index {
                    array: Box::new(e),
                    index: Box::new(index),
                };
                (e, h) = self.node(indexed, 1 + h.max(ih))?;
            } else if self.peek() == &Tok::Sym(".") {
                if !matches!(self.peek_at(1), Tok::Ident(m) if m == "len") {
                    self.next();
                    return Err(self.err("`len()` (the only method is `.len()`)"));
                }
                self.next();
                self.next();
                self.expect_sym("(")?;
                self.expect_sym(")")?;
                (e, h) = self.node(Expr::Len { array: Box::new(e) }, h + 1)?;
            } else {
                break;
            }
        }
        for neg in prefix.into_iter().rev() {
            let arg = Box::new(e);
            let op = if neg {
                Expr::Neg { arg }
            } else {
                Expr::Not { arg }
            };
            (e, h) = self.node(op, h + 1)?;
        }
        Ok((e, h))
    }

    fn parse_primary(&mut self) -> Result<Parsed, ProgramError> {
        let line = self.line();
        match self.peek().clone() {
            // `-` directly before a number literal (see
            // `negative_literal_ahead`): one negative literal, which is how
            // `-9223372036854775808` fits an `i64`.
            Tok::Sym("-") => {
                self.next();
                match self.next() {
                    Tok::Float(v) => self.node(Expr::F64 { value: -v }, 1),
                    Tok::Int(u) => match i64::try_from(-i128::from(u)) {
                        Ok(value) => self.node(Expr::Int { value }, 1),
                        Err(_) => Err(syntax(
                            line,
                            format!("integer literal `-{u}` is out of range for i64"),
                        )),
                    },
                    _ => unreachable!("checked by negative_literal_ahead"),
                }
            }
            Tok::Int(u) => {
                self.next();
                match i64::try_from(u) {
                    Ok(value) => self.node(Expr::Int { value }, 1),
                    Err(_) => Err(syntax(
                        line,
                        format!("integer literal `{u}` is out of range for i64"),
                    )),
                }
            }
            Tok::Float(value) => {
                self.next();
                self.node(Expr::F64 { value }, 1)
            }
            Tok::Sym("(") => {
                self.next();
                self.enter()?;
                let inner = self.parse_expr();
                self.leave();
                let inner = inner?;
                self.expect_sym(")")?;
                Ok(inner)
            }
            Tok::Sym("[") => {
                self.next();
                self.enter()?;
                let items = self.parse_list("]");
                self.leave();
                let (items, h) = unzip(items?);
                self.node(Expr::Array { items }, h)
            }
            Tok::Ident(name) => match name.as_str() {
                "true" | "false" => {
                    self.next();
                    self.node(
                        Expr::Bool {
                            value: name == "true",
                        },
                        1,
                    )
                }
                kw if RESERVED.contains(&kw) => Err(self.err("an expression")),
                _ => {
                    self.next();
                    if self.eat_sym("(") {
                        self.enter()?;
                        let args = self.parse_list(")");
                        self.leave();
                        let (args, h) = unzip(args?);
                        self.node(Expr::Call { func: name, args }, h)
                    } else {
                        self.node(Expr::Var { name }, 1)
                    }
                }
            },
            _ => Err(self.err("an expression")),
        }
    }
}

/// Split parsed list items into expressions and the height of the node that
/// holds them: the node, its (always written) array, then the tallest item.
fn unzip(items: Vec<Parsed>) -> (Vec<Expr>, usize) {
    let h = 2 + items.iter().map(|(_, h)| *h).max().unwrap_or(0);
    (items.into_iter().map(|(e, _)| e).collect(), h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Program {
        parse_program(src).unwrap_or_else(|e| panic!("{src:?}: {e}"))
    }

    fn ret(src: &str) -> Expr {
        parse(&format!("pure({src})")).ret
    }

    #[test]
    fn precedence_matches_rust() {
        assert_eq!(ret("1 + 2 * 3").to_string(), "1 + 2 * 3");
        assert!(matches!(ret("1 + 2 * 3"), Expr::Add { .. }));
        assert!(matches!(ret("a || b && c"), Expr::Or { .. }));
        assert!(matches!(ret("a && b == c"), Expr::And { .. }));
        assert!(matches!(ret("a < b + 1"), Expr::Lt { .. }));
        assert!(matches!(ret("!a && b"), Expr::And { .. }));
        assert!(matches!(ret("-x.len()"), Expr::Neg { .. }));
        assert!(
            matches!(ret("a - b - c"), Expr::Sub { lhs, .. } if matches!(*lhs, Expr::Sub { .. }))
        );
    }

    #[test]
    fn negative_literals_fold_and_arrow_is_contextual() {
        assert_eq!(ret("-2.0"), Expr::F64 { value: -2.0 });
        assert_eq!(ret("-9223372036854775808"), Expr::Int { value: i64::MIN });
        assert!(parse_program("pure(9223372036854775808)").is_err());
        assert!(matches!(ret("-(2.0)"), Expr::Neg { .. }));
        // `x<-1` is a comparison with -1, as in Rust.
        assert!(matches!(ret("x<-1"), Expr::Lt { rhs, .. } if *rhs == Expr::Int { value: -1 }));
        // ...and `let x <- sample` still samples, with or without a space.
        for src in [
            "let x <- sample(addr!(\"x\"), Normal(0.0, 1.0)); pure(x)",
            "let x < - sample(addr!(\"x\"), Normal(0.0, 1.0)); pure(x)",
        ] {
            assert!(matches!(parse(src).body[0], Stmt::Sample { .. }));
        }
    }

    #[test]
    fn lexer_details() {
        // Rust string escapes are honoured, so names match `addr!` in Rust.
        let p = parse(r#"observe(addr!("a\\#1\"q"), Normal(0.0, 1.0), 0.0); pure(0)"#);
        let Stmt::Observe { addr, .. } = &p.body[0] else {
            panic!()
        };
        assert_eq!(addr.name, "a\\#1\"q");
        assert!(parse_program(r#"observe(addr!("a\q"), Normal(0.0, 1.0), 0.0); pure(0)"#).is_err());
        // Numbers: `0..n` ranges, exponents, typed literals.
        assert_eq!(ret("1e3"), Expr::F64 { value: 1000.0 });
        assert_eq!(ret("2.5E-1"), Expr::F64 { value: 0.25 });
        assert_eq!(ret("7"), Expr::Int { value: 7 });
        assert!(parse_program("pure(1e999)").is_err());
        let p = parse("for i in 0..3 { } pure(0)");
        assert!(matches!(
            &p.body[0],
            Stmt::For {
                start: Expr::Int { value: 0 },
                ..
            }
        ));
        // Comments and non-ASCII whitespace are skipped.
        assert_eq!(ret("1 // one\n\u{a0}"), Expr::Int { value: 1 });
    }

    #[test]
    fn syntax_errors_carry_lines() {
        for (src, line) in [
            ("let x = 1.0;\nlet y = ;\npure(x)", 2),
            ("let x = 1.0;\n\npure(x) extra", 3),
            ("let x = 1.0;", 1),
            ("pure(a < b < c)", 1),
            ("let x = @;\npure(x)", 1),
            ("for i in 0..3 {\n pure(i)\n}\npure(0)", 2),
            ("pure(x.foo())", 1),
            (
                "let x <- sample(addr!(\"x\"), Normal::create(0.0, 1.0)); pure(x)",
                1,
            ),
            ("let if = 1; pure(0)", 1),
        ] {
            match parse_program(src) {
                Err(ProgramError::Syntax { line: l, .. }) => assert_eq!(l, line, "{src:?}"),
                other => panic!("{src:?}: expected a syntax error, got {other:?}"),
            }
        }
    }

    #[test]
    fn deep_nesting_fails_cleanly() {
        let too_deep = |src: &str| match parse_program(src) {
            Err(ProgramError::Syntax { message, .. }) => {
                assert!(message.contains("nested too deeply"), "{message}")
            }
            other => panic!("{other:?}"),
        };
        // Recursion (parentheses, brackets, blocks) is bounded...
        too_deep(&format!(
            "pure({}1{})",
            "(".repeat(10_000),
            ")".repeat(10_000)
        ));
        too_deep(&format!(
            "pure({}1{})",
            "[".repeat(10_000),
            "]".repeat(10_000)
        ));
        too_deep(&format!(
            "pure({}1.0{})",
            "exp(".repeat(10_000),
            ")".repeat(10_000)
        ));
        too_deep(&format!(
            "{}{}pure(0)",
            "if true { ".repeat(10_000),
            "} ".repeat(10_000)
        ));
        let mut ladder = String::new();
        for _ in 0..10_000 {
            ladder.push_str("if true { } else ");
        }
        too_deep(&format!("{ladder}{{ }} pure(0)"));
        // ...and so are the trees built by loops, which would otherwise be
        // deep enough to overflow on drop: prefix and postfix runs, operator
        // chains.
        too_deep(&format!("pure({}x)", "-".repeat(100_000)));
        too_deep(&format!("pure({}x)", "!".repeat(100_000)));
        too_deep(&format!("pure(x{})", ".len()".repeat(100_000)));
        too_deep(&format!("pure(x{})", "[0]".repeat(100_000)));
        too_deep(&format!("pure(x{})", " + x".repeat(100_000)));
        too_deep(&format!("pure(x{})", " * x - x".repeat(50_000)));
    }
}
