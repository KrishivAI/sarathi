// SPDX-License-Identifier: Apache-2.0
//! Permission rules: an effect and a `Tool(pattern)` matcher.

/// What a matching rule does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Run the call without asking.
    Allow,
    /// Ask the host.
    Ask,
    /// Refuse; beats every allow and ask.
    Deny,
}

/// A rule did not parse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuleError {
    /// The tool part is empty or holds a space or a parenthesis.
    #[error("rule `{0}`: the tool name is empty or malformed")]
    BadTool(Box<str>),
    /// An opening parenthesis without the closing one at the end.
    #[error("rule `{0}`: `(` is not closed by a final `)`")]
    Unclosed(Box<str>),
    /// `Tool()`: say `Tool` to match every call.
    #[error("rule `{0}`: the pattern is empty (write the tool name alone to match every call)")]
    EmptyPattern(Box<str>),
}

/// `Tool` or `Tool(pattern)` with an effect.
///
/// The tool part and the pattern are globs: `*` matches any run of characters (`/` included), `?` one character. A
/// pattern is matched against the call's *subject* (the command, path or URL the tool reports through
/// [`Toolbox::subject`](crate::Toolbox::subject)); a call with no subject is not matched by a rule that has a pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    effect: Effect,
    tool: Box<str>,
    pattern: Option<Box<str>>,
    source: Box<str>,
}

impl Rule {
    /// Parses `Tool` or `Tool(pattern)`.
    ///
    /// # Errors
    /// [`RuleError`] for an empty or malformed tool name, a missing `)` or an empty pattern.
    pub fn parse(effect: Effect, text: &str) -> Result<Rule, RuleError> {
        let text = text.trim();
        let (tool, pattern) = match text.find('(') {
            None => (text, None),
            Some(open) => {
                let inner = text[open + 1..]
                    .strip_suffix(')')
                    .ok_or_else(|| RuleError::Unclosed(text.into()))?;
                if inner.is_empty() {
                    return Err(RuleError::EmptyPattern(text.into()));
                }
                (&text[..open], Some(inner))
            }
        };
        if tool.is_empty() || tool.contains(|c: char| c.is_whitespace() || c == ')') {
            return Err(RuleError::BadTool(text.into()));
        }
        Ok(Rule {
            effect,
            tool: tool.into(),
            pattern: pattern.map(Into::into),
            source: text.into(),
        })
    }

    /// The effect of a match.
    pub fn effect(&self) -> Effect {
        self.effect
    }

    /// The rule as written.
    pub fn as_str(&self) -> &str {
        &self.source
    }

    /// Whether the rule covers a call of `tool` with `subject`.
    pub fn matches(&self, tool: &str, subject: Option<&str>) -> bool {
        if !glob(&self.tool, tool) {
            return false;
        }
        match (&self.pattern, subject) {
            (None, _) => true,
            (Some(p), Some(s)) => glob(p, s),
            (Some(_), None) => false,
        }
    }
}

/// `*` any run, `?` one character; everything else literal. Iterative, no recursion, linear space.
fn glob(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) && p[pi] != '*' {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some((pi, ti));
            pi += 1;
        } else if let Some((sp, st)) = star {
            pi = sp + 1;
            ti = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(s: &str) -> Rule {
        Rule::parse(Effect::Allow, s).unwrap()
    }

    #[test]
    fn globs() {
        assert!(glob("*", ""));
        assert!(glob("git *", "git status"));
        assert!(!glob("git *", "gitx status"));
        assert!(glob("src/**/*.rs", "src/a/b/c.rs"));
        assert!(glob("a?c", "abc"));
        assert!(!glob("a?c", "ac"));
        assert!(glob("*a*b*", "xxaxxbxx"));
        assert!(!glob("*a*b*", "xxbxxaxx"));
        assert!(glob("é*", "éa"));
    }

    #[test]
    fn parse_and_match() {
        let r = rule("edit");
        assert!(r.matches("edit", None) && r.matches("edit", Some("x")));
        assert!(!r.matches("edits", None));
        let r = rule("bash(git status*)");
        assert!(r.matches("bash", Some("git status -s")));
        assert!(!r.matches("bash", Some("rm -rf /")));
        assert!(!r.matches("bash", None), "a pattern needs a subject");
        assert!(rule("mcp_*").matches("mcp_files", None));
        assert!(rule("*").matches("anything", None));
        assert_eq!(rule(" look ").as_str(), "look");
        assert_eq!(rule("f((x))").as_str(), "f((x))");
        assert!(rule("f((x))").matches("f", Some("(x)")));
    }

    #[test]
    fn malformed_rules_are_refused() {
        for bad in ["", "(x)", "a b", "a)"] {
            assert!(
                matches!(Rule::parse(Effect::Deny, bad), Err(RuleError::BadTool(_))),
                "{bad}"
            );
        }
        assert!(matches!(
            Rule::parse(Effect::Deny, "a(x"),
            Err(RuleError::Unclosed(_))
        ));
        assert!(matches!(
            Rule::parse(Effect::Deny, "a()"),
            Err(RuleError::EmptyPattern(_))
        ));
    }
}
