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
/// [`Toolbox::subject`](crate::Toolbox::subject)). A call with no subject fails closed: a deny or ask rule with a
/// pattern matches it (the tool cannot show it is outside the rule), an allow rule with a pattern does not.
///
/// Matching limits (a rule is a convenience, not a sandbox): the subject is matched as written, plus, for a
/// whitespace-free path, in its lexically normalised form (`./`, `//` and `a/../` removed), so `secret/*` also
/// refuses `./secret/x`. A deny or ask rule matches either form; an allow rule needs both. An allow rule with a pattern
/// never matches a subject holding a shell control character (`;`, `&`, `|`, `<`, `>`, `` ` ``, `$`, a newline),
/// so `git status*` does not allow `git status; rm x`. Symlinks and a shell's own expansions are not resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    effect: Effect,
    tool: Box<str>,
    pattern: Option<Box<str>>,
    /// The pattern is compared as a literal string, not as a glob.
    literal: bool,
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
            literal: false,
            source: text.into(),
        })
    }

    /// A rule for exactly one tool and, when given, exactly one subject (no glob characters are special in it). Used
    /// for an "allow always" answer, which covers what was asked and nothing wider.
    pub fn exact(effect: Effect, tool: &str, subject: Option<&str>) -> Rule {
        Rule {
            effect,
            tool: tool.into(),
            pattern: subject.map(Into::into),
            literal: true,
            source: match subject {
                Some(s) => format!("{tool}({s})").into(),
                None => tool.into(),
            },
        }
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
            (Some(_), None) => self.effect != Effect::Allow,
            (Some(p), Some(s)) => self.matches_subject(p, s),
        }
    }

    fn matches_subject(&self, pattern: &str, subject: &str) -> bool {
        let one = |t: &str| {
            if self.literal {
                pattern == t
            } else {
                glob(pattern, t)
            }
        };
        let norm = normalise(subject);
        let raw = one(subject);
        match self.effect {
            Effect::Allow => {
                raw && norm.as_deref().is_none_or(one)
                    && (self.literal || !subject.contains(is_shell_control))
            }
            Effect::Ask | Effect::Deny => raw || norm.as_deref().is_some_and(one),
        }
    }
}

fn is_shell_control(c: char) -> bool {
    matches!(c, ';' | '&' | '|' | '<' | '>' | '`' | '$' | '\n' | '\r')
}

/// The lexically normalised form of a whitespace-free path (`./` and `//` dropped, `a/..` folded), or `None` when
/// the subject is not such a path or is already normal.
fn normalise(subject: &str) -> Option<String> {
    if subject.contains(char::is_whitespace) || subject.contains("://") || !subject.contains('/') {
        return None;
    }
    let rooted = subject.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for seg in subject.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|l| *l != "..") {
                    parts.pop();
                } else if !rooted {
                    parts.push("..");
                }
            }
            s => parts.push(s),
        }
    }
    let joined = parts.join("/");
    let out = if rooted { format!("/{joined}") } else { joined };
    (out != subject).then_some(out)
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
        assert!(!r.matches("bash", None), "an allow pattern needs a subject");
        assert!(rule("mcp_*").matches("mcp_files", None));
        assert!(rule("*").matches("anything", None));
        assert_eq!(rule(" look ").as_str(), "look");
        assert_eq!(rule("f((x))").as_str(), "f((x))");
        assert!(rule("f((x))").matches("f", Some("(x)")));
    }

    #[test]
    fn a_pattern_deny_or_ask_fails_closed_without_a_subject() {
        let deny = Rule::parse(Effect::Deny, "bash(rm *)").unwrap();
        let ask = Rule::parse(Effect::Ask, "bash(rm *)").unwrap();
        assert!(deny.matches("bash", None) && ask.matches("bash", None));
        assert!(!deny.matches("look", None));
    }

    #[test]
    fn paths_are_normalised_and_compound_commands_do_not_ride_an_allow() {
        let deny = Rule::parse(Effect::Deny, "look(secret/*)").unwrap();
        assert!(deny.matches("look", Some("./secret/k")));
        assert!(deny.matches("look", Some("a/../secret/k")));
        assert!(deny.matches("look", Some("secret//k")));
        assert!(!deny.matches("look", Some("public/k")));
        let allow = Rule::parse(Effect::Allow, "bash(git status*)").unwrap();
        assert!(allow.matches("bash", Some("git status -s")));
        for bad in [
            "git status; rm x",
            "git status && rm x",
            "git status | sh",
            "git status $(x)",
            "git status\nrm x",
        ] {
            assert!(!allow.matches("bash", Some(bad)), "{bad}");
        }
        let allow_path = Rule::parse(Effect::Allow, "look(src/*)").unwrap();
        assert!(allow_path.matches("look", Some("src/a")));
        assert!(
            !allow_path.matches("look", Some("src/../secret/a")),
            "the normal form is outside"
        );
    }

    #[test]
    fn an_exact_rule_covers_one_subject_only() {
        let r = Rule::exact(Effect::Allow, "bash", Some("ls *"));
        assert!(r.matches("bash", Some("ls *")));
        assert!(!r.matches("bash", Some("ls a")), "no glob in an exact rule");
        assert!(!r.matches("bash", None));
        assert_eq!(r.as_str(), "bash(ls *)");
        assert!(Rule::exact(Effect::Allow, "bash", None).matches("bash", Some("x")));
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
