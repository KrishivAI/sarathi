// SPDX-License-Identifier: Apache-2.0
//! The policy: layered rules and a mode, giving a verdict for one call.

use crate::rule::{Effect, Rule};
use crate::tool::ToolSpec;

/// Who set a rule. All scopes weigh the same: deny wins whatever its scope. The scope is kept for the reason given to
/// the model and the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
    /// The organisation's rules.
    Organisation,
    /// The project's rules.
    Project,
    /// The user's rules.
    User,
    /// Rules made during this session (an "allow always" answer).
    Session,
}

impl Scope {
    fn name(self) -> &'static str {
        match self {
            Scope::Organisation => "organisation",
            Scope::Project => "project",
            Scope::User => "user",
            Scope::Session => "session",
        }
    }
}

/// What happens to a call no rule matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Mode {
    /// Read-only tools run; everything else asks.
    #[default]
    Default,
    /// Read-only tools and edits run; everything else asks.
    AcceptEdits,
    /// Read-only tools run; everything else is refused (the model plans and does not act).
    Plan,
    /// Everything runs. Deny rules still win.
    Bypass,
}

/// The policy's answer for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Run it.
    Allow,
    /// Ask the host.
    Ask,
    /// Refuse, with the reason the model is told.
    Deny(String),
}

/// Rules by scope and a mode.
#[derive(Debug, Clone, Default)]
pub struct Policy {
    mode: Mode,
    rules: Vec<(Scope, Rule)>,
}

impl Policy {
    /// No rules, the given mode.
    pub fn new(mode: Mode) -> Self {
        Policy {
            mode,
            rules: Vec::new(),
        }
    }

    /// The mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Changes the mode.
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
    }

    /// Adds a rule.
    pub fn add(&mut self, scope: Scope, rule: Rule) {
        self.rules.push((scope, rule));
    }

    /// The verdict for a call of `spec` with `subject`: a matching deny rule, else a matching ask rule, else a
    /// matching allow rule, else the mode's answer.
    pub fn evaluate(&self, spec: &ToolSpec, subject: Option<&str>) -> Verdict {
        let mut ask = false;
        let mut allow = false;
        for (scope, rule) in &self.rules {
            if !rule.matches(&spec.name, subject) {
                continue;
            }
            match rule.effect() {
                Effect::Deny => {
                    return Verdict::Deny(format!(
                        "`{}` is refused by the {} rule `{}`",
                        spec.name,
                        scope.name(),
                        rule.as_str()
                    ));
                }
                Effect::Ask => ask = true,
                Effect::Allow => allow = true,
            }
        }
        if ask {
            return Verdict::Ask;
        }
        if allow {
            return Verdict::Allow;
        }
        match self.mode {
            Mode::Bypass => Verdict::Allow,
            Mode::Plan if spec.read_only => Verdict::Allow,
            Mode::Plan => Verdict::Deny(format!(
                "`{}` changes things and the session is in plan mode",
                spec.name
            )),
            Mode::AcceptEdits if spec.read_only || spec.edits => Verdict::Allow,
            Mode::Default if spec.read_only => Verdict::Allow,
            Mode::Default | Mode::AcceptEdits => Verdict::Ask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(name: &str) -> ToolSpec {
        ToolSpec::new(name, "d", json!({}))
    }
    fn rule(e: Effect, s: &str) -> Rule {
        Rule::parse(e, s).unwrap()
    }

    #[test]
    fn deny_beats_ask_beats_allow_across_scopes() {
        let mut p = Policy::new(Mode::Bypass);
        p.add(Scope::User, rule(Effect::Allow, "bash"));
        p.add(Scope::Project, rule(Effect::Ask, "bash(git *)"));
        assert_eq!(p.evaluate(&spec("bash"), Some("git push")), Verdict::Ask);
        assert_eq!(p.evaluate(&spec("bash"), Some("ls")), Verdict::Allow);
        p.add(Scope::Organisation, rule(Effect::Deny, "bash(git push*)"));
        let Verdict::Deny(why) = p.evaluate(&spec("bash"), Some("git push -f")) else {
            panic!("deny wins");
        };
        assert!(
            why.contains("organisation") && why.contains("bash(git push*)"),
            "{why}"
        );
        assert_eq!(p.evaluate(&spec("bash"), Some("git pull")), Verdict::Ask);
    }

    #[test]
    fn the_mode_answers_when_no_rule_matches() {
        let ro = spec("look").read_only();
        let ed = spec("edit").edits();
        let other = spec("bash");
        let at = |m| Policy::new(m);
        assert_eq!(at(Mode::Default).evaluate(&ro, None), Verdict::Allow);
        assert_eq!(at(Mode::Default).evaluate(&ed, None), Verdict::Ask);
        assert_eq!(at(Mode::AcceptEdits).evaluate(&ed, None), Verdict::Allow);
        assert_eq!(at(Mode::AcceptEdits).evaluate(&other, None), Verdict::Ask);
        assert_eq!(at(Mode::Plan).evaluate(&ro, None), Verdict::Allow);
        assert!(matches!(
            at(Mode::Plan).evaluate(&ed, None),
            Verdict::Deny(_)
        ));
        assert_eq!(at(Mode::Bypass).evaluate(&other, None), Verdict::Allow);
    }

    #[test]
    fn a_rule_can_make_a_read_only_tool_ask() {
        let mut p = Policy::default();
        p.add(Scope::Project, rule(Effect::Ask, "look"));
        assert_eq!(p.evaluate(&spec("look").read_only(), None), Verdict::Ask);
    }
}
