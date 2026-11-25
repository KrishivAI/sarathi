// SPDX-License-Identifier: Apache-2.0
//! The permission gate: the policy, then the host callback for an ask.

use std::sync::Arc;

use sarathi_protocol::{CallId, Decision};
use serde_json::Value;

use crate::policy::{Policy, Scope, Verdict};
use crate::rule::{Effect, Rule};
use crate::tool::{BoxFuture, ToolSpec};

/// What the host is asked.
#[derive(Debug, Clone, PartialEq)]
pub struct AskRequest {
    /// The call.
    pub call: CallId,
    /// The tool.
    pub tool: String,
    /// The call's input.
    pub input: Value,
    /// What a rule pattern would match (command, path, URL), when the tool says.
    pub subject: Option<String>,
}

/// The host's permission callback. The future may stay pending as long as the host likes (a person is deciding); the
/// run waits, and an interrupt drops the future.
pub trait PermissionHost: Send + Sync {
    /// Answers one ask.
    fn ask<'a>(&'a self, request: &'a AskRequest) -> BoxFuture<'a, Decision>;
}

/// Whether a call may run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    /// Run it. `asked` is the host's answer when the host was asked (the caller logs it, source included).
    Allowed {
        /// The answer, or `None` when a rule or the mode allowed the call without asking.
        asked: Option<Decision>,
        /// The session rule an "allow always" added, as text, so a resumed session can add it again.
        added_rule: Option<String>,
    },
    /// Do not run it; the model is told this.
    Denied {
        /// What the model is told.
        reason: String,
        /// The host's answer when the host was asked and refused.
        asked: Option<Decision>,
    },
}

impl Access {
    fn allowed() -> Self {
        Access::Allowed {
            asked: None,
            added_rule: None,
        }
    }
    fn denied(reason: impl Into<String>) -> Self {
        Access::Denied {
            reason: reason.into(),
            asked: None,
        }
    }
}

/// A [`Policy`] plus the host that answers its asks. With no host an ask is a refusal, so a host that forgets to set
/// one fails closed.
pub struct Gate {
    policy: Policy,
    host: Option<Arc<dyn PermissionHost>>,
}

impl Gate {
    /// A gate over `policy`, asking `host` (if any).
    pub fn new(policy: Policy, host: Option<Arc<dyn PermissionHost>>) -> Self {
        Gate { policy, host }
    }

    /// The policy.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// The policy, to change its mode or add rules mid-session.
    pub fn policy_mut(&mut self) -> &mut Policy {
        &mut self.policy
    }

    /// Decides one call. An "allow always" answer adds a session-scope allow for this tool and subject only (deny
    /// rules still win next time). With no subject the rule covers every later call of the tool, subject or not; a tool
    /// that can report one should.
    pub async fn check(
        &mut self,
        call: &CallId,
        spec: &ToolSpec,
        input: &Value,
        subject: Option<String>,
    ) -> Access {
        match self.policy.evaluate(spec, subject.as_deref()) {
            Verdict::Allow => Access::allowed(),
            Verdict::Deny(why) => Access::denied(why),
            Verdict::Ask => {
                let Some(host) = self.host.clone() else {
                    return Access::denied(format!(
                        "`{}` needs permission and there is no one to ask",
                        spec.name
                    ));
                };
                let req = AskRequest {
                    call: call.clone(),
                    tool: spec.name.clone(),
                    input: input.clone(),
                    subject: subject.clone(),
                };
                let answer = host.ask(&req).await;
                match &answer {
                    Decision::AllowOnce => Access::Allowed {
                        asked: Some(answer),
                        added_rule: None,
                    },
                    Decision::AllowAlways => {
                        let rule = Rule::exact(Effect::Allow, &spec.name, subject.as_deref());
                        let added_rule = Some(rule.as_str().to_owned());
                        self.policy.add(Scope::Session, rule);
                        Access::Allowed {
                            asked: Some(answer),
                            added_rule,
                        }
                    }
                    Decision::Deny { reason } => Access::Denied {
                        reason: reason.to_string(),
                        asked: Some(answer.clone()),
                    },
                    // A decision this version does not know is not a yes.
                    _ => Access::Denied {
                        reason: "the host's answer was not understood".to_owned(),
                        asked: Some(answer),
                    },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Mode;
    use crate::testutil::{YieldOnce, block_on};
    use serde_json::json;
    use std::sync::Mutex;

    struct Host {
        answer: Decision,
        seen: Mutex<Vec<AskRequest>>,
    }
    impl PermissionHost for Host {
        fn ask<'a>(&'a self, request: &'a AskRequest) -> BoxFuture<'a, Decision> {
            self.seen.lock().unwrap().push(request.clone());
            let a = self.answer.clone();
            Box::pin(async move {
                YieldOnce(false).await;
                a
            })
        }
    }

    fn host(answer: Decision) -> Arc<Host> {
        Arc::new(Host {
            answer,
            seen: Mutex::new(Vec::new()),
        })
    }
    fn spec() -> ToolSpec {
        ToolSpec::new("bash", "d", json!({}))
    }
    fn id() -> CallId {
        "c1".into()
    }

    #[test]
    fn an_ask_waits_for_the_host() {
        let h = host(Decision::AllowOnce);
        let mut g = Gate::new(Policy::new(Mode::Default), Some(h.clone()));
        let r = block_on(g.check(&id(), &spec(), &json!({"cmd":"ls"}), Some("ls".into())));
        assert!(matches!(
            r,
            Access::Allowed {
                asked: Some(Decision::AllowOnce),
                ..
            }
        ));
        let seen = h.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].subject.as_deref(), Some("ls"));
    }

    #[test]
    fn allow_always_is_remembered_but_deny_rules_still_win() {
        let h = host(Decision::AllowAlways);
        let mut g = Gate::new(Policy::default(), Some(h.clone()));
        let first = block_on(g.check(&id(), &spec(), &json!({}), Some("ls".into())));
        assert!(matches!(
            &first,
            Access::Allowed { added_rule: Some(r), .. } if r == "bash(ls)"
        ));
        assert!(matches!(
            block_on(g.check(&id(), &spec(), &json!({}), Some("ls".into()))),
            Access::Allowed { asked: None, .. }
        ));
        assert_eq!(
            h.seen.lock().unwrap().len(),
            1,
            "the second call is not asked"
        );
        block_on(g.check(&id(), &spec(), &json!({}), Some("rm x".into())));
        assert_eq!(
            h.seen.lock().unwrap().len(),
            2,
            "another subject is asked again"
        );
        g.policy_mut().add(
            Scope::Organisation,
            Rule::parse(Effect::Deny, "bash").unwrap(),
        );
        assert!(matches!(
            block_on(g.check(&id(), &spec(), &json!({}), Some("ls".into()))),
            Access::Denied { .. }
        ));
    }

    #[test]
    fn a_denial_carries_the_hosts_reason_and_no_host_fails_closed() {
        let h = host(Decision::Deny {
            reason: "outside the project".into(),
        });
        let mut g = Gate::new(Policy::default(), Some(h));
        let Access::Denied { reason, asked } = block_on(g.check(&id(), &spec(), &json!({}), None))
        else {
            panic!("the host said no");
        };
        assert_eq!(reason, "outside the project");
        assert!(matches!(asked, Some(Decision::Deny { .. })));
        let mut g = Gate::new(Policy::default(), None);
        let Access::Denied { reason: why, .. } =
            block_on(g.check(&id(), &spec(), &json!({}), None))
        else {
            panic!("no host, no yes");
        };
        assert!(why.contains("no one to ask"), "{why}");
    }
}
