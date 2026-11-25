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
    /// Run it.
    Allowed,
    /// Do not run it; the model is told this.
    Denied(String),
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

    /// Decides one call. An "allow always" answer adds a session-scope allow for the tool (deny rules still win next
    /// time).
    pub async fn check(
        &mut self,
        call: &CallId,
        spec: &ToolSpec,
        input: &Value,
        subject: Option<String>,
    ) -> Access {
        match self.policy.evaluate(spec, subject.as_deref()) {
            Verdict::Allow => Access::Allowed,
            Verdict::Deny(why) => Access::Denied(why),
            Verdict::Ask => {
                let Some(host) = self.host.clone() else {
                    return Access::Denied(format!(
                        "`{}` needs permission and there is no one to ask",
                        spec.name
                    ));
                };
                let req = AskRequest {
                    call: call.clone(),
                    tool: spec.name.clone(),
                    input: input.clone(),
                    subject,
                };
                match host.ask(&req).await {
                    Decision::AllowOnce => Access::Allowed,
                    Decision::AllowAlways => {
                        if let Ok(rule) = Rule::parse(Effect::Allow, &spec.name) {
                            self.policy.add(Scope::Session, rule);
                        }
                        Access::Allowed
                    }
                    Decision::Deny { reason } => Access::Denied(reason.into()),
                    // A decision this version does not know is not a yes.
                    _ => Access::Denied("the host's answer was not understood".to_owned()),
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
        assert_eq!(r, Access::Allowed);
        let seen = h.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].subject.as_deref(), Some("ls"));
    }

    #[test]
    fn allow_always_is_remembered_but_deny_rules_still_win() {
        let h = host(Decision::AllowAlways);
        let mut g = Gate::new(Policy::default(), Some(h.clone()));
        assert_eq!(
            block_on(g.check(&id(), &spec(), &json!({}), None)),
            Access::Allowed
        );
        assert_eq!(
            block_on(g.check(&id(), &spec(), &json!({}), None)),
            Access::Allowed
        );
        assert_eq!(
            h.seen.lock().unwrap().len(),
            1,
            "the second call is not asked"
        );
        g.policy_mut().add(
            Scope::Organisation,
            Rule::parse(Effect::Deny, "bash").unwrap(),
        );
        assert!(matches!(
            block_on(g.check(&id(), &spec(), &json!({}), None)),
            Access::Denied(_)
        ));
    }

    #[test]
    fn a_denial_carries_the_hosts_reason_and_no_host_fails_closed() {
        let h = host(Decision::Deny {
            reason: "outside the project".into(),
        });
        let mut g = Gate::new(Policy::default(), Some(h));
        assert_eq!(
            block_on(g.check(&id(), &spec(), &json!({}), None)),
            Access::Denied("outside the project".into())
        );
        let mut g = Gate::new(Policy::default(), None);
        let Access::Denied(why) = block_on(g.check(&id(), &spec(), &json!({}), None)) else {
            panic!("no host, no yes");
        };
        assert!(why.contains("no one to ask"), "{why}");
    }
}
