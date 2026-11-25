// SPDX-License-Identifier: Apache-2.0
//! The completion gate: the model states criteria up front, and a run may finish only when every criterion has
//! evidence gathered after the last edit.
//!
//! The gate is a pure value. The driver calls [`Gate::record_edit`] after every tool call that changed the
//! project, [`Gate::record_evidence`] when a host tool returns evidence, and [`Gate::check`] when the model asks to
//! complete. The host decides which evidence kinds exist (a design host: a render taken with `look`; a data host:
//! a cross-check query) and passes them in as [`EvidenceKinds`]; the gate knows nothing else about them.

use std::collections::BTreeMap;
use std::fmt;

/// The evidence kinds a host can supply, by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceKinds(Vec<Box<str>>);

impl EvidenceKinds {
    /// The kinds a host offers; duplicates are dropped, order is kept.
    pub fn new<I, S>(kinds: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<Box<str>>,
    {
        let mut out: Vec<Box<str>> = Vec::new();
        for kind in kinds {
            let kind = kind.into();
            if !out.contains(&kind) {
                out.push(kind);
            }
        }
        Self(out)
    }

    /// Whether the host offers `kind`.
    #[must_use]
    pub fn contains(&self, kind: &str) -> bool {
        self.0.iter().any(|k| &**k == kind)
    }

    /// The kinds, for naming them to the model.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|k| &**k)
    }
}

/// One thing the model says must hold before it finishes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Criterion {
    /// A short id the model uses when it later cites evidence.
    pub id: Box<str>,
    /// What must hold, in the model's words.
    pub text: Box<str>,
    /// The kind of evidence that can show it.
    pub kind: Box<str>,
}

/// Why a plan was not accepted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PlanError {
    /// The plan has no criteria.
    #[error("a plan needs at least one criterion")]
    Empty,
    /// Two criteria share an id.
    #[error("criterion id `{0}` is used twice")]
    DuplicateId(Box<str>),
    /// A criterion has an empty id or text.
    #[error("a criterion needs a non-empty id, text and evidence kind")]
    Blank,
    /// A criterion names an evidence kind the host does not offer.
    #[error(
        "criterion `{id}` needs evidence of kind `{kind}`, which this host does not offer (offered: {offered})"
    )]
    UnknownKind {
        /// The criterion.
        id: Box<str>,
        /// The kind it named.
        kind: Box<str>,
        /// The offered kinds, comma separated.
        offered: Box<str>,
    },
}

/// Why a piece of evidence was not recorded.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EvidenceError {
    /// The id is not in the plan.
    #[error("`{0}` is not a criterion in the plan")]
    UnknownCriterion(Box<str>),
    /// The evidence is not of the kind the criterion asked for.
    #[error("criterion `{id}` needs `{expected}` evidence, not `{got}`")]
    WrongKind {
        /// The criterion.
        id: Box<str>,
        /// The kind it asked for.
        expected: Box<str>,
        /// The kind that was supplied.
        got: Box<str>,
    },
}

/// Why one criterion is not met.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Gap {
    /// No evidence was recorded.
    Missing,
    /// Evidence exists but was gathered before the last edit.
    Stale,
}

/// A criterion that blocks completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unmet {
    /// The criterion's id.
    pub id: Box<str>,
    /// The criterion's text.
    pub text: Box<str>,
    /// The kind of evidence it needs.
    pub kind: Box<str>,
    /// Why it is not met.
    pub gap: Gap,
}

/// The gate's answer to a request to complete.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Verdict {
    /// Every criterion has evidence from after the last edit.
    Accepted,
    /// The model never stated a plan.
    NoPlan,
    /// These criteria are not met.
    Refused(Vec<Unmet>),
}

impl Verdict {
    /// Whether the run may finish.
    #[must_use]
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }
}

impl fmt::Display for Verdict {
    /// The text the model is shown when completion is refused (and a short note when accepted).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accepted => f.write_str("Completion accepted."),
            Self::NoPlan => f.write_str(
                "Completion refused: state a plan first, with the criteria that must hold and the evidence \
                 that shows each.",
            ),
            Self::Refused(unmet) => {
                f.write_str("Completion refused. These criteria have no evidence from after your last edit:")?;
                for u in unmet {
                    let why = match u.gap {
                        Gap::Missing => "no evidence yet",
                        Gap::Stale => "evidence is older than your last edit",
                    };
                    write!(f, "\n- {} ({}): needs `{}`; {why}", u.id, u.text, u.kind)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Evidence {
    /// The edit count when it was gathered.
    at: u64,
    summary: Box<str>,
}

/// The completion gate for one run.
#[derive(Debug, Clone)]
pub struct Gate {
    kinds: EvidenceKinds,
    criteria: Vec<Criterion>,
    evidence: BTreeMap<Box<str>, Evidence>,
    /// How many edits have happened; evidence is fresh when it was gathered at this count.
    edits: u64,
}

impl Gate {
    /// A gate with no plan, for a host offering `kinds`.
    #[must_use]
    pub fn new(kinds: EvidenceKinds) -> Self {
        Self {
            kinds,
            criteria: Vec::new(),
            evidence: BTreeMap::new(),
            edits: 0,
        }
    }

    /// State (or restate) the plan. A restated plan replaces the old criteria and drops all evidence.
    ///
    /// # Errors
    ///
    /// [`PlanError`] when the plan is empty, has blank or repeated ids, or names an evidence kind the host does not
    /// offer. The previous plan is kept on error.
    pub fn set_plan(&mut self, criteria: Vec<Criterion>) -> Result<(), PlanError> {
        if criteria.is_empty() {
            return Err(PlanError::Empty);
        }
        for (i, c) in criteria.iter().enumerate() {
            if c.id.trim().is_empty() || c.text.trim().is_empty() || c.kind.trim().is_empty() {
                return Err(PlanError::Blank);
            }
            if criteria[..i].iter().any(|p| p.id.trim() == c.id.trim()) {
                return Err(PlanError::DuplicateId(c.id.clone()));
            }
            if !self.kinds.contains(&c.kind) {
                return Err(PlanError::UnknownKind {
                    id: c.id.clone(),
                    kind: c.kind.clone(),
                    offered: self.kinds.iter().collect::<Vec<_>>().join(", ").into(),
                });
            }
        }
        self.criteria = criteria
            .into_iter()
            .map(|mut c| {
                c.id = c.id.trim().into();
                c
            })
            .collect();
        self.evidence.clear();
        Ok(())
    }

    /// A tool call changed the project: all evidence gathered so far is now stale.
    pub fn record_edit(&mut self) {
        self.edits += 1;
    }

    /// Evidence of `kind` for criterion `id`, gathered now (after every edit recorded so far).
    ///
    /// Only a successful host result is evidence: the driver must not call this for a failed or denied tool result.
    ///
    /// # Errors
    ///
    /// [`EvidenceError`] when `id` is not in the plan or `kind` is not the kind the criterion asked for; nothing is
    /// recorded, so the driver can tell the model.
    pub fn record_evidence(
        &mut self,
        id: &str,
        kind: &str,
        summary: impl Into<Box<str>>,
    ) -> Result<(), EvidenceError> {
        let Some(c) = self.criteria.iter().find(|c| &*c.id == id.trim()) else {
            return Err(EvidenceError::UnknownCriterion(id.into()));
        };
        if &*c.kind != kind {
            return Err(EvidenceError::WrongKind {
                id: c.id.clone(),
                expected: c.kind.clone(),
                got: kind.into(),
            });
        }
        self.evidence.insert(
            c.id.clone(),
            Evidence {
                at: self.edits,
                summary: summary.into(),
            },
        );
        Ok(())
    }

    /// Evidence of `kind` for every criterion that asked for that kind, gathered now. For a host tool that does not
    /// know which criterion it serves (a `look` shows the whole page): the model's plan says what each criterion
    /// needs, and one fresh result of the kind serves them all. Returns how many criteria it served (zero when the
    /// plan has none of that kind; nothing is recorded then).
    pub fn record_evidence_of_kind(&mut self, kind: &str, summary: &str) -> usize {
        let mut served = 0;
        for c in self.criteria.iter().filter(|c| &*c.kind == kind) {
            self.evidence.insert(
                c.id.clone(),
                Evidence {
                    at: self.edits,
                    summary: summary.into(),
                },
            );
            served += 1;
        }
        served
    }

    /// The summary of the evidence held for `id`, fresh or not.
    #[must_use]
    pub fn evidence_for(&self, id: &str) -> Option<&str> {
        self.evidence.get(id).map(|e| &*e.summary)
    }

    /// Whether the model asked to complete is allowed to.
    #[must_use]
    pub fn check(&self) -> Verdict {
        if self.criteria.is_empty() {
            return Verdict::NoPlan;
        }
        let unmet: Vec<Unmet> = self
            .criteria
            .iter()
            .filter_map(|c| {
                let gap = match self.evidence.get(&c.id) {
                    None => Gap::Missing,
                    Some(e) if e.at < self.edits => Gap::Stale,
                    Some(_) => return None,
                };
                Some(Unmet {
                    id: c.id.clone(),
                    text: c.text.clone(),
                    kind: c.kind.clone(),
                    gap,
                })
            })
            .collect();
        if unmet.is_empty() {
            Verdict::Accepted
        } else {
            Verdict::Refused(unmet)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crit(id: &str, kind: &str) -> Criterion {
        Criterion {
            id: id.into(),
            text: format!("{id} holds").into(),
            kind: kind.into(),
        }
    }

    fn gate() -> Gate {
        let mut g = Gate::new(EvidenceKinds::new(["look", "query"]));
        g.set_plan(vec![crit("hero", "look"), crit("count", "query")])
            .unwrap();
        g
    }

    #[test]
    fn a_plan_id_with_surrounding_whitespace_is_stored_trimmed() {
        let mut g = Gate::new(EvidenceKinds::new(["look"]));
        g.set_plan(vec![crit("a ", "look")]).unwrap();
        g.record_evidence("a", "look", "seen").unwrap();
        g.record_evidence(" a ", "look", "seen again").unwrap();
        assert!(g.check().is_accepted());
    }

    #[test]
    fn evidence_of_a_kind_serves_every_criterion_asking_for_it_and_none_else() {
        let mut g = gate();
        assert_eq!(g.record_evidence_of_kind("look", "seen"), 1);
        assert_eq!(g.record_evidence_of_kind("render", "x"), 0);
        assert_eq!(g.evidence_for("hero"), Some("seen"));
        assert_eq!(g.evidence_for("count"), None);
        g.record_edit();
        assert!(matches!(g.check(), Verdict::Refused(u) if u.len() == 2));
    }

    #[test]
    fn no_plan_refuses() {
        let g = Gate::new(EvidenceKinds::new(["look"]));
        assert_eq!(g.check(), Verdict::NoPlan);
        assert!(g.check().to_string().contains("state a plan"));
    }

    #[test]
    fn a_run_without_evidence_is_refused_and_names_every_criterion() {
        let g = gate();
        let Verdict::Refused(unmet) = g.check() else {
            panic!("expected a refusal")
        };
        assert_eq!(unmet.len(), 2);
        assert!(unmet.iter().all(|u| u.gap == Gap::Missing));
    }

    #[test]
    fn evidence_after_the_last_edit_passes() {
        let mut g = gate();
        g.record_edit();
        assert!(
            g.record_evidence("hero", "look", "render shows the hero")
                .is_ok()
        );
        assert!(
            g.record_evidence("count", "query", "42 rows both ways")
                .is_ok()
        );
        assert!(g.check().is_accepted());
        assert_eq!(g.evidence_for("hero"), Some("render shows the hero"));
    }

    #[test]
    fn an_edit_after_the_evidence_makes_it_stale() {
        let mut g = gate();
        g.record_evidence("hero", "look", "ok").unwrap();
        g.record_evidence("count", "query", "ok").unwrap();
        assert!(g.check().is_accepted(), "no edit yet: evidence is fresh");
        g.record_edit();
        let Verdict::Refused(unmet) = g.check() else {
            panic!("expected a refusal")
        };
        assert!(unmet.iter().all(|u| u.gap == Gap::Stale));
        assert!(g.check().to_string().contains("older than your last edit"));
        g.record_evidence("hero", "look", "ok again").unwrap();
        let Verdict::Refused(unmet) = g.check() else {
            panic!("expected a refusal")
        };
        assert_eq!(unmet.len(), 1, "only the criterion still stale is named");
        assert_eq!(&*unmet[0].id, "count");
    }

    #[test]
    fn evidence_for_an_unknown_criterion_is_not_recorded() {
        let mut g = gate();
        assert_eq!(
            g.record_evidence("nope", "look", "x"),
            Err(EvidenceError::UnknownCriterion("nope".into()))
        );
        assert_eq!(g.evidence_for("nope"), None);
    }

    #[test]
    fn plans_are_validated_and_a_bad_plan_keeps_the_old_one() {
        let mut g = gate();
        g.record_evidence("hero", "look", "ok").unwrap();
        assert_eq!(g.set_plan(vec![]), Err(PlanError::Empty));
        assert_eq!(
            g.set_plan(vec![crit("a", "look"), crit("a", "look")]),
            Err(PlanError::DuplicateId("a".into()))
        );
        assert_eq!(g.set_plan(vec![crit(" ", "look")]), Err(PlanError::Blank));
        let err = g.set_plan(vec![crit("a", "screenshot")]).unwrap_err();
        assert!(err.to_string().contains("offered: look, query"), "{err}");
        assert_eq!(
            g.evidence_for("hero"),
            Some("ok"),
            "old plan and evidence kept"
        );
    }

    #[test]
    fn restating_the_plan_drops_the_evidence() {
        let mut g = gate();
        g.record_evidence("hero", "look", "ok").unwrap();
        g.set_plan(vec![crit("hero", "look")]).unwrap();
        assert_eq!(g.evidence_for("hero"), None);
        assert!(!g.check().is_accepted());
    }

    #[test]
    fn duplicate_kinds_are_dropped() {
        let k = EvidenceKinds::new(["look", "look", "query"]);
        assert_eq!(k.iter().collect::<Vec<_>>(), ["look", "query"]);
    }

    #[test]
    fn evidence_of_the_wrong_kind_is_refused_and_not_recorded() {
        let mut g = gate();
        let err = g
            .record_evidence("hero", "query", "rows match")
            .unwrap_err();
        assert_eq!(
            err,
            EvidenceError::WrongKind {
                id: "hero".into(),
                expected: "look".into(),
                got: "query".into()
            }
        );
        assert!(err.to_string().contains("needs `look` evidence"));
        assert_eq!(g.evidence_for("hero"), None);
    }

    #[test]
    fn evidence_edit_evidence_for_one_criterion_ends_fresh() {
        let mut g = gate();
        g.record_evidence("hero", "look", "first").unwrap();
        g.record_edit();
        g.record_evidence("hero", "look", "second").unwrap();
        g.record_evidence("count", "query", "fresh").unwrap();
        assert!(g.check().is_accepted());
        assert_eq!(g.evidence_for("hero"), Some("second"));
    }

    #[test]
    fn ids_that_differ_only_by_whitespace_are_duplicates_and_a_blank_kind_is_refused() {
        let mut g = Gate::new(EvidenceKinds::new(["look"]));
        assert_eq!(
            g.set_plan(vec![crit("a", "look"), crit("a ", "look")]),
            Err(PlanError::DuplicateId("a ".into()))
        );
        assert_eq!(g.set_plan(vec![crit("a", " ")]), Err(PlanError::Blank));
    }

    /// The scripted run the slice asks for: finishing without post-edit evidence is refused, and supplying it passes.
    #[test]
    fn a_scripted_run_is_refused_until_it_has_evidence_after_its_last_edit() {
        let mut g = Gate::new(EvidenceKinds::new(["look"]));
        assert_eq!(g.check(), Verdict::NoPlan);
        g.set_plan(vec![crit("hero", "look")]).unwrap();
        g.record_edit();
        assert!(!g.check().is_accepted(), "tries to finish after an edit");
        g.record_evidence("hero", "look", "render ok").unwrap();
        g.record_edit();
        assert!(!g.check().is_accepted(), "edited again after the look");
        g.record_evidence("hero", "look", "render ok again")
            .unwrap();
        assert!(g.check().is_accepted());
    }
}
