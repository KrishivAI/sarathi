// SPDX-License-Identifier: Apache-2.0
//! The structural acceptance check: each rule refuses on its own, and says which.

use sarathi_eval::runner::HarnessOutput;
use sarathi_eval::structural::{Contrast, Proposal, Rejection, Structural};

fn good() -> Proposal {
    Proposal {
        proposal: 1,
        errors: Vec::new(),
        required_sections: vec!["header".into(), "footer".into()],
        sections: vec!["header".into(), "hero".into(), "footer".into()],
        contrast: vec![
            Contrast {
                pair: "text on background".into(),
                ratio: 4.5,
            },
            Contrast {
                pair: "link on background".into(),
                ratio: 21.0,
            },
        ],
    }
}

fn output(artifact: String, valid_ops: u32) -> HarnessOutput {
    HarnessOutput {
        artifact,
        valid_ops,
        total_ops: 2,
        ..HarnessOutput::default()
    }
}

fn check(edit: impl FnOnce(&mut Proposal)) -> Result<(), Rejection> {
    let mut p = good();
    edit(&mut p);
    Structural::default().check(&output(p.to_artifact().unwrap_or_default(), 1))
}

#[test]
fn a_sound_proposal_is_accepted() {
    assert_eq!(check(|_| {}), Ok(()));
}

#[test]
fn a_run_with_no_valid_operation_is_refused() {
    let a = good().to_artifact().unwrap();
    assert_eq!(
        Structural::default().check(&output(a, 0)),
        Err(Rejection::NoValidOp)
    );
}

#[test]
fn each_rule_refuses_and_names_what_broke_it() {
    assert_eq!(
        check(|p| p.errors = vec!["unknown token".into(), "second".into()]),
        Err(Rejection::Invalid("unknown token".into()))
    );
    assert_eq!(
        check(|p| p.required_sections.clear()),
        Err(Rejection::NoRequiredSections)
    );
    assert_eq!(
        check(|p| p.sections.retain(|s| s != "footer")),
        Err(Rejection::MissingSection("footer".into()))
    );
    assert_eq!(check(|p| p.contrast.clear()), Err(Rejection::NoContrast));
}

#[test]
fn every_pair_needs_the_minimum_contrast_and_a_real_ratio() {
    for ratio in [4.49, 0.0, 21.1, -5.0] {
        assert_eq!(
            check(|p| p.contrast[1].ratio = ratio),
            Err(Rejection::Contrast {
                pair: "link on background".into(),
                ratio,
            }),
            "{ratio}"
        );
    }
    // A minimum below 1 does not let a number that is no contrast ratio through.
    let mut p = good();
    p.contrast[0].ratio = 0.5;
    let lax = Structural { min_contrast: 0.0 };
    assert!(matches!(
        lax.check(&output(p.to_artifact().unwrap(), 1)),
        Err(Rejection::Contrast { .. })
    ));
    // A stricter minimum is applied.
    let strict = Structural { min_contrast: 7.0 };
    assert!(matches!(
        strict.check(&output(good().to_artifact().unwrap(), 1)),
        Err(Rejection::Contrast { .. })
    ));
}

#[test]
fn an_artifact_that_is_not_a_proposal_is_refused() {
    let mut other_version = good();
    other_version.proposal = 2;
    let mut not_finite = good();
    not_finite.contrast[0].ratio = f64::NAN;
    let mut unknown: serde_json::Value =
        serde_json::from_str(&good().to_artifact().unwrap()).unwrap();
    unknown["extra"] = serde_json::json!(1);
    for artifact in [
        "a theme".to_owned(),
        "{}".to_owned(),
        unknown.to_string(),
        other_version.to_artifact().unwrap(),
        not_finite.to_artifact().unwrap(),
    ] {
        let r = Structural::default().check(&output(artifact.clone(), 1));
        assert!(matches!(r, Err(Rejection::Unreadable(_))), "{artifact}");
    }
}
