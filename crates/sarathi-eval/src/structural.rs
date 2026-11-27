// SPDX-License-Identifier: Apache-2.0
//! The structural acceptance check ([`Structural`]): the proposal changed something through a valid operation, the
//! host's validator reports no error on the result, every required section is still there, and every text colour pair
//! has enough contrast.
//!
//! The check knows nothing about a host's page format. A harness reports those facts as a [`Proposal`] and puts its
//! JSON in [`HarnessOutput::artifact`]; both harnesses of a comparison must fill it with the same host code, so the
//! rule falls on both alike.

use serde::{Deserialize, Serialize};

use crate::brief::Brief;
use crate::runner::{Acceptance, HarnessOutput};

/// The [`Proposal`] format version this crate writes.
pub const PROPOSAL_VERSION: u32 = 1;

/// The name a brief's `acceptance` field gives the structural check.
pub const STRUCTURAL: &str = "structural";

/// The WCAG AA contrast ratio for normal-size text.
pub const AA_CONTRAST: f64 = 4.5;

/// What a host says about the result of one run, as the harness's artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    /// Format version, [`PROPOSAL_VERSION`].
    pub proposal: u32,
    /// What the host's validator finds wrong with the final state; empty when it is valid.
    pub errors: Vec<String>,
    /// The sections the result must keep (for a page: those it had before the run).
    pub required_sections: Vec<String>,
    /// The sections the result has.
    pub sections: Vec<String>,
    /// The contrast of each text colour pair of the result.
    pub contrast: Vec<Contrast>,
}

/// The contrast of one text colour on its background.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contrast {
    /// Which pair, for the reason a run was refused (`text on background`).
    pub pair: String,
    /// The WCAG contrast ratio, from 1 to 21.
    pub ratio: f64,
}

impl Proposal {
    /// The artifact text: this proposal as JSON.
    ///
    /// # Errors
    /// Serialisation failure (not expected for these types; a ratio that is not finite is written as `null`, which the
    /// check refuses to read).
    pub fn to_artifact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Why the structural check refused a run. The first rule broken, in the order of the variants.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Rejection {
    /// The artifact is not a proposal this crate reads.
    #[error("the artifact is not a proposal: {0}")]
    Unreadable(String),
    /// No operation of the run was valid, so nothing was proposed.
    #[error("no valid operation: nothing was proposed")]
    NoValidOp,
    /// The host's validator refuses the result.
    #[error("the result does not validate: {0}")]
    Invalid(String),
    /// The host named no section to keep, so the section rule cannot be checked.
    #[error("no required section was named")]
    NoRequiredSections,
    /// A required section is gone.
    #[error("required section `{0}` is missing")]
    MissingSection(String),
    /// The host measured no colour pair, so the contrast rule cannot be checked.
    #[error("no contrast was measured")]
    NoContrast,
    /// A colour pair is below the minimum, or its ratio is not a contrast ratio.
    #[error("contrast of {pair} is {ratio}, below the minimum or not a ratio from 1 to 21")]
    Contrast {
        /// The pair.
        pair: String,
        /// Its reported ratio.
        ratio: f64,
    },
}

/// The structural acceptance check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Structural {
    /// The least contrast every pair must have.
    pub min_contrast: f64,
}

impl Default for Structural {
    fn default() -> Self {
        Self {
            min_contrast: AA_CONTRAST,
        }
    }
}

impl Structural {
    /// Whether `out` passes, and if not, why.
    ///
    /// # Errors
    /// The first [`Rejection`] that applies.
    pub fn check(&self, out: &HarnessOutput) -> Result<(), Rejection> {
        let p: Proposal = serde_json::from_str(&out.artifact)
            .map_err(|e| Rejection::Unreadable(e.to_string()))?;
        if p.proposal != PROPOSAL_VERSION {
            return Err(Rejection::Unreadable(format!(
                "version {}, this crate reads {PROPOSAL_VERSION}",
                p.proposal
            )));
        }
        if out.valid_ops == 0 {
            return Err(Rejection::NoValidOp);
        }
        if let Some(e) = p.errors.first() {
            return Err(Rejection::Invalid(e.clone()));
        }
        if p.required_sections.is_empty() {
            return Err(Rejection::NoRequiredSections);
        }
        if let Some(s) = p.required_sections.iter().find(|s| !p.sections.contains(s)) {
            return Err(Rejection::MissingSection(s.clone()));
        }
        if p.contrast.is_empty() {
            return Err(Rejection::NoContrast);
        }
        // A range check, so a NaN fails it too.
        if let Some(c) = p
            .contrast
            .iter()
            .find(|c| !(self.min_contrast..=21.0).contains(&c.ratio) || c.ratio < 1.0)
        {
            return Err(Rejection::Contrast {
                pair: c.pair.clone(),
                ratio: c.ratio,
            });
        }
        Ok(())
    }
}

impl Acceptance for Structural {
    fn handles(&self, check: &str) -> bool {
        check == STRUCTURAL
    }

    fn accepts(&self, _brief: &Brief, out: &HarnessOutput) -> bool {
        self.check(out).is_ok()
    }
}
