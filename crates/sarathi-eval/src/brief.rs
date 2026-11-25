// SPDX-License-Identifier: Apache-2.0
//! The brief file format (the brief-set protocol). One JSONL file per set: a header line, then one brief per line. Unknown fields
//! are refused, so a typo in a brief is an error and not a silently ignored setting.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// The brief-set format version this crate writes. It reads this version and the one before.
pub const BRIEF_SET_VERSION: u32 = 1;

/// Which part of the set a brief belongs to. Only `Tuning` briefs may drive a prompt or profile change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    /// Developed against.
    Tuning,
    /// Never used to tune; decides the kill line.
    Measurement,
}

/// One brief.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Brief {
    /// Unique id within the set.
    pub id: String,
    /// Tuning or measurement.
    pub split: Split,
    /// The site context the brief runs against.
    pub site_context: String,
    /// The brief text.
    pub text: String,
    /// Model ids this brief may run on; at least one.
    pub allowed_models: Vec<String>,
    /// Name of the acceptance check that judges the run.
    pub acceptance: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    sarathi_brief_set: u32,
}

/// A brief file could not be read.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BriefError {
    /// No header line.
    #[error("brief set is empty: the first line must be the header")]
    Empty,
    /// The first line is not a header.
    #[error("line 1 is not a brief-set header: {0}")]
    BadHeader(String),
    /// The file is from a newer or unknown version.
    #[error(
        "brief set version {0} is not supported (this reader handles up to {BRIEF_SET_VERSION})"
    )]
    Version(u32),
    /// A brief line does not parse.
    #[error("line {line}: {message}")]
    BadBrief {
        /// 1-based line number.
        line: usize,
        /// What was wrong.
        message: String,
    },
    /// Two briefs share an id.
    #[error("line {line}: duplicate brief id `{id}`")]
    Duplicate {
        /// 1-based line number of the second one.
        line: usize,
        /// The id.
        id: String,
    },
}

impl Brief {
    fn check(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("empty id".into());
        }
        if self.text.trim().is_empty() {
            return Err(format!("brief `{}` has no text", self.id));
        }
        if self.allowed_models.is_empty() {
            return Err(format!("brief `{}` allows no model", self.id));
        }
        if self.acceptance.trim().is_empty() {
            return Err(format!("brief `{}` names no acceptance check", self.id));
        }
        Ok(())
    }
}

/// Parse a brief set.
///
/// # Errors
/// [`BriefError`] naming the line.
pub fn parse_brief_set(text: &str) -> Result<Vec<Brief>, BriefError> {
    let mut lines = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty());
    let (_, first) = lines.next().ok_or(BriefError::Empty)?;
    let header: Header =
        serde_json::from_str(first).map_err(|e| BriefError::BadHeader(e.to_string()))?;
    if header.sarathi_brief_set == 0 || header.sarathi_brief_set > BRIEF_SET_VERSION {
        return Err(BriefError::Version(header.sarathi_brief_set));
    }
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (i, l) in lines {
        let line = i + 1;
        let b: Brief = serde_json::from_str(l).map_err(|e| BriefError::BadBrief {
            line,
            message: e.to_string(),
        })?;
        b.check()
            .map_err(|message| BriefError::BadBrief { line, message })?;
        if !seen.insert(b.id.clone()) {
            return Err(BriefError::Duplicate { line, id: b.id });
        }
        out.push(b);
    }
    Ok(out)
}

/// Write a brief set (header, then one line per brief).
///
/// # Errors
/// A brief that fails its own checks, or does not serialise.
pub fn write_brief_set(briefs: &[Brief]) -> Result<String, BriefError> {
    let mut out = format!("{{\"sarathi_brief_set\":{BRIEF_SET_VERSION}}}\n");
    for (i, b) in briefs.iter().enumerate() {
        let line = i + 2;
        b.check()
            .map_err(|message| BriefError::BadBrief { line, message })?;
        let s = serde_json::to_string(b).map_err(|e| BriefError::BadBrief {
            line,
            message: e.to_string(),
        })?;
        out.push_str(&s);
        out.push('\n');
    }
    Ok(out)
}

/// Briefs of one split, in file order.
#[must_use]
pub fn split_of(briefs: &[Brief], split: Split) -> Vec<&Brief> {
    briefs.iter().filter(|b| b.split == split).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brief(id: &str, split: Split) -> Brief {
        Brief {
            id: id.into(),
            split,
            site_context: "a bakery site".into(),
            text: "warm palette".into(),
            allowed_models: vec!["m".into()],
            acceptance: "structural".into(),
        }
    }

    #[test]
    fn round_trips_and_splits() {
        let set = [brief("a", Split::Tuning), brief("b", Split::Measurement)];
        let text = write_brief_set(&set).unwrap();
        assert_eq!(parse_brief_set(&text).unwrap(), set);
        assert_eq!(split_of(&set, Split::Measurement).len(), 1);
    }

    #[test]
    fn duplicate_id_names_the_second_line() {
        let text =
            write_brief_set(&[brief("a", Split::Tuning), brief("a", Split::Tuning)]).unwrap();
        assert_eq!(
            parse_brief_set(&text),
            Err(BriefError::Duplicate {
                line: 3,
                id: "a".into()
            })
        );
    }

    #[test]
    fn unknown_field_and_bad_split_are_refused() {
        let h = "{\"sarathi_brief_set\":1}\n";
        let extra = format!(
            "{h}{{\"id\":\"a\",\"split\":\"tuning\",\"site_context\":\"\",\"text\":\"t\",\"allowed_models\":[\"m\"],\"acceptance\":\"s\",\"tempo\":1}}"
        );
        assert!(matches!(
            parse_brief_set(&extra),
            Err(BriefError::BadBrief { line: 2, .. })
        ));
        let split = format!(
            "{h}{{\"id\":\"a\",\"split\":\"holdout\",\"site_context\":\"\",\"text\":\"t\",\"allowed_models\":[\"m\"],\"acceptance\":\"s\"}}"
        );
        assert!(matches!(
            parse_brief_set(&split),
            Err(BriefError::BadBrief { line: 2, .. })
        ));
    }

    #[test]
    fn invalid_briefs_are_refused() {
        let mut b = brief("a", Split::Tuning);
        b.allowed_models.clear();
        assert!(write_brief_set(&[b]).is_err());
        let mut b = brief(" ", Split::Tuning);
        b.id = " ".into();
        assert!(write_brief_set(&[b]).is_err());
    }

    #[test]
    fn header_rules() {
        assert_eq!(parse_brief_set(""), Err(BriefError::Empty));
        assert_eq!(
            parse_brief_set("{\"sarathi_brief_set\":2}"),
            Err(BriefError::Version(2))
        );
        assert!(matches!(
            parse_brief_set("{\"id\":\"a\"}"),
            Err(BriefError::BadHeader(_))
        ));
        assert_eq!(
            parse_brief_set("{\"sarathi_brief_set\":1}\n").unwrap(),
            vec![]
        );
    }
}
