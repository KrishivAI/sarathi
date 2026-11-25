// SPDX-License-Identifier: Apache-2.0
//! An offline run end to end: run briefs on a scripted harness, write the run directory, aggregate, report. Run twice,
//! the report is byte-identical.

use sarathi_eval::brief::{Brief, Split};
use sarathi_eval::metrics::{PriceTable, arm_stats};
use sarathi_eval::report::{Report, to_json, to_markdown};
use sarathi_eval::runner::{
    Acceptance, Arm, Harness, HarnessError, HarnessOutput, Purpose, run_briefs, write_records,
};

struct Scripted;
impl Harness for Scripted {
    fn run(&self, b: &Brief, _m: &str, run: u32) -> Result<HarnessOutput, HarnessError> {
        if b.id == "b2" && run == 1 {
            return Err(HarnessError("cut stream".into()));
        }
        Ok(HarnessOutput {
            artifact: "ok".into(),
            valid_ops: 4,
            total_ops: 5,
            steps: 2,
            wall_ms: 100 + u64::from(run),
            ..Default::default()
        })
    }
}
struct Always;
impl Acceptance for Always {
    fn accepts(&self, _b: &Brief, _o: &HarnessOutput) -> bool {
        true
    }
}

fn once() -> Result<(String, String), String> {
    let briefs: Vec<Brief> = ["b1", "b2"]
        .iter()
        .map(|id| Brief {
            id: (*id).into(),
            split: Split::Measurement,
            site_context: "c".into(),
            text: "t".into(),
            allowed_models: vec!["new".into()],
            acceptance: "a".into(),
        })
        .collect();
    let h = Scripted;
    let arms = [Arm {
        name: "new".into(),
        slot: "new".into(),
        model: "m".into(),
        harness: &h,
    }];
    let recs = run_briefs(&briefs, &arms, &Always, 3, Purpose::Measure)
        .map_err(|e| e.to_string())
        .unwrap_or_default();
    let dir = std::env::temp_dir().join(format!("sarathi-eval-{}", std::process::id()));
    write_records(&dir, "2026-11-28", &recs)
        .map_err(|e| e.to_string())
        .unwrap_or_default();
    assert!(dir.join("2026-11-28/new/b2/1.json").is_file());
    let _ = std::fs::remove_dir_all(&dir);
    let prices: PriceTable =
        serde_json::from_str(r#"{"models":{"m":{"input":1,"output":2,"cached_input":0}}}"#)
            .map_err(|e| e.to_string())?;
    let stats = arm_stats("new", &recs, &prices).map_err(|e| e.to_string())?;
    assert_eq!(stats.runs, 6);
    assert_eq!(stats.accepted, 5, "the failed run counts as a miss");
    let report = Report {
        date: "2026-11-28".into(),
        arms: vec![stats],
        kill_line: None,
        deviations: 0,
    };
    Ok((
        to_json(&report).map_err(|e| e.to_string())?,
        to_markdown(&report),
    ))
}

#[test]
fn offline_run_report_is_reproducible() -> Result<(), String> {
    let (j1, m1) = once()?;
    let (j2, m2) = once()?;
    assert_eq!(j1, j2);
    assert_eq!(m1, m2);
    assert!(!j1.is_empty());
    Ok(())
}
