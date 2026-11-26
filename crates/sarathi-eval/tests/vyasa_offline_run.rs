// SPDX-License-Identifier: Apache-2.0
//! The real Theme Studio brief set through the offline pipeline (A9, no model spend): every measurement brief runs
//! three times on a scripted harness, tuning briefs are refused for a measurement run, and the fault of `adv-04` reaches
//! the harness.

use sarathi_eval::brief::{Brief, Split, parse_brief_set, split_of};
use sarathi_eval::runner::{
    Acceptance, Arm, Harness, HarnessError, HarnessOutput, Purpose, run_briefs,
};

const SET: &str = include_str!("../../../evals/vyasa/briefs.jsonl");

/// Fails the injected-fault brief once, as a model returning malformed steps would on its first try.
struct Scripted;
impl Harness for Scripted {
    fn run(&self, b: &Brief, _m: &str, run: u32) -> Result<HarnessOutput, HarnessError> {
        if b.fault.as_deref() == Some("malformed_steps") && run == 1 {
            return Err(HarnessError::new("malformed steps"));
        }
        Ok(HarnessOutput {
            artifact: format!("theme for {}", b.id),
            valid_ops: 5,
            total_ops: 5,
            steps: 3,
            wall_ms: 10,
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

fn arms(h: &Scripted) -> [Arm<'_>; 1] {
    [Arm {
        name: "new".into(),
        slot: "strong".into(),
        model: "m".into(),
        harness: h,
    }]
}

#[test]
fn the_twenty_measurement_briefs_run_three_times_each() -> Result<(), String> {
    let set = parse_brief_set(SET).map_err(|e| e.to_string())?;
    let measurement: Vec<Brief> = split_of(&set, Split::Measurement)
        .into_iter()
        .cloned()
        .collect();
    assert_eq!(measurement.len(), 20);
    let h = Scripted;
    let recs = run_briefs(&measurement, &arms(&h), &Always, 3, Purpose::Measure)
        .map_err(|e| e.to_string())?;
    assert_eq!(recs.len(), 60);
    Ok(())
}

#[test]
fn a_measurement_run_over_the_whole_set_is_refused() -> Result<(), String> {
    let set = parse_brief_set(SET).map_err(|e| e.to_string())?;
    let h = Scripted;
    let r = run_briefs(&set, &arms(&h), &Always, 3, Purpose::Measure);
    assert!(r.is_err(), "tuning briefs must not enter a measurement run");
    Ok(())
}

#[test]
fn the_injected_fault_reaches_the_harness() -> Result<(), String> {
    let set = parse_brief_set(SET).map_err(|e| e.to_string())?;
    let faulty: Vec<Brief> = set.into_iter().filter(|b| b.fault.is_some()).collect();
    assert_eq!(faulty.len(), 1);
    let h = Scripted;
    let recs =
        run_briefs(&faulty, &arms(&h), &Always, 3, Purpose::Measure).map_err(|e| e.to_string())?;
    assert_eq!(recs.len(), 3);
    let missed: Vec<u32> = recs.iter().filter(|r| !r.accepted).map(|r| r.run).collect();
    assert_eq!(missed, vec![1], "only the first run hit the injected fault");
    Ok(())
}
