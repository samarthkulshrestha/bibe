//! Spectrum-based fault localization (Ochiai, Tarantula) as attribution
//! baselines. Elements are function ids; a trace covers a function if any
//! event calls it. Built from trace-level labels only (no cause labels).

use crate::data::{Trace, Vocabulary};

/// Per-function coverage counts over passed (normal) and failed (anomalous)
/// traces, from which suspiciousness scores are computed.
pub struct Spectrum {
    passed: Vec<f32>,
    failed: Vec<f32>,
    total_passed: f32,
    total_failed: f32,
}

impl Spectrum {
    /// Build coverage counts from labeled traces (trace-level labels only).
    pub fn build(traces: &[Trace], vocab: &Vocabulary) -> Self {
        let n = vocab.len();
        let (mut passed, mut failed) = (vec![0.0; n], vec![0.0; n]);
        let (mut tp, mut tf) = (0.0, 0.0);
        for t in traces {
            let mut covered = vec![false; n];
            for e in &t.events {
                let id = vocab.encode(&e.function);
                if id < n {
                    covered[id] = true;
                }
            }
            let bucket = if t.is_anomalous() {
                tf += 1.0;
                &mut failed
            } else {
                tp += 1.0;
                &mut passed
            };
            for (id, c) in covered.iter().enumerate() {
                if *c {
                    bucket[id] += 1.0;
                }
            }
        }
        Spectrum { passed, failed, total_passed: tp, total_failed: tf }
    }

    /// Ochiai suspiciousness: `ef / sqrt(F * (ef + ep))`.
    pub fn ochiai(&self, function_id: usize) -> f32 {
        let (ef, ep) = match (self.failed.get(function_id), self.passed.get(function_id)) {
            (Some(&ef), Some(&ep)) => (ef, ep),
            _ => return 0.0,
        };
        let denom = (self.total_failed * (ef + ep)).sqrt();
        if denom == 0.0 { 0.0 } else { ef / denom }
    }

    /// Tarantula suspiciousness: `(ef/F) / (ef/F + ep/P)`.
    pub fn tarantula(&self, function_id: usize) -> f32 {
        let (ef, ep) = match (self.failed.get(function_id), self.passed.get(function_id)) {
            (Some(&ef), Some(&ep)) => (ef, ep),
            _ => return 0.0,
        };
        if self.total_failed == 0.0 || self.total_passed == 0.0 {
            return 0.0;
        }
        let (fr, pr) = (ef / self.total_failed, ep / self.total_passed);
        if fr + pr == 0.0 { 0.0 } else { fr / (fr + pr) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Trace, TraceEvent, TraceLabel, Vocabulary};

    fn ev(f: &str) -> TraceEvent {
        TraceEvent {
            function: f.to_string(),
            timestamp_us: 0,
            call_depth: 0,
            l1_misses: 0,
            l2_misses: 0,
            llc_misses: 0,
            branch_misses: 0,
            object_id: 0,
        }
    }

    fn trace(funcs: &[&str], anomalous: bool) -> Trace {
        Trace {
            events: funcs.iter().map(|f| ev(f)).collect(),
            label: if anomalous {
                TraceLabel::Anomalous { root_cause: 0, cause: 0 }
            } else {
                TraceLabel::Normal
            },
        }
    }

    #[test]
    fn test_ochiai_isolates_fail_only_function() {
        // "bad" appears in the failing trace only; "common" in all traces.
        let traces = vec![
            trace(&["common", "bad"], true),
            trace(&["common"], false),
            trace(&["common"], false),
        ];
        let vocab = Vocabulary::build(&traces, 1);
        let spec = Spectrum::build(&traces, &vocab);
        let bad = spec.ochiai(vocab.encode("bad"));
        let common = spec.ochiai(vocab.encode("common"));
        // bad: ef=1, F=1, ep=0 -> 1/sqrt(1*1) = 1.0
        assert!((bad - 1.0).abs() < 1e-6, "bad ochiai = {bad}");
        // common: ef=1, F=1, ep=2 -> 1/sqrt(1*3) ≈ 0.577
        assert!((common - 0.577).abs() < 1e-2, "common ochiai = {common}");
        assert!(bad > common);
    }

    #[test]
    fn test_tarantula_isolates_fail_only_function() {
        let traces = vec![
            trace(&["common", "bad"], true),
            trace(&["common"], false),
            trace(&["common"], false),
        ];
        let vocab = Vocabulary::build(&traces, 1);
        let spec = Spectrum::build(&traces, &vocab);
        let bad = spec.tarantula(vocab.encode("bad"));
        let common = spec.tarantula(vocab.encode("common"));
        // bad: (1/1) / (1/1 + 0/2) = 1.0
        assert!((bad - 1.0).abs() < 1e-6, "bad tarantula = {bad}");
        // common: (1/1) / (1/1 + 2/2) = 0.5
        assert!((common - 0.5).abs() < 1e-6, "common tarantula = {common}");
    }

    #[test]
    fn test_unseen_function_scores_zero() {
        let traces = vec![trace(&["a"], true), trace(&["a"], false)];
        let vocab = Vocabulary::build(&traces, 1);
        let spec = Spectrum::build(&traces, &vocab);
        assert_eq!(spec.ochiai(vocab.len() + 5), 0.0);
        assert_eq!(spec.tarantula(vocab.len() + 5), 0.0);
    }
}
