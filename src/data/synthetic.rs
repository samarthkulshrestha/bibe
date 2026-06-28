use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::trace::{Trace, TraceEvent, TraceLabel};

/// A class of injected bug, each with a distinctive event signature at the
/// labeled root cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BugKind {
    /// A `free` precursor followed by a freed-memory access with an LLC spike.
    UseAfterFree,
    /// A lock-wait event with a branch-misprediction spike and deep stack.
    Deadlock,
    /// An allocation event with an L2-miss signature.
    MemoryLeak,
    /// A hot loop with cache misses spiking across all levels.
    PerfRegression,
}

impl BugKind {
    /// The bug kinds in a fixed order, for round-robin dataset generation.
    pub const ALL: [BugKind; 4] = [
        BugKind::UseAfterFree,
        BugKind::Deadlock,
        BugKind::MemoryLeak,
        BugKind::PerfRegression,
    ];

    fn function_name(self) -> &'static str {
        match self {
            BugKind::UseAfterFree => "use_after_free",
            BugKind::Deadlock => "lock_wait",
            BugKind::MemoryLeak => "malloc_leak",
            BugKind::PerfRegression => "hot_loop",
        }
    }
}

/// Generator settings.
pub struct GeneratorConfig {
    pub num_functions: usize,
    pub min_len: usize,
    pub max_len: usize,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        GeneratorConfig { num_functions: 32, min_len: 16, max_len: 64 }
    }
}

/// Deterministic generator of synthetic execution traces with optional,
/// labeled injected bugs. Useful both as initial training data and as a
/// controlled validation set where the ground-truth root cause is known.
pub struct TraceGenerator {
    rng: StdRng,
    config: GeneratorConfig,
}

impl TraceGenerator {
    /// Seeded generator with default settings.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, GeneratorConfig::default())
    }

    /// Seeded generator with explicit settings.
    pub fn with_config(seed: u64, config: GeneratorConfig) -> Self {
        TraceGenerator { rng: StdRng::seed_from_u64(seed), config }
    }

    /// Generate a clean trace of normal events.
    pub fn normal_trace(&mut self) -> Trace {
        let len = self.rng.random_range(self.config.min_len..=self.config.max_len);
        let events = (0..len).map(|i| self.normal_event(i)).collect();
        Trace { events, label: TraceLabel::Normal }
    }

    /// Generate a trace with one injected bug of the given kind, labeled with
    /// the root-cause event index.
    pub fn anomalous_trace(&mut self, kind: BugKind) -> Trace {
        let len = self.rng.random_range(self.config.min_len..=self.config.max_len);
        let mut events: Vec<TraceEvent> = (0..len).map(|i| self.normal_event(i)).collect();

        // Root cause sits away from the very start so precursors have room.
        let root_cause = self.rng.random_range(3..len);
        events[root_cause] = self.bug_event(kind, root_cause);

        if kind == BugKind::UseAfterFree {
            // A `free` a few events before the freed-memory access.
            let free_idx = root_cause - 2;
            events[free_idx].function = "free".to_string();
        }

        Trace { events, label: TraceLabel::Anomalous { root_cause, cause: root_cause } }
    }

    /// Generate a labeled dataset: `n_normal` clean traces followed by
    /// `n_anomalous` buggy traces cycling through the bug kinds.
    pub fn dataset(&mut self, n_normal: usize, n_anomalous: usize) -> Vec<Trace> {
        let mut traces = Vec::with_capacity(n_normal + n_anomalous);
        for _ in 0..n_normal {
            traces.push(self.normal_trace());
        }
        for i in 0..n_anomalous {
            let kind = BugKind::ALL[i % BugKind::ALL.len()];
            traces.push(self.anomalous_trace(kind));
        }
        traces
    }

    /// A typical low-noise event from the normal function pool.
    fn normal_event(&mut self, index: usize) -> TraceEvent {
        TraceEvent {
            function: format!("func_{}", self.rng.random_range(0..self.config.num_functions)),
            timestamp_us: index as u64 * 10 + self.rng.random_range(0..5),
            call_depth: self.rng.random_range(0..6),
            l1_misses: self.rng.random_range(0..3),
            l2_misses: self.rng.random_range(0..2),
            llc_misses: self.rng.random_range(0..2),
            branch_misses: self.rng.random_range(0..2),
        }
    }

    /// The signature event for a bug kind at the given position.
    fn bug_event(&mut self, kind: BugKind, index: usize) -> TraceEvent {
        let mut e = self.normal_event(index);
        e.function = kind.function_name().to_string();
        match kind {
            BugKind::UseAfterFree => {
                e.llc_misses = self.rng.random_range(40..100);
                e.branch_misses = self.rng.random_range(5..20);
            }
            BugKind::Deadlock => {
                e.branch_misses = self.rng.random_range(40..100);
                e.call_depth = self.rng.random_range(8..16);
            }
            BugKind::MemoryLeak => {
                e.l2_misses = self.rng.random_range(40..100);
                e.call_depth = self.rng.random_range(6..12);
            }
            BugKind::PerfRegression => {
                e.l1_misses = self.rng.random_range(40..100);
                e.l2_misses = self.rng.random_range(40..100);
                e.llc_misses = self.rng.random_range(40..100);
            }
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_with_seed() {
        let a = TraceGenerator::new(7).normal_trace();
        let b = TraceGenerator::new(7).normal_trace();
        assert_eq!(a.events, b.events, "same seed must reproduce the same trace");
    }

    #[test]
    fn test_normal_trace_shape() {
        let mut g = TraceGenerator::new(1);
        let t = g.normal_trace();
        assert_eq!(t.label, TraceLabel::Normal);
        assert!(t.len() >= 16 && t.len() <= 64, "length {} out of range", t.len());
        assert!(t.events.iter().all(|e| e.function.starts_with("func_")));
    }

    #[test]
    fn test_anomalous_trace_labels_root_cause() {
        let mut g = TraceGenerator::new(2);
        let t = g.anomalous_trace(BugKind::PerfRegression);
        let rc = t.root_cause().expect("should be anomalous");
        assert!(rc < t.len(), "root cause {rc} out of range {}", t.len());
        // The root-cause event carries the bug's signature.
        assert_eq!(t.events[rc].function, "hot_loop");
        assert!(t.events[rc].l1_misses > 10, "perf regression should spike cache misses");
    }

    #[test]
    fn test_each_bug_kind_has_distinct_signature() {
        let mut g = TraceGenerator::new(3);
        for kind in BugKind::ALL {
            let t = g.anomalous_trace(kind);
            let rc = t.root_cause().unwrap();
            assert_eq!(t.events[rc].function, kind.function_name());
        }
    }

    #[test]
    fn test_use_after_free_has_free_precursor() {
        let mut g = TraceGenerator::new(4);
        let t = g.anomalous_trace(BugKind::UseAfterFree);
        let rc = t.root_cause().unwrap();
        // A `free` appears somewhere before the freed-memory access.
        assert!(
            t.events[..rc].iter().any(|e| e.function == "free"),
            "use-after-free should have a free precursor"
        );
    }

    #[test]
    fn test_dataset_counts_and_labels() {
        let mut g = TraceGenerator::new(5);
        let ds = g.dataset(5, 3);
        assert_eq!(ds.len(), 8);
        assert_eq!(ds.iter().filter(|t| !t.is_anomalous()).count(), 5);
        assert_eq!(ds.iter().filter(|t| t.is_anomalous()).count(), 3);
    }

    #[test]
    fn test_generated_trace_round_trips_through_text() {
        use super::super::parser::{parse_trace, serialize_trace};
        let mut g = TraceGenerator::new(9);
        let t = g.anomalous_trace(BugKind::Deadlock);
        let restored = parse_trace(&serialize_trace(&t)).unwrap();
        assert_eq!(restored.label, t.label);
        assert_eq!(restored.events, t.events);
    }

    #[test]
    fn test_generated_data_flows_through_pipeline() {
        use crate::autograd::Var;
        use crate::data::loader::collate;
        use crate::data::window::extract_windows;
        use crate::data::{Vocabulary, N_AUX};
        use crate::model::{BibeConfig, BibeModel};

        let mut g = TraceGenerator::new(11);
        let dataset = g.dataset(3, 2);
        let vocab = Vocabulary::build(&dataset, 1);

        let mut windows = Vec::new();
        for t in &dataset {
            windows.extend(extract_windows(t, 32, 32));
        }
        let batch = collate(&windows, &vocab);

        let config = BibeConfig {
            vocab_size: vocab.len(),
            d_model: 16,
            num_heads: 2,
            d_ff: 32,
            num_layers: 2,
            n_aux: N_AUX,
            max_len: 64,
            dropout_p: 0.0,
        };
        let model = BibeModel::new(&config);
        let aux = Var::new(batch.aux.clone(), false);
        let out = model.forward(&batch.function_ids, &aux, batch.batch, batch.seq, false);
        assert_eq!(out.anomaly_scores.tensor().shape(), &[batch.batch, batch.seq]);
        assert!(out.anomaly_scores.tensor().data.iter().all(|v| v.is_finite()));
    }
}
