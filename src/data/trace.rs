/// A single execution-trace event.
///
/// Function names are kept as raw strings here; the vocabulary maps them to
/// integer ids later. The five numeric counters become the auxiliary feature
/// vector after normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceEvent {
    pub function: String,
    pub timestamp_us: u64,
    pub call_depth: u32,
    pub l1_misses: u32,
    pub l2_misses: u32,
    pub llc_misses: u32,
    pub branch_misses: u32,
}

/// Trace-level label: either a clean run or an anomalous one with the
/// known root-cause event index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceLabel {
    Normal,
    Anomalous { root_cause: usize },
}

/// A full execution trace: an ordered sequence of events plus its label.
#[derive(Debug, Clone)]
pub struct Trace {
    pub events: Vec<TraceEvent>,
    pub label: TraceLabel,
}

impl Trace {
    /// Number of events in the trace.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the trace has no events.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the trace is labeled anomalous.
    pub fn is_anomalous(&self) -> bool {
        matches!(self.label, TraceLabel::Anomalous { .. })
    }

    /// The root-cause event index, if the trace is anomalous.
    pub fn root_cause(&self) -> Option<usize> {
        match self.label {
            TraceLabel::Anomalous { root_cause } => Some(root_cause),
            TraceLabel::Normal => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(name: &str) -> TraceEvent {
        TraceEvent {
            function: name.to_string(),
            timestamp_us: 0,
            call_depth: 0,
            l1_misses: 0,
            l2_misses: 0,
            llc_misses: 0,
            branch_misses: 0,
        }
    }

    #[test]
    fn test_len_counts_events() {
        let t = Trace { events: vec![ev("a"), ev("b"), ev("c")], label: TraceLabel::Normal };
        assert_eq!(t.len(), 3);
        assert!(!t.is_empty());
    }

    #[test]
    fn test_normal_trace_has_no_root_cause() {
        let t = Trace { events: vec![ev("a")], label: TraceLabel::Normal };
        assert!(!t.is_anomalous());
        assert_eq!(t.root_cause(), None);
    }

    #[test]
    fn test_anomalous_trace_exposes_root_cause() {
        let t = Trace {
            events: vec![ev("a"), ev("b")],
            label: TraceLabel::Anomalous { root_cause: 1 },
        };
        assert!(t.is_anomalous());
        assert_eq!(t.root_cause(), Some(1));
    }
}
