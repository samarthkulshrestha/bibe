use super::trace::{Trace, TraceEvent};
use super::vocab::PAD_TOKEN;

/// A fixed-length window over a trace, ready for batching.
///
/// `events`, `labels`, and `pad_mask` all have length `window_size`. Positions
/// beyond the end of the trace are padding: a `<PAD>` event, a zero label, and
/// `pad_mask = false`.
#[derive(Debug, Clone)]
pub struct TraceWindow {
    pub events: Vec<TraceEvent>,
    /// Per-position anomaly label: 1.0 at the root-cause event, else 0.0.
    pub labels: Vec<f32>,
    /// true for real events, false for padding.
    pub pad_mask: Vec<bool>,
}

/// A padding event: the `<PAD>` token with all counters zero.
fn pad_event() -> TraceEvent {
    TraceEvent {
        function: PAD_TOKEN.to_string(),
        timestamp_us: 0,
        call_depth: 0,
        l1_misses: 0,
        l2_misses: 0,
        llc_misses: 0,
        branch_misses: 0,
    }
}

/// Slice a trace into overlapping windows of `window_size`, advancing by
/// `stride`. Short or trailing windows are padded to `window_size`. An empty
/// trace yields no windows.
pub fn extract_windows(trace: &Trace, window_size: usize, stride: usize) -> Vec<TraceWindow> {
    assert!(window_size > 0 && stride > 0, "window_size and stride must be positive");

    let len = trace.len();
    if len == 0 {
        return Vec::new();
    }

    let root_cause = trace.root_cause();
    let mut windows = Vec::new();

    let mut start = 0;
    while start < len {
        let mut events = Vec::with_capacity(window_size);
        let mut labels = Vec::with_capacity(window_size);
        let mut pad_mask = Vec::with_capacity(window_size);

        for p in 0..window_size {
            let idx = start + p;
            if idx < len {
                events.push(trace.events[idx].clone());
                labels.push(if root_cause == Some(idx) { 1.0 } else { 0.0 });
                pad_mask.push(true);
            } else {
                events.push(pad_event());
                labels.push(0.0);
                pad_mask.push(false);
            }
        }

        windows.push(TraceWindow { events, labels, pad_mask });
        start += stride;
    }

    windows
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::trace::TraceLabel;

    fn trace(len: usize, label: TraceLabel) -> Trace {
        let events = (0..len)
            .map(|i| TraceEvent {
                function: format!("f{i}"),
                timestamp_us: i as u64,
                call_depth: 0,
                l1_misses: 0,
                l2_misses: 0,
                llc_misses: 0,
                branch_misses: 0,
            })
            .collect();
        Trace { events, label }
    }

    #[test]
    fn test_exact_fit_single_window_no_padding() {
        let w = extract_windows(&trace(4, TraceLabel::Normal), 4, 4);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].events.len(), 4);
        assert!(w[0].pad_mask.iter().all(|&m| m));
        assert_eq!(w[0].events[0].function, "f0");
    }

    #[test]
    fn test_short_trace_is_padded() {
        let w = extract_windows(&trace(2, TraceLabel::Normal), 4, 4);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].events.len(), 4);
        assert_eq!(w[0].pad_mask, vec![true, true, false, false]);
        assert_eq!(w[0].events[3].function, PAD_TOKEN);
        assert_eq!(w[0].labels, vec![0.0; 4]);
    }

    #[test]
    fn test_window_count_and_overlap() {
        // len 10, window 4, stride 4 -> starts 0,4,8 -> 3 windows.
        let w = extract_windows(&trace(10, TraceLabel::Normal), 4, 4);
        assert_eq!(w.len(), 3);
        // Last window starts at event 8, positions 10,11 are padding.
        assert_eq!(w[2].events[0].function, "f8");
        assert_eq!(w[2].pad_mask, vec![true, true, false, false]);
    }

    #[test]
    fn test_root_cause_label_inside_window() {
        let w = extract_windows(&trace(4, TraceLabel::Anomalous { root_cause: 2, cause: 2 }), 4, 4);
        assert_eq!(w[0].labels, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_root_cause_outside_window_is_unlabeled() {
        // root cause at global index 5; first window covers 0..4 only.
        let w = extract_windows(&trace(10, TraceLabel::Anomalous { root_cause: 5, cause: 5 }), 4, 4);
        assert_eq!(w[0].labels, vec![0.0; 4]);
        // Second window covers events 4..8, so local index 1 (global 5) is set.
        assert_eq!(w[1].labels, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_empty_trace_yields_no_windows() {
        let w = extract_windows(&trace(0, TraceLabel::Normal), 4, 4);
        assert!(w.is_empty());
    }
}
