use super::trace::TraceEvent;

/// Number of auxiliary per-event features fed to the model.
pub const N_AUX: usize = 5;

/// Build the normalized auxiliary feature vector for one event.
///
/// All five counters (call depth and L1/L2/LLC cache and branch misses) are
/// log-compressed with `ln(1 + x)`, which tames their wide dynamic range and
/// maps zero to zero. The model's learned aux projection and the encoder's
/// LayerNorm handle any remaining scaling.
///
/// Order: `[depth, l1, l2, llc, branch]`.
///
/// Note: timestamps are intentionally not included here — event order is
/// carried by the positional encoding, not by a timestamp feature.
pub fn aux_features(event: &TraceEvent) -> [f32; N_AUX] {
    let log1p = |x: u32| (1.0 + x as f32).ln();
    [
        log1p(event.call_depth),
        log1p(event.l1_misses),
        log1p(event.l2_misses),
        log1p(event.llc_misses),
        log1p(event.branch_misses),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(depth: u32, l1: u32, l2: u32, llc: u32, branch: u32) -> TraceEvent {
        TraceEvent {
            function: "f".to_string(),
            timestamp_us: 0,
            call_depth: depth,
            l1_misses: l1,
            l2_misses: l2,
            llc_misses: llc,
            branch_misses: branch,
            object_id: 0,
        }
    }

    #[test]
    fn test_n_aux_is_five() {
        assert_eq!(N_AUX, 5);
    }

    #[test]
    fn test_zeros_map_to_zeros() {
        let f = aux_features(&event(0, 0, 0, 0, 0));
        assert_eq!(f, [0.0; 5]);
    }

    #[test]
    fn test_log_compression_values() {
        let f = aux_features(&event(3, 88, 4, 1, 0));
        let expected = [
            4.0_f32.ln(),  // ln(1+3)
            89.0_f32.ln(), // ln(1+88)
            5.0_f32.ln(),  // ln(1+4)
            2.0_f32.ln(),  // ln(1+1)
            0.0,           // ln(1+0)
        ];
        for (a, e) in f.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "{a} != {e}");
        }
    }

    #[test]
    fn test_monotonic_in_counts() {
        let small = aux_features(&event(0, 10, 0, 0, 0))[1];
        let large = aux_features(&event(0, 1000, 0, 0, 0))[1];
        assert!(large > small, "more cache misses should give a larger feature");
    }
}
