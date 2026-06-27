use std::f32::consts::PI;

/// Learning rate at `step` under linear warmup followed by cosine decay.
///
/// ```text
/// step < warmup:  base_lr · step / warmup           (linear ramp from 0)
/// step ≥ warmup:  base_lr · 0.5·(1 + cos(π·progress))
///                 where progress = (step - warmup) / (total - warmup)
/// ```
///
/// The rate peaks at `base_lr` exactly at `step == warmup` and decays to 0 at
/// `step == total_steps`; beyond `total_steps` it stays at 0.
pub fn lr_at(base_lr: f32, step: usize, warmup_steps: usize, total_steps: usize) -> f32 {
    if warmup_steps > 0 && step < warmup_steps {
        return base_lr * step as f32 / warmup_steps as f32;
    }

    let decay_span = total_steps.saturating_sub(warmup_steps).max(1);
    let progress = (step - warmup_steps) as f32 / decay_span as f32;
    let progress = progress.clamp(0.0, 1.0);
    base_lr * 0.5 * (1.0 + (PI * progress).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_peak_at_warmup_step() {
        let lr = lr_at(1.0, 100, 100, 1000);
        assert!((lr - 1.0).abs() < 1e-5, "peak should be base_lr, got {lr}");
    }

    #[test]
    fn test_linear_during_warmup() {
        // Halfway through warmup -> half of base_lr.
        let lr = lr_at(2.0, 50, 100, 1000);
        assert!((lr - 1.0).abs() < 1e-5, "got {lr}");
    }

    #[test]
    fn test_zero_at_start() {
        assert!(lr_at(1.0, 0, 100, 1000) < 1e-6);
    }

    #[test]
    fn test_decays_to_zero_at_end() {
        let lr = lr_at(1.0, 1000, 100, 1000);
        assert!(lr < 1e-5, "should decay to ~0, got {lr}");
    }

    #[test]
    fn test_midpoint_of_decay_is_half() {
        // Halfway through the cosine decay window -> base_lr/2.
        let lr = lr_at(1.0, 550, 100, 1000); // (550-100)/(1000-100) = 0.5
        assert!((lr - 0.5).abs() < 1e-5, "got {lr}");
    }

    #[test]
    fn test_clamped_beyond_total() {
        assert!(lr_at(1.0, 5000, 100, 1000) < 1e-6);
    }
}
