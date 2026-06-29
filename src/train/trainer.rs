use crate::attention::attention_rollout_var;
use crate::autograd::Var;
use crate::data::loader::Batch;
use crate::data::DataLoader;
use crate::data::Vocabulary;
use crate::model::BibeModel;
use crate::optim::{clip_grad_norm, lr_at, Adam};
use crate::train::loss::attention_sparsity_loss;
use crate::train::objective::{
    attribution_margin_loss, attribution_supervision_loss, batch_contrastive_loss,
    masked_focal_loss, masked_mean_pool, rollout_supervision_loss,
};

/// Which attention signal the attribution supervision acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttributionTarget {
    /// Raw last-layer head-averaged attention (reward mass on the cause).
    #[default]
    RawAttention,
    /// The differentiable cross-layer rollout.
    Rollout,
    /// Margin/contrastive: rank the cause above decoy candidates.
    Margin,
}

/// Training hyperparameters.
pub struct TrainConfig {
    pub lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub grad_clip: f32,
    pub focal_alpha: f32,
    pub focal_gamma: f32,
    pub sparsity_lambda: f32,
    pub contrastive_lambda: f32,
    pub contrastive_temp: f32,
    /// Weight on the attention attribution-supervision loss.
    pub attribution_lambda: f32,
    /// Which attention signal the attribution supervision acts on.
    pub attribution_target: AttributionTarget,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            lr: 3e-4,
            warmup_steps: 1000,
            total_steps: 100_000,
            grad_clip: 1.0,
            focal_alpha: 0.75,
            focal_gamma: 2.0,
            sparsity_lambda: 0.01,
            // Off by default: the leave-one-out sweep (examples/ood_study.rs)
            // showed the trace-pooled contrastive term hurts both in- and
            // out-of-distribution detection at this scale. Kept available for
            // noisier, less-separable data where it may help.
            contrastive_lambda: 0.0,
            contrastive_temp: 0.07,
            attribution_lambda: 0.0,
            attribution_target: AttributionTarget::RawAttention,
        }
    }
}

/// Per-step training diagnostics.
pub struct StepStats {
    pub loss: f32,
    pub grad_norm: f32,
    pub lr: f32,
}

/// Owns the model and optimizer and runs the training loop.
pub struct Trainer {
    model: BibeModel,
    optimizer: Adam,
    config: TrainConfig,
    step: usize,
}

impl Trainer {
    pub fn new(model: BibeModel, config: TrainConfig) -> Self {
        let optimizer = Adam::new(model.parameters(), config.lr);
        Trainer { model, optimizer, config, step: 0 }
    }

    /// Borrow the underlying model (e.g. for evaluation or checkpointing).
    pub fn model(&self) -> &BibeModel {
        &self.model
    }

    /// Run one optimization step on a batch and return diagnostics.
    ///
    /// The total loss is the masked per-position focal loss plus the
    /// attention-sparsity regularizer over every layer plus, when the batch
    /// contains both anomalous and normal traces, the contrastive trace loss.
    pub fn train_step(&mut self, batch: &Batch) -> StepStats {
        self.step += 1;
        let lr = lr_at(self.config.lr, self.step, self.config.warmup_steps, self.config.total_steps);
        self.optimizer.set_lr(lr);

        let aux = Var::new(batch.aux.clone(), false);
        let out = self
            .model
            .forward(&batch.function_ids, &batch.object_ids, &aux, batch.batch, batch.seq, true);

        let labels = Var::new(batch.labels.clone(), false);
        let mask = Var::new(batch.pad_mask.clone(), false);

        // Per-event anomaly loss (masked focal).
        let mut loss = masked_focal_loss(
            &out.anomaly_scores,
            &labels,
            &mask,
            self.config.focal_alpha,
            self.config.focal_gamma,
        );

        // Attention sparsity regularizer over every layer.
        for attn in &out.attention_weights {
            loss = loss.add(&attention_sparsity_loss(attn, self.config.sparsity_lambda));
        }

        // Contrastive trace loss, when the batch has a usable composition.
        let pooled = masked_mean_pool(&out.hidden, &mask);
        let is_anomalous = window_anomaly_flags(batch);
        if let Some(c) = batch_contrastive_loss(&pooled, &is_anomalous, self.config.contrastive_temp) {
            loss = loss.add(&c.mul_scalar(self.config.contrastive_lambda));
        }

        // Attribution supervision: steer the symptom's attention to the cause,
        // but only where the cause is a distinct event.
        if self.config.attribution_lambda > 0.0 {
            let supervised = supervised_triples(batch);
            let heads = self.model.num_heads();
            let term = match self.config.attribution_target {
                AttributionTarget::RawAttention => {
                    attribution_supervision_loss(&out.attention_weights, &supervised, heads)
                }
                AttributionTarget::Rollout => {
                    let rollout =
                        attention_rollout_var(&out.attention_weights, heads, batch.batch);
                    rollout_supervision_loss(&rollout, &supervised)
                }
                AttributionTarget::Margin => {
                    let masked: Vec<(usize, usize, usize, Vec<f32>)> = supervised
                        .iter()
                        .map(|&(w, sym, cause)| (w, sym, cause, candidate_mask(batch, w, sym)))
                        .collect();
                    let last = out.attention_weights.last().unwrap();
                    attribution_margin_loss(last, heads, &masked)
                }
            };
            if let Some(a) = term {
                loss = loss.add(&a.mul_scalar(self.config.attribution_lambda));
            }
        }

        let loss_val = loss.tensor().data[0];

        self.optimizer.zero_grad();
        loss.backward();
        let grad_norm = clip_grad_norm(&self.model.parameters(), self.config.grad_clip);
        self.optimizer.step();

        StepStats { loss: loss_val, grad_norm, lr }
    }

    /// Run one pass over the loader, returning the mean per-batch loss.
    pub fn train_epoch(&mut self, loader: &DataLoader, vocab: &Vocabulary) -> f32 {
        let batches = loader.batches(vocab);
        if batches.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for batch in &batches {
            total += self.train_step(batch).loss;
        }
        total / batches.len() as f32
    }
}

/// A window counts as anomalous if any real position carries a positive label.
fn window_anomaly_flags(batch: &Batch) -> Vec<bool> {
    (0..batch.batch)
        .map(|b| {
            (0..batch.seq).any(|s| batch.labels.data[b * batch.seq + s] > 0.5)
        })
        .collect()
}

/// Candidate-source mask `[seq]` for a window's symptom query: 1.0 for real
/// positions other than the symptom itself, 0.0 elsewhere.
fn candidate_mask(batch: &Batch, window: usize, symptom: usize) -> Vec<f32> {
    (0..batch.seq)
        .map(|s| {
            let real = batch.pad_mask.data[window * batch.seq + s] > 0.5;
            if real && s != symptom { 1.0 } else { 0.0 }
        })
        .collect()
}

/// `(window, symptom, cause)` triples for windows whose cause is a distinct
/// event from the symptom — the only ones worth supervising attention on.
fn supervised_triples(batch: &Batch) -> Vec<(usize, usize, usize)> {
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    (0..batch.batch)
        .filter_map(|b| {
            let span = b * batch.seq..(b + 1) * batch.seq;
            let labels = &batch.labels.data[span.clone()];
            let cause = &batch.cause.data[span];
            let symptom = argmax(labels);
            let cause_pos = argmax(cause);
            let anomalous = labels[symptom] > 0.5 && cause[cause_pos] > 0.5;
            (anomalous && symptom != cause_pos).then_some((b, symptom, cause_pos))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::loader::collate;
    use crate::data::trace::{Trace, TraceEvent, TraceLabel};
    use crate::data::window::extract_windows;
    use crate::data::N_AUX;
    use crate::model::BibeConfig;

    fn event(name: &str, l1: u32) -> TraceEvent {
        TraceEvent {
            function: name.to_string(),
            timestamp_us: 0,
            call_depth: 1,
            l1_misses: l1,
            l2_misses: 0,
            llc_misses: 0,
            branch_misses: 0,
            object_id: 0,
        }
    }

    fn fixture() -> (Batch, BibeConfig) {
        // Two short traces (one normal, one anomalous) -> a 2-window batch with
        // both classes, so the contrastive term engages.
        let normal = Trace {
            events: vec![event("a", 0), event("b", 1), event("c", 0), event("d", 0)],
            label: TraceLabel::Normal,
        };
        let anomalous = Trace {
            events: vec![event("a", 0), event("x", 9), event("c", 0), event("d", 0)],
            label: TraceLabel::Anomalous { root_cause: 1, cause: 1 },
        };
        let vocab = crate::data::Vocabulary::build(&[normal.clone(), anomalous.clone()], 1);
        let mut windows = extract_windows(&normal, 4, 4);
        windows.extend(extract_windows(&anomalous, 4, 4));
        let batch = collate(&windows, &vocab);

        let config = BibeConfig {
            vocab_size: vocab.len(),
            d_model: 16,
            num_heads: 2,
            d_ff: 32,
            num_layers: 2,
            n_aux: N_AUX,
            num_objects: 8,
            max_len: 16,
            dropout_p: 0.0,
        };
        (batch, config)
    }

    #[test]
    fn test_train_step_returns_finite_stats() {
        let (batch, config) = fixture();
        let mut trainer = Trainer::new(BibeModel::new(&config), TrainConfig::default());
        let stats = trainer.train_step(&batch);
        assert!(stats.loss.is_finite(), "loss not finite");
        assert!(stats.grad_norm.is_finite(), "grad norm not finite");
        assert!(stats.lr > 0.0, "lr should be positive during warmup");
    }

    #[test]
    fn test_overfits_single_batch() {
        // The whole loop end to end: loss must fall substantially when
        // overfitting one fixed batch.
        let (batch, config) = fixture();
        let cfg = TrainConfig {
            lr: 1e-3,
            warmup_steps: 5,
            total_steps: 200,
            ..TrainConfig::default()
        };
        let mut trainer = Trainer::new(BibeModel::new(&config), cfg);

        let first = trainer.train_step(&batch).loss;
        let mut last = first;
        for _ in 0..120 {
            last = trainer.train_step(&batch).loss;
        }
        assert!(last < first * 0.7, "loss did not fall enough: {first} -> {last}");
    }

    #[test]
    fn test_train_step_with_attribution_supervision() {
        // A window with a distinct cause exercises the attribution-supervision
        // path; the step must still produce a finite loss.
        let trace = Trace {
            events: vec![event("a", 0), event("free", 5), event("c", 0), event("use", 9)],
            label: TraceLabel::Anomalous { root_cause: 3, cause: 1 },
        };
        let vocab = crate::data::Vocabulary::build(&[trace.clone()], 1);
        let windows = extract_windows(&trace, 4, 4);
        let batch = collate(&windows, &vocab);
        let config = BibeConfig {
            vocab_size: vocab.len(),
            d_model: 16,
            num_heads: 2,
            d_ff: 32,
            num_layers: 2,
            n_aux: N_AUX,
            num_objects: 8,
            max_len: 16,
            dropout_p: 0.0,
        };
        let cfg = TrainConfig { attribution_lambda: 0.5, ..TrainConfig::default() };
        let mut trainer = Trainer::new(BibeModel::new(&config), cfg);
        let stats = trainer.train_step(&batch);
        assert!(stats.loss.is_finite(), "loss not finite with attribution supervision");

        // The rollout and margin supervision paths must also stay finite.
        for target in [AttributionTarget::Rollout, AttributionTarget::Margin] {
            let cfg2 = TrainConfig {
                attribution_lambda: 0.5,
                attribution_target: target,
                ..TrainConfig::default()
            };
            let mut t2 = Trainer::new(BibeModel::new(&config), cfg2);
            assert!(t2.train_step(&batch).loss.is_finite(), "{target:?} supervision not finite");
        }
    }

    #[test]
    fn test_parameters_change_after_step() {
        let (batch, config) = fixture();
        let mut trainer = Trainer::new(BibeModel::new(&config), TrainConfig::default());
        let before = trainer.model().parameters()[0].tensor().data.clone();
        trainer.train_step(&batch);
        let after = trainer.model().parameters()[0].tensor().data.clone();
        let changed = before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "parameters should change after a step");
    }
}
