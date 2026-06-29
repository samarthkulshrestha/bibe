use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::tensor::Tensor;

use super::normalize::{aux_features, N_AUX};
use super::vocab::Vocabulary;
use super::window::TraceWindow;

/// A model-ready batch of windows.
///
/// Shapes line up with [`crate::model::BibeModel::forward`]: `function_ids` is
/// a flat `[batch*seq]` id list, `aux` is `[batch, seq, N_AUX]`, and `labels`
/// / `pad_mask` are `[batch, seq]` (mask: 1.0 for real events, 0.0 padding).
pub struct Batch {
    pub function_ids: Vec<usize>,
    pub aux: Tensor,
    pub labels: Tensor,
    /// Per-position causal-event labels `[batch, seq]` (1.0 at the cause).
    pub cause: Tensor,
    /// Per-position object id `[batch*seq]`, tying events on the same object
    /// together (0 = no object). See [`object_id_from_name`].
    pub object_ids: Vec<usize>,
    pub pad_mask: Tensor,
    pub batch: usize,
    pub seq: usize,
}

/// Derive an object id from a function name.
///
/// A stopgap for the templated corpus: `free_N` / `use_N` call sites map to
/// object `N + 1` (so events touching the same object share an id), everything
/// else to 0. Real captures would assign object ids from allocation addresses.
pub fn object_id_from_name(name: &str) -> usize {
    for prefix in ["free_", "use_"] {
        if let Some(suffix) = name.strip_prefix(prefix)
            && let Ok(n) = suffix.parse::<usize>()
        {
            return n + 1;
        }
    }
    0
}

/// Encode and stack a slice of equal-length windows into a [`Batch`].
pub fn collate(windows: &[TraceWindow], vocab: &Vocabulary) -> Batch {
    let batch = windows.len();
    let seq = if batch > 0 { windows[0].events.len() } else { 0 };

    let mut function_ids = Vec::with_capacity(batch * seq);
    let mut object_ids = Vec::with_capacity(batch * seq);
    let mut aux = Vec::with_capacity(batch * seq * N_AUX);
    let mut labels = Vec::with_capacity(batch * seq);
    let mut cause = Vec::with_capacity(batch * seq);
    let mut mask = Vec::with_capacity(batch * seq);

    for w in windows {
        for (i, ev) in w.events.iter().enumerate() {
            function_ids.push(vocab.encode(&ev.function));
            object_ids.push(object_id_from_name(&ev.function));
            aux.extend_from_slice(&aux_features(ev));
            labels.push(w.labels[i]);
            cause.push(w.cause_labels[i]);
            mask.push(if w.pad_mask[i] { 1.0 } else { 0.0 });
        }
    }

    Batch {
        function_ids,
        aux: Tensor::new(aux, vec![batch, seq, N_AUX]),
        labels: Tensor::new(labels, vec![batch, seq]),
        cause: Tensor::new(cause, vec![batch, seq]),
        object_ids,
        pad_mask: Tensor::new(mask, vec![batch, seq]),
        batch,
        seq,
    }
}

/// Batches windows into model-ready tensors, optionally shuffled.
pub struct DataLoader {
    windows: Vec<TraceWindow>,
    batch_size: usize,
}

impl DataLoader {
    pub fn new(windows: Vec<TraceWindow>, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be positive");
        DataLoader { windows, batch_size }
    }

    /// Number of batches per epoch (the final batch may be smaller).
    pub fn num_batches(&self) -> usize {
        self.windows.len().div_ceil(self.batch_size)
    }

    /// Collate all windows into batches, in order.
    pub fn batches(&self, vocab: &Vocabulary) -> Vec<Batch> {
        self.windows
            .chunks(self.batch_size)
            .map(|group| collate(group, vocab))
            .collect()
    }

    /// Collate all windows into batches after a seeded shuffle of window order.
    pub fn shuffled_batches(&self, vocab: &Vocabulary, seed: u64) -> Vec<Batch> {
        let mut order: Vec<usize> = (0..self.windows.len()).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        order.shuffle(&mut rng);

        order
            .chunks(self.batch_size)
            .map(|idxs| {
                let group: Vec<TraceWindow> =
                    idxs.iter().map(|&i| self.windows[i].clone()).collect();
                collate(&group, vocab)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Var;
    use crate::data::trace::{Trace, TraceEvent, TraceLabel};
    use crate::data::vocab::PAD_ID;
    use crate::data::window::extract_windows;
    use crate::model::{BibeConfig, BibeModel};

    fn event(name: &str, l1: u32) -> TraceEvent {
        TraceEvent {
            function: name.to_string(),
            timestamp_us: 0,
            call_depth: 0,
            l1_misses: l1,
            l2_misses: 0,
            llc_misses: 0,
            branch_misses: 0,
        }
    }

    fn corpus() -> (Vec<TraceWindow>, Vocabulary) {
        let trace = Trace {
            events: vec![event("malloc", 1), event("free", 2), event("use", 3)],
            label: TraceLabel::Anomalous { root_cause: 2, cause: 0 },
        };
        let windows = extract_windows(&trace, 4, 4); // one padded window of length 4
        let vocab = Vocabulary::build(&[trace], 1);
        (windows, vocab)
    }

    #[test]
    fn test_collate_shapes() {
        let (windows, vocab) = corpus();
        let b = collate(&windows, &vocab);
        assert_eq!(b.batch, 1);
        assert_eq!(b.seq, 4);
        assert_eq!(b.function_ids.len(), 4);
        assert_eq!(b.aux.shape(), &[1, 4, N_AUX]);
        assert_eq!(b.labels.shape(), &[1, 4]);
        assert_eq!(b.pad_mask.shape(), &[1, 4]);
    }

    #[test]
    fn test_collate_encodes_ids_and_padding() {
        let (windows, vocab) = corpus();
        let b = collate(&windows, &vocab);
        // First three positions are real functions, fourth is padding -> PAD_ID.
        assert_eq!(b.function_ids[0], vocab.encode("malloc"));
        assert!(b.function_ids[0] >= 2);
        assert_eq!(b.function_ids[3], PAD_ID);
    }

    #[test]
    fn test_collate_labels_and_mask() {
        let (windows, vocab) = corpus();
        let b = collate(&windows, &vocab);
        // root cause at index 2.
        assert_eq!(b.labels.data, vec![0.0, 0.0, 1.0, 0.0]);
        // first three real, last padded.
        assert_eq!(b.pad_mask.data, vec![1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_object_id_from_name() {
        assert_eq!(object_id_from_name("free_3"), 4);
        assert_eq!(object_id_from_name("use_3"), 4); // same object as free_3
        assert_eq!(object_id_from_name("free_0"), 1);
        assert_eq!(object_id_from_name("free"), 0);
        assert_eq!(object_id_from_name("work_2"), 0);
        assert_eq!(object_id_from_name("main"), 0);
    }

    #[test]
    fn test_collate_object_ids_link_same_object() {
        let trace = Trace {
            events: vec![event("use_1", 0), event("work_0", 0), event("free_1", 0)],
            label: TraceLabel::Anomalous { root_cause: 0, cause: 2 },
        };
        let windows = extract_windows(&trace, 4, 4);
        let vocab = Vocabulary::build(std::slice::from_ref(&trace), 1);
        let b = collate(&windows, &vocab);
        // use_1 and free_1 share object id 2; work_0 and padding are 0.
        assert_eq!(b.object_ids, vec![2, 0, 2, 0]);
    }

    #[test]
    fn test_collate_cause_channel() {
        let (windows, vocab) = corpus();
        let b = collate(&windows, &vocab);
        // Cause at index 0 (distinct from the symptom at index 2).
        assert_eq!(b.cause.shape(), &[1, 4]);
        assert_eq!(b.cause.data, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(b.labels.data, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_collate_aux_log_compressed() {
        let (windows, vocab) = corpus();
        let b = collate(&windows, &vocab);
        // Event 0 has l1=1 -> aux index 1 = ln(2); padding row is all zeros.
        assert!((b.aux.get(&[0, 0, 1]) - 2.0_f32.ln()).abs() < 1e-6);
        for k in 0..N_AUX {
            assert_eq!(b.aux.get(&[0, 3, k]), 0.0, "padding aux must be zero");
        }
    }

    #[test]
    fn test_num_batches_partitions_windows() {
        let trace = Trace {
            events: (0..10).map(|i| event(&format!("f{i}"), 0)).collect(),
            label: TraceLabel::Normal,
        };
        let windows = extract_windows(&trace, 2, 2); // 5 windows
        let loader = DataLoader::new(windows, 2);
        assert_eq!(loader.num_batches(), 3); // 2 + 2 + 1
    }

    #[test]
    fn test_batches_cover_all_windows() {
        let trace = Trace {
            events: (0..10).map(|i| event(&format!("f{i}"), 0)).collect(),
            label: TraceLabel::Normal,
        };
        let windows = extract_windows(&trace, 2, 2);
        let vocab = Vocabulary::build(&[trace], 1);
        let loader = DataLoader::new(windows, 2);
        let batches = loader.batches(&vocab);
        let total: usize = batches.iter().map(|b| b.batch).sum();
        assert_eq!(total, 5);
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_shuffle_is_deterministic_with_seed() {
        let trace = Trace {
            events: (0..10).map(|i| event(&format!("f{i}"), i as u32)).collect(),
            label: TraceLabel::Normal,
        };
        let windows = extract_windows(&trace, 2, 2);
        let vocab = Vocabulary::build(&[trace], 1);
        let loader = DataLoader::new(windows, 2);
        let a = loader.shuffled_batches(&vocab, 42);
        let b = loader.shuffled_batches(&vocab, 42);
        assert_eq!(a[0].function_ids, b[0].function_ids);
    }

    #[test]
    fn test_batch_feeds_model() {
        // The batch tensors must satisfy BibeModel's forward contract.
        let (windows, vocab) = corpus();
        let batch = collate(&windows, &vocab);
        let config = BibeConfig {
            vocab_size: vocab.len(),
            d_model: 16,
            num_heads: 2,
            d_ff: 32,
            num_layers: 2,
            n_aux: N_AUX,
            max_len: 16,
            dropout_p: 0.0,
        };
        let model = BibeModel::new(&config);
        let aux = Var::new(batch.aux.clone(), false);
        let out = model.forward(&batch.function_ids, &aux, batch.batch, batch.seq, false);
        assert_eq!(out.anomaly_scores.tensor().shape(), &[batch.batch, batch.seq]);
        assert!(out.anomaly_scores.tensor().data.iter().all(|v| v.is_finite()));
    }
}
