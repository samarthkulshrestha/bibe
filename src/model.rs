use crate::attention::head_average;
use crate::data::vocab::PAD_ID;
use crate::autograd::Var;
use crate::nn::{Embedding, Linear, PositionalEncoding};
use crate::tensor::Tensor;
use crate::transformer::{AnomalyHead, TransformerEncoder};

/// Hyperparameters for [`BibeModel`].
pub struct BibeConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    /// Number of auxiliary per-event features (call depth, cache misses, ...).
    pub n_aux: usize,
    /// Size of the object-id vocabulary (0-id reserved for "no object").
    pub num_objects: usize,
    /// Additive attention bias favoring keys that share the query's object
    /// (0 disables it). Structurally points a use toward its same-object free.
    pub object_bias: f32,
    /// Mask future keys (j > i), turning the encoder into a causal
    /// (backward-only) model. Ablation for the bidirectionality claim.
    pub causal: bool,
    pub max_len: usize,
    pub dropout_p: f32,
}

/// Output of a [`BibeModel`] forward pass.
pub struct ModelOutput {
    /// Per-position anomaly probabilities, shape `[batch, seq]`.
    pub anomaly_scores: Var,
    /// Encoder hidden states, shape `[batch, seq, d_model]`. Exposed so the
    /// trainer can pool a trace-level representation for the contrastive loss.
    pub hidden: Var,
    /// Self-attention weights from each layer, shape `[batch*heads, seq, seq]`.
    pub attention_weights: Vec<Var>,
    /// Attribution map (last layer's head-averaged attention), shape
    /// `[batch, seq, seq]`: how much each source influences each query.
    pub attribution: Tensor,
}

/// The full BiBE encoder model: function-ID and object-id embeddings plus
/// projected auxiliary features and sinusoidal positions, encoded by a
/// bidirectional transformer stack, with a per-position anomaly head and
/// head-averaged attention attribution.
pub struct BibeModel {
    embedding: Embedding,
    object_embedding: Embedding,
    aux_projection: Linear,
    pos_encoding: PositionalEncoding,
    encoder: TransformerEncoder,
    anomaly_head: AnomalyHead,
    num_heads: usize,
    object_bias: f32,
    causal: bool,
}

impl BibeModel {
    /// Build a model from a configuration.
    pub fn new(config: &BibeConfig) -> Self {
        BibeModel {
            embedding: Embedding::new(config.vocab_size, config.d_model),
            object_embedding: Embedding::new(config.num_objects, config.d_model),
            aux_projection: Linear::new(config.n_aux, config.d_model, true),
            pos_encoding: PositionalEncoding::new(config.max_len, config.d_model),
            encoder: TransformerEncoder::new(
                config.num_layers,
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout_p,
            ),
            anomaly_head: AnomalyHead::new(config.d_model),
            num_heads: config.num_heads,
            object_bias: config.object_bias,
            causal: config.causal,
        }
    }

    /// Forward pass.
    ///
    /// `function_ids` is a flat `[batch*seq]` list of token ids; `aux` holds the
    /// auxiliary features as a `[batch, seq, n_aux]` (non-trainable) variable.
    pub fn forward(
        &self,
        function_ids: &[usize],
        object_ids: &[usize],
        aux: &Var,
        batch: usize,
        seq: usize,
        training: bool,
    ) -> ModelOutput {
        // Function-ID + object-id embeddings + projected aux features + positions.
        // The object embedding ties events touching the same object together.
        let tok = self.embedding.forward(function_ids, &[batch, seq]);
        let obj = self.object_embedding.forward(object_ids, &[batch, seq]);
        let aux_emb = self.aux_projection.forward(aux);
        let pos = self.pos_encoding.forward(seq);
        let x = tok.add(&obj).add(&aux_emb).add(&pos);

        // Bidirectional encoder: padded keys masked out, and (optionally) keys
        // sharing the query's object favored. Per-position anomaly scores.
        let mask =
            attention_bias(function_ids, object_ids, self.object_bias, self.causal, batch, seq);
        let (hidden, attention_weights) = self.encoder.forward(&x, training, mask.as_ref());
        let anomaly_scores = self.anomaly_head.forward(&hidden);

        // Attribution from the last layer's head-averaged attention, which
        // pinpoints causes more sharply than the cross-layer rollout.
        let last = attention_weights.last().unwrap().tensor();
        let attribution = head_average(&last, self.num_heads, batch);

        ModelOutput {
            anomaly_scores,
            hidden,
            attention_weights,
            attribution,
        }
    }

    /// Number of attention heads per layer.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

/// Additive attention bias `[batch, seq, seq]` combining two effects on the
/// score for query `i` attending to key `j`:
///   * padding: `-1e9` when key `j` is a `PAD_ID` position (softmax -> ~0),
///   * object: `+object_bias` when `i` and `j` share the same nonzero object.
///
/// Returns `None` when there is no padding and the object bias is disabled.
fn attention_bias(
    function_ids: &[usize],
    object_ids: &[usize],
    object_bias: f32,
    causal: bool,
    batch: usize,
    seq: usize,
) -> Option<Var> {
    let has_pad = function_ids.contains(&PAD_ID);
    if !has_pad && object_bias == 0.0 && !causal {
        return None;
    }

    let mut data = vec![0.0f32; batch * seq * seq];
    for b in 0..batch {
        for i in 0..seq {
            let oi = object_ids[b * seq + i];
            for j in 0..seq {
                let cell = &mut data[b * seq * seq + i * seq + j];
                if function_ids[b * seq + j] == PAD_ID || (causal && j > i) {
                    *cell = -1e9;
                } else if object_bias != 0.0 && oi != 0 && oi == object_ids[b * seq + j] {
                    *cell = object_bias;
                }
            }
        }
    }
    Some(Var::new(Tensor::new(data, vec![batch, seq, seq]), false))
}

impl BibeModel {

    /// Collect all trainable parameters for the optimizer.
    pub fn parameters(&self) -> Vec<Var> {
        let mut params = self.embedding.parameters();
        params.extend(self.object_embedding.parameters());
        params.extend(self.aux_projection.parameters());
        params.extend(self.encoder.parameters());
        params.extend(self.anomaly_head.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> BibeConfig {
        BibeConfig {
            vocab_size: 20,
            d_model: 16,
            num_heads: 2,
            d_ff: 32,
            num_layers: 3,
            n_aux: 4,
            num_objects: 8,
            object_bias: 0.0,
            causal: false,
            max_len: 32,
            dropout_p: 0.0,
        }
    }

    fn sample_input(batch: usize, seq: usize, n_aux: usize) -> (Vec<usize>, Vec<usize>, Var) {
        let ids: Vec<usize> = (0..batch * seq).map(|i| i % 20).collect();
        let objects: Vec<usize> = (0..batch * seq).map(|i| i % 8).collect();
        let aux = Var::new(Tensor::randn(&[batch, seq, n_aux]), false);
        (ids, objects, aux)
    }

    #[test]
    fn test_output_shapes() {
        let model = BibeModel::new(&tiny_config());
        let (ids, obj, aux) = sample_input(2, 6, 4);
        let out = model.forward(&ids, &obj, &aux, 2, 6, false);

        assert_eq!(out.anomaly_scores.tensor().shape(), &[2, 6]);
        assert_eq!(out.attention_weights.len(), 3);
        assert_eq!(out.attribution.shape(), &[2, 6, 6]);
    }

    #[test]
    fn test_hidden_states_exposed() {
        let model = BibeModel::new(&tiny_config());
        let (ids, obj, aux) = sample_input(2, 6, 4);
        let out = model.forward(&ids, &obj, &aux, 2, 6, false);
        // [batch, seq, d_model]
        assert_eq!(out.hidden.tensor().shape(), &[2, 6, 16]);
    }

    #[test]
    fn test_attention_bias_favors_same_object_and_masks_padding() {
        // seq 3: ids [1, 2, PAD], objects [5, 5, 0], bias 2.0.
        let fids = vec![1usize, 2, 0];
        let objs = vec![5usize, 5, 0];
        let bias = attention_bias(&fids, &objs, 2.0, false, 1, 3).unwrap();
        let t = bias.tensor();
        // Padded key column 2 is masked for every query.
        for i in 0..3 {
            assert!(t.get(&[0, i, 2]) < -1e8);
        }
        // Queries 0 and 1 share object 5 -> +2.0 to each other and themselves.
        assert!((t.get(&[0, 0, 1]) - 2.0).abs() < 1e-6);
        assert!((t.get(&[0, 1, 0]) - 2.0).abs() < 1e-6);
        assert!((t.get(&[0, 0, 0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_padding_keys_get_no_attention() {
        // 3 real tokens then 3 padding (id 0). Padded key columns should
        // receive ~0 attention in the attribution map.
        let model = BibeModel::new(&tiny_config());
        let ids = vec![1usize, 2, 3, 0, 0, 0];
        let obj = vec![0usize; 6];
        let aux = Var::new(Tensor::zeros(&[1, 6, 4]), false);
        let out = model.forward(&ids, &obj, &aux, 1, 6, false);
        for i in 0..6 {
            for pad_j in 3..6 {
                assert!(
                    out.attribution.get(&[0, i, pad_j]) < 1e-4,
                    "query {i} attends {} to padded key {pad_j}",
                    out.attribution.get(&[0, i, pad_j])
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_zeroes_future_attention() {
        let mut config = tiny_config();
        config.causal = true;
        let model = BibeModel::new(&config);
        // Non-PAD ids: with the causal mask, query 0's only valid key is
        // itself, so a PAD at position 0 would leave it keyless.
        let ids: Vec<usize> = (0..5).map(|i| i + 1).collect();
        let obj: Vec<usize> = (0..5).map(|i| i % 8).collect();
        let aux = Var::new(Tensor::randn(&[1, 5, 4]), false);
        let out = model.forward(&ids, &obj, &aux, 1, 5, false);
        for i in 0..5 {
            for j in (i + 1)..5 {
                assert!(
                    out.attribution.get(&[0, i, j]) < 1e-4,
                    "query {i} attends {} to future key {j}",
                    out.attribution.get(&[0, i, j])
                );
            }
        }
    }

    #[test]
    fn test_scores_are_probabilities() {
        let model = BibeModel::new(&tiny_config());
        let (ids, obj, aux) = sample_input(2, 5, 4);
        let out = model.forward(&ids, &obj, &aux, 2, 5, false);
        for &v in &out.anomaly_scores.tensor().data {
            assert!(v > 0.0 && v < 1.0, "score {v} is not a probability");
        }
    }

    #[test]
    fn test_attribution_rows_sum_to_one() {
        let model = BibeModel::new(&tiny_config());
        let (ids, obj, aux) = sample_input(1, 5, 4);
        let out = model.forward(&ids, &obj, &aux, 1, 5, false);
        for i in 0..5 {
            let sum: f32 = (0..5).map(|j| out.attribution.get(&[0, i, j])).sum();
            assert!((sum - 1.0).abs() < 1e-5, "attribution row {i} sums to {sum}");
        }
    }

    #[test]
    fn test_output_is_finite() {
        let model = BibeModel::new(&tiny_config());
        let (ids, obj, aux) = sample_input(2, 5, 4);
        let out = model.forward(&ids, &obj, &aux, 2, 5, false);
        assert!(out.anomaly_scores.tensor().data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_end_to_end_gradients() {
        // A loss on the anomaly scores must reach every parameter, including
        // the embedding table and the auxiliary-feature projection.
        let model = BibeModel::new(&tiny_config());
        let (ids, obj, aux) = sample_input(2, 4, 4);
        let out = model.forward(&ids, &obj, &aux, 2, 4, true);
        let loss = out.anomaly_scores.sum();
        loss.backward();

        for (i, p) in model.parameters().iter().enumerate() {
            let g = p.grad().unwrap_or_else(|| panic!("parameter {i} has no gradient"));
            assert!(g.data.iter().all(|v| v.is_finite()), "parameter {i} grad not finite");
        }
    }
}
