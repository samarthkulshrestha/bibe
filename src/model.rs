use crate::attention::head_average;
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

/// The full BiBE encoder model: function-ID embeddings plus projected
/// auxiliary features and sinusoidal positions, encoded by a bidirectional
/// transformer stack, with a per-position anomaly head and attention-rollout
/// attribution.
pub struct BibeModel {
    embedding: Embedding,
    aux_projection: Linear,
    pos_encoding: PositionalEncoding,
    encoder: TransformerEncoder,
    anomaly_head: AnomalyHead,
    num_heads: usize,
}

impl BibeModel {
    /// Build a model from a configuration.
    pub fn new(config: &BibeConfig) -> Self {
        BibeModel {
            embedding: Embedding::new(config.vocab_size, config.d_model),
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
        }
    }

    /// Forward pass.
    ///
    /// `function_ids` is a flat `[batch*seq]` list of token ids; `aux` holds the
    /// auxiliary features as a `[batch, seq, n_aux]` (non-trainable) variable.
    pub fn forward(
        &self,
        function_ids: &[usize],
        aux: &Var,
        batch: usize,
        seq: usize,
        training: bool,
    ) -> ModelOutput {
        // Function-ID embeddings + projected auxiliary features + positions.
        let tok = self.embedding.forward(function_ids, &[batch, seq]);
        let aux_emb = self.aux_projection.forward(aux);
        let pos = self.pos_encoding.forward(seq);
        let x = tok.add(&aux_emb).add(&pos);

        // Bidirectional encoder, per-position anomaly scores.
        let (hidden, attention_weights) = self.encoder.forward(&x, training);
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

    /// Collect all trainable parameters for the optimizer.
    pub fn parameters(&self) -> Vec<Var> {
        let mut params = self.embedding.parameters();
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
            max_len: 32,
            dropout_p: 0.0,
        }
    }

    fn sample_input(batch: usize, seq: usize, n_aux: usize) -> (Vec<usize>, Var) {
        let ids: Vec<usize> = (0..batch * seq).map(|i| i % 20).collect();
        let aux = Var::new(Tensor::randn(&[batch, seq, n_aux]), false);
        (ids, aux)
    }

    #[test]
    fn test_output_shapes() {
        let model = BibeModel::new(&tiny_config());
        let (ids, aux) = sample_input(2, 6, 4);
        let out = model.forward(&ids, &aux, 2, 6, false);

        assert_eq!(out.anomaly_scores.tensor().shape(), &[2, 6]);
        assert_eq!(out.attention_weights.len(), 3);
        assert_eq!(out.attribution.shape(), &[2, 6, 6]);
    }

    #[test]
    fn test_hidden_states_exposed() {
        let model = BibeModel::new(&tiny_config());
        let (ids, aux) = sample_input(2, 6, 4);
        let out = model.forward(&ids, &aux, 2, 6, false);
        // [batch, seq, d_model]
        assert_eq!(out.hidden.tensor().shape(), &[2, 6, 16]);
    }

    #[test]
    fn test_scores_are_probabilities() {
        let model = BibeModel::new(&tiny_config());
        let (ids, aux) = sample_input(2, 5, 4);
        let out = model.forward(&ids, &aux, 2, 5, false);
        for &v in &out.anomaly_scores.tensor().data {
            assert!(v > 0.0 && v < 1.0, "score {v} is not a probability");
        }
    }

    #[test]
    fn test_attribution_rows_sum_to_one() {
        let model = BibeModel::new(&tiny_config());
        let (ids, aux) = sample_input(1, 5, 4);
        let out = model.forward(&ids, &aux, 1, 5, false);
        for i in 0..5 {
            let sum: f32 = (0..5).map(|j| out.attribution.get(&[0, i, j])).sum();
            assert!((sum - 1.0).abs() < 1e-5, "attribution row {i} sums to {sum}");
        }
    }

    #[test]
    fn test_output_is_finite() {
        let model = BibeModel::new(&tiny_config());
        let (ids, aux) = sample_input(2, 5, 4);
        let out = model.forward(&ids, &aux, 2, 5, false);
        assert!(out.anomaly_scores.tensor().data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_end_to_end_gradients() {
        // A loss on the anomaly scores must reach every parameter, including
        // the embedding table and the auxiliary-feature projection.
        let model = BibeModel::new(&tiny_config());
        let (ids, aux) = sample_input(2, 4, 4);
        let out = model.forward(&ids, &aux, 2, 4, true);
        let loss = out.anomaly_scores.sum();
        loss.backward();

        for (i, p) in model.parameters().iter().enumerate() {
            let g = p.grad().unwrap_or_else(|| panic!("parameter {i} has no gradient"));
            assert!(g.data.iter().all(|v| v.is_finite()), "parameter {i} grad not finite");
        }
    }
}
