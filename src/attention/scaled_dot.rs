use crate::autograd::Var;

/// Scaled dot-product attention.
///
/// ```text
/// scores = Q @ K^T / sqrt(d_k)
/// if mask: scores = scores + mask  (mask should contain 0 for attend, -1e9 for ignore)
/// attn_weights = softmax(scores, dim=-1)
/// output = attn_weights @ V
/// ```
///
/// # Arguments
/// - `query`:  [batch, seq_len, d_k]
/// - `key`:    [batch, seq_len, d_k]
/// - `value`:  [batch, seq_len, d_v]
/// - `mask`:   Optional [batch, seq_len, seq_len] with 0.0 / -1e9 values
///
/// # Returns
/// `(output, attn_weights)` where:
/// - `output`:       [batch, seq_len, d_v]
/// - `attn_weights`: [batch, seq_len, seq_len]
pub fn scaled_dot_product_attention(
    query: &Var,
    key: &Var,
    value: &Var,
    mask: Option<&Var>,
) -> (Var, Var) {
    let q_shape = query.tensor().shape().to_vec();
    assert_eq!(q_shape.len(), 3, "query must be 3D [batch, seq, d_k]");
    let d_k = q_shape[2];

    // scores = Q @ K^T / sqrt(d_k)
    let kt = key.transpose_last2(); // [batch, d_k, seq_len]
    let scores = query.matmul(&kt); // [batch, seq_len, seq_len]
    let scale = (d_k as f32).sqrt();
    let scores = scores.mul_scalar(1.0 / scale);

    // Apply mask
    let scores = match mask {
        Some(m) => scores.add(m),
        None => scores,
    };

    // Softmax over last dim (keys)
    let last_dim = scores.tensor().shape().len() - 1;
    let attn_weights = scores.softmax(last_dim);

    // output = attn_weights @ V
    let output = attn_weights.matmul(value); // [batch, seq_len, d_v]

    (output, attn_weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::autograd::gradcheck;

    #[test]
    fn test_output_shape() {
        let q = Var::new(Tensor::randn(&[2, 4, 8]), false);
        let k = Var::new(Tensor::randn(&[2, 4, 8]), false);
        let v = Var::new(Tensor::randn(&[2, 4, 8]), false);

        let (out, attn) = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.tensor().shape(), &[2, 4, 8]);
        assert_eq!(attn.tensor().shape(), &[2, 4, 4]);
    }

    #[test]
    fn test_output_shape_dv_differs() {
        // d_v != d_k
        let q = Var::new(Tensor::randn(&[2, 5, 8]), false);
        let k = Var::new(Tensor::randn(&[2, 5, 8]), false);
        let v = Var::new(Tensor::randn(&[2, 5, 16]), false);

        let (out, attn) = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.tensor().shape(), &[2, 5, 16]);
        assert_eq!(attn.tensor().shape(), &[2, 5, 5]);
    }

    #[test]
    fn test_attn_weights_sum_to_one() {
        let q = Var::new(Tensor::randn(&[2, 4, 8]), false);
        let k = Var::new(Tensor::randn(&[2, 4, 8]), false);
        let v = Var::new(Tensor::randn(&[2, 4, 8]), false);

        let (_, attn) = scaled_dot_product_attention(&q, &k, &v, None);
        let at = attn.tensor();

        for b in 0..2 {
            for i in 0..4 {
                let row_sum: f32 = (0..4).map(|j| at.get(&[b, i, j])).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "attn row [{}, {}] sums to {} instead of 1.0", b, i, row_sum
                );
            }
        }
    }

    #[test]
    fn test_identity_qkv() {
        // When Q = K = V = identity-like, attention should roughly
        // reproduce V (each query attends most to its matching key).
        // Use orthogonal rows so each query has a clear best match.
        let data = vec![
            // batch 0
            1.0, 0.0,
            0.0, 1.0,
        ];
        let x = Tensor::new(data, vec![1, 2, 2]);
        let q = Var::new(x.clone(), false);
        let k = Var::new(x.clone(), false);
        let v = Var::new(x.clone(), false);

        let (out, _) = scaled_dot_product_attention(&q, &k, &v, None);
        let ot = out.tensor();

        // Each row should approximately equal the corresponding V row
        // (softmax will distribute some weight to other positions, but
        // the dominant weight should be on the matching position)
        for i in 0..2 {
            for j in 0..2 {
                let expected = x.get(&[0, i, j]);
                let actual = ot.get(&[0, i, j]);
                // The dominant attention should make output close to input
                // With d_k=2, scale=sqrt(2)≈1.41, dot with self=1/1.41≈0.71
                // softmax([0.71, 0]) ≈ [0.57, 0.43], so output ≈ 0.57*v_i + 0.43*v_j
                // This won't be exact, but row 0 output[0] should be > output[1]
                let _ = (expected, actual); // just verify no crash
            }
        }
        // Row 0 of output: attn weights favor pos 0, so out[0,0] > out[0,1]
        assert!(ot.get(&[0, 0, 0]) > ot.get(&[0, 0, 1]));
        // Row 1 of output: attn weights favor pos 1, so out[1,1] > out[1,0]
        assert!(ot.get(&[0, 1, 1]) > ot.get(&[0, 1, 0]));
    }

    #[test]
    fn test_mask_blocks_attention() {
        let q = Var::new(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2]), false);
        let k = Var::new(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2]), false);
        let v = Var::new(
            Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 2, 2]),
            false,
        );

        // Mask out position 1 for query 0: query 0 can only attend to position 0
        let mask = Var::new(
            Tensor::new(vec![0.0, -1e9, 0.0, 0.0], vec![1, 2, 2]),
            false,
        );

        let (out, attn) = scaled_dot_product_attention(&q, &k, &v, Some(&mask));
        let at = attn.tensor();

        // Query 0 should attend almost entirely to position 0
        assert!(at.get(&[0, 0, 0]) > 0.99);
        assert!(at.get(&[0, 0, 1]) < 0.01);

        // Output row 0 should be ≈ V[0] = [10, 20]
        let ot = out.tensor();
        assert!((ot.get(&[0, 0, 0]) - 10.0).abs() < 0.1);
        assert!((ot.get(&[0, 0, 1]) - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_gradients_flow() {
        let q = Var::new(Tensor::randn(&[1, 3, 4]), true);
        let k = Var::new(Tensor::randn(&[1, 3, 4]), true);
        let v = Var::new(Tensor::randn(&[1, 3, 4]), true);

        let (out, _) = scaled_dot_product_attention(&q, &k, &v, None);
        let loss = out.sum();
        loss.backward();

        assert!(q.grad().is_some(), "query has no gradient");
        assert!(k.grad().is_some(), "key has no gradient");
        assert!(v.grad().is_some(), "value has no gradient");

        assert_eq!(q.grad().unwrap().shape(), &[1, 3, 4]);
        assert_eq!(k.grad().unwrap().shape(), &[1, 3, 4]);
        assert_eq!(v.grad().unwrap().shape(), &[1, 3, 4]);
    }

    #[test]
    fn gradcheck_attention_query() {
        let q_data = Tensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec![1, 2, 4],
        );
        let k_data = Tensor::new(
            vec![0.2, 0.3, 0.1, 0.4, 0.5, 0.1, 0.3, 0.2],
            vec![1, 2, 4],
        );
        let v_data = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 4],
        );

        let (ok, err) = gradcheck(
            &|q| {
                let k = Var::new(k_data.clone(), false);
                let v = Var::new(v_data.clone(), false);
                let (out, _) = scaled_dot_product_attention(q, &k, &v, None);
                out.sum()
            },
            &q_data, 5e-4, 1e-2,
        );
        assert!(ok, "attention gradcheck (query) failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_attention_key() {
        let q_data = Tensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec![1, 2, 4],
        );
        let k_data = Tensor::new(
            vec![0.2, 0.3, 0.1, 0.4, 0.5, 0.1, 0.3, 0.2],
            vec![1, 2, 4],
        );
        let v_data = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 4],
        );

        let (ok, err) = gradcheck(
            &|k| {
                let q = Var::new(q_data.clone(), false);
                let v = Var::new(v_data.clone(), false);
                let (out, _) = scaled_dot_product_attention(&q, k, &v, None);
                out.sum()
            },
            &k_data, 5e-4, 1e-2,
        );
        assert!(ok, "attention gradcheck (key) failed: max_rel_err={}", err);
    }

    #[test]
    fn gradcheck_attention_value() {
        let q_data = Tensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec![1, 2, 4],
        );
        let k_data = Tensor::new(
            vec![0.2, 0.3, 0.1, 0.4, 0.5, 0.1, 0.3, 0.2],
            vec![1, 2, 4],
        );
        let v_data = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 4],
        );

        let (ok, err) = gradcheck(
            &|v| {
                let q = Var::new(q_data.clone(), false);
                let k = Var::new(k_data.clone(), false);
                let (out, _) = scaled_dot_product_attention(&q, &k, v, None);
                out.sum()
            },
            &v_data, 5e-4, 1e-2,
        );
        assert!(ok, "attention gradcheck (value) failed: max_rel_err={}", err);
    }
}
