use crate::autograd::Var;
use crate::tensor::Tensor;

/// Build a sinusoidal positional encoding table of shape `[max_len, d_model]`
/// (Vaswani et al., 2017):
///
/// ```text
/// PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
/// ```
///
/// Even dimensions use sine, odd dimensions cosine, with geometrically
/// increasing wavelengths. The encoding is fixed (not learned).
pub fn sinusoidal_encoding(max_len: usize, d_model: usize) -> Tensor {
    let mut data = vec![0.0f32; max_len * d_model];
    for pos in 0..max_len {
        for i in 0..d_model {
            // Both dims of a sin/cos pair share the same frequency: the pair
            // index is i/2, so the exponent uses 2*(i/2).
            let exponent = (2 * (i / 2)) as f32 / d_model as f32;
            let angle = pos as f32 / 10000f32.powf(exponent);
            data[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Tensor::new(data, vec![max_len, d_model])
}

/// Precomputed sinusoidal positional encodings, sliced per sequence length.
pub struct PositionalEncoding {
    encoding: Tensor,
    d_model: usize,
}

impl PositionalEncoding {
    /// Precompute encodings up to `max_len` positions.
    pub fn new(max_len: usize, d_model: usize) -> Self {
        PositionalEncoding {
            encoding: sinusoidal_encoding(max_len, d_model),
            d_model,
        }
    }

    /// Return the `[seq_len, d_model]` encoding prefix as a non-trainable
    /// variable, ready to broadcast-add onto a `[batch, seq_len, d_model]`
    /// embedding.
    pub fn forward(&self, seq_len: usize) -> Var {
        let max_len = self.encoding.shape()[0];
        assert!(seq_len <= max_len, "seq_len {seq_len} exceeds max_len {max_len}");
        let slice = self.encoding.data[..seq_len * self.d_model].to_vec();
        Var::new(Tensor::new(slice, vec![seq_len, self.d_model]), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let pe = sinusoidal_encoding(16, 8);
        assert_eq!(pe.shape(), &[16, 8]);
    }

    #[test]
    fn test_first_position_is_sin0_cos0() {
        // pos = 0: sin(0)=0 on even dims, cos(0)=1 on odd dims.
        let pe = sinusoidal_encoding(4, 6);
        let row0 = &pe.data[..6];
        assert_eq!(row0, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_known_values_at_position_one() {
        // pos = 1, dim 0: sin(1/10000^0) = sin(1) ≈ 0.84147
        //          dim 1: cos(1)            ≈ 0.54030
        let pe = sinusoidal_encoding(4, 4);
        let row1 = &pe.data[4..8];
        assert!((row1[0] - 1.0_f32.sin()).abs() < 1e-5);
        assert!((row1[1] - 1.0_f32.cos()).abs() < 1e-5);
    }

    #[test]
    fn test_values_bounded() {
        let pe = sinusoidal_encoding(50, 16);
        assert!(pe.data.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_positions_are_distinct() {
        let pe = sinusoidal_encoding(10, 8);
        let row1 = &pe.data[8..16];
        let row2 = &pe.data[16..24];
        let differs = row1.iter().zip(row2).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "different positions should have different encodings");
    }

    #[test]
    fn test_struct_forward_slices_prefix() {
        let pe = PositionalEncoding::new(32, 8);
        let v = pe.forward(5);
        assert_eq!(v.tensor().shape(), &[5, 8]);
        assert!(!v.requires_grad(), "positional encoding must be constant");
        // Prefix matches the full table.
        let full = sinusoidal_encoding(32, 8);
        assert_eq!(v.tensor().data, full.data[..5 * 8].to_vec());
    }
}
