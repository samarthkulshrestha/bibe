use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::autograd::Var;

const MAGIC: &[u8; 4] = b"BIBE";
const VERSION: u32 = 1;

/// Serialize parameter tensors to a binary checkpoint.
///
/// Layout: magic `BIBE`, a `u32` version, a `u32` tensor count, then for each
/// tensor a `u32` rank, that many `u32` dimensions, and the row-major `f32`
/// data (all little-endian). Only parameter values are stored; the model must
/// be reconstructed from its config before loading.
pub fn save_parameters(path: &Path, params: &[Var]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(params.len() as u32).to_le_bytes())?;

    for p in params {
        let t = p.tensor();
        let shape = t.shape();
        w.write_all(&(shape.len() as u32).to_le_bytes())?;
        for &dim in shape {
            w.write_all(&(dim as u32).to_le_bytes())?;
        }
        for &x in &t.data {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    w.flush()
}

/// Load parameter tensors from a checkpoint into `params` in order.
///
/// `params` must match the saved checkpoint in count and per-tensor shape
/// (i.e. the same model config); otherwise an error is returned.
pub fn load_parameters(path: &Path, params: &[Var]) -> io::Result<()> {
    let mut r = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("not a BiBE checkpoint (bad magic)"));
    }
    if read_u32(&mut r)? != VERSION {
        return Err(invalid("unsupported checkpoint version"));
    }

    let count = read_u32(&mut r)? as usize;
    if count != params.len() {
        return Err(invalid(&format!(
            "checkpoint has {count} tensors, model has {}",
            params.len()
        )));
    }

    for p in params {
        let ndim = read_u32(&mut r)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u32(&mut r)? as usize);
        }

        if shape != p.tensor().shape() {
            return Err(invalid(&format!(
                "shape mismatch: checkpoint {shape:?} vs model {:?}",
                p.tensor().shape()
            )));
        }

        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            data.push(f32::from_le_bytes(buf));
        }

        p.with_data_mut(|t| t.data = data);
    }

    Ok(())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("bibe_ckpt_{name}.bin"))
    }

    #[test]
    fn test_round_trip_restores_values() {
        let saved = vec![
            Var::new(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]), true),
            Var::new(Tensor::new(vec![4.0, 5.0, 6.0, 7.0], vec![2, 2]), true),
        ];
        let path = tmp("round_trip");
        save_parameters(&path, &saved).unwrap();

        // Fresh params of matching shape but different values.
        let loaded = vec![
            Var::new(Tensor::zeros(&[3]), true),
            Var::new(Tensor::zeros(&[2, 2]), true),
        ];
        load_parameters(&path, &loaded).unwrap();

        assert_eq!(loaded[0].tensor().data, vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded[1].tensor().data, vec![4.0, 5.0, 6.0, 7.0]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_shape_mismatch_is_error() {
        let saved = vec![Var::new(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]), true)];
        let path = tmp("shape_mismatch");
        save_parameters(&path, &saved).unwrap();

        let wrong = vec![Var::new(Tensor::zeros(&[4]), true)];
        assert!(load_parameters(&path, &wrong).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_count_mismatch_is_error() {
        let saved = vec![Var::new(Tensor::new(vec![1.0], vec![1]), true)];
        let path = tmp("count_mismatch");
        save_parameters(&path, &saved).unwrap();

        let two = vec![
            Var::new(Tensor::zeros(&[1]), true),
            Var::new(Tensor::zeros(&[1]), true),
        ];
        assert!(load_parameters(&path, &two).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bad_magic_is_error() {
        let path = tmp("bad_magic");
        std::fs::write(&path, b"not a checkpoint at all").unwrap();
        let params = vec![Var::new(Tensor::zeros(&[1]), true)];
        assert!(load_parameters(&path, &params).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_model_checkpoint_reproduces_output() {
        use crate::autograd::Var;
        use crate::data::N_AUX;
        use crate::model::{BibeConfig, BibeModel};

        let config = BibeConfig {
            vocab_size: 12,
            d_model: 16,
            num_heads: 2,
            d_ff: 32,
            num_layers: 2,
            n_aux: N_AUX,
            max_len: 16,
            dropout_p: 0.0,
        };
        let ids = vec![1usize, 2, 3, 4];
        let aux = Var::new(Tensor::randn(&[1, 4, N_AUX]), false);

        let trained = BibeModel::new(&config);
        let path = tmp("model_round_trip");
        save_parameters(&path, &trained.parameters()).unwrap();

        // A fresh model has different random weights, hence different output...
        let fresh = BibeModel::new(&config);
        let before = fresh.forward(&ids, &aux, 1, 4, false).anomaly_scores.tensor().data;
        let target = trained.forward(&ids, &aux, 1, 4, false).anomaly_scores.tensor().data;
        assert!(before.iter().zip(&target).any(|(a, b)| (a - b).abs() > 1e-6));

        // ...until we load the saved parameters into it.
        load_parameters(&path, &fresh.parameters()).unwrap();
        let after = fresh.forward(&ids, &aux, 1, 4, false).anomaly_scores.tensor().data;
        for (a, t) in after.iter().zip(&target) {
            assert!((a - t).abs() < 1e-6, "loaded model output differs: {a} vs {t}");
        }
        let _ = std::fs::remove_file(&path);
    }
}
