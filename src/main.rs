use bibe::tensor::Tensor;
use bibe::tensor::stability::{stable_softmax, logsumexp, has_nan, has_inf, all_finite};

fn main() {
    // Softmax on safe values
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let sm = stable_softmax(&x, 1);
    println!("softmax(x, dim=1):");
    for i in 0..2 {
        let row_sum: f32 = (0..3).map(|j| sm.get(&[i, j])).sum();
        println!("  row {}: [{:.4}, {:.4}, {:.4}] sum={:.6}",
            i, sm.get(&[i, 0]), sm.get(&[i, 1]), sm.get(&[i, 2]), row_sum);
    }

    // Softmax on large values (overflow test)
    let big = Tensor::new(vec![1000.0, 1000.1, 1000.2], vec![1, 3]);
    let sm_big = stable_softmax(&big, 1);
    println!("\nsoftmax([1000.0, 1000.1, 1000.2]):");
    println!("  [{:.4}, {:.4}, {:.4}]",
        sm_big.get(&[0, 0]), sm_big.get(&[0, 1]), sm_big.get(&[0, 2]));
    println!("  all_finite: {}", all_finite(&sm_big));
    println!("  has_nan: {}", has_nan(&sm_big));
    println!("  has_inf: {}", has_inf(&sm_big));

    // Logsumexp
    let lse = logsumexp(&x, 1);
    println!("\nlogsumexp(x, dim=1): [{:.4}, {:.4}]",
        lse.get(&[0]), lse.get(&[1]));
}
