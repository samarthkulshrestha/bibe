use bibe::tensor::Tensor;
use bibe::autograd::Var;
use bibe::nn::Linear;
use bibe::nn::{relu, sigmoid, gelu};

fn main() {
    println!("=== Linear Layer ===\n");

    let layer = Linear::new(4, 3, true);
    let x = Var::new(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]),
        true,
    );

    let y = layer.forward(&x);
    let yt = y.tensor();
    println!("input:  [2, 4]");
    println!("output: {:?}", yt.shape());
    println!("  row 0: [{:.4}, {:.4}, {:.4}]", yt.get(&[0, 0]), yt.get(&[0, 1]), yt.get(&[0, 2]));
    println!("  row 1: [{:.4}, {:.4}, {:.4}]", yt.get(&[1, 0]), yt.get(&[1, 1]), yt.get(&[1, 2]));

    // backward
    let loss = y.sum();
    loss.backward();
    let xg = x.grad().unwrap();
    let wg = layer.weight.grad().unwrap();
    let bg = layer.bias.as_ref().unwrap().grad().unwrap();
    println!("\nafter backward:");
    println!("  input grad shape:  {:?}", xg.shape());
    println!("  weight grad shape: {:?}", wg.shape());
    println!("  bias grad:         [{:.4}, {:.4}, {:.4}]", bg.data[0], bg.data[1], bg.data[2]);

    println!("\n=== Activation Functions ===\n");

    let a = Var::new(
        Tensor::new(vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], vec![7]),
        true,
    );

    let r = relu(&a).tensor();
    let s = sigmoid(&a).tensor();
    let g = gelu(&a).tensor();

    println!("{:<6} {:>8} {:>8} {:>8}", "x", "relu", "sigmoid", "gelu");
    println!("{}", "-".repeat(38));
    for i in 0..7 {
        println!(
            "{:<6.1} {:>8.4} {:>8.4} {:>8.4}",
            a.tensor().data[i], r.data[i], s.data[i], g.data[i]
        );
    }

    // gradient through gelu
    let g_var = gelu(&a);
    let loss = g_var.sum();
    loss.backward();
    let ag = a.grad().unwrap();
    println!("\ngelu gradients:");
    print!("  [");
    for i in 0..7 {
        if i > 0 { print!(", "); }
        print!("{:.4}", ag.data[i]);
    }
    println!("]");

    println!("\n=== Chain: Linear -> GeLU -> Linear ===\n");

    let l1 = Linear::new(4, 8, true);
    let l2 = Linear::new(8, 2, true);

    let input = Var::new(Tensor::randn(&[3, 4]), true);
    let h = gelu(&l1.forward(&input));
    let out = l2.forward(&h);
    let loss = out.sum();

    println!("input:  {:?}", input.tensor().shape());
    println!("hidden: {:?} (after gelu)", h.tensor().shape());
    println!("output: {:?}", out.tensor().shape());
    println!("loss:   {:.4}", loss.tensor().data[0]);

    loss.backward();
    println!("\nall gradients computed:");
    println!("  input grad:     {:?}", input.grad().unwrap().shape());
    println!("  l1.weight grad: {:?}", l1.weight.grad().unwrap().shape());
    println!("  l1.bias grad:   {:?}", l1.bias.as_ref().unwrap().grad().unwrap().shape());
    println!("  l2.weight grad: {:?}", l2.weight.grad().unwrap().shape());
    println!("  l2.bias grad:   {:?}", l2.bias.as_ref().unwrap().grad().unwrap().shape());
}
