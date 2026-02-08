// use std::env;
// use std::error::Error;

fn main() {
    let rn = bibe::tensor::Tensor::randn(&[2, 2]);
    let rn_t = rn.t();
    let flat = rn.reshape(&[rn.data.len()]);

    println!("rn = {:#?}", rn);
    println!("rn_0 = {:#?}", rn[[0, 0]]);
    println!("rn_t = {:#?}", rn_t);
    println!("flat = {:#?}", flat);
}
