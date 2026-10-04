use bibe::data::synthetic::{BugKind, TraceGenerator};

/// `cargo run` shows what BiBE works on: one labeled synthetic trace with a
/// known symptom and a known cause. It does not run the model; attribution
/// numbers only mean something after training, which lives in the examples
/// printed at the end.
fn main() {
    let mut generator = TraceGenerator::new(7);
    let trace = generator.anomalous_trace(BugKind::UseAfterFree);
    let symptom = trace.root_cause().unwrap();
    let cause = trace.cause().unwrap();

    println!("synthetic use-after-free trace: {} events", trace.len());
    println!("  symptom (the crash the detector should flag): event {symptom}");
    println!("  cause (what attribution should point back to): event {cause}\n");

    let lo = cause.saturating_sub(2);
    let hi = (symptom + 3).min(trace.len());
    println!("{:>5}  {:<18} {:>6} {:>7} {:>7} {:>7}", "idx", "function", "depth", "l1", "llc", "branch");
    for i in lo..hi {
        let e = &trace.events[i];
        let tag = if i == symptom {
            "  <- symptom"
        } else if i == cause {
            "  <- cause"
        } else {
            ""
        };
        println!(
            "{:>5}  {:<18} {:>6} {:>7} {:>7} {:>7}{}",
            i, e.function, e.call_depth, e.l1_misses, e.llc_misses, e.branch_misses, tag
        );
    }

    println!("\nordinary work sits between the free and the use, so flagging the");
    println!("crash is not enough: attribution has to reach back past it to the");
    println!("free that caused it.\n");

    println!("to actually train and score a model:");
    println!("  cargo run --release --example train        synthetic end-to-end run");
    println!("  cargo run --release --example train_real   traces captured from real C programs");
    println!("  ./scripts/bench.sh                         the full benchmark matrix");
}
