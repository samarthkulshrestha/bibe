//! Generate synthetic "distal cause" traces where recency heuristics fail.
//!
//! Each object gets several `write`s and `read`s. Exactly one write is causal:
//! the one immediately following a `trigger` event. The victim object later
//! `crash`es (the symptom); its cause is that trigger-preceded write, which is
//! neither the most-recent event, nor the most-recent same-object event, nor
//! the most-recent same-object *write* — so recency, same-object-recency, and
//! same-object-write baselines all fail. The model must learn the relational
//! `trigger -> write` pattern. Sanitizers don't catch this class (no oracle),
//! so labels are injected synthetically.
//!
//! Emits BiBE `.trace` files (with the object column) that
//! `examples/train_real.rs` can train on and score against the baselines.
//!
//! ```text
//! cargo run --example synth_distal_gen -- <count> <out_dir> [seed]
//! ```

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use bibe::data::{serialize_trace, Trace, TraceEvent, TraceLabel};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: synth_distal_gen <count> <out_dir> [seed]");
        std::process::exit(2);
    }
    let count: usize = args[1].parse().expect("count");
    let out_dir = &args[2];
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    std::fs::create_dir_all(out_dir).expect("create out dir");
    let mut rng = StdRng::seed_from_u64(seed);

    let mut anom = 0;
    for i in 0..count {
        let anomalous = rng.random::<bool>();
        if anomalous {
            anom += 1;
        }
        let trace = gen_trace(&mut rng, anomalous);
        std::fs::write(format!("{out_dir}/trace_{i:04}.trace"), serialize_trace(&trace))
            .expect("write trace");
    }
    println!("wrote {count} traces to {out_dir} ({anom} anomalous)");
}

fn event(func: &str, ts: u64, object_id: u32) -> TraceEvent {
    TraceEvent {
        function: func.to_string(),
        timestamp_us: ts,
        call_depth: 0,
        l1_misses: 0,
        l2_misses: 0,
        llc_misses: 0,
        branch_misses: 0,
        object_id,
    }
}

/// One emission step: a run of consecutive `(function, object_id)` events kept
/// adjacent through interleaving (so `trigger`->write stays together).
type Step = Vec<(&'static str, u32)>;

/// Steps for one object. Every object has a `trigger`->`write` step (its one
/// causal-shaped write) plus benign writes/reads; only the victim crashes.
fn object_steps(rng: &mut StdRng, obj: usize, is_victim: bool) -> Vec<Step> {
    let oid = obj as u32 + 1;
    let mut steps: Vec<Step> = vec![vec![("alloc", oid)], vec![("write", oid)], vec![("read", oid)]];
    // The causal-shaped step: a trigger (no object) immediately before a write.
    steps.push(vec![("trigger", 0), ("write", oid)]);
    // Trailing benign writes/reads (distractors after the causal write).
    for _ in 0..rng.random_range(1..=2) {
        let f = if rng.random::<bool>() { "write" } else { "read" };
        steps.push(vec![(f, oid)]);
    }
    steps.push(vec![("read", oid)]);
    if is_victim {
        steps.push(vec![("crash", oid)]);
    }
    steps
}

fn gen_trace(rng: &mut StdRng, anomalous: bool) -> Trace {
    let k = rng.random_range(2..=4);
    let victim = if anomalous { Some(rng.random_range(0..k)) } else { None };

    let mut queues: Vec<Vec<Step>> =
        (0..k).map(|i| object_steps(rng, i, victim == Some(i))).collect();

    let mut events: Vec<TraceEvent> = Vec::new();
    let (mut cause, mut symptom) = (0usize, 0usize);
    let mut ts = 0u64;

    loop {
        let ready: Vec<usize> = (0..k).filter(|&i| !queues[i].is_empty()).collect();
        if ready.is_empty() {
            break;
        }
        if rng.random_range(0..10) < 4 {
            events.push(event("work", ts, 0));
            ts += 1;
        }
        let obj = ready[rng.random_range(0..ready.len())];
        let step = queues[obj].remove(0);
        let is_causal_step = victim == Some(obj) && step.iter().any(|&(f, _)| f == "trigger");
        for (func, oid) in step {
            if is_causal_step && func == "write" {
                cause = events.len(); // the trigger-preceded write is the cause
            }
            if func == "crash" {
                symptom = events.len();
            }
            events.push(event(func, ts, oid));
            ts += 1;
        }
    }

    let label = match victim {
        Some(_) => TraceLabel::Anomalous { root_cause: symptom, cause },
        None => TraceLabel::Normal,
    };
    Trace { events, label }
}
