//! Generate synthetic "distal cause" traces where recency heuristics fail.
//!
//! Each object is allocated, `write`-n (the cause for the victim), then `read`
//! several times (valid, same-object distractors), and the victim finally
//! `crash`es (the symptom). So the most-recent same-object event before the
//! crash is a *read distractor*, not the causal write — recency and
//! same-object-recency baselines cannot find the cause. Sanitizers don't catch
//! this class (no oracle), so labels are injected synthetically.
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

fn gen_trace(rng: &mut StdRng, anomalous: bool) -> Trace {
    let k = rng.random_range(2..=4);
    let victim = if anomalous { Some(rng.random_range(0..k)) } else { None };

    // Per-object op queues. Object `i` uses object id `i + 1`. Each object is
    // allocated, written once (the potential cause), then read a few times; the
    // victim finally crashes.
    let mut queues: Vec<Vec<&str>> = (0..k)
        .map(|i| {
            let reads = rng.random_range(1..=3);
            let mut q = vec!["alloc", "write"];
            q.extend(std::iter::repeat_n("read", reads));
            if victim == Some(i) {
                q.push("crash");
            }
            q
        })
        .collect();

    let mut events: Vec<TraceEvent> = Vec::new();
    let (mut cause, mut symptom) = (0usize, 0usize);
    let mut ts = 0u64;

    loop {
        let ready: Vec<usize> = (0..k).filter(|&i| !queues[i].is_empty()).collect();
        if ready.is_empty() {
            break;
        }
        // Occasional unrelated filler (no object).
        if rng.random_range(0..10) < 4 {
            events.push(event("work", ts, 0));
            ts += 1;
        }
        let obj = ready[rng.random_range(0..ready.len())];
        let func = queues[obj].remove(0);
        if victim == Some(obj) && func == "write" {
            cause = events.len();
        }
        if func == "crash" {
            symptom = events.len();
        }
        events.push(event(func, ts, obj as u32 + 1));
        ts += 1;
    }

    let label = match victim {
        Some(_) => TraceLabel::Anomalous { root_cause: symptom, cause },
        None => TraceLabel::Normal,
    };
    Trace { events, label }
}
