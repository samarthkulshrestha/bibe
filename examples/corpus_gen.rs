//! Generate a corpus of small C programs for real-trace capture.
//!
//! Each program allocates several pointers, each with its own `free_k`/`use_k`
//! call site, and interleaves their operations with `work_*` filler. Clean and
//! buggy programs contain the *same* set of frees and uses — only in a buggy
//! program is exactly one pointer (the victim) used after it is freed, while
//! the others are used correctly (decoy frees). So detection cannot rely on
//! token presence, and attribution must pick the victim's free among several.
//! AddressSanitizer labels each run automatically at capture time.
//!
//! ```text
//! cargo run --example corpus_gen -- <count> <out_dir> [seed]
//! ```

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const WORK_POOL: usize = 6;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: corpus_gen <count> <out_dir> [seed]");
        std::process::exit(2);
    }
    let count: usize = args[1].parse().expect("count");
    let out_dir = &args[2];
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    std::fs::create_dir_all(out_dir).expect("create out dir");
    let mut rng = StdRng::seed_from_u64(seed);

    let mut buggy = 0;
    for i in 0..count {
        let anomalous = rng.random::<bool>();
        if anomalous {
            buggy += 1;
        }
        let src = program(&mut rng, anomalous);
        let path = format!("{out_dir}/prog_{i:04}.c");
        std::fs::write(&path, src).expect("write program");
    }

    println!("wrote {count} programs to {out_dir} ({buggy} buggy, {} clean)", count - buggy);
}

/// Emit a C program with `2..=4` pointers. When `anomalous`, one randomly
/// chosen victim pointer is used after it is freed; the rest are used correctly.
fn program(rng: &mut StdRng, anomalous: bool) -> String {
    let k = rng.random_range(2..=4);
    let victim = if anomalous { Some(rng.random_range(0..k)) } else { None };

    let mut s = String::new();
    s.push_str("#include <stdio.h>\n#include <stdlib.h>\n\n");
    // Provided by the instrumentation shim; records the object (real pointer
    // address) the current function touches.
    s.push_str("void bibe_obj_event(void *p);\n\n");
    // allocate does not emit an object event: for use-after-free attribution
    // the cause is the free, so we link the free/use of an object but not its
    // allocation (which would otherwise be a same-object competitor).
    s.push_str("char *allocate(void) { return (char *)malloc(16); }\n");
    for i in 0..k {
        s.push_str(&format!("void free_{i}(char *p) {{ bibe_obj_event(p); free(p); }}\n"));
        s.push_str(&format!("char use_{i}(char *p) {{ bibe_obj_event(p); return p[0]; }}\n"));
    }
    for w in 0..WORK_POOL {
        s.push_str(&format!(
            "void work_{w}(void) {{ volatile int x = 0; for (int i = 0; i < 3; i++) x += i; }}\n"
        ));
    }

    s.push_str("\nint main(void) {\n    volatile char sink = 0;\n");
    for i in 0..k {
        s.push_str(&format!("    char *ptr_{i} = allocate();\n    ptr_{i}[0] = 'A';\n"));
    }

    // Per-pointer op queue, in required order (true = free). Each object gets
    // several valid uses before its free; the victim adds one use AFTER its
    // free (the bug). So the symptom's same-object events include many valid
    // uses, and identifying the causal free needs object + is-free + order, not
    // just "attend to the same object".
    let mut queues: Vec<Vec<(bool, usize)>> = (0..k)
        .map(|i| {
            let n_uses = rng.random_range(1..=3);
            let mut q: Vec<(bool, usize)> = (0..n_uses).map(|_| (false, i)).collect();
            q.push((true, i)); // free
            if victim == Some(i) {
                q.push((false, i)); // use-after-free
            }
            q
        })
        .collect();

    // Interleave the queues (preserving each pointer's order) with filler.
    loop {
        let ready: Vec<usize> = (0..k).filter(|&i| !queues[i].is_empty()).collect();
        if ready.is_empty() {
            break;
        }
        if rng.random_range(0..10) < 4 {
            s.push_str(&format!("    work_{}();\n", rng.random_range(0..WORK_POOL)));
        }
        let pick = ready[rng.random_range(0..ready.len())];
        let (is_free, idx) = queues[pick].remove(0);
        if is_free {
            s.push_str(&format!("    free_{idx}(ptr_{idx});\n"));
        } else {
            s.push_str(&format!("    sink = use_{idx}(ptr_{idx});\n"));
        }
    }

    s.push_str("    printf(\"%d\\n\", (int)sink);\n    return 0;\n}\n");
    s
}
