//! Generate a corpus of small C programs for real-trace capture.
//!
//! Each program shares the same function vocabulary (a pool of `work_*` filler
//! functions plus allocate/do_free/do_use) and differs only in the filler
//! sequence and whether the use comes after the free. Roughly half are buggy
//! (use-after-free); AddressSanitizer labels them automatically at capture
//! time, so the generator does not record labels.
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

/// Emit a C program. When `anomalous`, the use happens after the free.
fn program(rng: &mut StdRng, anomalous: bool) -> String {
    let mut s = String::new();
    s.push_str("#include <stdio.h>\n#include <stdlib.h>\n\n");
    s.push_str("char *allocate(void) { return (char *)malloc(16); }\n");
    s.push_str("void do_free(char *p) { free(p); }\n");
    s.push_str("char do_use(char *p) { return p[0]; }\n");
    for w in 0..WORK_POOL {
        s.push_str(&format!(
            "void work_{w}(void) {{ volatile int x = 0; for (int i = 0; i < 3; i++) x += i; }}\n"
        ));
    }
    s.push_str("\nint main(void) {\n    char *p = allocate();\n    p[0] = 'A';\n");

    // Filler before, between, and after the two memory operations.
    let filler = |rng: &mut StdRng, s: &mut String| {
        for _ in 0..rng.random_range(1..5) {
            s.push_str(&format!("    work_{}();\n", rng.random_range(0..WORK_POOL)));
        }
    };

    filler(rng, &mut s);
    // Anomalous: free then (later) use. Clean: use then free.
    let (first, second) = if anomalous {
        ("    do_free(p);\n", "    char c = do_use(p);\n")
    } else {
        ("    char c = do_use(p);\n", "    do_free(p);\n")
    };
    s.push_str(first);
    filler(rng, &mut s);
    s.push_str(second);
    filler(rng, &mut s);

    s.push_str("    printf(\"%c\\n\", c);\n    return 0;\n}\n");
    s
}
