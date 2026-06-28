//! Convert a captured function-call log + AddressSanitizer report into a BiBE
//! trace (TSV). Part of Stage B (B0).
//!
//! ```text
//! cargo run --example trace_convert -- <trace.log> <asan.txt> <out.trace>
//! ```
//!
//! The log holds `E <function> <timestamp_us> <depth>` lines (from the
//! instrumentation shim). If the ASan report shows a heap error, the symptom is
//! the access function's event and the cause is the free function's event;
//! otherwise the trace is labeled normal.

use std::collections::HashSet;

use bibe::data::{
    collate, extract_windows, parse_trace, serialize_trace, Trace, TraceEvent, TraceLabel,
    Vocabulary,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: trace_convert <trace.log> <asan.txt> <out.trace>");
        std::process::exit(2);
    }
    let (log_path, asan_path, out_path) = (&args[1], &args[2], &args[3]);

    let log = std::fs::read_to_string(log_path).expect("read trace log");
    let asan = std::fs::read_to_string(asan_path).unwrap_or_default();

    let events = parse_events(&log);
    assert!(!events.is_empty(), "no events parsed from {log_path}");
    let names: HashSet<String> = events.iter().map(|e| e.function.clone()).collect();

    let label = if asan.contains("AddressSanitizer") {
        let symptom = first_app_frame(&asan, "of size", &names)
            .expect("could not find symptom function in ASan report");
        let cause = first_app_frame(&asan, "freed by", &names)
            .expect("could not find cause function in ASan report");
        let root_cause = last_index(&events, &symptom).expect("symptom event not in trace");
        let cause = last_index(&events, &cause).expect("cause event not in trace");
        TraceLabel::Anomalous { root_cause, cause }
    } else {
        TraceLabel::Normal
    };

    let trace = Trace { events, label };
    let text = serialize_trace(&trace);
    std::fs::write(out_path, &text).expect("write output trace");

    // Close the loop: the output must parse back AND reach model-input form.
    let reparsed = parse_trace(&text).expect("generated trace failed to parse");
    let vocab = Vocabulary::build(std::slice::from_ref(&reparsed), 1);
    let windows = extract_windows(&reparsed, 64, 64);
    let batch = collate(&windows, &vocab);
    let pipeline = format!(
        "vocab {}, batch [{}, {}]",
        vocab.len(),
        batch.batch,
        batch.seq
    );

    match reparsed.label {
        TraceLabel::Normal => {
            println!("{out_path}: normal, {} events -> {pipeline}", reparsed.len());
        }
        TraceLabel::Anomalous { root_cause, cause } => {
            println!(
                "{out_path}: anomalous, {} events, symptom={} (#{root_cause}), cause={} (#{cause}) -> {pipeline}",
                reparsed.len(),
                reparsed.events[root_cause].function,
                reparsed.events[cause].function,
            );
        }
    }
}

/// Parse `E <function> <ts> <depth>` lines into events (counters unmeasured -> 0).
fn parse_events(log: &str) -> Vec<TraceEvent> {
    log.lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            if parts.next() != Some("E") {
                return None;
            }
            let function = parts.next()?.to_string();
            let timestamp_us = parts.next()?.parse().ok()?;
            let call_depth = parts.next()?.parse().ok()?;
            Some(TraceEvent {
                function,
                timestamp_us,
                call_depth,
                l1_misses: 0,
                l2_misses: 0,
                llc_misses: 0,
                branch_misses: 0,
            })
        })
        .collect()
}

/// First stack frame (after the line containing `header`) whose function name
/// is one of our instrumented functions — skipping libc frames like `free`.
fn first_app_frame(asan: &str, header: &str, names: &HashSet<String>) -> Option<String> {
    let lines: Vec<&str> = asan.lines().collect();
    let start = lines.iter().position(|l| l.contains(header))?;
    for line in &lines[start..] {
        if let Some(idx) = line.find(" in ") {
            let name = line[idx + 4..].split_whitespace().next().unwrap_or("");
            if names.contains(name) {
                return Some(name.to_string());
            }
        }
    }
    None
}

fn last_index(events: &[TraceEvent], name: &str) -> Option<usize> {
    events.iter().rposition(|e| e.function == name)
}
