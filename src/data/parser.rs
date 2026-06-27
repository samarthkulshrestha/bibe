use std::path::Path;

use super::trace::{Trace, TraceEvent, TraceLabel};

/// Parse the trace-delimited text format into a [`Trace`].
///
/// ```text
/// # label=anomalous root_cause=2
/// # func  ts_us  depth  l1  l2  llc  branch   (optional column comment)
/// malloc  1034   3      2   0   0    1
/// memcpy  1051   3      88  4   1    0
/// free    1090   3      0   0   0    0
/// ```
///
/// Lines beginning with `#` are headers/comments; the first `# label=...`
/// line sets the trace label. Blank lines are ignored. Every data row must
/// have seven whitespace-separated fields:
/// `function timestamp_us depth l1 l2 llc branch`.
pub fn parse_trace(text: &str) -> Result<Trace, String> {
    let mut label: Option<TraceLabel> = None;
    let mut events = Vec::new();

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            // Only the label header carries meaning; other comments are skipped.
            if let Some(spec) = rest.trim().strip_prefix("label=") {
                label = Some(parse_label(spec, lineno + 1)?);
            }
            continue;
        }

        events.push(parse_event(line, lineno + 1)?);
    }

    let label = label.ok_or_else(|| "trace is missing a '# label=' header".to_string())?;
    Ok(Trace { events, label })
}

fn parse_label(spec: &str, lineno: usize) -> Result<TraceLabel, String> {
    let mut parts = spec.split_whitespace();
    match parts.next() {
        Some("normal") => Ok(TraceLabel::Normal),
        Some("anomalous") => {
            let rc = parts
                .next()
                .and_then(|kv| kv.strip_prefix("root_cause="))
                .ok_or_else(|| format!("line {lineno}: anomalous label needs root_cause="))?;
            let root_cause = rc
                .parse::<usize>()
                .map_err(|e| format!("line {lineno}: bad root_cause '{rc}': {e}"))?;
            Ok(TraceLabel::Anomalous { root_cause })
        }
        other => Err(format!("line {lineno}: unknown label '{other:?}'")),
    }
}

fn parse_event(line: &str, lineno: usize) -> Result<TraceEvent, String> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() != 7 {
        return Err(format!(
            "line {lineno}: expected 7 fields, got {}",
            fields.len()
        ));
    }

    let num = |idx: usize, name: &str| -> Result<u32, String> {
        fields[idx]
            .parse::<u32>()
            .map_err(|e| format!("line {lineno}: bad {name} '{}': {e}", fields[idx]))
    };

    Ok(TraceEvent {
        function: fields[0].to_string(),
        timestamp_us: fields[1]
            .parse::<u64>()
            .map_err(|e| format!("line {lineno}: bad timestamp '{}': {e}", fields[1]))?,
        call_depth: num(2, "depth")?,
        l1_misses: num(3, "l1")?,
        l2_misses: num(4, "l2")?,
        llc_misses: num(5, "llc")?,
        branch_misses: num(6, "branch")?,
    })
}

/// Read and parse a trace file from disk.
pub fn parse_trace_file(path: &Path) -> Result<Trace, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_trace(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NORMAL: &str = "\
# label=normal
# func ts depth l1 l2 llc branch
malloc 1034 3 2 0 0 1
memcpy 1051 3 88 4 1 0
";

    const ANOMALOUS: &str = "\
# label=anomalous root_cause=2

malloc 1000 0 1 0 0 0
free 1010 0 0 0 0 0
use 1020 1 5 2 1 3
";

    #[test]
    fn test_parse_normal_trace() {
        let t = parse_trace(NORMAL).unwrap();
        assert_eq!(t.label, TraceLabel::Normal);
        assert_eq!(t.len(), 2);
        assert_eq!(
            t.events[0],
            TraceEvent {
                function: "malloc".to_string(),
                timestamp_us: 1034,
                call_depth: 3,
                l1_misses: 2,
                l2_misses: 0,
                llc_misses: 0,
                branch_misses: 1,
            }
        );
        assert_eq!(t.events[1].function, "memcpy");
        assert_eq!(t.events[1].l1_misses, 88);
    }

    #[test]
    fn test_parse_anomalous_trace_with_root_cause() {
        let t = parse_trace(ANOMALOUS).unwrap();
        assert_eq!(t.label, TraceLabel::Anomalous { root_cause: 2 });
        assert_eq!(t.len(), 3);
        assert_eq!(t.events[2].function, "use");
    }

    #[test]
    fn test_blank_and_comment_lines_ignored() {
        // ANOMALOUS has a blank line and a comment; only 3 events should parse.
        let t = parse_trace(ANOMALOUS).unwrap();
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn test_missing_label_is_error() {
        let text = "malloc 1000 0 0 0 0 0\n";
        assert!(parse_trace(text).is_err());
    }

    #[test]
    fn test_wrong_column_count_is_error() {
        let text = "# label=normal\nmalloc 1000 0 0\n";
        assert!(parse_trace(text).is_err());
    }

    #[test]
    fn test_non_numeric_field_is_error() {
        let text = "# label=normal\nmalloc abc 0 0 0 0 0\n";
        assert!(parse_trace(text).is_err());
    }

    #[test]
    fn test_anomalous_without_root_cause_is_error() {
        let text = "# label=anomalous\nmalloc 1000 0 0 0 0 0\n";
        assert!(parse_trace(text).is_err());
    }
}
