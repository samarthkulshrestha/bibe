use std::collections::HashMap;

use super::trace::Trace;

/// Padding token / id (reserved).
pub const PAD_TOKEN: &str = "<PAD>";
pub const PAD_ID: usize = 0;
/// Unknown token / id (reserved).
pub const UNK_TOKEN: &str = "<UNK>";
pub const UNK_ID: usize = 1;

/// Maps function names to integer ids, reserving `<PAD>=0` and `<UNK>=1`.
pub struct Vocabulary {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
}

impl Vocabulary {
    /// Build a vocabulary from a corpus of traces, keeping only function names
    /// that occur at least `min_freq` times. Token ids are assigned in sorted
    /// order so the build is deterministic.
    pub fn build(traces: &[Trace], min_freq: usize) -> Self {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for trace in traces {
            for event in &trace.events {
                *counts.entry(event.function.as_str()).or_insert(0) += 1;
            }
        }

        // Sort kept tokens so id assignment is deterministic across builds.
        let mut kept: Vec<&str> = counts
            .iter()
            .filter(|&(_, &c)| c >= min_freq)
            .map(|(&t, _)| t)
            .collect();
        kept.sort_unstable();

        let mut token_to_id = HashMap::from([
            (PAD_TOKEN.to_string(), PAD_ID),
            (UNK_TOKEN.to_string(), UNK_ID),
        ]);
        let mut id_to_token = vec![PAD_TOKEN.to_string(), UNK_TOKEN.to_string()];

        for tok in kept {
            token_to_id.insert(tok.to_string(), id_to_token.len());
            id_to_token.push(tok.to_string());
        }

        Vocabulary { token_to_id, id_to_token }
    }

    /// Encode a function name to its id, or `<UNK>` if not in the vocabulary.
    pub fn encode(&self, token: &str) -> usize {
        self.token_to_id.get(token).copied().unwrap_or(UNK_ID)
    }

    /// Decode an id back to its token.
    pub fn decode(&self, id: usize) -> &str {
        self.id_to_token.get(id).map(|s| s.as_str()).unwrap_or(UNK_TOKEN)
    }

    /// Total number of tokens, including the reserved ones.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::trace::{TraceEvent, TraceLabel};

    fn trace_of(names: &[&str]) -> Trace {
        let events = names
            .iter()
            .map(|n| TraceEvent {
                function: n.to_string(),
                timestamp_us: 0,
                call_depth: 0,
                l1_misses: 0,
                l2_misses: 0,
                llc_misses: 0,
                branch_misses: 0,
                object_id: 0,
            })
            .collect();
        Trace { events, label: TraceLabel::Normal }
    }

    #[test]
    fn test_reserved_tokens() {
        let v = Vocabulary::build(&[], 1);
        assert_eq!(v.decode(PAD_ID), PAD_TOKEN);
        assert_eq!(v.decode(UNK_ID), UNK_TOKEN);
        assert_eq!(v.encode(PAD_TOKEN), PAD_ID);
        assert_eq!(v.encode(UNK_TOKEN), UNK_ID);
    }

    #[test]
    fn test_known_token_roundtrips() {
        let v = Vocabulary::build(&[trace_of(&["malloc", "free", "malloc"])], 1);
        let id = v.encode("malloc");
        assert!(id >= 2, "real tokens come after the reserved ids");
        assert_eq!(v.decode(id), "malloc");
    }

    #[test]
    fn test_unknown_token_maps_to_unk() {
        let v = Vocabulary::build(&[trace_of(&["malloc"])], 1);
        assert_eq!(v.encode("never_seen"), UNK_ID);
    }

    #[test]
    fn test_min_freq_filters_rare_tokens() {
        // "rare" appears once, "common" three times; min_freq=2 drops "rare".
        let v = Vocabulary::build(
            &[trace_of(&["common", "common", "common", "rare"])],
            2,
        );
        assert_ne!(v.encode("common"), UNK_ID);
        assert_eq!(v.encode("rare"), UNK_ID);
    }

    #[test]
    fn test_len_includes_reserved_and_kept_tokens() {
        let v = Vocabulary::build(&[trace_of(&["a", "b", "c"])], 1);
        // 2 reserved + 3 functions
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_build_is_deterministic() {
        let traces = [trace_of(&["b", "a", "c", "a"])];
        let v1 = Vocabulary::build(&traces, 1);
        let v2 = Vocabulary::build(&traces, 1);
        assert_eq!(v1.encode("a"), v2.encode("a"));
        assert_eq!(v1.encode("b"), v2.encode("b"));
    }
}
