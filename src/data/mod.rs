pub mod trace;
pub mod parser;
pub mod vocab;
pub mod normalize;
pub mod window;
pub mod loader;
pub mod synthetic;

pub use trace::{Trace, TraceEvent, TraceLabel};
pub use parser::{parse_trace, parse_trace_file, serialize_trace};
pub use vocab::Vocabulary;
pub use normalize::{aux_features, N_AUX};
pub use window::{extract_windows, TraceWindow};
pub use loader::{collate, Batch, DataLoader};
pub use synthetic::{BugKind, GeneratorConfig, TraceGenerator};
