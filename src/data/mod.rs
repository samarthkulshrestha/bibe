pub mod trace;
pub mod parser;

pub use trace::{Trace, TraceEvent, TraceLabel};
pub use parser::{parse_trace, parse_trace_file};
