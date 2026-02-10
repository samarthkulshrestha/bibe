pub mod node;
pub mod backward;
pub mod graph;
pub mod gradcheck;

pub use graph::Var;
pub use node::GradFn;
pub use gradcheck::{numerical_gradient, check_gradient, gradcheck};
