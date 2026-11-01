//! A* pathfinding core, ported from baritone's `baritone.pathing.calc`.
//!
//! The search operates over integer block positions. For each node position it
//! expands all reachable neighbour positions via the [`Moves`] enum, computing
//! movement costs from the [`crate::world::LiveWorld`] (which queries Horizon
//! for block states).

pub mod movement;
pub mod search;

pub use movement::{MoveResult, Moves};
pub use search::{BinaryHeapOpenSet, Path, PathNode, SearchConfig, path_finder};
