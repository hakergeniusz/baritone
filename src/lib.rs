//! Baritone — a pathfinding bot for Horizon (a Rust rewrite of Minecraft).
//!
//! This is a Rust port of Baritone (https://github.com/cabaletta/baritone),
//! originally licensed under the GNU Lesser General Public License v3.0 or later.
//!
//! The port inherits the LGPL-3.0-or-later license from the original source.
//!
//! # Architecture
//!
//! - [`goals`]: Pathing goals (GoalXZ, GoalBlock, etc.) and action cost constants.
//! - [`pathing`]: A* search engine (`path_finder`), binary heap open set, and
//!   movement cost calculations (walk, ascend, descend, fall, diagonal, parkour).
//! - [`world`]: World model that queries Horizon's control socket for block
//!   states and player position.

pub mod executor;
pub mod goals;
pub mod pathing;
pub mod world;

pub use executor::execute_path;
pub use goals::{Goal, GoalBlock, GoalNear, GoalTwoBlocks, GoalXZ, GoalYLevel};
pub use pathing::{
    BinaryHeapOpenSet, MoveResult, Moves, Path, PathNode, SearchConfig, path_finder,
};
pub use world::{BlockCache, LiveWorld, World};
