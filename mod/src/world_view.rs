//! The pathfinder's view of the live world, fed by the game through a block
//! reader. Pure Rust: the helper (`tick.rs`) hands in a closure that reads
//! the game's own chunks, and this module caches the answers for the search.
//!
//! Mirrors the standalone port's `world::LiveWorld` (same `get_block`
//! signature the movement costs were written against), with the cache filled
//! lazily instead of prefetched over the control socket.

use std::cell::RefCell;
use std::collections::HashMap;

/// Reads one block's name (`minecraft:stone`) from the game. Unloaded
/// chunks read as `minecraft:air`, which the planner treats as passable but
/// not walkable: paths reach into unloaded terrain only as far as there is
/// ground, and the bot re-paths as more of the world streams in.
pub type BlockReader<'a> = dyn Fn(i32, i32, i32) -> String + 'a;

/// A cached world for one pathfinding run. The game's blocks cannot change
/// during a tick, so a fresh `LiveWorld` per plan is always consistent.
pub struct LiveWorld<'a> {
    reader: &'a BlockReader<'a>,
    cache: RefCell<HashMap<(i32, i32, i32), String>>,
}

impl<'a> LiveWorld<'a> {
    pub fn new(reader: &'a BlockReader<'a>) -> Self {
        LiveWorld {
            reader,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// The block's name at a position, from the cache or the game.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> String {
        let key = (x, y, z);
        if let Some(block) = self.cache.borrow().get(&key) {
            return block.clone();
        }
        let block = (self.reader)(x, y, z);
        self.cache.borrow_mut().insert(key, block.clone());
        block
    }

    /// The block's name without caching it. A `/mine` scan reads tens of
    /// thousands of positions once; caching them would pin a stale picture
    /// of the world that the next plan would trust after the mining has
    /// changed the terrain.
    pub fn peek(&self, x: i32, y: i32, z: i32) -> String {
        let key = (x, y, z);
        if let Some(block) = self.cache.borrow().get(&key) {
            return block.clone();
        }
        (self.reader)(x, y, z)
    }

    /// How many positions the search has read this run.
    pub fn cached(&self) -> usize {
        self.cache.borrow().len()
    }
}
