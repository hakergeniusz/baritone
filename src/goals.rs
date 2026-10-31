//! Pathing goals and action cost constants.
//!
//! Ported from baritone's `baritone.api.pathing.goals` and
//! `baritone.api.pathing.movement.ActionCosts`.
//!
//! A [`Goal`] defines a target for the pathfinder. Each goal provides:
//! - `is_in_goal(x, y, z)` — whether a position satisfies the goal.
//! - `heuristic(x, y, z)` — an *admissible* estimate of cost-to-goal (in ticks).
//!   Must never overestimate, so A* finds optimal paths.

use std::sync::OnceLock;

/// Movement cost constants and precomputations.
///
/// Costs are measured roughly in game ticks (20 ticks/second).
/// Source: baritone's `ActionCosts.java` interface.
pub mod action_costs {
    use super::OnceLock;

    /// Walking one horizontal block on flat ground.
    pub const WALK_ONE_BLOCK_COST: f64 = 20.0 / 4.317;
    /// Walking through water (no sprint).
    pub const WALK_ONE_IN_WATER_COST: f64 = 20.0 / 2.2;
    /// Sprint-swimming cost.
    pub const SWIM_ONE_BLOCK_COST: f64 = 20.0 / 2.86;
    /// Walking on soul sand — half speed.
    pub const WALK_ONE_OVER_SOUL_SAND_COST: f64 = WALK_ONE_BLOCK_COST * 2.0;
    /// Climbing a ladder upward.
    pub const LADDER_UP_ONE_COST: f64 = 20.0 / 2.35;
    /// Climbing a ladder downward.
    pub const LADDER_DOWN_ONE_COST: f64 = 20.0 / 3.0;
    /// Sneaking one block.
    pub const SNEAK_ONE_BLOCK_COST: f64 = 20.0 / 1.3;
    /// Sprinting one block.
    pub const SPRINT_ONE_BLOCK_COST: f64 = 20.0 / 5.612;
    /// Sprint vs walk ratio.
    pub const SPRINT_MULTIPLIER: f64 = SPRINT_ONE_BLOCK_COST / WALK_ONE_BLOCK_COST;
    /// Cost to walk off a block edge (0.8 block distance).
    pub const WALK_OFF_BLOCK_COST: f64 = WALK_ONE_BLOCK_COST * 0.8;
    /// Cost to recenter after a 1-block fall.
    pub const CENTER_AFTER_FALL_COST: f64 = WALK_ONE_BLOCK_COST - WALK_OFF_BLOCK_COST;
    /// "Infinite" cost — unreachable. Not MAX to avoid overflow when added.
    pub const COST_INF: f64 = 1_000_000.0;

    /// Fall cost table, lazily initialized (indexed by fall distance in blocks).
    static FALL_N_BLOCKS_COST: OnceLock<Vec<f64>> = OnceLock::new();

    fn fall_table() -> &'static Vec<f64> {
        FALL_N_BLOCKS_COST.get_or_init(|| {
            let mut v = Vec::with_capacity(4097);
            v.push(0.0);
            for i in 1..=4096 {
                v.push(distance_to_ticks(i as f64));
            }
            v
        })
    }

    /// Cost of falling `n` blocks (integer).
    pub fn fall_n_blocks_cost(n: usize) -> f64 {
        fall_table()[n.min(4096)]
    }

    /// Cost of falling 1.25 blocks (jump apex).
    pub fn fall_1_25_blocks_cost() -> f64 {
        distance_to_ticks(1.25)
    }

    /// Cost of falling 0.25 blocks.
    pub fn fall_0_25_blocks_cost() -> f64 {
        distance_to_ticks(0.25)
    }

    /// Jump cost = cost to ascend 1 block.
    pub fn jump_one_block_cost() -> f64 {
        fall_1_25_blocks_cost() - fall_0_25_blocks_cost()
    }

    /// Velocity (fall distance per tick) under gravity after `ticks` ticks,
    /// using Minecraft's drag formula `v = (0.98^ticks - 1) * -3.92`.
    fn velocity(ticks: i32) -> f64 {
        let mut p: f64 = 1.0;
        for _ in 0..ticks {
            p *= 0.98;
        }
        (p - 1.0) * -3.92
    }

    /// Convert a fall distance (in blocks) to the number of game ticks it
    /// takes, integrating velocity over time. Ported from `ActionCosts.distanceToTicks`.
    pub fn distance_to_ticks(distance: f64) -> f64 {
        if distance == 0.0 {
            return 0.0;
        }
        let mut tmp = distance;
        let mut tick_count: i32 = 0;
        loop {
            let fall_distance = velocity(tick_count);
            if tmp <= fall_distance {
                return tick_count as f64 + tmp / fall_distance;
            }
            tmp -= fall_distance;
            tick_count += 1;
        }
    }
}

pub use action_costs::*;

/// A pathing goal. Ported from `baritone.api.pathing.goals.Goal`.
pub trait Goal: Send + Sync {
    /// Whether the given position satisfies the goal.
    fn is_in_goal(&self, x: i32, y: i32, z: i32) -> bool;

    /// Admissible heuristic estimate of cost (in ticks) to reach the goal
    /// from (x, y, z). Must never overestimate.
    fn heuristic(&self, x: i32, y: i32, z: i32) -> f64;

    /// The heuristic value evaluated *at* a goal-satisfying position.
    fn heuristic_at_goal(&self) -> f64 {
        0.0
    }
}

/// Goal: stand on a specific X and Z (any Y). Useful for long-distance pathing.
/// Ported from `GoalXZ`.
pub struct GoalXZ {
    pub x: i32,
    pub z: i32,
}

impl GoalXZ {
    pub fn new(x: i32, z: i32) -> Self {
        GoalXZ { x, z }
    }

    /// Combined pythagorean/manhattan heuristic used by baritone.
    fn calculate(x_diff: f64, z_diff: f64) -> f64 {
        use action_costs::WALK_ONE_BLOCK_COST;
        const SQRT_2: f64 = 1.4142135623730951;

        let x = x_diff.abs();
        let z = z_diff.abs();
        let (straight, diagonal) = if x < z {
            (z - x, x)
        } else {
            (x - z, z)
        };
        let diagonal = diagonal * SQRT_2;
        (diagonal + straight) * WALK_ONE_BLOCK_COST
    }
}

impl Goal for GoalXZ {
    fn is_in_goal(&self, x: i32, _y: i32, z: i32) -> bool {
        x == self.x && z == self.z
    }

    fn heuristic(&self, x: i32, _y: i32, z: i32) -> f64 {
        let x_diff = (x - self.x) as f64;
        let z_diff = (z - self.z) as f64;
        Self::calculate(x_diff, z_diff)
    }
}

/// Goal: stand on a specific block (X, Y, Z).
/// Ported from `GoalBlock`.
pub struct GoalBlock {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl GoalBlock {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        GoalBlock { x, y, z }
    }
}

impl Goal for GoalBlock {
    fn is_in_goal(&self, x: i32, y: i32, z: i32) -> bool {
        x == self.x && y == self.y && z == self.z
    }

    fn heuristic(&self, x: i32, y: i32, z: i32) -> f64 {
        use action_costs::WALK_ONE_BLOCK_COST;
        const SQRT_2: f64 = 1.4142135623730951;
        let x_diff = (x - self.x) as f64;
        let y_diff = (y - self.y) as f64;
        let z_diff = (z - self.z) as f64;
        let x = x_diff.abs();
        let y = y_diff.abs();
        let z = z_diff.abs();

        let (straight, diag) = if x < z {
            (z - x, x)
        } else {
            (x - z, z)
        };
        let vert = if y > 0.0 {
            y * action_costs::jump_one_block_cost()
        } else {
            action_costs::fall_n_blocks_cost((-y) as usize)
        };
        (diag * SQRT_2 + straight + vert) * WALK_ONE_BLOCK_COST
    }
}

/// Goal: get within a radius of a block position.
/// Ported from `GoalNear`.
pub struct GoalNear {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub range: f64,
}

impl Goal for GoalNear {
    fn is_in_goal(&self, x: i32, y: i32, z: i32) -> bool {
        let dx = (x - self.x) as f64;
        let dy = (y - self.y) as f64;
        let dz = (z - self.z) as f64;
        dx * dx + dy * dy + dz * dz <= self.range * self.range
    }

    fn heuristic(&self, x: i32, y: i32, z: i32) -> f64 {
        let dx = (x - self.x) as f64;
        let dy = (y - self.y) as f64;
        let dz = (z - self.z) as f64;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        (dist - self.range).max(0.0) * action_costs::WALK_ONE_BLOCK_COST
    }
}

/// Goal: reach a given Y level.
/// Ported from `GoalYLevel`.
pub struct GoalYLevel {
    pub y: i32,
}

impl Goal for GoalYLevel {
    fn is_in_goal(&self, _x: i32, y: i32, _z: i32) -> bool {
        y == self.y
    }

    fn heuristic(&self, _x: i32, y: i32, _z: i32) -> f64 {
        let y_diff = (y - self.y) as f64;
        if y_diff > 0.0 {
            y_diff * action_costs::jump_one_block_cost()
        } else {
            action_costs::fall_n_blocks_cost((-y_diff) as usize)
        }
    }
}

/// Goal: stand at foot or eye level of a 2-block-tall position.
/// Ported from `GoalTwoBlocks`.
pub struct GoalTwoBlocks {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Goal for GoalTwoBlocks {
    fn is_in_goal(&self, x: i32, y: i32, z: i32) -> bool {
        x == self.x && z == self.z && (y == self.y || y == self.y - 1)
    }

    fn heuristic(&self, x: i32, y: i32, z: i32) -> f64 {
        let g_low = GoalBlock::new(self.x, self.y, self.z);
        let g_high = GoalBlock::new(self.x, self.y - 1, self.z);
        g_low.heuristic(x, y, z).min(g_high.heuristic(x, y, z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_xz() {
        let goal = GoalXZ::new(10, 10);
        assert!(goal.is_in_goal(10, 64, 10));
        assert!(!goal.is_in_goal(11, 64, 10));
        let h = goal.heuristic(0, 64, 0);
        assert!(h > 0.0);
        assert!(!h.is_nan());
    }

    #[test]
    fn test_goal_block() {
        let goal = GoalBlock::new(10, 64, 10);
        assert!(goal.is_in_goal(10, 64, 10));
        assert!(!goal.is_in_goal(10, 65, 10));
    }

    #[test]
    fn test_distance_to_ticks() {
        let c = action_costs::distance_to_ticks(1.0);
        assert!(c > 0.0 && c < 10.0);
    }

    #[test]
    fn test_fall_costs() {
        assert_eq!(action_costs::fall_n_blocks_cost(0), 0.0);
        let c1 = action_costs::fall_n_blocks_cost(1);
        let c3 = action_costs::fall_n_blocks_cost(3);
        assert!(c1 > 0.0 && c1 < c3);
        // Falling more blocks costs more.
        assert!(c3 > c1);
    }
}
