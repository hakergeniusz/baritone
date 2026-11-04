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
    /// Placing one block (aim at the support's face, click use): a click
    /// lands the same tick and the block appears the next, with a little
    /// aim-settling slack.
    pub const PLACE_BLOCK_COST: f64 = 4.0;
    /// Swimming up one block in water. Baritone never prices vertical
    /// swimming (`MovementDescend` leaves a "TODO incorporate water swim
    /// up cost?" where it would belong) and so cannot escape deep water;
    /// the port adds the move and prices it at the in-water walk cost —
    /// a surfaced swim-up takes about a second.
    pub const SWIM_UP_ONE_BLOCK_COST: f64 = 20.0;
    /// "Infinite" cost — unreachable. Not MAX to avoid overflow when added.
    pub const COST_INF: f64 = 1_000_000.0;
    /// Deepest unprotected fall (no water, no bucket, no ladder) the planner
    /// will accept onto solid ground. Baritone's `maxFallHeightNoWater`
    /// setting: falls deeper than this deal too much damage. The descend
    /// move allows one extra block (`+ 1` in MovementDescend).
    pub const MAX_FALL_HEIGHT_NO_WATER: i32 = 3;

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
    /// Ported from `GoalXZ.calculate` (baritone reads the `costHeuristic`
    /// setting here; it defaults to `WALK_ONE_BLOCK_COST`).
    pub fn calculate(x_diff: f64, z_diff: f64) -> f64 {
        use action_costs::WALK_ONE_BLOCK_COST;
        use std::f64::consts::SQRT_2;

        let x = x_diff.abs();
        let z = z_diff.abs();
        let (straight, diagonal) = if x < z { (z - x, x) } else { (x - z, z) };
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

    /// Ported from `GoalBlock.calculate`: vertical cost via `GoalYLevel`
    /// plus horizontal cost via `GoalXZ`. The vertical component is *not*
    /// scaled by the walk cost — scaling it would overestimate climbing and
    /// make the heuristic inadmissible.
    pub fn calculate(x_diff: f64, y_diff: i32, z_diff: f64) -> f64 {
        GoalYLevel::calculate(0, y_diff) + GoalXZ::calculate(x_diff, z_diff)
    }
}

impl Goal for GoalBlock {
    fn is_in_goal(&self, x: i32, y: i32, z: i32) -> bool {
        x == self.x && y == self.y && z == self.z
    }

    fn heuristic(&self, x: i32, y: i32, z: i32) -> f64 {
        Self::calculate((x - self.x) as f64, y - self.y, (z - self.z) as f64)
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
        // Java adds a half-block grace (`GoalNear.isInRange` builds
        // `(range + 0.5)²`): the goal reads the BLOCK the feet stand in,
        // whose center can sit half a block outside the bare radius.
        // Without it a mine target three under the player reads as out
        // of reach (Euclidean 11 > 9) and the bot plans a descend-walk
        // instead of digging straight toward it.
        let range_sq = (self.range + 0.5) * (self.range + 0.5);
        dx * dx + dy * dy + dz * dz <= range_sq
    }

    /// Ported verbatim from `GoalNear.heuristic`: baritone reuses the
    /// GoalBlock estimate to the center point during search (this is not
    /// strictly admissible for range > 0, but it is what baritone does).
    fn heuristic(&self, x: i32, y: i32, z: i32) -> f64 {
        GoalBlock::calculate((x - self.x) as f64, y - self.y, (z - self.z) as f64)
    }
}

/// Goal: reach a given Y level.
/// Ported from `GoalYLevel`.
pub struct GoalYLevel {
    pub y: i32,
}

impl GoalYLevel {
    /// Ported from `GoalYLevel.calculate(goalY, currentY)`. Descending uses
    /// half the cost of a 2-block fall per level; ascending uses the jump
    /// cost per level. Both are per-level lower bounds, keeping the
    /// heuristic admissible.
    pub fn calculate(goal_y: i32, current_y: i32) -> f64 {
        if current_y > goal_y {
            // need to descend
            return action_costs::fall_n_blocks_cost(2) / 2.0 * (current_y - goal_y) as f64;
        }
        if current_y < goal_y {
            // need to ascend
            return (goal_y - current_y) as f64 * action_costs::jump_one_block_cost();
        }
        0.0
    }
}

impl Goal for GoalYLevel {
    fn is_in_goal(&self, _x: i32, y: i32, _z: i32) -> bool {
        y == self.y
    }

    fn heuristic(&self, _x: i32, y: i32, _z: i32) -> f64 {
        Self::calculate(self.y, y)
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
        let x_diff = x - self.x;
        let y_diff = y - self.y;
        let z_diff = z - self.z;
        // Ported verbatim from GoalTwoBlocks.heuristic: the y difference is
        // folded so that standing one block below the target counts as 0.
        GoalBlock::calculate(
            x_diff as f64,
            if y_diff < 0 { y_diff + 1 } else { y_diff },
            z_diff as f64,
        )
    }
}

/// Goal: any position on a straight axis-aligned segment at a fixed floor
/// level — the shape of baritone's `#tunnel` command, which digs a 1-wide,
/// 2-tall corridor from where you stand toward where you face.
pub struct GoalCorridor {
    pub x0: i32,
    pub z0: i32,
    pub x1: i32,
    pub z1: i32,
    pub y: i32,
}

impl GoalCorridor {
    /// The segment's far end must share either the start's X or Z.
    fn axis(&self) -> (i32, i32, i32, i32) {
        // (from, to, low, high) along the moving axis.
        if self.x0 != self.x1 {
            (self.x0, self.x1, self.x0.min(self.x1), self.x0.max(self.x1))
        } else {
            (self.z0, self.z1, self.z0.min(self.z1), self.z0.max(self.z1))
        }
    }

    /// Whether (x, z) lies on the segment.
    fn on_segment(&self, x: i32, z: i32) -> bool {
        let (_, _, low, high) = self.axis();
        if self.x0 != self.x1 {
            z == self.z0 && x >= low && x <= high
        } else {
            x == self.x0 && z >= low && z <= high
        }
    }

    /// The nearest point on the segment to (x, z).
    fn nearest(&self, x: i32, z: i32) -> (i32, i32) {
        let (_, _, low, high) = self.axis();
        if self.x0 != self.x1 {
            (x.clamp(low, high), self.z0)
        } else {
            (self.x0, z.clamp(low, high))
        }
    }
}

impl Goal for GoalCorridor {
    fn is_in_goal(&self, x: i32, y: i32, z: i32) -> bool {
        y == self.y && self.on_segment(x, z)
    }

    fn heuristic(&self, x: i32, y: i32, z: i32) -> f64 {
        let (px, pz) = self.nearest(x, z);
        GoalYLevel::calculate(self.y, y) + GoalXZ::calculate((x - px) as f64, (z - pz) as f64)
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

    #[test]
    fn test_corridor_goal() {
        let goal = GoalCorridor {
            x0: 0,
            z0: 5,
            x1: 10,
            z1: 5,
            y: 64,
        };
        assert!(goal.is_in_goal(0, 64, 5));
        assert!(goal.is_in_goal(10, 64, 5));
        assert!(!goal.is_in_goal(5, 65, 5), "floor level is fixed");
        assert!(!goal.is_in_goal(11, 64, 5), "past the far end");
        assert!(!goal.is_in_goal(5, 64, 6), "off the segment");
        assert_eq!(goal.heuristic(5, 64, 5), 0.0);
        let h = goal.heuristic(0, 64, 9);
        assert!(h > 0.0 && !h.is_nan());
    }
}
