//! Movement definitions and cost calculations.
//!
//! Ported from baritone's `baritone.pathing.movement` package:
//! - `Moves.java` — enum of all movement types with offsets
//! - `Movement.java` base + `MovementHelper.java` — cost logic
//! - `MovementTraverse.java`, `MovementAscend.java`, `MovementDescend.java`,
//!   `MovementFall.java`, `MovementDiagonal.java`, `MovementParkour.java`

use crate::goals::action_costs;
use crate::world::LiveWorld;

/// Direction enum mirroring Minecraft's `Direction` (N/S/E/W, no UP/DOWN).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    pub fn dx(self) -> i32 {
        match self {
            Direction::East => 1,
            Direction::West => -1,
            _ => 0,
        }
    }
    pub fn dz(self) -> i32 {
        match self {
            Direction::North => -1,
            Direction::South => 1,
            _ => 0,
        }
    }
}

/// The result of evaluating a movement's cost at a position.
#[derive(Clone, Debug)]
pub struct MoveResult {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    /// Cost in game ticks; `COST_INF` means unreachable.
    pub cost: f64,
}

/// All possible movement types, ported from baritone's `Moves` enum.
/// Each entry has fixed (x,y,z) offsets from the source position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Moves {
    // --- Flat ground traversal (4) ---
    TraverseNorth,
    TraverseSouth,
    TraverseEast,
    TraverseWest,

    // --- Ascending (4) ---
    AscendNorth,
    AscendSouth,
    AscendEast,
    AscendWest,

    // --- Descending (4) — 1-block drop, walk off edge ---
    DescendNorth,
    DescendSouth,
    DescendEast,
    DescendWest,

    // --- Diagonal (4) ---
    DiagonalNE, // +x, -z
    DiagonalNW, // -x, -z
    DiagonalSE, // +x, +z
    DiagonalSW, // -x, +z

    // --- Parkour (4) — 4-block sprint jump ---
    ParkourNorth,
    ParkourSouth,
    ParkourEast,
    ParkourWest,
}

/// The (x, y, z) offset for each move type (when `dynamicXZ`/`dynamicY` is false).
fn static_offset(mv: Moves) -> (i32, i32, i32) {
    match mv {
        Moves::TraverseNorth => (0, 0, -1),
        Moves::TraverseSouth => (0, 0, 1),
        Moves::TraverseEast => (1, 0, 0),
        Moves::TraverseWest => (-1, 0, 0),

        Moves::AscendNorth => (0, 1, -1),
        Moves::AscendSouth => (0, 1, 1),
        Moves::AscendEast => (1, 1, 0),
        Moves::AscendWest => (-1, 1, 0),

        Moves::DescendNorth => (0, -1, -1),
        Moves::DescendSouth => (0, -1, 1),
        Moves::DescendEast => (1, -1, 0),
        Moves::DescendWest => (-1, -1, 0),

        Moves::DiagonalNE => (1, 0, -1),
        Moves::DiagonalNW => (-1, 0, -1),
        Moves::DiagonalSE => (1, 0, 1),
        Moves::DiagonalSW => (-1, 0, 1),

        // Parkour offsets are "dynamic" (computed at runtime), use sentinel.
        Moves::ParkourNorth => (0, 0, -4),
        Moves::ParkourSouth => (0, 0, 4),
        Moves::ParkourEast => (4, 0, 0),
        Moves::ParkourWest => (-4, 0, 0),
    }
}

impl Moves {
    /// Returns all movement types (equivalent to Java `Moves.values()`).
    pub fn all() -> &'static [Moves] {
        &[
            Moves::TraverseNorth,
            Moves::TraverseSouth,
            Moves::TraverseEast,
            Moves::TraverseWest,
            Moves::AscendNorth,
            Moves::AscendSouth,
            Moves::AscendEast,
            Moves::AscendWest,
            Moves::DescendNorth,
            Moves::DescendSouth,
            Moves::DescendEast,
            Moves::DescendWest,
            Moves::DiagonalNE,
            Moves::DiagonalNW,
            Moves::DiagonalSE,
            Moves::DiagonalSW,
            Moves::ParkourNorth,
            Moves::ParkourSouth,
            Moves::ParkourEast,
            Moves::ParkourWest,
        ]
    }

    /// Convert a position delta into the nearest canonical move, if possible.
    /// Used for path reconstruction.
    pub fn from_delta(dx: i32, dy: i32, dz: i32) -> Option<Moves> {
        for mv in Moves::all() {
            let (mx, my, mz) = static_offset(*mv);
            if matches!(
                mv,
                Moves::ParkourNorth | Moves::ParkourSouth | Moves::ParkourEast | Moves::ParkourWest
            ) {
                let (pdx, _pdy, pdz) = static_offset(*mv);
                let pd = pdx.abs() + pdz.abs();
                let d = dx.abs() + dz.abs();
                if d == pd && dy == 0 {
                    return Some(*mv);
                }
                continue;
            }
            if dx == mx && dy == my && dz == mz {
                return Some(*mv);
            }
        }
        None
    }

    /// Compute the cost (and effective destination) of this movement at
    /// position (x, y, z), using the world for block queries.
    pub fn cost(&self, world: &LiveWorld, x: i32, y: i32, z: i32) -> MoveResult {
        match self {
            Moves::TraverseNorth => traverse_cost(world, x, y, z, 0, -1),
            Moves::TraverseSouth => traverse_cost(world, x, y, z, 0, 1),
            Moves::TraverseEast => traverse_cost(world, x, y, z, 1, 0),
            Moves::TraverseWest => traverse_cost(world, x, y, z, -1, 0),

            Moves::AscendNorth => ascend_cost(world, x, y, z, 0, -1),
            Moves::AscendSouth => ascend_cost(world, x, y, z, 0, 1),
            Moves::AscendEast => ascend_cost(world, x, y, z, 1, 0),
            Moves::AscendWest => ascend_cost(world, x, y, z, -1, 0),

            Moves::DescendNorth => descend_cost(world, x, y, z, 0, -1),
            Moves::DescendSouth => descend_cost(world, x, y, z, 0, 1),
            Moves::DescendEast => descend_cost(world, x, y, z, 1, 0),
            Moves::DescendWest => descend_cost(world, x, y, z, -1, 0),

            Moves::DiagonalNE => diagonal_cost(world, x, y, z, 1, -1),
            Moves::DiagonalNW => diagonal_cost(world, x, y, z, -1, -1),
            Moves::DiagonalSE => diagonal_cost(world, x, y, z, 1, 1),
            Moves::DiagonalSW => diagonal_cost(world, x, y, z, -1, 1),

            Moves::ParkourNorth => parkour_cost(world, x, y, z, 0, -1, Direction::North),
            Moves::ParkourSouth => parkour_cost(world, x, y, z, 0, 1, Direction::South),
            Moves::ParkourEast => parkour_cost(world, x, y, z, 1, 0, Direction::East),
            Moves::ParkourWest => parkour_cost(world, x, y, z, -1, 0, Direction::West),
        }
    }
}

// ---------------------------------------------------------------------------
// Block property checks (ported from MovementHelper)
// ---------------------------------------------------------------------------

/// Block tag helpers, ported from baritone's block-type checks.
/// All functions strip Minecraft block state properties first
/// (e.g. `minecraft:oak_leaves[distance=2]` → `minecraft:oak_leaves`).
pub mod block_tags {
    /// Strip Minecraft block state properties: `minecraft:oak_leaves[distance=2]` → `minecraft:oak_leaves`.
    pub fn strip_props(block: &str) -> &str {
        match block.find('[') {
            Some(idx) => &block[..idx],
            None => block,
        }
    }

    /// Blocks that baritone will never break (e.g. bedrock, barriers).
    pub fn never_break() -> &'static [&'static str] {
        &["minecraft:bedrock", "minecraft:barrier", "minecraft:command_block",
          "minecraft:light_blue_stained_glass"]
    }

    /// Check if a block is in the never-break list (after stripping props).
    pub fn is_never_break(block: &str) -> bool {
        let base = strip_props(block);
        never_break().contains(&base)
    }

    /// Blocks the player can climb (ladders, vines, scaffolding).
    pub fn is_climbable(block: &str) -> bool {
        let base = strip_props(block);
        matches!(
            base,
            "minecraft:ladder" | "minecraft:vine" | "minecraft:scaffolding" | "minecraft:thin_ice"
        )
    }

    /// Blocks that are leaves (walkable but slow).
    pub fn is_leaves(block: &str) -> bool {
        let base = strip_props(block);
        base.ends_with("_leaves") || base == "minecraft:leaves"
    }

    /// Leaf litter and similar ground cover (dirt, grass, sand, etc.).
    pub fn is_ground_cover(block: &str) -> bool {
        let base = strip_props(block);
        matches!(
            base,
            "minecraft:leaf_litter" | "minecraft:coarse_dirt" | "minecraft:rooted_dirt"
                | "minecraft:podzol" | "minecraft:mycelium" | "minecraft:dirt"
                | "minecraft:grass_block" | "minecraft:crimson_nyllis" | "minecraft:warped_nylium"
                | "minecraft:gravel" | "minecraft:sand" | "minecraft:red_sand"
                | "minecraft:clay" | "minecraft:soul_soil" | "minecraft:muck"
        )
    }

    /// Air blocks.
    pub fn is_air(block: &str) -> bool {
        let base = strip_props(block);
        matches!(base, "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air")
    }

    /// Water blocks.
    pub fn is_water(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:water"
    }

    /// Lava blocks.
    pub fn is_lava(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:lava"
    }

    /// Fire-type blocks to avoid.
    pub fn is_fire(block: &str) -> bool {
        let base = strip_props(block);
        matches!(
            base,
            "minecraft:fire" | "minecraft:soul_fire" | "minecraft:crimson_fire" | "minecraft:warped_fire"
        )
    }

    /// Magma block — deals damage when stood on.
    pub fn is_magma(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:magma_block"
    }

    /// Soul sand — slow movement.
    pub fn is_soul_sand(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:soul_sand"
    }

    /// Cobweb — slows movement.
    pub fn is_web(block: &str) -> bool {
        let base = strip_props(block);
        matches!(base, "minecraft:cobweb")
    }

    /// Cactus — damages on contact.
    pub fn is_cactus(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:cactus"
    }

    /// Sweet berry bush — slows and can be broken.
    pub fn is_sweet_berry(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:sweet_berry_bush"
    }

    /// Fence gates — passable when open.
    pub fn is_fence_gate(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:oak_fence_gate" || base.ends_with("_fence_gate")
    }

    /// Doors (wooden doors are walkable after opening).
    pub fn is_wooden_door(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:oak_door" || base.ends_with("_door")
    }

    /// Trapdoors (wooden are passable when open).
    pub fn is_trapdoor(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:trapdoor" || base.ends_with("_trapdoor")
    }

    /// Slabs — half-height blocks.
    pub fn is_slab(block: &str) -> bool {
        let base = strip_props(block);
        base.ends_with("_slab")
    }

    /// Stairs — climbable but not walkable without care.
    pub fn is_stairs(block: &str) -> bool {
        let base = strip_props(block);
        base.ends_with("_stairs")
    }

    /// Falling blocks (sand, gravel, anvils, etc.) that could suffocate.
    pub fn is_falling_block(block: &str) -> bool {
        let base = strip_props(block);
        base.ends_with("_sand") || base.ends_with("_gravel")
            || base == "minecraft:anvil"
            || base.ends_with("_anvil")
            || base.ends_with("_cactus") // falling cactus
    }
}

/// Check if the player can walk *through* (occupy the same space as) a block.
fn can_walk_through(world: &LiveWorld, x: i32, y: i32, z: i32) -> bool {
    let block = world.get_block(x, y, z);
    if block_tags::is_air(&block) {
        return true;
    }
    // Leaves and ground cover are passable (walk-through-able).
    if block_tags::is_leaves(&block) || block_tags::is_ground_cover(&block) {
        return true;
    }
    // Doors, fence gates are walk-through when open (assume open for pathing).
    if block_tags::is_fence_gate(&block) {
        return true;
    }
    if block_tags::is_wooden_door(&block) {
        return true;
    }
    // Avoid dangerous blocks.
    if block_tags::is_fire(&block)
        || block_tags::is_web(&block)
        || block_tags::is_lava(&block)
        || block_tags::is_cactus(&block)
        || block_tags::is_sweet_berry(&block)
    {
        return false;
    }
    false
}

/// Check if a block can be stood *on* (walked upon). Mirrors `canWalkOn`.
pub fn can_walk_on(world: &LiveWorld, x: i32, y: i32, z: i32) -> bool {
    let block = world.get_block(x, y, z);
    if block_tags::is_air(&block) {
        return false;
    }
    if block_tags::is_climbable(&block) {
        return true;
    }
    if block_tags::is_water(&block) {
        return true; // can walk on water surface (or with frost walker)
    }
    if block_tags::is_magma(&block) {
        return true; // can walk, but takes damage
    }
    // Leaves have a collision box — can be stood on.
    if block_tags::is_leaves(&block) {
        return true;
    }
    // Ground cover / vegetation like grass, dirt, leaf litter.
    if block_tags::is_ground_cover(&block) {
        return true;
    }
    // Solid blocks. We approximate by treating anything with a collision box
    // that's not air/water/lava/fire.
    !block_tags::is_air(&block)
        && !block_tags::is_water(&block)
        && !block_tags::is_lava(&block)
        && !block_tags::is_fire(&block)
        && !block_tags::is_web(&block)
        && !block_tags::is_cactus(&block)
        && !block_tags::is_sweet_berry(&block)
}

/// Whether the block at (x,y,z) is replaceable (can be broken/placed against).
fn is_replaceable(world: &LiveWorld, x: i32, y: i32, z: i32) -> bool {
    let block = world.get_block(x, y, z);
    block_tags::is_air(&block) || block_tags::is_water(&block) || block_tags::is_lava(&block)
}

// ---------------------------------------------------------------------------
// Movement cost calculations
// ---------------------------------------------------------------------------

/// Cost to traverse one block horizontally on flat ground (or water).
/// Ported from `MovementTraverse.cost`.
fn traverse_cost(world: &LiveWorld, x: i32, y: i32, z: i32, dx: i32, dz: i32) -> MoveResult {
    let dest_x = x + dx;
    let dest_z = z + dz;

    let pb0 = world.get_block(dest_x, y + 1, dest_z);
    let pb1 = world.get_block(dest_x, y, dest_z);
    let dest_on = world.get_block(dest_x, y - 1, dest_z);
    let src_down = world.get_block(x, y - 1, z);

    let water = block_tags::is_water(&pb0) || block_tags::is_water(&pb1);

    if can_walk_on(world, dest_x, y - 1, dest_z) {
        let mut wc = action_costs::WALK_ONE_BLOCK_COST;

        if water {
            wc = action_costs::WALK_ONE_IN_WATER_COST;
        } else {
            if block_tags::is_soul_sand(&src_down) {
                wc += (action_costs::WALK_ONE_OVER_SOUL_SAND_COST - action_costs::WALK_ONE_BLOCK_COST) / 2.0;
            }
            if block_tags::is_magma(&dest_on) {
                wc += (action_costs::SNEAK_ONE_BLOCK_COST - action_costs::WALK_ONE_BLOCK_COST) / 2.0;
            }
        }

        let hardness1 = mining_cost(&world, dest_x, y, dest_z, &pb1, false);
        if hardness1 >= action_costs::COST_INF {
            return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
        }
        let hardness2 = mining_cost(&world, dest_x, y + 1, dest_z, &pb0, true);
        if hardness1 == 0.0 && hardness2 == 0.0 && !water {
            wc *= action_costs::SPRINT_MULTIPLIER;
            return MoveResult { x: dest_x, y, z: dest_z, cost: wc };
        }
        if block_tags::is_climbable(&src_down) {
            return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
        }
        return MoveResult { x: dest_x, y, z: dest_z, cost: wc + hardness1 + hardness2 };
    }

    // Can't walk on — need to place a block (bridge). Simplified to COST_INF.
    MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF }
}

/// Cost to ascend one block (jump up). Ported from `MovementAscend.cost`.
fn ascend_cost(world: &LiveWorld, x: i32, y: i32, z: i32, dx: i32, dz: i32) -> MoveResult {
    let dest_x = x + dx;
    let dest_z = z + dz;
    let dest_y = y + 1;

    // Must be able to walk through the destination (foot + head level).
    if !can_walk_through(world, dest_x, dest_y, dest_z) {
        return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
    }
    if !can_walk_through(world, dest_x, dest_y + 1, dest_z) {
        return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
    }

    // Source headroom must be clear.
    if !can_walk_through(world, x, y + 2, z) {
        return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
    }

    // Dest must be walk-onto-able (or placeable).
    if !can_walk_on(world, dest_x, dest_y, dest_z) {
        let to_place = world.get_block(dest_x, dest_y, dest_z);
        if !is_replaceable(world, dest_x, dest_y, dest_z) {
            return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
        }
        let placement_cost = placement_cost(world, dest_x, dest_y, dest_z, &to_place);
        if placement_cost >= action_costs::COST_INF {
            return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
        }
        let found = placeable_adjacent(world, dest_x, dest_y, dest_z, x, z);
        if !found {
            return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
        }
        return MoveResult {
            x: dest_x, y: dest_y, z: dest_z,
            cost: placement_cost + action_costs::WALK_ONE_BLOCK_COST,
        };
    }

    // Walkable — standard jump cost.
    let walk_cost = action_costs::WALK_ONE_BLOCK_COST;
    let jump_cost = action_costs::jump_one_block_cost();
    let total = walk_cost + jump_cost;

    // Check for falling blocks above that could suffocate.
    let above1 = world.get_block(dest_x, dest_y + 1, dest_z);
    let above2 = world.get_block(dest_x, dest_y + 2, dest_z);
    if block_tags::is_falling_block(&above2)
        && (can_walk_through(world, x, y + 1, z) || !block_tags::is_falling_block(&above1))
    {
        return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
    }

    MoveResult { x: dest_x, y: dest_y, z: dest_z, cost: total }
}

/// Cost to descend one block (walk off edge and fall 1 block).
/// Ported from `MovementDescend.cost`.
fn descend_cost(world: &LiveWorld, x: i32, y: i32, z: i32, dx: i32, dz: i32) -> MoveResult {
    let dest_x = x + dx;
    let dest_z = z + dz;
    let dest_y = y - 1;

    // Block below destination must be walkable.
    if !can_walk_on(world, dest_x, dest_y - 1, dest_z) {
        // Dynamic fall — could fall through water, etc.
        return dynamic_fall_cost(world, x, y, z, dest_x, dest_z);
    }

    // Standard 1-block descend: walk off, fall 1, recenter.
    let walk = action_costs::WALK_OFF_BLOCK_COST;
    let fall = action_costs::fall_n_blocks_cost(1);
    let recenter = action_costs::CENTER_AFTER_FALL_COST;

    MoveResult { x: dest_x, y: dest_y, z: dest_z, cost: walk + fall.max(recenter) }
}

/// Dynamic fall cost — falling through air/water to a landing spot.
/// Ported from `MovementDescend.dynamicFallCost`.
fn dynamic_fall_cost(
    world: &LiveWorld,
    _x: i32,
    y: i32,
    _z: i32,
    dest_x: i32,
    dest_z: i32,
) -> MoveResult {
    let front_block = world.get_block(dest_x, y, dest_z);
    let front_break = if block_tags::is_air(&front_block) {
        0.0
    } else {
        mining_cost(world, dest_x, y, dest_z, &front_block, false)
    };
    let front_break = if front_break >= action_costs::COST_INF { 0.0 } else { front_break };

    let cost_so_far = 0.0;

    for fall_height in 3..100 {
        let new_y = y - fall_height;
        if new_y < -64 {
            return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
        }

        let onto_block = world.get_block(dest_x, new_y, dest_z);

        // Landing in water.
        if block_tags::is_water(&onto_block) {
            if !can_walk_on(world, dest_x, new_y - 1, dest_z) {
                return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
            }
            let tentative = action_costs::WALK_OFF_BLOCK_COST
                + action_costs::fall_n_blocks_cost(fall_height as usize)
                + front_break
                + cost_so_far;
            return MoveResult { x: dest_x, y: new_y, z: dest_z, cost: tentative };
        }

        // Landing on a solid block.
        if can_walk_on(world, dest_x, new_y, dest_z) {
            let tentative = action_costs::WALK_OFF_BLOCK_COST
                + action_costs::fall_n_blocks_cost(fall_height as usize)
                + front_break
                + cost_so_far;
            return MoveResult { x: dest_x, y: new_y, z: dest_z, cost: tentative };
        }

        // Still falling — if block is solid, we can't continue.
        if !can_walk_through(world, dest_x, new_y, dest_z) {
            break;
        }
    }

    MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF }
}

/// Cost for diagonal movement (including ascend/descend on the diagonal).
/// Ported from `MovementDiagonal.cost`.
fn diagonal_cost(world: &LiveWorld, x: i32, y: i32, z: i32, dx: i32, dz: i32) -> MoveResult {
    let dest_x = x + dx;
    let dest_z = z + dz;

    let diag_base = action_costs::WALK_ONE_BLOCK_COST * 0.7071067811865476; // sqrt(2)/2 scaling

    if !can_walk_through(world, dest_x, y, dest_z) {
        // Could be an ascend on diagonal.
        if !can_walk_through(world, x, y + 2, z) {
            return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
        }
        if can_walk_on(world, dest_x, y, dest_z) && can_walk_through(world, dest_x, y + 1, dest_z) {
            return MoveResult {
                x: dest_x, y: y + 1, z: dest_z,
                cost: diag_base + action_costs::jump_one_block_cost(),
            };
        }
        return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
    }

    // Flat diagonal.
    if can_walk_on(world, dest_x, y - 1, dest_z) && can_walk_through(world, dest_x, y + 1, dest_z) {
        return MoveResult { x: dest_x, y, z: dest_z, cost: diag_base };
    }

    // Descending diagonal.
    if can_walk_on(world, dest_x, y - 2, dest_z) {
        return MoveResult {
            x: dest_x, y: y - 1, z: dest_z,
            cost: action_costs::WALK_OFF_BLOCK_COST + action_costs::fall_n_blocks_cost(1),
        };
    }

    MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF }
}

/// Cost for a 4-block sprint jump (parkour) over a gap.
/// Ported conceptually from `MovementParkour.cost`.
fn parkour_cost(world: &LiveWorld, x: i32, y: i32, z: i32, dx: i32, dz: i32, _dir: Direction) -> MoveResult {
    let dest_x = x + dx;
    let dest_z = z + dz;

    // Check the gap is air for 4 blocks.
    for step in 1..=3 {
        let bx = x + dx * step;
        let _bz = z + dz * step;
        if !can_walk_through(world, bx, y, dest_z) && !can_walk_through(world, bx, y, z) {
            return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
        }
    }

    if !can_walk_on(world, dest_x, y - 1, dest_z) {
        return MoveResult { x: 0, y: 0, z: 0, cost: action_costs::COST_INF };
    }

    let cost = action_costs::SPRINT_ONE_BLOCK_COST * 4.0 + action_costs::jump_one_block_cost();
    MoveResult { x: dest_x, y: y - 1, z: dest_z, cost }
}

// ---------------------------------------------------------------------------
// Mining / placement cost helpers
// ---------------------------------------------------------------------------

/// Estimate the cost (in ticks) to break a block.
/// Simplified from `MovementHelper.getMiningDurationTicks`.
fn mining_cost(_world: &LiveWorld, _x: i32, _y: i32, _z: i32, block: &str, _is_upper: bool) -> f64 {
    if block_tags::is_air(block) {
        return 0.0;
    }
    if block_tags::is_never_break(block) {
        return action_costs::COST_INF;
    }
    // Simple approximation: 1 tick to break, 0 if air.
    1.0
}

/// Cost to place a block (placeholder — full implementation needs inventory info).
fn placement_cost(_world: &LiveWorld, _x: i32, _y: i32, _z: i32, _block: &str) -> f64 {
    action_costs::WALK_ONE_BLOCK_COST * 0.5
}

/// Check if there's a solid adjacent block to place against.
fn placeable_adjacent(world: &LiveWorld, x: i32, y: i32, z: i32, src_x: i32, src_z: i32) -> bool {
    const DIRS: [(i32, i32, i32); 5] = [
        (0, 1, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    for (dx, dy, dz) in DIRS {
        let ax = x + dx;
        let ay = y + dy;
        let az = z + dz;
        if ax == src_x && ay == y && az == src_z {
            continue;
        }
        let block = world.get_block(ax, ay, az);
        if !block_tags::is_air(&block) && !block_tags::is_water(&block) && !block_tags::is_lava(&block) {
            return true;
        }
    }
    false
}
