//! Movement definitions and cost calculations.
//!
//! Ported from baritone's `baritone.pathing.movement` package:
//! - `Moves.java` — enum of all movement types with offsets
//! - `Movement.java` base + `MovementHelper.java` — cost logic
//! - `MovementTraverse.java`, `MovementAscend.java`, `MovementDescend.java`,
//!   `MovementFall.java`, `MovementDiagonal.java`, `MovementParkour.java`
//!
//! Divergence from baritone: the executor mines and *places* blocks, but
//! only from the hotbar it actually carries — movements that need a block
//! placed (pillar jumps, bridges) are impassable unless the hotbar holds a
//! placeable block, and baritone's assumption of infinite blocks is kept
//! beyond that (a `/give` of 64 does not last a 300-block bridge).
//! Mining is priced with the player's actual tools ([`ToolSet`], ported
//! from baritone's `utils/ToolSet`), and the executor holds the selected
//! slot while breaking.

use crate::goals::action_costs;
use crate::world::LiveWorld;

pub use self::tools::ToolSet;

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

    // --- Placing (5) — need a block from the hotbar ---
    /// Straight up: jump and place beneath ourselves (MovementPillar).
    PillarUp,
    /// One block over a hole: hold the edge and fill the hole below the
    /// destination (MovementBridge).
    BridgeNorth,
    BridgeSouth,
    BridgeEast,
    BridgeWest,

    // --- Swimming (1) — rise one block through water ---
    /// Straight up while in water: swim to the surface (an addition over
    /// baritone, which wades on the bottom and cannot escape deep water).
    SwimUp,
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

        Moves::PillarUp => (0, 1, 0),
        Moves::BridgeNorth => (0, 0, -1),
        Moves::BridgeSouth => (0, 0, 1),
        Moves::BridgeEast => (1, 0, 0),
        Moves::BridgeWest => (-1, 0, 0),
        Moves::SwimUp => (0, 1, 0),
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
            Moves::PillarUp,
            Moves::BridgeNorth,
            Moves::BridgeSouth,
            Moves::BridgeEast,
            Moves::BridgeWest,
            Moves::SwimUp,
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
                // Parkour is a flat axis-aligned jump; match the exact offset
                // (direction matters — north and south share the same
                // distance, so only comparing distances mislabels them).
                let (pdx, pdy, pdz) = static_offset(*mv);
                if dx == pdx && dy == pdy && dz == pdz {
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
    /// position (x, y, z), using the world for block queries and the
    /// player's tools for mining times.
    pub fn cost(&self, world: &LiveWorld, tools: &ToolSet, x: i32, y: i32, z: i32) -> MoveResult {
        match self {
            Moves::TraverseNorth => traverse_cost(world, tools, x, y, z, 0, -1),
            Moves::TraverseSouth => traverse_cost(world, tools, x, y, z, 0, 1),
            Moves::TraverseEast => traverse_cost(world, tools, x, y, z, 1, 0),
            Moves::TraverseWest => traverse_cost(world, tools, x, y, z, -1, 0),

            Moves::AscendNorth => ascend_cost(world, tools, x, y, z, 0, -1),
            Moves::AscendSouth => ascend_cost(world, tools, x, y, z, 0, 1),
            Moves::AscendEast => ascend_cost(world, tools, x, y, z, 1, 0),
            Moves::AscendWest => ascend_cost(world, tools, x, y, z, -1, 0),

            Moves::DescendNorth => descend_cost(world, tools, x, y, z, 0, -1),
            Moves::DescendSouth => descend_cost(world, tools, x, y, z, 0, 1),
            Moves::DescendEast => descend_cost(world, tools, x, y, z, 1, 0),
            Moves::DescendWest => descend_cost(world, tools, x, y, z, -1, 0),

            Moves::DiagonalNE => diagonal_cost(world, x, y, z, 1, -1),
            Moves::DiagonalNW => diagonal_cost(world, x, y, z, -1, -1),
            Moves::DiagonalSE => diagonal_cost(world, x, y, z, 1, 1),
            Moves::DiagonalSW => diagonal_cost(world, x, y, z, -1, 1),

            Moves::ParkourNorth => parkour_cost(world, x, y, z, 0, -1, Direction::North),
            Moves::ParkourSouth => parkour_cost(world, x, y, z, 0, 1, Direction::South),
            Moves::ParkourEast => parkour_cost(world, x, y, z, 1, 0, Direction::East),
            Moves::ParkourWest => parkour_cost(world, x, y, z, -1, 0, Direction::West),

            Moves::PillarUp => pillar_cost(world, tools, x, y, z),
            Moves::BridgeNorth => bridge_cost(world, tools, x, y, z, 0, -1),
            Moves::BridgeSouth => bridge_cost(world, tools, x, y, z, 0, 1),
            Moves::BridgeEast => bridge_cost(world, tools, x, y, z, 1, 0),
            Moves::BridgeWest => bridge_cost(world, tools, x, y, z, -1, 0),
            Moves::SwimUp => swim_up_cost(world, tools, x, y, z),
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
        &[
            "minecraft:bedrock",
            "minecraft:barrier",
            "minecraft:command_block",
            "minecraft:light_blue_stained_glass",
        ]
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

    /// Non-solid ground cover plants (no collision box) — passable to walk
    /// through, but not standable on. Solid terrain (dirt, grass blocks,
    /// sand, ...) is intentionally NOT here: those are full cubes, so they
    /// must not be treated as walk-through.
    pub fn is_ground_cover(block: &str) -> bool {
        let base = strip_props(block);
        matches!(
            base,
            "minecraft:leaf_litter"
                | "minecraft:pink_petals"
                | "minecraft:short_grass"
                | "minecraft:fern"
                | "minecraft:dead_bush"
        )
    }

    /// Air blocks.
    pub fn is_air(block: &str) -> bool {
        let base = strip_props(block);
        matches!(
            base,
            "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air"
        )
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
            "minecraft:fire"
                | "minecraft:soul_fire"
                | "minecraft:crimson_fire"
                | "minecraft:warped_fire"
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

    /// Powder snow — the player sinks in and freezes (baritone's
    /// `canWalkThroughBlockState` refuses it outright).
    pub fn is_powder_snow(block: &str) -> bool {
        let base = strip_props(block);
        base == "minecraft:powder_snow"
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
    /// `soul_sand` is deliberately not matched: it only shares the suffix.
    pub fn is_falling_block(block: &str) -> bool {
        let base = strip_props(block);
        matches!(
            base,
            "minecraft:sand"
                | "minecraft:red_sand"
                | "minecraft:suspicious_sand"
                | "minecraft:suspicious_gravel"
                | "minecraft:anvil"
                | "minecraft:chipped_anvil"
                | "minecraft:damaged_anvil"
        ) || base.ends_with("_gravel")
    }
}

/// Whether a hotbar item places as a block when used (right click). The
/// brain cannot see the game's item registry, so this is a denylist of the
/// common non-block items — tools, armor, food, materials, buckets —
/// plus suffix rules for the rest (spawn eggs, dyes, music discs);
/// anything unrecognized counts as a block, because block items share
/// their block's name (`dirt`, `cobblestone`, `oak_planks`, ...).
pub fn is_placeable(item: &str) -> bool {
    let name = item.strip_prefix("minecraft:").unwrap_or(item);
    const NON_PLACE: &[&str] = &[
        // Tools and combat (the `_sword` suffix also parses as a tool).
        "bow",
        "crossbow",
        "trident",
        "mace",
        "fishing_rod",
        "shears",
        "flint_and_steel",
        "shield",
        "spyglass",
        "elytra",
        // Food.
        "bread",
        "apple",
        "golden_apple",
        "enchanted_golden_apple",
        "carrot",
        "golden_carrot",
        "potato",
        "baked_potato",
        "poisonous_potato",
        "beetroot",
        "beetroot_soup",
        "mushroom_stew",
        "rabbit_stew",
        "suspicious_stew",
        "melon_slice",
        "sweet_berries",
        "glow_berries",
        "cookie",
        "pumpkin_pie",
        "cake",
        "dried_kelp",
        "honey_bottle",
        "milk_bucket",
        "potion",
        "splash_potion",
        "lingering_potion",
        "rotten_flesh",
        "spider_eye",
        "chorus_fruit",
        "popped_chorus_fruit",
        // Materials and misc.
        "stick",
        "coal",
        "charcoal",
        "flint",
        "wheat",
        "wheat_seeds",
        "pumpkin_seeds",
        "melon_seeds",
        "torchflower_seeds",
        "pitcher_pod",
        "sugar",
        "egg",
        "string",
        "feather",
        "leather",
        "paper",
        "book",
        "writable_book",
        "written_book",
        "enchanted_book",
        "arrow",
        "bone",
        "bone_meal",
        "snowball",
        "bucket",
        "water_bucket",
        "lava_bucket",
        "ender_pearl",
        "eye_of_ender",
        "experience_bottle",
        "firework_rocket",
        "compass",
        "clock",
        "map",
        "filled_map",
        "name_tag",
        "lead",
        "saddle",
        "totem_of_undying",
        "iron_ingot",
        "gold_ingot",
        "copper_ingot",
        "netherite_ingot",
        "netherite_scrap",
        "diamond",
        "emerald",
        "amethyst_shard",
        "lapis_lazuli",
        "quartz",
        "redstone",
        "glowstone_dust",
        "gunpowder",
        "blaze_powder",
        "blaze_rod",
        "magma_cream",
        "slime_ball",
        "clay_ball",
        "brick",
        "nether_brick",
        "prismarine_shard",
        "prismarine_crystals",
        "ink_sac",
        "glow_ink_sac",
        "honeycomb",
    ];
    if NON_PLACE.contains(&name) {
        return false;
    }
    // Tool, armor and other item-name suffixes; no block item ends in any
    // of these (`_axe` also ends `_pickaxe`, and the pickaxe check is
    // first, so a pickaxe never reads as an axe here — both deny anyway).
    for suffix in [
        "_pickaxe",
        "_axe",
        "_shovel",
        "_hoe",
        "_sword",
        "_helmet",
        "_chestplate",
        "_leggings",
        "_boots",
        "_horse_armor",
        "_spawn_egg",
        "_dye",
        "_pottery_sherd",
        "_smithing_template",
        "_disc_fragment",
        "_bucket",
        "_boat",
    ] {
        if name.ends_with(suffix) {
            return false;
        }
    }
    if name.starts_with("music_disc_")
        || name.ends_with("_minecart")
        || name == "minecart"
        || name == "boat"
    {
        return false;
    }
    true
}

/// Check if the player can walk *through* (occupy the same space as) a block.
fn can_walk_through(world: &LiveWorld, x: i32, y: i32, z: i32) -> bool {
    let block = world.get_block(x, y, z);
    if block_tags::is_air(&block) {
        return true;
    }
    // Non-solid ground cover plants are passable. Leaves are NOT: they are
    // full solid cubes, so walking through a canopy would need mining,
    // which the executor can't do.
    if block_tags::is_ground_cover(&block) {
        return true;
    }
    // Water is passable (you wade or swim through it).
    if block_tags::is_water(&block) {
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
        || block_tags::is_powder_snow(&block)
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
    // Water is not standable (baritone's canWalkOn returns false for water
    // unless assumeWalkOnWater is set, which defaults to false).
    if block_tags::is_water(&block) {
        return false;
    }
    if block_tags::is_magma(&block) {
        return true; // can walk, but takes damage
    }
    // Leaves have a collision box — can be stood on.
    if block_tags::is_leaves(&block) {
        return true;
    }
    // Non-solid ground cover plants can't be stood on.
    if block_tags::is_ground_cover(&block) {
        return false;
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
        && !block_tags::is_powder_snow(&block)
}

// ---------------------------------------------------------------------------
// Movement cost calculations
// ---------------------------------------------------------------------------

/// Cost to traverse one block horizontally on flat ground (or water).
/// Ported from `MovementTraverse.cost`.
fn traverse_cost(
    world: &LiveWorld,
    tools: &ToolSet,
    x: i32,
    y: i32,
    z: i32,
    dx: i32,
    dz: i32,
) -> MoveResult {
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
                wc += (action_costs::WALK_ONE_OVER_SOUL_SAND_COST
                    - action_costs::WALK_ONE_BLOCK_COST)
                    / 2.0;
            }
            if block_tags::is_magma(&dest_on) {
                wc +=
                    (action_costs::SNEAK_ONE_BLOCK_COST - action_costs::WALK_ONE_BLOCK_COST) / 2.0;
            }
        }

        let hardness1 = mining_cost(world, tools, dest_x, y, dest_z, &pb1, false);
        if hardness1 >= action_costs::COST_INF {
            return MoveResult {
                x: 0,
                y: 0,
                z: 0,
                cost: action_costs::COST_INF,
            };
        }
        let hardness2 = mining_cost(world, tools, dest_x, y + 1, dest_z, &pb0, true);
        if hardness1 == 0.0 && hardness2 == 0.0 && !water {
            wc *= action_costs::SPRINT_MULTIPLIER;
            return MoveResult {
                x: dest_x,
                y,
                z: dest_z,
                cost: wc,
            };
        }
        if block_tags::is_climbable(&src_down) {
            return MoveResult {
                x: 0,
                y: 0,
                z: 0,
                cost: action_costs::COST_INF,
            };
        }
        return MoveResult {
            x: dest_x,
            y,
            z: dest_z,
            cost: wc + hardness1 + hardness2,
        };
    }

    // Can't walk on — deep water, then: swim into it. The destination's
    // feet hold water and its bottom is out of reach, so the move is a
    // surface (or underwater) swim (baritone's traverse only wades on the
    // bottom and refuses the rest; the swim-up move handles rising).
    if block_tags::is_water(&world.get_block(dest_x, y, dest_z)) {
        let head = world.get_block(dest_x, y + 1, dest_z);
        let mine_head = mining_cost(world, tools, dest_x, y + 1, dest_z, &head, true);
        if mine_head < action_costs::COST_INF {
            return MoveResult {
                x: dest_x,
                y,
                z: dest_z,
                cost: action_costs::WALK_ONE_IN_WATER_COST + mine_head,
            };
        }
    }
    MoveResult {
        x: 0,
        y: 0,
        z: 0,
        cost: action_costs::COST_INF,
    }
}

/// Cost to swim straight up one block while in water. No baritone
/// equivalent (it has no vertical water move at all): the start must be
/// submerged at feet or head, the landing space is mined first if solid.
fn swim_up_cost(world: &LiveWorld, tools: &ToolSet, x: i32, y: i32, z: i32) -> MoveResult {
    if !block_tags::is_water(&world.get_block(x, y, z))
        && !block_tags::is_water(&world.get_block(x, y + 1, z))
    {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }
    let mut cost = action_costs::SWIM_UP_ONE_BLOCK_COST;
    for (by, upper) in [(y + 1, false), (y + 2, true)] {
        let block = world.get_block(x, by, z);
        let mine = mining_cost(world, tools, x, by, z, &block, upper);
        if mine >= action_costs::COST_INF {
            return MoveResult {
                x: 0,
                y: 0,
                z: 0,
                cost: action_costs::COST_INF,
            };
        }
        cost += mine;
    }
    MoveResult {
        x,
        y: y + 1,
        z,
        cost,
    }
}

/// Cost to pillar straight up: jump and place the block beneath ourselves.
/// Ported from `MovementPillar`. Needs a placeable block in the hotbar and
/// a solid support under the source to place against; the landing space at
/// the destination's feet and head is mined first if solid.
fn pillar_cost(world: &LiveWorld, tools: &ToolSet, x: i32, y: i32, z: i32) -> MoveResult {
    if !tools.has_placeable() || !can_walk_on(world, x, y - 1, z) {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }
    let mut cost = action_costs::WALK_ONE_BLOCK_COST
        + action_costs::jump_one_block_cost()
        + action_costs::PLACE_BLOCK_COST;
    for (bx, by, upper) in [(x, y + 1, false), (x, y + 2, true)] {
        let block = world.get_block(bx, by, z);
        let mine = mining_cost(world, tools, bx, by, z, &block, upper);
        if mine >= action_costs::COST_INF {
            return MoveResult {
                x: 0,
                y: 0,
                z: 0,
                cost: action_costs::COST_INF,
            };
        }
        cost += mine;
    }
    MoveResult {
        x,
        y: y + 1,
        z,
        cost,
    }
}

/// Cost to bridge one block over a hole: hold the edge, aim at a solid
/// neighbour of the hole and place. Ported from `MovementBridge`. Needs a
/// placeable block, a hole to fill (passable but not standable — a step
/// down is the descend move's job), and a solid block beside the hole to
/// place against. The source's own support is never aimable: from above,
/// its top face occludes the side face the placement needs, so it does
/// not count.
fn bridge_cost(
    world: &LiveWorld,
    tools: &ToolSet,
    x: i32,
    y: i32,
    z: i32,
    dx: i32,
    dz: i32,
) -> MoveResult {
    let dest_x = x + dx;
    let dest_z = z + dz;
    if !tools.has_placeable()
        || !can_walk_on(world, x, y - 1, z)
        || !can_walk_through(world, dest_x, y - 1, dest_z)
        || can_walk_on(world, dest_x, y - 1, dest_z)
    {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }
    let aimable = [
        (dest_x + dx, y - 1, dest_z + dz),
        (dest_x + dz, y - 1, dest_z + dx),
        (dest_x - dz, y - 1, dest_z - dx),
    ]
    .iter()
    .any(|(bx, by, bz)| !can_walk_through(world, *bx, *by, *bz));
    if !aimable {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }
    let mut cost = action_costs::WALK_ONE_BLOCK_COST + action_costs::PLACE_BLOCK_COST;
    for (bx, by, upper) in [(dest_x, y, false), (dest_x, y + 1, true)] {
        let block = world.get_block(bx, by, dest_z);
        let mine = mining_cost(world, tools, bx, by, dest_z, &block, upper);
        if mine >= action_costs::COST_INF {
            return MoveResult {
                x: 0,
                y: 0,
                z: 0,
                cost: action_costs::COST_INF,
            };
        }
        cost += mine;
    }
    MoveResult {
        x: dest_x,
        y,
        z: dest_z,
        cost,
    }
}

/// Cost to ascend one block (jump up). Ported from `MovementAscend.cost`.
fn ascend_cost(
    world: &LiveWorld,
    tools: &ToolSet,
    x: i32,
    y: i32,
    z: i32,
    dx: i32,
    dz: i32,
) -> MoveResult {
    let dest_x = x + dx;
    let dest_z = z + dz;
    let dest_y = y + 1;

    // Land on the step block at (dest_x, y, dest_z) — the block *beneath*
    // the landing position (baritone checks `canWalkOn(destX, src.y, destZ)`).
    // Baritone falls back to placing a block and jumping onto it when there
    // is no step; the executor can't place, so that's impassable here.
    if !can_walk_on(world, dest_x, y, dest_z) {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }

    // Check for falling blocks above the source that could suffocate
    // (baritone's `srcUp3` falling check).
    let above1 = world.get_block(x, y + 1, z);
    if block_tags::is_falling_block(&world.get_block(x, y + 3, z))
        && (can_walk_through(world, x, y + 1, z) || !block_tags::is_falling_block(&above1))
    {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }

    // Walkable step — standard jump cost (baritone: WALK + JUMP).
    let walk_cost = action_costs::WALK_ONE_BLOCK_COST;
    let jump_cost = action_costs::jump_one_block_cost();

    // Landing space and headroom above it, plus the headroom above the
    // source: baritone mines all three if solid (`MovementAscend.cost` adds
    // `getMiningDurationTicks` for (x, y+2), (destX, y+1) and (destX, y+2)).
    let mut cost = walk_cost + jump_cost;
    for (bx, by, bz, upper) in [
        (x, y + 2, z, false),
        (dest_x, dest_y, dest_z, false),
        (dest_x, dest_y + 1, dest_z, true),
    ] {
        let block = world.get_block(bx, by, bz);
        let mine = mining_cost(world, tools, bx, by, bz, &block, upper);
        if mine >= action_costs::COST_INF {
            return MoveResult {
                x: 0,
                y: 0,
                z: 0,
                cost: action_costs::COST_INF,
            };
        }
        cost += mine;
    }

    // Walkable step — standard jump cost (baritone: WALK + JUMP).
    MoveResult {
        x: dest_x,
        y: dest_y,
        z: dest_z,
        cost,
    }
}

/// Cost to descend one block (walk off edge and fall 1 block).
/// Ported from `MovementDescend.cost`.
fn descend_cost(
    world: &LiveWorld,
    tools: &ToolSet,
    x: i32,
    y: i32,
    z: i32,
    dx: i32,
    dz: i32,
) -> MoveResult {
    let dest_x = x + dx;
    let dest_z = z + dz;
    let dest_y = y - 1;

    // The move passes through the front column at feet and head level (and
    // one above the head): baritone mines all three (`MovementDescend.cost`
    // prices (destX, y-1), (destX, y) and (destX, y+1) up front).
    let mut mining = 0.0;
    for (bx, by, upper) in [
        (dest_x, dest_y, false),
        (dest_x, y, false),
        (dest_x, y + 1, true),
    ] {
        let block = world.get_block(bx, by, dest_z);
        let mine = mining_cost(world, tools, bx, by, dest_z, &block, upper);
        if mine >= action_costs::COST_INF {
            return MoveResult {
                x: 0,
                y: 0,
                z: 0,
                cost: action_costs::COST_INF,
            };
        }
        mining += mine;
    }

    // Don't descend off a ladder or vine (baritone's `isClimbable(fromDown)`).
    if block_tags::is_climbable(&world.get_block(x, y - 1, z)) {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }

    // Block two below the destination must be walkable for a standard
    // descend (feet land at y-1, standing on y-2); otherwise it may be a
    // dynamic fall into water or a deeper drop.
    if !can_walk_on(world, dest_x, dest_y - 1, dest_z) {
        return dynamic_fall_cost(world, x, y, z, dest_x, dest_z, mining);
    }

    // A ladder in the landing column is the climber's job, not a descend.
    if block_tags::is_climbable(&world.get_block(dest_x, dest_y, dest_z)) {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }

    let walk = action_costs::WALK_OFF_BLOCK_COST;
    let fall = action_costs::fall_n_blocks_cost(1);
    let recenter = action_costs::CENTER_AFTER_FALL_COST;

    MoveResult {
        x: dest_x,
        y: dest_y,
        z: dest_z,
        cost: mining + walk + fall.max(recenter),
    }
}

/// Dynamic fall cost — falling through air/water to a landing spot.
/// Ported from `MovementDescend.dynamicFallCost`. `front_break` is the
/// caller's mining cost for the front passage (baritone's `frontBreak`).
fn dynamic_fall_cost(
    world: &LiveWorld,
    _x: i32,
    y: i32,
    _z: i32,
    dest_x: i32,
    dest_z: i32,
    front_break: f64,
) -> MoveResult {
    // Breaking blocks in front can shake loose a falling block stacked on
    // top of the column (baritone's `FallingBlock` guard): don't fall in.
    if front_break > 0.0 && block_tags::is_falling_block(&world.get_block(dest_x, y + 2, dest_z)) {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }
    // The fall column's first free-fall block must already be passable —
    // the caller mines the front passage, not the fall column itself.
    if !can_walk_through(world, dest_x, y - 2, dest_z) {
        return MoveResult {
            x: 0,
            y: 0,
            z: 0,
            cost: action_costs::COST_INF,
        };
    }

    for fall_height in 3..100 {
        let new_y = y - fall_height;
        if new_y < -64 {
            return MoveResult {
                x: 0,
                y: 0,
                z: 0,
                cost: action_costs::COST_INF,
            };
        }

        let onto_block = world.get_block(dest_x, new_y, dest_z);

        // Landing in water: feet end up in the water block itself.
        if block_tags::is_water(&onto_block) {
            if !can_walk_on(world, dest_x, new_y - 1, dest_z) {
                return MoveResult {
                    x: 0,
                    y: 0,
                    z: 0,
                    cost: action_costs::COST_INF,
                };
            }
            let tentative = action_costs::WALK_OFF_BLOCK_COST
                + action_costs::fall_n_blocks_cost(fall_height as usize)
                + front_break;
            return MoveResult {
                x: dest_x,
                y: new_y,
                z: dest_z,
                cost: tentative,
            };
        }

        // Landing on a solid block: feet end up standing ON it
        // (baritone: `res.y = newY + 1` — "the block we're falling onto").
        if can_walk_on(world, dest_x, new_y, dest_z) {
            // Baritone rejects the move when the unprotected fall exceeds
            // `maxFallHeightNoWater + 1` (default 3 + 1): deeper drops deal
            // too much fall damage, and the solid block ends the column —
            // there is nothing deeper to consider.
            if fall_height > action_costs::MAX_FALL_HEIGHT_NO_WATER + 1 {
                return MoveResult {
                    x: 0,
                    y: 0,
                    z: 0,
                    cost: action_costs::COST_INF,
                };
            }
            let tentative = action_costs::WALK_OFF_BLOCK_COST
                + action_costs::fall_n_blocks_cost(fall_height as usize)
                + front_break;
            return MoveResult {
                x: dest_x,
                y: new_y + 1,
                z: dest_z,
                cost: tentative,
            };
        }

        // Still falling — if block is solid, we can't continue.
        if !can_walk_through(world, dest_x, new_y, dest_z) {
            break;
        }
    }

    MoveResult {
        x: 0,
        y: 0,
        z: 0,
        cost: action_costs::COST_INF,
    }
}

/// Cost for diagonal movement (including ascend/descend on the diagonal).
/// Ported from `MovementDiagonal.cost`.
fn diagonal_cost(world: &LiveWorld, x: i32, y: i32, z: i32, dx: i32, dz: i32) -> MoveResult {
    let dest_x = x + dx;
    let dest_z = z + dz;
    let inf = MoveResult {
        x: 0,
        y: 0,
        z: 0,
        cost: action_costs::COST_INF,
    };

    let diag_base = action_costs::WALK_ONE_BLOCK_COST * std::f64::consts::FRAC_1_SQRT_2;

    // Corner analysis (baritone MovementDiagonal): the move needs at most
    // one of the two corner columns cleared — baritone mines the other one,
    // we can't, so at least one column must be fully passable already.
    // Also never cut a corner over lava (baritone checks the same).
    let a_clear =
        can_walk_through(world, x, y, z + dz) && can_walk_through(world, x, y + 1, z + dz);
    let b_clear =
        can_walk_through(world, x + dx, y, z) && can_walk_through(world, x + dx, y + 1, z);
    if (!a_clear && !b_clear)
        || block_tags::is_lava(&world.get_block(x, y - 1, z + dz))
        || block_tags::is_lava(&world.get_block(x + dx, y - 1, z))
    {
        return inf;
    }

    if !can_walk_through(world, dest_x, y, dest_z) {
        // Diagonal ascend: land on the step block at (dest_x, y, dest_z).
        // Baritone additionally requires one corner column clear all the
        // way up past the head (otherwise the jump bonks).
        let a_top = can_walk_through(world, x, y + 2, z + dz);
        let b_top = can_walk_through(world, x + dx, y + 2, z);
        if !can_walk_through(world, x, y + 2, z) || !(a_clear && a_top || b_clear && b_top) {
            return inf;
        }
        if can_walk_on(world, dest_x, y, dest_z)
            && can_walk_through(world, dest_x, y + 1, dest_z)
            && can_walk_through(world, dest_x, y + 2, dest_z)
        {
            return MoveResult {
                x: dest_x,
                y: y + 1,
                z: dest_z,
                cost: diag_base + action_costs::jump_one_block_cost(),
            };
        }
        return inf;
    }

    // Flat diagonal.
    if can_walk_on(world, dest_x, y - 1, dest_z) && can_walk_through(world, dest_x, y + 1, dest_z) {
        return MoveResult {
            x: dest_x,
            y,
            z: dest_z,
            cost: diag_base,
        };
    }

    // Descending diagonal: the dropped-into space must be enterable.
    if can_walk_through(world, dest_x, y - 1, dest_z) && can_walk_on(world, dest_x, y - 2, dest_z) {
        return MoveResult {
            x: dest_x,
            y: y - 1,
            z: dest_z,
            cost: action_costs::WALK_OFF_BLOCK_COST + action_costs::fall_n_blocks_cost(1),
        };
    }

    inf
}

/// Cost for a 4-block sprint jump (parkour) over a gap.
/// Ported conceptually from `MovementParkour.cost` (fixed 4-block distance).
///
/// `dx`/`dz` are a unit direction; the landing is [`PARKOUR_DIST`] blocks out
/// (the takeoff edge clears three gap columns onto the fourth block, matching
/// `Moves::from_delta`'s (±4, 0) offsets).
const PARKOUR_DIST: i32 = 4;

fn parkour_cost(
    world: &LiveWorld,
    x: i32,
    y: i32,
    z: i32,
    dx: i32,
    dz: i32,
    _dir: Direction,
) -> MoveResult {
    let dest_x = x + dx * PARKOUR_DIST;
    let dest_z = z + dz * PARKOUR_DIST;
    let inf = MoveResult {
        x: 0,
        y: 0,
        z: 0,
        cost: action_costs::COST_INF,
    };

    // Can't jump out of water, and jumping off a climbable block doesn't work.
    if block_tags::is_water(&world.get_block(x, y, z))
        || block_tags::is_climbable(&world.get_block(x, y - 1, z))
    {
        return inf;
    }
    // Headroom above the takeoff point.
    if !can_walk_through(world, x, y + 2, z) {
        return inf;
    }
    // The adjacent block must be passable at feet and head level...
    if !can_walk_through(world, x + dx, y, z + dz)
        || !can_walk_through(world, x + dx, y + 1, z + dz)
    {
        return inf;
    }
    // ...and if we could just walk onto the adjacent floor, don't jump
    // (baritone: "don't parkour if we could just traverse").
    if can_walk_on(world, x + dx, y - 1, z + dz) {
        return inf;
    }

    // The gap columns between takeoff and landing must be passable at feet
    // and head level. These are the *intermediate* columns — the old code
    // checked the takeoff/landing columns here instead, so north/south
    // gaps went entirely unverified.
    for step in 1..=3 {
        let bx = x + dx * step;
        let bz = z + dz * step;
        if !can_walk_through(world, bx, y, bz) || !can_walk_through(world, bx, y + 1, bz) {
            return inf;
        }
    }

    // Landing floor must be walkable; landing is at the same Y (flat jump).
    if !can_walk_on(world, dest_x, y - 1, dest_z) {
        return inf;
    }

    // Overshoot safety: the two blocks past the landing must be enterable
    // (baritone's checkOvershootSafety — don't jump into a wall or lava).
    if !can_walk_through(world, dest_x + dx, y, dest_z + dz)
        || !can_walk_through(world, dest_x + dx, y + 1, dest_z + dz)
    {
        return inf;
    }

    let cost = action_costs::SPRINT_ONE_BLOCK_COST * 4.0 + action_costs::jump_one_block_cost();
    MoveResult {
        x: dest_x,
        y,
        z: dest_z,
        cost,
    }
}

// ---------------------------------------------------------------------------
// Mining cost helper
// ---------------------------------------------------------------------------

/// Block hardness data and hand-mining times.
///
/// Ported from baritone's cost model (`MovementHelper.getMiningDurationTicks`
/// over a `ToolSet` with no tools): Minecraft breaks a block in
/// `hardness × 30` ticks when the hand can harvest it and
/// `hardness × 100` when a tool is required, and cannot break blocks of
/// hardness < 0 at all. Horizon's own mining loop uses the same formula
/// (`progress = speed / hardness / (30 if harvest else 100)`), so these
/// times match what the executor will actually experience.
pub mod hardness {
    use super::action_costs::COST_INF;

    /// (hardness, requires_correct_tool) for the blocks worth naming.
    /// Vanilla values; suffix rules below cover whole material families.
    /// Accepts bare and `minecraft:`-prefixed names, with or without
    /// state properties. Public so the sibling [`super::tools`] module
    /// reads the same table.
    pub fn entry(base: &str) -> Option<(f64, bool)> {
        let name = super::block_tags::strip_props(base);
        let name = name.strip_prefix("minecraft:").unwrap_or(name);
        let table: &[(&str, f64, bool)] = &[
            // unbreakable (hardness < 0: the hand cannot break them)
            ("bedrock", -1.0, true),
            ("barrier", -1.0, true),
            ("command_block", -1.0, true),
            ("chain_command_block", -1.0, true),
            ("repeating_command_block", -1.0, true),
            ("end_portal", -1.0, true),
            ("end_portal_frame", -1.0, true),
            ("end_gateway", -1.0, true),
            ("nether_portal", -1.0, true),
            // stone family
            ("stone", 1.5, true),
            ("cobblestone", 2.0, true),
            ("mossy_cobblestone", 2.0, true),
            ("smooth_stone", 2.0, true),
            ("stone_bricks", 1.5, true),
            ("mossy_stone_bricks", 1.5, true),
            ("cracked_stone_bricks", 1.5, true),
            ("chiseled_stone_bricks", 1.5, true),
            ("granite", 1.5, true),
            ("diorite", 1.5, true),
            ("andesite", 1.5, true),
            ("deepslate", 3.0, true),
            ("cobbled_deepslate", 3.5, true),
            ("polished_deepslate", 3.5, true),
            ("deepslate_bricks", 3.5, true),
            ("deepslate_tiles", 3.5, true),
            ("tuff", 1.5, true),
            ("calcite", 0.75, true),
            ("dripstone_block", 1.5, true),
            ("basalt", 1.25, true),
            ("blackstone", 1.5, true),
            ("obsidian", 50.0, true),
            ("crying_obsidian", 50.0, true),
            ("netherrack", 0.4, true),
            ("nether_bricks", 2.0, true),
            ("red_nether_bricks", 2.0, true),
            ("end_stone", 3.0, true),
            ("end_stone_bricks", 3.0, true),
            ("purpur_block", 1.5, true),
            ("quartz_block", 0.8, true),
            ("quartz_bricks", 0.8, true),
            ("amethyst_block", 1.5, true),
            ("budding_amethyst", 1.5, true),
            ("sandstone", 0.8, true),
            ("red_sandstone", 0.8, true),
            ("terracotta", 1.25, true),
            ("bricks", 2.0, true),
            ("mud_bricks", 1.5, true),
            ("packed_mud", 1.0, false),
            ("mud", 0.5, false),
            ("clay", 0.6, false),
            // soft terrain
            ("dirt", 0.5, false),
            ("coarse_dirt", 0.5, false),
            ("rooted_dirt", 0.5, false),
            ("grass_block", 0.6, false),
            ("podzol", 0.6, false),
            ("mycelium", 0.6, false),
            ("dirt_path", 0.65, false),
            ("farmland", 0.6, false),
            ("sand", 0.5, false),
            ("red_sand", 0.5, false),
            ("gravel", 0.6, false),
            ("soul_sand", 0.5, false),
            ("soul_soil", 0.5, false),
            ("snow_block", 0.2, true),
            ("powder_snow", 0.2, false),
            ("moss_block", 0.1, false),
            ("sponge", 0.6, false),
            ("wet_sponge", 0.6, false),
            ("hay_block", 0.5, false),
            ("dried_kelp_block", 0.5, false),
            // wood
            ("oak_log", 2.0, false),
            ("spruce_log", 2.0, false),
            ("birch_log", 2.0, false),
            ("jungle_log", 2.0, false),
            ("acacia_log", 2.0, false),
            ("dark_oak_log", 2.0, false),
            ("mangrove_log", 2.0, false),
            ("cherry_log", 2.0, false),
            ("pale_oak_log", 2.0, false),
            ("crimson_stem", 2.0, false),
            ("warped_stem", 2.0, false),
            ("bamboo_block", 2.0, false),
            ("bookshelf", 1.5, false),
            ("chiseled_bookshelf", 1.5, false),
            ("crafting_table", 2.5, false),
            ("chest", 2.5, false),
            ("trapped_chest", 2.5, false),
            ("barrel", 2.5, false),
            ("lectern", 2.5, false),
            ("jukebox", 2.0, false),
            ("note_block", 0.8, false),
            ("loom", 2.5, false),
            ("cartography_table", 2.5, false),
            ("fletching_table", 2.5, false),
            ("smithing_table", 2.5, false),
            ("composter", 0.6, false),
            // ores and metal blocks
            ("coal_ore", 3.0, true),
            ("iron_ore", 3.0, true),
            ("copper_ore", 3.0, true),
            ("gold_ore", 3.0, true),
            ("redstone_ore", 3.0, true),
            ("lapis_ore", 3.0, true),
            ("diamond_ore", 3.0, true),
            ("emerald_ore", 3.0, true),
            ("deepslate_coal_ore", 4.5, true),
            ("deepslate_iron_ore", 4.5, true),
            ("deepslate_copper_ore", 4.5, true),
            ("deepslate_gold_ore", 4.5, true),
            ("deepslate_redstone_ore", 4.5, true),
            ("deepslate_lapis_ore", 4.5, true),
            ("deepslate_diamond_ore", 4.5, true),
            ("deepslate_emerald_ore", 4.5, true),
            ("nether_gold_ore", 3.0, true),
            ("nether_quartz_ore", 3.0, true),
            ("ancient_debris", 30.0, true),
            ("iron_block", 5.0, true),
            ("gold_block", 3.0, true),
            ("diamond_block", 5.0, true),
            ("emerald_block", 5.0, true),
            ("copper_block", 3.0, true),
            ("netherite_block", 50.0, true),
            ("raw_iron_block", 5.0, true),
            ("raw_gold_block", 5.0, true),
            ("raw_copper_block", 5.0, true),
            // misc
            ("glass", 0.3, false),
            ("glowstone", 0.3, false),
            ("sea_lantern", 0.2, false),
            ("shroomlight", 1.0, false),
            ("pumpkin", 1.0, false),
            ("carved_pumpkin", 1.0, false),
            ("jack_o_lantern", 1.0, false),
            ("melon", 1.0, false),
            ("cobweb", 4.0, true),
            ("cactus", 0.4, false),
            ("sweet_berry_bush", 0.0, false),
            ("fire", 0.0, false),
            ("soul_fire", 0.0, false),
            ("torch", 0.0, false),
            ("wall_torch", 0.0, false),
            ("soul_torch", 0.0, false),
            ("ladder", 0.4, false),
            ("rail", 0.7, false),
            ("powered_rail", 0.7, false),
            ("detector_rail", 0.7, false),
            ("activator_rail", 0.7, false),
            ("lantern", 3.5, true),
            ("soul_lantern", 3.5, true),
            ("iron_bars", 5.0, true),
            ("chain", 5.0, true),
            ("iron_door", 5.0, true),
            ("shulker_box", 2.0, true),
            ("bed", 0.2, false),
            ("cake", 0.5, false),
            ("turtle_egg", 0.5, false),
            ("scaffolding", 0.0, false),
            ("mangrove_roots", 0.7, false),
            ("azalea", 0.2, false),
            ("flowering_azalea", 0.2, false),
            ("vine", 0.2, false),
            ("glow_lichen", 0.2, false),
            ("sugar_cane", 0.0, false),
            ("bamboo", 1.0, false),
            ("cocoa", 0.2, false),
            ("nether_wart", 0.0, false),
            ("wheat", 0.0, false),
            ("carrots", 0.0, false),
            ("potatoes", 0.0, false),
            ("beetroots", 0.0, false),
        ];
        for (n, h, tool) in table {
            if *n == name {
                return Some((*h, *tool));
            }
        }
        // Family fallbacks, widest suffix first.
        let suffixes: &[(&str, f64, bool)] = &[
            ("_glazed_terracotta", 1.4, true),
            ("_concrete_powder", 0.5, false),
            ("_concrete", 1.8, true),
            ("_shulker_box", 2.0, true),
            ("_hanging_sign", 1.5, false),
            ("_pressure_plate", 0.5, false),
            ("_fence_gate", 2.0, false),
            ("_trapdoor", 3.0, false),
            ("_sandstone", 0.8, true),
            ("_terracotta", 1.25, true),
            ("_bricks", 2.0, true),
            ("_planks", 2.0, false),
            ("_leaves", 0.2, false),
            ("_log", 2.0, false),
            ("_wood", 2.0, false),
            ("_stem", 2.0, false),
            ("_hyphae", 2.0, false),
            ("_carpet", 0.1, false),
            ("_wool", 0.8, false),
            ("_glass", 0.3, false),
            ("_ore", 3.0, true),
            ("_anvil", 5.0, true),
            ("_door", 3.0, false),
            ("_fence", 2.0, false),
            ("_stairs", 2.0, true),
            ("_slab", 2.0, true),
            ("_sign", 1.5, false),
            ("_banner", 1.0, false),
            ("_torch", 0.0, false),
            ("_rail", 0.7, false),
            ("_button", 0.5, false),
            ("_bed", 0.2, false),
        ];
        for (suffix, h, tool) in suffixes {
            if name.ends_with(suffix) {
                let h = if *suffix == "_ore" && name.starts_with("deepslate_") {
                    4.5
                } else {
                    *h
                };
                return Some((h, *tool));
            }
        }
        // Unknown block: assume stone-like (conservative for planning).
        Some((1.5, true))
    }

    /// Ticks to break a block bare-handed: `COST_INF` for unbreakables
    /// (hardness < 0), 0 for instant blocks, else hardness × 30/100.
    pub fn break_ticks(base: &str) -> f64 {
        let (h, tool) = match entry(base) {
            Some(e) => e,
            None => return COST_INF,
        };
        if h < 0.0 {
            COST_INF
        } else if h == 0.0 {
            0.0
        } else {
            h * if tool { 100.0 } else { 30.0 }
        }
    }
}

/// Tool selection, ported from baritone's `ToolSet` (utils/ToolSet.java):
/// how fast the hotbar's best item breaks a block, and which slot to hold
/// while breaking. Vanilla's break formula applies: `hardness × 30 / speed`
/// ticks when the block drops for the tool in hand, `hardness × 100 / speed`
/// when it does not — so the right tool is several times faster than the
/// hand, and tool-gated ores drop nothing for anything weaker.
pub mod tools {
    use super::action_costs::COST_INF;
    use super::hardness;

    /// Tool classes that speed up breaking (baritone's `ToolClass`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ToolClass {
        Pickaxe,
        Axe,
        Shovel,
        Hoe,
        Sword,
    }

    /// (material, class) of a tool item, or None for non-tools. `_pickaxe`
    /// is tried before `_axe` because pickaxe ends in axe.
    fn parse(item: &str) -> Option<(&str, ToolClass)> {
        let name = item.strip_prefix("minecraft:").unwrap_or(item);
        for (suffix, class) in [
            ("_pickaxe", ToolClass::Pickaxe),
            ("_axe", ToolClass::Axe),
            ("_shovel", ToolClass::Shovel),
            ("_hoe", ToolClass::Hoe),
            ("_sword", ToolClass::Sword),
        ] {
            if let Some(material) = name.strip_suffix(suffix) {
                return Some((material, class));
            }
        }
        None
    }

    /// (class, base speed, harvest tier) of a tool item — vanilla's tool
    /// materials, baritone's `ToolSet.Material`. The speed applies against
    /// blocks of the tool's own class; the tier is the harvest gate it
    /// satisfies (wood 0, stone 1, iron 2, diamond 3, netherite 4; gold is
    /// fast but harvests only tier-0 blocks, like wood).
    fn stats(item: &str) -> Option<(ToolClass, f64, u8)> {
        let (material, class) = parse(item)?;
        let (speed, tier) = match material {
            "wooden" => (2.0, 0),
            "stone" => (4.0, 1),
            "iron" => (6.0, 2),
            "golden" => (12.0, 0),
            "diamond" => (8.0, 3),
            "netherite" => (9.0, 4),
            _ => return None,
        };
        Some((class, speed, tier))
    }

    /// Tie-break cost between equally fast tools (Java's material order
    /// WOODEN, STONE, IRON, GOLD, DIAMOND, NETHERITE); non-tools lose.
    fn material_cost(item: &str) -> u8 {
        match parse(item) {
            Some(("wooden", _)) => 0,
            Some(("stone", _)) => 1,
            Some(("iron", _)) => 2,
            Some(("golden", _)) => 3,
            Some(("diamond", _)) => 4,
            Some(("netherite", _)) => 5,
            _ => u8::MAX,
        }
    }

    /// The tool class `base` prefers and the minimum harvest tier for its
    /// drops (vanilla's `isCorrectToolForDrops`; 0 = any tool of the
    /// class works). Ported from baritone's class assignments and the
    /// mineable-with tags.
    fn requirement(base: &str, tool_required: bool) -> (ToolClass, u8) {
        let class = if base == "cobweb" {
            ToolClass::Sword
        } else if is_shovel_block(base) {
            ToolClass::Shovel
        } else if is_axe_block(base) {
            ToolClass::Axe
        } else {
            ToolClass::Pickaxe
        };
        let gate = if !tool_required {
            0
        } else if matches!(
            base,
            "obsidian" | "crying_obsidian" | "ancient_debris" | "netherite_block"
        ) {
            3
        } else if matches!(
            base,
            "diamond_ore"
                | "deepslate_diamond_ore"
                | "gold_ore"
                | "deepslate_gold_ore"
                | "emerald_ore"
                | "deepslate_emerald_ore"
                | "redstone_ore"
                | "deepslate_redstone_ore"
                | "gold_block"
                | "diamond_block"
                | "emerald_block"
                | "raw_gold_block"
        ) {
            2
        } else if matches!(
            base,
            "iron_ore"
                | "deepslate_iron_ore"
                | "copper_ore"
                | "deepslate_copper_ore"
                | "lapis_ore"
                | "deepslate_lapis_ore"
                | "iron_block"
                | "copper_block"
                | "raw_iron_block"
                | "raw_copper_block"
        ) {
            1
        } else {
            0
        };
        (class, gate)
    }

    /// Blocks the shovel class speeds up (vanilla's mineable/shovel tag,
    /// the full-cube members that matter for pathing).
    fn is_shovel_block(base: &str) -> bool {
        matches!(
            base,
            "dirt"
                | "grass_block"
                | "podzol"
                | "mycelium"
                | "dirt_path"
                | "farmland"
                | "sand"
                | "red_sand"
                | "gravel"
                | "soul_sand"
                | "soul_soil"
                | "snow_block"
                | "powder_snow"
                | "mud"
                | "packed_mud"
                | "clay"
                | "moss_block"
                | "hay_block"
                | "dried_kelp_block"
                | "sponge"
                | "wet_sponge"
        ) || base.ends_with("_concrete_powder")
    }

    /// Blocks the axe class speeds up (vanilla's mineable/axe tag; walls
    /// and mixed-material slabs/stairs are left to the pickaxe default —
    /// the class only changes speed there, never drops).
    fn is_axe_block(base: &str) -> bool {
        matches!(
            base,
            "bookshelf"
                | "chiseled_bookshelf"
                | "crafting_table"
                | "chest"
                | "trapped_chest"
                | "barrel"
                | "lectern"
                | "jukebox"
                | "note_block"
                | "loom"
                | "cartography_table"
                | "fletching_table"
                | "smithing_table"
                | "composter"
                | "bamboo_block"
        ) || base.ends_with("_log")
            || base.ends_with("_wood")
            || base.ends_with("_stem")
            || base.ends_with("_hyphae")
            || base.ends_with("_planks")
            || base.ends_with("_fence")
            || (base.ends_with("_door") && base != "iron_door")
            || base.ends_with("_sign")
    }

    /// Ticks for one candidate holder (the hand when `item` is None) to
    /// break `base`, ported from `ToolSet.calculateSpeedVsBlock`. A tool
    /// of the wrong class swings at the hand's speed; a sword swings at
    /// 1.5 everywhere and shreds cobwebs at 15 (vanilla ItemSword speeds).
    fn ticks_vs(item: Option<&str>, base: &str) -> f64 {
        let base = super::block_tags::strip_props(base);
        let (h, tool_required) = match hardness::entry(base) {
            Some(e) => e,
            None => return COST_INF,
        };
        if h < 0.0 {
            return COST_INF;
        }
        if h == 0.0 {
            return 0.0;
        }
        let name = base.strip_prefix("minecraft:").unwrap_or(base);
        let (class, gate) = requirement(name, tool_required);
        let (speed, harvest) = match item.and_then(stats) {
            None => (1.0, !tool_required),
            Some((c, s, tier)) => {
                let speed = if c == ToolClass::Sword {
                    if class == ToolClass::Sword { 15.0 } else { 1.5 }
                } else if c == class {
                    s
                } else {
                    1.0
                };
                let harvest = !tool_required || (c == class && tier >= gate);
                (speed, harvest)
            }
        };
        h * if harvest { 30.0 } else { 100.0 } / speed
    }

    /// The tools the player can reach — the hotbar. The planner prices
    /// breaks with the best tool's time; the executor holds that tool
    /// while breaking ([`ToolSet::best_slot`]).
    pub struct ToolSet {
        /// Hotbar item names by slot (0–8); None when the slot is empty.
        pub slots: [Option<String>; 9],
    }

    impl ToolSet {
        /// No tools at all — the bare hand.
        pub fn hand() -> Self {
            ToolSet {
                slots: std::array::from_fn(|_| None),
            }
        }

    pub fn new(slots: [Option<String>; 9]) -> Self {
        ToolSet { slots }
    }

    /// Whether any hotbar slot holds a placeable block: the gate on the
    /// placing moves (`PillarUp`, the bridges) — without one they stay
    /// impassable and the planner routes around them.
    pub fn has_placeable(&self) -> bool {
        self.slots
            .iter()
            .any(|slot| slot.as_deref().is_some_and(super::is_placeable))
    }

        /// Fastest break time for `base` over the hand and every hotbar
        /// item.
        pub fn break_ticks_with(&self, base: &str) -> f64 {
            let mut best = ticks_vs(None, base);
            for slot in &self.slots {
                if let Some(item) = slot {
                    let t = ticks_vs(Some(item), base);
                    if t < best {
                        best = t;
                    }
                }
            }
            best
        }

        /// The hotbar slot to hold while breaking `base`: the strictly
        /// fastest tool, ties going to the cheaper material (Java's
        /// order). None when nothing beats the hand — the held slot stays
        /// as it is. (Java switches on ties; keeping the slot avoids
        /// needless churn.)
        pub fn best_slot(&self, base: &str) -> Option<usize> {
            let hand = ticks_vs(None, base);
            let mut best: Option<(f64, u8, usize)> = None;
            for (i, slot) in self.slots.iter().enumerate() {
                let Some(item) = slot else { continue };
                let t = ticks_vs(Some(item), base);
                if t >= hand || t >= COST_INF {
                    continue;
                }
                let cost = material_cost(item);
                let better = match best {
                    Some((bt, bc, _)) => t < bt || (t == bt && cost < bc),
                    None => true,
                };
                if better {
                    best = Some((t, cost, i));
                }
            }
            best.map(|(_, _, i)| i)
        }
    }
}

/// Cost of a movement entering the block `block` (in ticks). Ported from
/// `MovementHelper.getMiningDurationTicks`: 0 when the block can be walked
/// through as-is, the best-tool mining time when it must be broken first,
/// and `COST_INF` when breaking is forbidden.
///
/// `is_upper` is baritone's `includeFalling`: on the upper block of a
/// movement's break set, a stack of falling blocks above joins the cost.
fn mining_cost(
    world: &LiveWorld,
    tools: &ToolSet,
    x: i32,
    y: i32,
    z: i32,
    block: &str,
    is_upper: bool,
) -> f64 {
    if block_tags::is_air(block)
        || block_tags::is_water(block)
        || block_tags::is_ground_cover(block)
        || block_tags::is_climbable(block)
        || block_tags::is_fence_gate(block)
        || block_tags::is_wooden_door(block)
    {
        return 0.0;
    }
    let base = block_tags::strip_props(block);
    // Never break, ice (it becomes water), or infested blocks (silverfish).
    if block_tags::is_never_break(block)
        || base == "minecraft:ice"
        || base.starts_with("minecraft:infested_")
        || block_tags::is_lava(block)
    {
        return action_costs::COST_INF;
    }
    // Baritone's `avoidBreaking`: don't break next to a liquid (it floods
    // the path) or beside an unsupported falling block (it collapses
    // weirdly). The block directly above is exempt from the falling rule —
    // its cost is added via `includeFalling` instead.
    for (nx, ny, nz, above) in [
        (x, y + 1, z, true),
        (x + 1, y, z, false),
        (x - 1, y, z, false),
        (x, y, z + 1, false),
        (x, y, z - 1, false),
    ] {
        let nb = world.get_block(nx, ny, nz);
        if block_tags::is_water(&nb) || block_tags::is_lava(&nb) {
            return action_costs::COST_INF;
        }
        if !above && block_tags::is_falling_block(&nb) {
            let under = world.get_block(nx, ny - 1, nz);
            if block_tags::is_air(&under) || block_tags::is_water(&under) {
                return action_costs::COST_INF;
            }
        }
    }
    let mut ticks = tools.break_ticks_with(base);
    if is_upper {
        // A falling block resting on this one drops when it breaks: mine
        // the whole supported stack above (baritone's recursion).
        let mut ay = y + 1;
        for _ in 0..16 {
            let ab = world.get_block(x, ay, z);
            if !block_tags::is_falling_block(&ab) {
                break;
            }
            ticks += tools.break_ticks_with(block_tags::strip_props(&ab));
            ay += 1;
        }
    }
    ticks
}

/// The blocks a movement would break when taken from (x, y, z): exactly the
/// positions `mining_cost` prices above zero, in baritone's order. The path
/// stores these per step so the executor can mine before walking.
pub fn movement_breaks(
    mv: Moves,
    world: &LiveWorld,
    tools: &ToolSet,
    x: i32,
    y: i32,
    z: i32,
) -> Vec<(i32, i32, i32)> {
    let mut out = Vec::new();
    let price = |out: &mut Vec<(i32, i32, i32)>, bx: i32, by: i32, bz: i32, upper: bool| {
        let block = world.get_block(bx, by, bz);
        if mining_cost(world, tools, bx, by, bz, &block, upper) > 0.0 {
            out.push((bx, by, bz));
        }
    };
    // Straight-up moves (pillar hop, swim-up) break only the two cells
    // they rise into; no horizontal direction is involved.
    if matches!(mv, Moves::PillarUp | Moves::SwimUp) {
        price(&mut out, x, y + 1, z, false);
        price(&mut out, x, y + 2, z, true);
        return out;
    }
    let (dx, dz) = match mv {
        Moves::TraverseNorth | Moves::AscendNorth | Moves::DescendNorth | Moves::BridgeNorth => {
            (0, -1)
        }
        Moves::TraverseSouth | Moves::AscendSouth | Moves::DescendSouth | Moves::BridgeSouth => {
            (0, 1)
        }
        Moves::TraverseEast | Moves::AscendEast | Moves::DescendEast | Moves::BridgeEast => (1, 0),
        Moves::TraverseWest | Moves::AscendWest | Moves::DescendWest | Moves::BridgeWest => (-1, 0),
        // Diagonals edge through a clear corner without mining; parkour
        // jumps clear their own way. Neither breaks anything.
        _ => return Vec::new(),
    };
    let dest_x = x + dx;
    let dest_z = z + dz;
    match mv {
        Moves::TraverseNorth | Moves::TraverseSouth | Moves::TraverseEast | Moves::TraverseWest => {
            price(&mut out, dest_x, y, dest_z, false);
            price(&mut out, dest_x, y + 1, dest_z, true);
        }
        Moves::AscendNorth | Moves::AscendSouth | Moves::AscendEast | Moves::AscendWest => {
            price(&mut out, x, y + 2, z, false);
            price(&mut out, dest_x, y + 1, dest_z, false);
            price(&mut out, dest_x, y + 2, dest_z, true);
        }
        Moves::DescendNorth | Moves::DescendSouth | Moves::DescendEast | Moves::DescendWest => {
            price(&mut out, dest_x, y - 1, dest_z, false);
            price(&mut out, dest_x, y, dest_z, false);
            price(&mut out, dest_x, y + 1, dest_z, true);
        }
        // Placing moves: the destination's feet and head must be clear —
        // the hole below gets filled, not mined.
        Moves::BridgeNorth
        | Moves::BridgeSouth
        | Moves::BridgeEast
        | Moves::BridgeWest => {
            price(&mut out, dest_x, y, dest_z, false);
            price(&mut out, dest_x, y + 1, dest_z, true);
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::action_costs::COST_INF;
    use super::hardness;
    use super::tools::ToolSet;

    /// A hotbar from nine item names; the empty string is an empty slot.
    fn toolset(items: [&str; 9]) -> ToolSet {
        ToolSet::new(items.map(|i| (!i.is_empty()).then(|| i.to_string())))
    }

    #[test]
    fn tool_break_times_match_vanilla_formulas() {
        let hand = ToolSet::hand();

        // Stone (hardness 1.5, pickaxe, any tier): the hand pays the
        // no-harvest divisor (150 ticks); the right tool divides by its
        // speed at the harvest divisor.
        assert_eq!(hand.break_ticks_with("minecraft:stone"), 150.0);
        let wooden = toolset(["minecraft:wooden_pickaxe", "", "", "", "", "", "", "", ""]);
        assert_eq!(wooden.break_ticks_with("stone"), 22.5);
        let iron = toolset(["", "", "", "", "minecraft:iron_pickaxe", "", "", "", ""]);
        assert_eq!(iron.break_ticks_with("stone"), 7.5);
        let golden = toolset(["minecraft:golden_pickaxe", "", "", "", "", "", "", "", ""]);
        assert_eq!(golden.break_ticks_with("stone"), 3.75);

        // Iron ore (hardness 3.0, needs a stone-tier pickaxe): a wooden
        // pickaxe is the right class but under the gate — no harvest — so
        // it is barely faster than the hand. A stone pickaxe harvests.
        assert_eq!(hand.break_ticks_with("iron_ore"), 300.0);
        assert_eq!(wooden.break_ticks_with("iron_ore"), 150.0);
        let stone = toolset(["", "minecraft:stone_pickaxe", "", "", "", "", "", "", ""]);
        assert_eq!(stone.break_ticks_with("iron_ore"), 22.5);
        let diamond = toolset(["minecraft:diamond_pickaxe", "", "", "", "", "", "", "", ""]);
        assert_eq!(diamond.break_ticks_with("iron_ore"), 11.25);

        // Coal ore (gate 0): even a wooden pickaxe harvests it.
        assert_eq!(wooden.break_ticks_with("coal_ore"), 45.0);

        // Dirt needs no tool: every tool of its class divides by speed,
        // and a wrong-class pickaxe matches the hand.
        assert_eq!(hand.break_ticks_with("dirt"), 15.0);
        let shovel = toolset(["minecraft:stone_shovel", "", "", "", "", "", "", "", ""]);
        assert_eq!(shovel.break_ticks_with("dirt"), 3.75);
        assert_eq!(wooden.break_ticks_with("dirt"), 15.0);

        // The axe speeds up wood.
        assert_eq!(hand.break_ticks_with("oak_log"), 60.0);
        let axe = toolset(["minecraft:wooden_axe", "", "", "", "", "", "", "", ""]);
        assert_eq!(axe.break_ticks_with("oak_log"), 30.0);

        // Swords: 1.5 everywhere, 15 against cobweb.
        let sword = toolset(["minecraft:iron_sword", "", "", "", "", "", "", "", ""]);
        assert_eq!(sword.break_ticks_with("stone"), 100.0);
        assert_eq!(sword.break_ticks_with("cobweb"), 8.0);
        assert_eq!(hand.break_ticks_with("cobweb"), 400.0);

        // Non-tool items swing at the hand's speed.
        let junk = toolset(["minecraft:dirt", "", "", "", "", "", "", "", ""]);
        assert_eq!(junk.break_ticks_with("stone"), 150.0);

        // Unbreakable and instant blocks are tool-independent.
        assert_eq!(hand.break_ticks_with("bedrock"), COST_INF);
        assert_eq!(hand.break_ticks_with("torch"), 0.0);
        assert_eq!(hand.break_ticks_with("minecraft:torch"), 0.0);
    }

    #[test]
    fn best_slot_picks_the_fastest_then_the_cheaper_tool() {
        // The iron pickaxe beats the wooden one for stone.
        let tools = toolset([
            "minecraft:wooden_pickaxe",
            "",
            "minecraft:iron_pickaxe",
            "",
            "",
            "",
            "",
            "",
            "",
        ]);
        assert_eq!(tools.best_slot("stone"), Some(2));

        // Ties go to the cheaper material: two wooden pickaxes, the first
        // slot wins.
        let tied = toolset([
            "",
            "minecraft:wooden_pickaxe",
            "",
            "minecraft:wooden_pickaxe",
            "",
            "",
            "",
            "",
            "",
        ]);
        assert_eq!(tied.best_slot("stone"), Some(1));

        // Nothing beats the hand for dirt when only pickaxes are held.
        assert_eq!(
            tools.best_slot("dirt"),
            None,
            "wrong-class tools are never selected"
        );

        // A sword is strictly faster than the hand on stone, so it is
        // selected (Java's useSwordToMine).
        let sword = toolset(["", "", "", "minecraft:iron_sword", "", "", "", "", ""]);
        assert_eq!(sword.best_slot("stone"), Some(3));

        // An empty hotbar never selects.
        assert_eq!(ToolSet::hand().best_slot("stone"), None);
    }

    #[test]
    fn powder_snow_is_breakable_but_not_standable_ground() {
        // Vanilla: hardness 0.2, no tool required — the hand punches
        // through in six ticks.
        assert_eq!(hardness::entry("powder_snow"), Some((0.2, false)));
        assert_eq!(hardness::entry("minecraft:powder_snow[holding=false]"), Some((0.2, false)));
        assert_eq!(
            ToolSet::hand().break_ticks_with("powder_snow"),
            6.0
        );
        let base = super::block_tags::strip_props("minecraft:powder_snow[holding=false]");
        assert!(super::block_tags::is_powder_snow(base));
    }
}
