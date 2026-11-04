//! A* path search and path representation.
//!
//! Ported from:
//! - `AStarPathFinder.java` — main search loop
//! - `PathNode.java` — search node
//! - `BinaryHeapOpenSet.java` — min-heap priority queue
//! - `PathBase.java` / `IPath.java` — path result representation

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::goals::{Goal, action_costs};
use crate::pathing::movement::{Moves, ToolSet, movement_breaks};
use crate::world::LiveWorld;

/// Cost coefficients for the partial-path bookkeeping
/// (`AbstractNodeCostSearch.COEFFICIENTS`): the best node per coefficient
/// of `cost / coefficient + heuristic`, tracked while the search runs.
/// A high coefficient devalues the cost paid so far, so its track reaches
/// past expensive terrain (a wall to mine) that the plain-cost track
/// never would.
///
/// The last two entries extend baritone's list: a mining move through
/// tool-grade terrain pays ~2 blocks × 30 ticks of planner mining cost per
/// step, and even coefficient 10 (which tolerates ~46 ticks/step at the
/// walk-cost heuristic) can't advance through it. 25 covers flat
/// traverses with two breaks, 40 ascends with three.
const PARTIAL_COEFFICIENTS: [f64; 9] = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 25.0, 40.0];
/// A partial path must leave this much Euclidean distance from the start
/// (`AbstractNodeCostSearch.MIN_DIST_PATH`): less is not worth
/// considering ("If a path goes less than 5 blocks and doesn't make it to
/// its goal, it's not worth considering").
const MIN_DIST_PATH: f64 = 5.0;

// ---------------------------------------------------------------------------
// PathNode
// ---------------------------------------------------------------------------

/// A node in the A* open/closed set, holding cost state and back-pointer.
///
/// The back-pointer is an `Arc` chain: relaxing a node points it at the
/// popped node without copying the chain, so reconstruction stays O(depth)
/// instead of O(depth²) per relaxation.
#[derive(Clone)]
pub struct PathNode {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub estimated_cost_to_goal: f64,
    pub cost: f64,
    pub combined_cost: f64,
    pub previous: Option<Arc<PathNode>>,
    /// The move that created this node from `previous`: kept because the
    /// delta alone cannot tell the move types apart — a bridge step and a
    /// traverse share their offset, and the executor behaves differently
    /// for the two (it stops and places a block over the bridge's hole).
    pub previous_move: Option<Moves>,
    pub heap_position: i32,
    pub favor: f64,
}

impl PathNode {
    pub fn new(x: i32, y: i32, z: i32, goal: &dyn Goal) -> Self {
        let estimated = goal.heuristic(x, y, z);
        // Baritone throws if heuristic is NaN; we keep that invariant.
        assert!(!estimated.is_nan(), "heuristic returned NaN at {x} {y} {z}");
        PathNode {
            x,
            y,
            z,
            estimated_cost_to_goal: estimated,
            cost: action_costs::COST_INF,
            combined_cost: action_costs::COST_INF,
            previous: None,
            previous_move: None,
            heap_position: -1,
            favor: 0.0,
        }
    }

    /// A node is open if it's currently in the binary heap.
    pub fn is_open(&self) -> bool {
        self.heap_position != -1
    }

    pub fn pos_key(x: i32, y: i32, z: i32) -> (i32, i32, i32) {
        (x, y, z)
    }
}

// ---------------------------------------------------------------------------
// BinaryHeapOpenSet (min-heap on combinedCost)
// ---------------------------------------------------------------------------

/// A min-binary-heap priority queue keyed on `combined_cost`.
/// Ported from baritone's `BinaryHeapOpenSet`.
///
/// Uses a `Vec<PathNode>` and a parallel `keys` array for cache-friendly
/// double comparisons (same trick the Java version uses).
pub struct BinaryHeapOpenSet {
    // 1-indexed array (index 0 is unused, matching the Java implementation)
    nodes: Vec<Option<PathNode>>,
    keys: Vec<f64>,
    size: usize,
}

impl BinaryHeapOpenSet {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1);
        BinaryHeapOpenSet {
            nodes: vec![None; cap],
            keys: vec![0.0; cap],
            size: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn insert(&mut self, node: PathNode) {
        if self.size + 1 >= self.nodes.len() {
            let new_cap = self.nodes.len() * 2;
            self.nodes.resize(new_cap, None);
            self.keys.resize(new_cap, 0.0);
        }
        self.size += 1;
        self.sift_up(self.size, node);
    }

    fn sift_up(&mut self, mut index: usize, node: PathNode) {
        let cost = node.combined_cost;
        while index > 1 {
            let parent = index >> 1;
            if self.keys[parent] <= cost {
                break;
            }
            // Move parent down.
            let parent_node = self.nodes[parent].take().unwrap();
            self.nodes[index] = Some(parent_node);
            self.keys[index] = self.keys[parent];
            if let Some(pn) = &mut self.nodes[index] {
                pn.heap_position = index as i32;
            }
            index = parent;
        }
        self.nodes[index] = Some(node);
        self.keys[index] = cost;
        if let Some(n) = &mut self.nodes[index] {
            n.heap_position = index as i32;
        }
    }

    pub fn remove_lowest(&mut self) -> Option<PathNode> {
        if self.size == 0 {
            return None;
        }
        let size = self.size;

        // The element to return: the root (index 1).
        let mut result = self.nodes[1].clone().unwrap();
        result.heap_position = -1;

        // The last element (index = size), to move to the root.
        let last = self.nodes[size].clone().unwrap();
        let last_cost = self.keys[size];

        // Clear both slots.
        self.nodes[1] = None;
        if size != 1 {
            self.nodes[size] = None;
        }
        self.size -= 1;

        // If there was only one element, we're done.
        if self.size == 0 {
            return Some(result);
        }

        let mut index = 1usize;
        let cost = last_cost;
        let val = last;

        loop {
            let child = index << 1; // left child
            if child > self.size {
                break;
            }
            let mut child_idx = child;
            let mut child_cost = self.keys[child];
            if child < self.size && self.keys[child + 1] < child_cost {
                child_idx = child + 1;
                child_cost = self.keys[child_idx];
            }
            if cost <= child_cost {
                break;
            }
            let child_node = self.nodes[child_idx].take().unwrap();
            self.nodes[index] = Some(child_node);
            self.keys[index] = child_cost;
            if let Some(cn) = &mut self.nodes[index] {
                cn.heap_position = index as i32;
            }
            index = child_idx;
        }

        self.nodes[index] = Some(val);
        self.keys[index] = cost;
        if let Some(n) = &mut self.nodes[index] {
            n.heap_position = index as i32;
        }

        Some(result)
    }
}

impl Default for BinaryHeapOpenSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Path (result of A* search)
// ---------------------------------------------------------------------------

/// A computed path: a sequence of block positions and per-step movements.
/// Ported from `PathBase` / `IPath`. `breaks[i]` lists the blocks the
/// executor must mine before walking step `i` (baritone's
/// `positionsToBreak`), parallel to `movements`.
pub struct Path {
    pub positions: Vec<(i32, i32, i32)>,
    pub movements: Vec<Moves>,
    pub breaks: Vec<Vec<(i32, i32, i32)>>,
    pub num_nodes_considered: i32,
    pub goal: Box<dyn Goal>,
}

impl Path {
    pub fn length(&self) -> usize {
        self.positions.len()
    }

    pub fn src(&self) -> (i32, i32, i32) {
        self.positions[0]
    }

    pub fn positions(&self) -> &[(i32, i32, i32)] {
        &self.positions
    }

    pub fn movements(&self) -> &[Moves] {
        &self.movements
    }

    pub fn breaks(&self) -> &[Vec<(i32, i32, i32)>] {
        &self.breaks
    }

    pub fn dest(&self) -> (i32, i32, i32) {
        self.positions[self.positions.len() - 1]
    }
}

/// Reconstruct a Path by walking back `previous` pointers.
fn reconstruct_path(
    node: &PathNode,
    num_nodes: i32,
    goal: Box<dyn Goal>,
    world: &LiveWorld,
    tools: &ToolSet,
) -> Path {
    let mut positions: Vec<(i32, i32, i32)> = Vec::new();
    let mut movements: Vec<Moves> = Vec::new();
    let mut cur: Option<&PathNode> = Some(node);
    while let Some(n) = cur {
        positions.push((n.x, n.y, n.z));
        // The move that led INTO this node — None only on the start node.
        if let Some(mv) = n.previous_move {
            movements.push(mv);
        }
        cur = n.previous.as_deref();
    }
    positions.reverse();
    movements.reverse();

    // Blocks to mine per step, from the same pricing the search used.
    let mut breaks: Vec<Vec<(i32, i32, i32)>> = Vec::with_capacity(movements.len());
    for i in 1..positions.len() {
        let (px, py, pz) = positions[i - 1];
        breaks.push(match movements.get(i - 1) {
            Some(mv) => movement_breaks(*mv, world, tools, px, py, pz),
            None => Vec::new(),
        });
    }

    Path {
        positions,
        movements,
        breaks,
        num_nodes_considered: num_nodes,
        goal,
    }
}

// ---------------------------------------------------------------------------
// A* main entry point
// ---------------------------------------------------------------------------

/// Configuration for the A* pathfinder.
#[derive(Clone)]
pub struct SearchConfig {
    /// Maximum number of nodes to consider (timeout).
    pub max_nodes: i32,
    /// Maximum fall height before pathing considers it unsafe.
    pub max_fall_height: i32,
    /// Minimum Y level (to prevent void falling in the End).
    pub min_y: i32,
    /// Prefer stairs/slabs for pathfinding (not yet implemented).
    pub allow_1_by_1_tunnel: bool,
    /// Whether the player has sprint available.
    pub can_sprint: bool,
    /// Water walking speed modifier (Jesus enchantment).
    pub water_walk_speed: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            max_nodes: 100_000,
            max_fall_height: 3,
            min_y: -64,
            allow_1_by_1_tunnel: false,
            can_sprint: true,
            water_walk_speed: action_costs::WALK_ONE_IN_WATER_COST,
        }
    }
}

/// Run A* pathfinding from `start` to `goal`, using the [`LiveWorld`] for
/// block queries and `tools` (the hotbar) for mining times.
///
/// Ported from `AStarPathFinder.findPath`.
pub fn path_finder(
    world: &LiveWorld,
    tools: &ToolSet,
    goal: Box<dyn Goal>,
    start: (i32, i32, i32),
    config: &SearchConfig,
) -> Option<Path> {
    let (sx, sy, sz) = start;

    // The player often stands across block boundaries: the floor of the
    // feet position may be air while a neighboring column holds them up.
    // Prefer the exact block; else any of the four horizontal neighbors
    // that is actually walked on.
    let starts = [(sx, sz), (sx + 1, sz), (sx - 1, sz), (sx, sz + 1), (sx, sz - 1)];
    let (sx, sz) = starts
        .iter()
        .copied()
        .find(|&(x, z)| crate::pathing::movement::can_walk_on(world, x, sy - 1, z))
        .unwrap_or((sx, sz));

    let mut open_set = BinaryHeapOpenSet::new();
    let mut node_map: HashMap<(i32, i32, i32), PathNode> = HashMap::new();
    // Positions already expanded. The heap keeps stale entries when a node
    // is relaxed (lazy deletion); they are skipped here when popped.
    let mut closed: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut nodes_considered: i32 = 0;

    // Verify the start position is walkable — or at least occupiable:
    // baritone's PathStart tolerates a water column (the player swims
    // from there) and the executor re-plans mid-fall, so a start the
    // player can occupy seeds the search instead of refusing it. The old
    // gate read the water below a swimmer as unwalkable ground and every
    // re-plan died, stranding the swim.
    // (Baritone checks this before searching.)
    if !crate::pathing::movement::can_walk_on(world, sx, sy - 1, sz)
        && !crate::pathing::movement::can_walk_through(world, sx, sy, sz)
    {
        log::warn!("start position {sx} {sy} {sz} is not walkable");
        return None;
    }

    // Seed the start node.
    let mut start_node = PathNode::new(sx, sy, sz, &*goal);
    start_node.cost = 0.0;
    start_node.combined_cost = start_node.estimated_cost_to_goal;
    open_set.insert(start_node.clone());
    node_map.insert((sx, sy, sz), start_node.clone());

    // Baritone's bestSoFar (`AStarPathFinder`): the best node per cost
    // coefficient. A high coefficient devalues the cost paid so far, so
    // its track reaches past expensive terrain — a wall to mine, which
    // the plain-cost track never wins, because mining raises the cost
    // faster than the heuristic falls.
    let mut best_heuristic = [f64::INFINITY; PARTIAL_COEFFICIENTS.len()];
    let mut best_nodes: Vec<Option<PathNode>> = vec![None; PARTIAL_COEFFICIENTS.len()];
    for i in 0..PARTIAL_COEFFICIENTS.len() {
        best_heuristic[i] = start_node.estimated_cost_to_goal;
        best_nodes[i] = Some(start_node.clone());
    }
    // The pure goal-estimate track, free of the cost/coeff mixing: the
    // coefficient tracks keep the search honest on cost, but in terrain
    // where every move is expensive (digging through stone bare-handed)
    // they never leave the start, and the partial below would refuse a
    // perfectly good one-block descend. This track follows pure progress
    // toward the goal instead.
    let mut best_progress = Some(start_node.clone());
    let mut best_progress_estimate = start_node.estimated_cost_to_goal;

    while nodes_considered < config.max_nodes {
        let Some(current) = open_set.remove_lowest() else {
            break;
        };
        let key = (current.x, current.y, current.z);

        // Skip stale heap entries: either already expanded, or superseded
        // by a cheaper route (whose entry sorts before this one).
        if !closed.insert(key) {
            continue;
        }
        nodes_considered += 1;

        // Goal check.
        if goal.is_in_goal(current.x, current.y, current.z) {
            return Some(reconstruct_path(&current, nodes_considered, goal, world, tools));
        }

        let current = Arc::new(current);

        // Expand neighbors.
        for mv in Moves::all() {
            let res = mv.cost(world, tools, current.x, current.y, current.z);
            if res.cost >= action_costs::COST_INF {
                continue;
            }

            let (nx, ny, nz) = (res.x, res.y, res.z);
            let tentative_g = current.cost + res.cost;

            let key = PathNode::pos_key(nx, ny, nz);
            let node = if let Some(existing) = node_map.get_mut(&key) {
                if tentative_g >= existing.cost {
                    continue;
                }
                existing.cost = tentative_g;
                existing.combined_cost = tentative_g + existing.estimated_cost_to_goal;
                existing.previous = Some(Arc::clone(&current));
                existing.previous_move = Some(*mv);
                open_set.insert(existing.clone());
                existing.clone()
            } else {
                let mut new_node = PathNode::new(nx, ny, nz, &*goal);
                new_node.cost = tentative_g;
                new_node.combined_cost = tentative_g + new_node.estimated_cost_to_goal;
                new_node.previous = Some(Arc::clone(&current));
                new_node.previous_move = Some(*mv);
                open_set.insert(new_node.clone());
                node_map.insert(key, new_node.clone());
                new_node
            };
            for (i, coeff) in PARTIAL_COEFFICIENTS.iter().enumerate() {
                let heuristic = node.estimated_cost_to_goal + node.cost / coeff;
                if best_heuristic[i] - heuristic > 0.01 {
                    best_heuristic[i] = heuristic;
                    best_nodes[i] = Some(node.clone());
                }
            }
            if node.estimated_cost_to_goal < best_progress_estimate - 0.01 {
                best_progress_estimate = node.estimated_cost_to_goal;
                best_progress = Some(node.clone());
            }
        }
    }

    // Baritone's bestPathSoFar: when the search ends without reaching the
    // goal (the node budget ran out, or no route exists), return the best
    // partial path — per cost coefficient, the first track whose best
    // node traveled at least MIN_DIST_PATH blocks from the start. Callers
    // re-path from wherever it ends (baritone's path splicing). Without
    // this, a goal outside the loaded chunks or past the budget would
    // yield nothing instead of a path toward it.
    for node in best_nodes.iter().flatten() {
        let (dx, dy, dz) = (
            f64::from(node.x - sx),
            f64::from(node.y - sy),
            f64::from(node.z - sz),
        );
        if dx * dx + dy * dy + dz * dz > MIN_DIST_PATH * MIN_DIST_PATH {
            log::warn!(
                "A* search fell short of the goal: {} nodes considered, {} created; \
                 returning a partial path",
                nodes_considered,
                node_map.len()
            );
            return Some(reconstruct_path(node, nodes_considered, goal, world, tools));
        }
    }

    // Enclosed spaces (a 1-wide shaft the bot dug itself) end the search
    // with every track shorter than MIN_DIST_PATH — refusing them would
    // strand the bot one plan after it starts digging down. Take the best
    // pure-progress track when it actually approaches the goal: a
    // one-block descend is exactly what a shaft needs, and it re-plans
    // from the new floor. A dead end (a flooded column: every breaking
    // move is COST_INF) offers no track that improves the estimate, so it
    // still comes back None and the mission can escape.
    if let Some(node) = best_progress
        && node.estimated_cost_to_goal < start_node.estimated_cost_to_goal - 0.01
    {
        log::warn!("A* search exhausted short of MIN_DIST_PATH: taking a progress partial");
        return Some(reconstruct_path(&node, nodes_considered, goal, world, tools));
    }

    log::warn!(
        "A* search exhausted: {} nodes considered, {} created",
        nodes_considered,
        node_map.len()
    );
    None
}
