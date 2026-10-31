//! A* path search and path representation.
//!
//! Ported from:
//! - `AStarPathFinder.java` — main search loop
//! - `PathNode.java` — search node
//! - `BinaryHeapOpenSet.java` — min-heap priority queue
//! - `PathBase.java` / `IPath.java` — path result representation

use std::collections::HashMap;

use crate::goals::{action_costs, Goal};
use crate::pathing::movement::Moves;
use crate::world::LiveWorld;

// ---------------------------------------------------------------------------
// PathNode
// ---------------------------------------------------------------------------

/// A node in the A* open/closed set, holding cost state and back-pointer.
#[derive(Clone)]
pub struct PathNode {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub estimated_cost_to_goal: f64,
    pub cost: f64,
    pub combined_cost: f64,
    pub previous: Option<Box<PathNode>>,
    pub heap_position: i32,
    pub favor: f64,
}

impl PathNode {
    pub fn new(x: i32, y: i32, z: i32, goal: &dyn Goal) -> Self {
        let estimated = goal.heuristic(x, y, z);
        // Baritone throws if heuristic is NaN; we keep that invariant.
        assert!(
            !estimated.is_nan(),
            "heuristic returned NaN at {x} {y} {z}"
        );
        PathNode {
            x,
            y,
            z,
            estimated_cost_to_goal: estimated,
            cost: action_costs::COST_INF,
            combined_cost: action_costs::COST_INF,
            previous: None,
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
/// Ported from `PathBase` / `IPath`.
pub struct Path {
    pub positions: Vec<(i32, i32, i32)>,
    pub movements: Vec<Moves>,
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

    pub fn dest(&self) -> (i32, i32, i32) {
        self.positions[self.positions.len() - 1]
    }
}

/// Reconstruct a Path by walking back `previous` pointers.
fn reconstruct_path(node: &PathNode, num_nodes: i32, goal: Box<dyn Goal>) -> Path {
    let mut positions: Vec<(i32, i32, i32)> = Vec::new();
    let mut cur: Option<&PathNode> = Some(node);
    while let Some(n) = cur {
        positions.push((n.x, n.y, n.z));
        cur = n.previous.as_deref();
    }
    positions.reverse();

    // Movements: for each step, determine the move type from the delta.
    let mut movements: Vec<Moves> = Vec::new();
    for i in 1..positions.len() {
        let (px, py, pz) = positions[i - 1];
        let (cx, cy, cz) = positions[i];
        let dx = cx - px;
        let dy = cy - py;
        let dz = cz - pz;
        if let Some(m) = Moves::from_delta(dx, dy, dz) {
            movements.push(m);
        }
    }

    Path {
        positions,
        movements,
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
/// block queries.
///
/// Ported from `AStarPathFinder.findPath`.
pub fn path_finder(
    world: &LiveWorld,
    goal: Box<dyn Goal>,
    start: (i32, i32, i32),
    config: &SearchConfig,
) -> Option<Path> {
    let (sx, sy, sz) = start;

    let mut open_set = BinaryHeapOpenSet::new();
    let mut node_map: HashMap<(i32, i32, i32), PathNode> = HashMap::new();
    let mut nodes_considered: i32 = 0;

    // Verify the start position is walkable.
    // (Baritone checks this before searching.)
    if !crate::pathing::movement::can_walk_on(world, sx, sy - 1, sz) {
        log::warn!("start position {sx} {sy} {sz} is not walkable");
        return None;
    }

    // Seed the start node.
    let mut start_node = PathNode::new(sx, sy, sz, &*goal);
    start_node.cost = 0.0;
    start_node.combined_cost = start_node.estimated_cost_to_goal;
    start_node.heap_position = 1; // set by insert
    open_set.insert(start_node.clone());
    node_map.insert((sx, sy, sz), start_node);

    let mut best_node: Option<PathNode> = None;
    let mut best_combined_cost = f64::INFINITY;

    while !open_set.is_empty() && nodes_considered < config.max_nodes {
        let current = open_set.remove_lowest()?;
        nodes_considered += 1;

        // Goal check.
        if goal.is_in_goal(current.x, current.y, current.z) {
            return Some(reconstruct_path(&current, nodes_considered, goal));
        }

        // Expand neighbors.
        for mv in Moves::all() {
            let res = mv.cost(world, current.x, current.y, current.z);
            if res.cost >= action_costs::COST_INF {
                continue;
            }

            let (nx, ny, nz) = (res.x, res.y, res.z);
            let tentative_g = current.cost + res.cost;

            let key = PathNode::pos_key(nx, ny, nz);
            if let Some(existing) = node_map.get_mut(&key) {
                if tentative_g < existing.cost {
                    existing.cost = tentative_g;
                    existing.combined_cost = tentative_g + existing.estimated_cost_to_goal;
                    existing.previous = Some(Box::new(current.clone()));
                    open_set.insert(existing.clone());
                    existing.heap_position = -1;
                }
            } else {
                let mut new_node = PathNode::new(nx, ny, nz, &*goal);
                new_node.cost = tentative_g;
                new_node.combined_cost = tentative_g + new_node.estimated_cost_to_goal;
                new_node.previous = Some(Box::new(current.clone()));
                if new_node.combined_cost < best_combined_cost {
                    best_combined_cost = new_node.combined_cost;
                    best_node = Some(new_node.clone());
                }
                open_set.insert(new_node.clone());
                node_map.insert(key, new_node);
            }
        }
    }

    // Return best path found so far (incremental cost backoff), if any.
    if let Some(node) = best_node {
        if node.cost < action_costs::COST_INF && goal.is_in_goal(node.x, node.y, node.z) {
            return Some(reconstruct_path(&node, nodes_considered, goal));
        }
    }

    log::warn!(
        "A* search exhausted: {} nodes considered, {} created",
        nodes_considered,
        node_map.len()
    );
    None
}
