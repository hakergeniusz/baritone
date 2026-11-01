//! Integration tests for the pathfinder's pure logic (goals, costs, heap),
//! run without a live Horizon game. The movement cost functions and the A*
//! search itself need a `LiveWorld` (socket-backed), so they are exercised
//! by the live verification instead.

use baritone::{Goal, GoalBlock, GoalXZ};

#[test]
fn test_goal_xz_simple() {
    // GoalXZ should find a path on flat ground.
    let goal: Box<dyn Goal> = Box::new(GoalXZ::new(10, 10));
    assert!(goal.is_in_goal(10, 64, 10));
    assert!(!goal.is_in_goal(5, 64, 5));

    let h = goal.heuristic(0, 64, 0);
    assert!(h > 0.0 && !h.is_nan());
}

#[test]
fn test_goal_block_simple() {
    let goal: Box<dyn Goal> = Box::new(GoalBlock::new(10, 64, 10));
    assert!(goal.is_in_goal(10, 64, 10));
    assert!(!goal.is_in_goal(10, 65, 10));
}

#[test]
fn test_action_costs() {
    use baritone::goals::action_costs;

    // Walk cost should be ~4.63 (a compile-time constant).
    const { assert!(action_costs::WALK_ONE_BLOCK_COST > 4.0 && action_costs::WALK_ONE_BLOCK_COST < 5.0) };

    // Fall N blocks cost should increase with N.
    let c0 = action_costs::fall_n_blocks_cost(0);
    let c1 = action_costs::fall_n_blocks_cost(1);
    let c3 = action_costs::fall_n_blocks_cost(3);
    assert_eq!(c0, 0.0);
    assert!(c1 > 0.0 && c1 < c3);

    // Jump cost is positive and small.
    let j = action_costs::jump_one_block_cost();
    assert!(j > 2.0 && j < 5.0);
}

#[test]
fn test_binary_heap_open_set() {
    use baritone::BinaryHeapOpenSet;
    use baritone::PathNode;

    // A simple goal for testing.
    struct TestGoal;
    impl Goal for TestGoal {
        fn is_in_goal(&self, _x: i32, _y: i32, _z: i32) -> bool {
            false
        }
        fn heuristic(&self, _x: i32, _y: i32, _z: i32) -> f64 {
            0.0
        }
    }

    let goal = TestGoal;
    let mut heap = BinaryHeapOpenSet::new();

    // Insert some nodes with different costs.
    let mut n1 = PathNode::new(0, 0, 0, &goal);
    n1.cost = 5.0;
    n1.combined_cost = 5.0;
    heap.insert(n1);

    let mut n2 = PathNode::new(1, 0, 0, &goal);
    n2.cost = 1.0;
    n2.combined_cost = 1.0;
    heap.insert(n2);

    let mut n3 = PathNode::new(0, 1, 0, &goal);
    n3.cost = 3.0;
    n3.combined_cost = 3.0;
    heap.insert(n3);

    // Lowest should come out first.
    let lowest = heap.remove_lowest().unwrap();
    assert_eq!(lowest.cost, 1.0);

    let next = heap.remove_lowest().unwrap();
    assert_eq!(next.cost, 3.0);

    let last = heap.remove_lowest().unwrap();
    assert_eq!(last.cost, 5.0);

    assert!(heap.is_empty());
}

#[test]
fn binary_heap_surfaces_relaxed_node_first() {
    use baritone::{BinaryHeapOpenSet, PathNode};

    struct TestGoal;
    impl Goal for TestGoal {
        fn is_in_goal(&self, _x: i32, _y: i32, _z: i32) -> bool {
            false
        }
        fn heuristic(&self, _x: i32, _y: i32, _z: i32) -> f64 {
            0.0
        }
    }

    let goal = TestGoal;
    let mut heap = BinaryHeapOpenSet::new();

    // Simulate relaxing a node: first discovered at cost 9, then improved
    // to 4. The heap keeps both entries (lazy deletion); the search skips
    // the stale one via its closed set, but the improved cost must surface
    // first so the node is expanded with the best known route.
    let mut first = PathNode::new(3, 0, 3, &goal);
    first.cost = 9.0;
    first.combined_cost = 9.0;
    heap.insert(first);

    let mut better = PathNode::new(3, 0, 3, &goal);
    better.cost = 4.0;
    better.combined_cost = 4.0;
    heap.insert(better);

    assert_eq!(heap.remove_lowest().unwrap().cost, 4.0);
    assert_eq!(heap.remove_lowest().unwrap().cost, 9.0);
    assert!(heap.is_empty());
}

/// The goal heuristics must match baritone's Java formulas exactly — in
/// particular the vertical component is *not* multiplied by the walk cost,
/// and ascend/descend branches are not swapped (either mistake makes the
/// heuristic inadmissible and A* miss optimal paths).
#[test]
fn goal_heuristics_match_baritone_formulas() {
    use baritone::goals::action_costs::{
        WALK_ONE_BLOCK_COST, fall_n_blocks_cost, jump_one_block_cost,
    };
    use baritone::{GoalBlock, GoalTwoBlocks, GoalXZ, GoalYLevel};

    // GoalYLevel.calculate: ascend → JUMP per level, descend → FALL[2]/2 per level.
    let goal = GoalYLevel { y: 64 };
    assert_eq!(goal.heuristic(0, 61, 0), 3.0 * jump_one_block_cost());
    assert_eq!(goal.heuristic(0, 68, 0), fall_n_blocks_cost(2) / 2.0 * 4.0);
    assert_eq!(goal.heuristic(0, 64, 0), 0.0);

    // GoalBlock.calculate = GoalYLevel.calculate(0, yDiff) + GoalXZ.calculate(xDiff, zDiff).
    let goal = GoalBlock::new(10, 64, 10);
    let h = goal.heuristic(7, 66, 13);
    let expected = GoalYLevel::calculate(0, 2) + GoalXZ::calculate(-3.0, -3.0);
    assert_eq!(h, expected);

    // One block below the goal costs exactly one jump — well under a real
    // ascend move (WALK + JUMP), so the heuristic stays admissible.
    let ascend_h = goal.heuristic(10, 63, 10);
    assert_eq!(ascend_h, jump_one_block_cost());
    assert!(
        ascend_h <= WALK_ONE_BLOCK_COST + jump_one_block_cost(),
        "heuristic for ascending 1 block must not exceed the ascend move cost"
    );
    // One block above the goal is a descend: half a 2-block fall per level.
    assert_eq!(goal.heuristic(10, 65, 10), fall_n_blocks_cost(2) / 2.0);

    // GoalTwoBlocks folds a y difference of -1 to 0 (standing one block
    // below the top counts as level).
    let two = GoalTwoBlocks { x: 5, y: 40, z: 5 };
    assert_eq!(two.heuristic(5, 39, 5), GoalXZ::calculate(0.0, 0.0));
    assert_eq!(two.heuristic(8, 41, 3), GoalBlock::calculate(3.0, 1, -2.0));
}

#[test]
fn goal_near_matches_baritone() {
    use baritone::{GoalBlock, GoalNear};

    // GoalNear.heuristic delegates to GoalBlock.calculate in baritone too.
    let goal = GoalNear {
        x: 0,
        y: 64,
        z: 0,
        range: 3.0,
    };
    assert!(goal.is_in_goal(1, 64, 2));
    assert!(!goal.is_in_goal(4, 64, 0));
    assert_eq!(
        goal.heuristic(7, 63, -2),
        GoalBlock::calculate(7.0, -1, -2.0)
    );
}

/// Parkour path labels must match the jump direction exactly; the old
/// distance-only check labeled every 4-block jump as ParkourNorth.
#[test]
fn from_delta_labels_parkour_directions() {
    use baritone::Moves;

    assert_eq!(Moves::from_delta(0, 0, -4), Some(Moves::ParkourNorth));
    assert_eq!(Moves::from_delta(0, 0, 4), Some(Moves::ParkourSouth));
    assert_eq!(Moves::from_delta(4, 0, 0), Some(Moves::ParkourEast));
    assert_eq!(Moves::from_delta(-4, 0, 0), Some(Moves::ParkourWest));

    // A diagonal 4-offset is not a parkour move.
    assert_eq!(Moves::from_delta(2, 0, 2), None);
    // Parkour lands flat; an ascend delta is not a parkour move.
    assert_eq!(Moves::from_delta(0, 1, -4), None);
}
