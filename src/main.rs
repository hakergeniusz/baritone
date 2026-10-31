use anyhow::{anyhow, Result};
use std::time::Instant;

use baritone::{execute_path, path_finder, Goal, GoalBlock, GoalXZ, SearchConfig};

/// Parse command-line args: `baritone X Y Z [--execute]`
/// - `baritone X Y Z` — find and print path only
/// - `baritone X Y Z --execute` — find path and execute it via horizon-ctl
/// - `baritone X Z` — treat as GoalXZ (long-distance, any Y)
fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let execute = args.iter().any(|a| a == "--execute" || a == "--run");

    let coords: Vec<&str> = args
        .iter()
        .skip(1)
        .filter(|a| a.as_str() != "--execute" && a.as_str() != "--run")
        .map(|s| s.as_str())
        .collect();

    let (goal, target_desc): (Box<dyn Goal>, String) = match coords.len() {
        2 => {
            let x = coords[0].parse::<i32>().map_err(|_| anyhow!("invalid X"))?;
            let z = coords[1].parse::<i32>().map_err(|_| anyhow!("invalid Z"))?;
            (
                Box::new(GoalXZ::new(x, z)),
                format!("XZ=({x}, {z})"),
            )
        }
        3 => {
            let x = coords[0].parse::<i32>().map_err(|_| anyhow!("invalid X"))?;
            let y = coords[1].parse::<i32>().map_err(|_| anyhow!("invalid Y"))?;
            let z = coords[2].parse::<i32>().map_err(|_| anyhow!("invalid Z"))?;
            (Box::new(GoalBlock::new(x, y, z)), format!("({x}, {y}, {z})"))
        }
        _ => {
            eprintln!("Usage: baritone <X> <Y> <Z> [--execute]");
            eprintln!("       baritone <X> <Z> [--execute]");
            return Ok(());
        }
    };

    // Connect to the Horizon game.
    let world = baritone::LiveWorld::new()?;
    let (sx, sy, sz) = world.player_pos()?;
    println!("Player at: ({sx}, {sy}, {sz})");
    println!("Pathing to: {target_desc}");

    // Run A* pathfinding.
    let start = Instant::now();
    println!("Searching for path (max {} nodes)...", SearchConfig::default().max_nodes);
    let path = path_finder(&world, goal, (sx, sy, sz), &SearchConfig::default());
    let elapsed = start.elapsed();

    match &path {
        Some(p) => {
            println!(
                "Path found in {:?}: {} steps, {} nodes considered",
                elapsed,
                p.length(),
                p.num_nodes_considered
            );
            for (i, pos) in p.positions().iter().enumerate() {
                let mv = if i < p.movements().len() {
                    format!("{:?}", p.movements()[i])
                } else {
                    "–".to_string()
                };
                println!("  {:>4}: {:?}  {}", i, pos, mv);
            }

            if execute {
                println!("\nExecuting path...");
                execute_path(&world, p)?;
            }
        }
        None => {
            println!("No path found in {:?}.", elapsed);
        }
    }

    Ok(())
}
