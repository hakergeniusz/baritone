# baritone-rs

A Rust port of [Baritone](https://github.com/cabaletta/baritone) — a Minecraft
pathfinding bot — for [Horizon](https://github.com/horizon-rust/horizon), a
Rust rewrite of Minecraft Java Edition 26.3.

## License

This project inherits the **GNU Lesser General Public License v3.0 or later**
from the original Baritone source. The full license text is in [`LICENCE`](LICENCE).

## Architecture

| Module | Description |
|--------|-------------|
| `goals` | Pathing goals (GoalXZ, GoalBlock, GoalNear, etc.) and action cost constants (movement speeds, fall costs, etc.). |
| `pathing` | A* search engine (`path_finder`), binary heap open-set, and movement cost calculations (walk, ascend, descend, fall, diagonal, parkour). |
| `world` | World model that queries Horizon's control socket for block states and player position, with a region cache. |
| `executor` | Path executor that sends `goto`/`break`/`place` commands to Horizon via its control socket. |

### Ported from Baritone (LGPL-3.0)

- `baritone.api.pathing.goals.Goal` → `goals::Goal`
- `baritone.api.pathing.movement.ActionCosts` → `goals::action_costs`
- `baritone.pathing.calc.AStarPathFinder` → `pathing::path_finder`
- `baritone.pathing.calc.PathNode` → `pathing::PathNode`
- `baritone.pathing.calc.openset.BinaryHeapOpenSet` → `pathing::BinaryHeapOpenSet`
- `baritone.pathing.path.{Path, SplicedPath}` → `pathing::Path`
- `baritone.pathing.movement.Moves` → `pathing::Moves`
- `baritone.pathing.movement.MovementHelper` → `pathing::movement::block_tags`
- `baritone.pathing.movement.movement.*` → `pathing::movement` cost functions
- `baritone.utils.BlockStateInterface` → `world::World`
- `baritone.pathing.path.PathExecutor` → `executor`

## Usage

```bash
# Ensure Horizon is running with --control and a known socket path
HORIZON_CONTROL=/tmp/my-horizon.sock baritone 100 65 -50      # find path only
HORIZON_CONTROL=/tmp/my-horizon.sock baritone 100 65 -50 --execute  # find + walk the path
baritone 100 -50                                             # GoalXZ (long-distance, any Y)
```

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```
