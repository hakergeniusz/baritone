# Horizon Baritone

A pathfinding bot for [Horizon](https://github.com/horizon-rust/horizon),
ported from [Baritone](https://github.com/cabaletta/baritone) (LGPL-3.0).
Like Baritone, it lives inside the game: in-game chat commands set a goal,
and a per-tick state machine drives the player to it — planning with the
A* planner ported in `../src/`, then walking, sprinting, swimming and
jumping by injecting the same inputs a player's keys would.

## Commands

Typed in chat, like the original's `#`-commands:

| Command | Meaning |
| --- | --- |
| `/goto <x> <y> <z>` | Walk to stand on that block |
| `/goto <x> <z>` | Walk to that column, any height |
| `/goto <y>` | Walk to that Y level |
| `/near [r]` | Walk to somewhere within `r` blocks (default 8) |
| `/path` | Re-plan toward the current goal |
| `/stop`, `/cancel` | Drop the goal and release the keys |
| `/help` | List the commands |

Coordinates accept `~` and `~N`, relative to the player's block.

While it walks, the bot re-plans whenever it is stuck for two seconds, has
drifted off its path, every five seconds (as terrain streams in), and at
most three times per stuck stretch before giving up. Partial paths walk
toward goals beyond the loaded chunks or the search budget. Opening a
menu, dying or changing dimension pauses or cancels it; Options > Mods has
the on/off switch.

## Layout

- `../src/` — the planner, ported module by module from Baritone's Java
  (`goals.rs`, `pathing/movement.rs`, `pathing/search.rs`), plus the
  standalone control-socket bot built on it.
- `src/bot.rs` — the brain: goals, commands, and the tick state machine.
  Pure Rust, no game types; tested by the standalone suite.
- `src/world_view.rs` — the planner's block source, cached.
- `mixins/baritone/` — the bundle: `runtime.rs` (generated — don't edit)
  and the helper files injected into the client (tick, chat command,
  command completion, mods list). `horizon-mixin.toml` wires them up.
- `tools/assemble.py` — flattens the sources into `runtime.rs`.
- `tools/test.py` — assemble, then `cargo fmt`/`clippy`/`test` on it.
- `tools/package.py` — deterministic `.hnmod` archive.
- `compatibility.json` — the Horizon revision the helpers were written
  against, with the SHA-256 of every file they patch.

## Building and testing

```sh
tools/test.py                 # assemble + fmt + clippy + tests
tools/package.py              # target/baritone-1.0.0.hnmod
```

To apply the bundle to the game's source and build a modded client
yourself (what the launcher does with the `.hnmod`):

```sh
tools/assemble.py
horizon-mixins apply \
  --workspace <pristine horizon source> \
  --mods mixins --out <staged source>
cargo build --profile fast -p horizon-client   # in the staged source
```

## Not ported (yet)

Mining and block placement mid-path (Baritone's builder and `#mine`),
crafting, following players, and the settings/config system. Movement
that needs blocks is impassable to the planner, so every path is
walk/jump/swim only.
