# HANDOFF 2 — baritone→Horizon port: stall fix written, not yet applied

> **RESOLVED 2026-11-04** — everything below was applied, extended by the
> snow-rest / water-start / step-up / ray-probe fixes, verified live
> (descent stall-free, /goto, chat-open drive), tested 58/58, committed as
> `3554f4d` and pushed. The known remaining issues are at the bottom of the
> memory file (`baritone-horizon-port.md`, Nov 4 entry): escape walks can
> freeze on freshly streamed terrain (LiveWorld reads air where the game
> rests the player — the 3-dead-end cap then ends /mine legally), powder
> snow is unpathable by design, and the live deep-water swim rig needs
> water settling time (sim-verified only).

Written 2026-11-03, mid-fix. Read `HANDOFF.md` first for the original four
failures and the environment. Everything there is resolved except what this
file says is still open. Repo is on this machine (`~/Projects/baritone`),
pushed through `d45ac9a`; the working tree holds the uncommitted fixes below.

## State when this file was written

- **Full test suite PASSES: 47/47, exit 0, ~52 s** (the old >25-min grind is
  gone — the revive-loop bug it exposed is fixed by the progress track +
  escape cap). Run via `/tmp/ai-agent/full_suite.sh` (nice 19).
- **Final binary built + installed 20:25** to
  `~/Downloads/horizon-linux-x86_64/horizon-client-baritone` (contains all
  HANDOFF.md-era fixes: snow swing, GoalNear grace, strata hunt, escape
  walk/revival/cap, blacklist preservation, at-dest fallthrough, watchdog).
- **Diamond hunt live-verified on that binary**: `/mine diamond_ore` →
  "no diamond_ore in range — digging to Y -59", segmented descent
  (Y51 → Y45 → Y14), then ore found in caves at y-32: **24 collected,
  33-34 in queue** when this file was written. The user's "quits instantly"
  is fixed and verified live.
- Test game still running: world `baritonefail2` seed 4242 on `:14`, socket
  `/tmp/.ai-agent-horizon-mine.sock`, log `/tmp/ai-agent/diamond2.log`
  (watchdog heartbeats), poll log `/tmp/ai-agent/diamond2.run.log`. Kill it
  before relaunching (`pgrep -f "horizon-client-baritone --world"`).
- Full suite watcher output: `/tmp/ai-agent/full_suite.log` (SUITE-EXIT:0).

## THE ONE OPEN BUG: descend-stall (root-caused, fix script ready, NOT applied)

**Symptom**: the bot froze at a pit rim `(-15.7, 21, -9.5)` for ~7 min,
jump-bobbing in place (y 21↔22.2), `mining: None`, plans flip-flopping
between length 10 and 14 (step 2/10 ↔ 3/14), `stuck 0/3` forever. It
un-stuck only because I probed blocks with `/fill` (broke its support → it
fell → re-planned). Same freeze had happened on the intermediate binary at
the same spot (the "cosmetic stale gravel stint" in the earlier session was
NOT cosmetic — it was this).

**Root cause chain, all three verified from probes + code reading**:

1. `WAYPOINT_Y_TOLERANCE = 1.1` (bot.rs:74) lets the waypoint-advance loop
   wave the player through a descend waypoint while it stands ABOVE it
   (Δy=1.0 < 1.1). The aim then targets a move corridor that assumes the
   drop happened; the player — one block higher — grinds its HEAD into
   solid terrain the plan never priced. Probe-verified: solid block at
   `(-17, 22, -10)` exactly where the walk pushed, while the plan's
   `breaks []` and the waypoint guard (`horizontal < 2.0`, checks the
   waypoint's own feet+head at y19/20) saw nothing.
2. `best_distance` progress reset: mid-hop the player's block alternates
   y21/y22 → each periodic replan starts from a different block → paths of
   different length → the aim point flips → `distance_xz(player, aim)`
   "improves" → `progress_tick` resets → `stalled` never reaches
   `STUCK_REPLAN_TICKS` → stuck replans/give-up never fire. The stuck-hop
   (60-tick) fires, which is the bob.
3. No mining fallback for plan-vs-reality mismatch beyond the waypoint
   guard: `breaks []` + solid block in front = push forever.

**Fix**: `/tmp/ai-agent/stall_fix.py` (on BOTH machines, corrected). It:

- **A**: signed Y gate in the advance loop — waypoints the path lands in
  from below-level (`positions[i].1 < positions[i-1].1`) require
  `player.y - center.1 <= 0.55` instead of `|Δy| < 1.1`;
- **B**: front-of-player mining fallback — when the plan's step-breaks
  yield nothing, probe the two blocks straight ahead of the player (feet
  and head height, aim direction, axes with |Δ|>0.3) and mine the first
  solid breakable one; chained via `break_target.or_else(...)` into the
  existing mining_tick match so Failed/Timeout handling is shared. This is
  Java baritone's "break what blocks you" behavior;
- **C**: displacement-anchored progress — `best_distance` field removed,
  `progress_anchor: (f64, f64)` + `mark_progress(player)` helper added;
  all `progress_tick` resets go through it; the walk branch resets only on
  >0.5 horizontal displacement from the anchor (aim flips no longer count
  as progress);
- **D**: per-tick `BARITONE_DEBUG=1` eprintln trace in the REAL walk branch
  (pos/yaw/fwd/jump/sprint/ground/mine/step/aim) — the sim drive() trace
  already existed, the real path didn't;
- plus 2 regression tests: `descend_waypoint_needs_the_actual_drop` (pit
  rim world, hover 1 above the landing waypoint must NOT advance) and
  `mines_a_solid_block_the_plan_missed_in_front` (plan walks east, player
  1 high, solid at (1,67,0) ahead → must mine it).

**Script status**: first run aborted on a wrong assert (counted 6
`progress_tick` sites, 5 remain after C-1/C-2 replacements) — **before
`write_text`, so bot.rs is UNCHANGED**. The local copy is corrected to
`== 5`. Next session: `scp /tmp/ai-agent/stall_fix.py acer:/tmp/ai-agent/`,
run it, `python3 tools/assemble.py`, `cargo test --locked --lib` (expect
49 passing), rebuild via `/tmp/ai-agent/build_variant.sh normal` (nice 19),
then re-verify.

## Remaining after the stall fix

1. Re-verify `/mine diamond_ore` on a FRESH world (`baritonefail3` —
   baritonefail2 has the old shaft/gravel mess baked in; the pre-dug shaft
   is why this run reached y21 in 45 s without mining) with
   `BARITONE_DEBUG=1 BARITONE_WATCHDOG=1` (extend `/tmp/ai-agent/diamond2.sh`).
2. `/goto` regression, chat-open drive (`xdotool key t` on `:14`), deep-water
   swim rig (`build_rig.sh` + `verify_swim.sh`) — none re-run on the final
   binary yet.
3. Full suite again, then commit (Conventional Commits + `Assisted-by: Pi`),
   push, and update memory `~/.zcode/cli/memories/projects/baritone-271b43da6c82ebb8/memory/baritone-horizon-port.md`.
4. Cleanup `/tmp/ai-agent/` scripts + test game.

## Diagnosis tricks that cracked this session (keep them)

- **Block probe without a query API**: `ctl command "/fill x y z x y z
  minecraft:air"` → "Successfully filled 1 block(s)" = was solid, "No
  blocks were filled" = was air. One /fill per block, serial, ~0.4 s apart.
  probe_blocks.sh / probe_head.sh in `/tmp/ai-agent/`.
- `ctl state` → `looking_at` gives the crosshair's first solid hit
  (block name + pos + face) — free ground truth on what the bot faces.
- Chat history via `ctl messages 20 styled` showed the whole hunt plan
  ("pathing to Y=… — N blocks from …") without touching the bot.
- The frozen-position signature: x/z frozen while y bobs = executor pushes
  into something; check looking_at, then probe the head-height block in the
  push direction.

## Hard requirements (unchanged, repeated)

- **LOW CPU PRIORITY for everything of ours** — the user runs their own
  qemu-aarch64 cross-build on this box: builds/tests `nice -n 19 ionice -c3`,
  test games `renice -n 10`, prefer renicing running processes over
  restarting them.
- Headless game launches MUST set `XDG_RUNTIME_DIR=/run/user/1000`
  (else libjack/pipewire deadlock in cpal enumeration on the main thread —
  the "bot pauses" mystery; not a bot bug).
- `python3 tools/assemble.py` before every cargo test/build.
- One ctl client at a time; bot command replies don't reach chat
  (`/status` via ctl returns no reply — use the watchdog heartbeat).
- Never `cp` over the installed binary while a game runs: cp-to-temp + `mv`
  (build_variant.sh does this).
- PATH for cargo: prepend
  `/home/hakergeniusz/.local/share/horizon-launcher/rust/1.99.1-x86_64-unknown-linux-gnu/bin`,
  `CARGO_HOME=~/.local/share/horizon-launcher/cargo`,
  `CARGO_TARGET_DIR=~/.cache/horizon-baritone-build`,
  `TMPDIR=~/.cache/tmp-build` (all in build_variant.sh / full_suite.sh).
