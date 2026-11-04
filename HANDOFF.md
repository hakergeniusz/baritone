# HANDOFF — baritone→Horizon port: user-reported failure investigation

Written 2026-11-02, mid-investigation. The repo is pushed through `d45ac9a`
(render + /follow + swimming, all live-verified). The user then reported:
**`/mine diamond_ore`, `/mine stone`, goto, and swimming "will fail"**, and
**"bot pauses when you open chat, click esc or other"**. This file is the
state of that investigation.

## Test environment (as of writing)

- Test game RUNNING: pid 4135820, world `baritonefail` seed 4242, on Xvfb
  `:13`, control socket `/tmp/.ai-agent-horizon-mine.sock`, log
  `/tmp/ai-agent/fail-test1.log`. Difficulty peaceful, time noon, player
  near `(-6.9, 63, 0.6)`.
- **Second game belongs to the user** (pid 4128351, bare launch on their
  Wayland session, default socket `/run/user/1000/horizon-control.sock`) —
  do not touch it, and never `cp` over `~/Downloads/horizon-linux-x86_64/
  horizon-client-baritone` while it runs (SIGBUS risk) — install with
  cp-to-temp + `mv`.
- ctl recipe: `~/Downloads/horizon-linux-x86_64/horizon-ctl --socket
  /tmp/.ai-agent-horizon-mine.sock state|command "/..."|messages N styled`.
- Screenshots: `ffmpeg -f x11grab -video_size 1280x720 -i :13 -frames:v 1
  out.png`. Evidence so far: `/tmp/ai-agent/stall{1,2,3}.png`,
  `follow1.png`, `swim1..5.png`, `scene1.png`, `fresh5.png`.
- Gotchas learned the hard way this session: `/kill @e` includes the player
  (died twice) — use `/difficulty peaceful`; `state.mining` is the GAME's
  block-break progress (None = no active break), not the mod's stint; a
  `/fill` rig must check surface height first (a y58–61 pool carved inside
  a mountain whose surface is y78); xdotool needs
  `env -u WAYLAND_DISPLAY DISPLAY=:13`.

## What is verified WORKING on the current binary (don't re-test)

- `/goto` 66-block path over a mountain → "reached (52, 1)".
- `/mine stone` on a mountain cliff: digs down through stone, target queue
  advances (game mining progress climbed, looking_at tracked stone).
- `/follow entities` tracks a moving zombie (retargets 30→8→1 blocks);
  graceful "nothing to follow — stopping" error paths.
- Swim: in-pond goal reached fully submerged (air 300→6); from a pond
  bottom the bot surfaced and reached a shore goal.
- Render: red path line, magenta walk-into boxes, green goal column visible
  in-game (stall1.png even shows the game's block-highlight on snow).

## Reproduced failures

### 1. `/mine stone` STALL on flat ground (root cause NOT yet confirmed)

Setup: player tp'd to `(-7, 63, 1)` (snowy flat, snow layer over dirt),
`/mine stone`. Target stone at `(-6, 60, 1)` — 3 under the surface, ~4.44
from the eye (within 4.5 reach, occluded).

Observed: announces `"pathing to within 3 of (-6, 60, 1) — 2 blocks"` every
~5 s, position frozen for 16 s+, `state.mining: None` the whole time (the
game never even starts a break), and eventually
`"path blocked by unbreakable terrain — giving up"` (from
`MiningOutcome::Failed`, bot.rs:1113 — three stint timeouts on one block).
Screenshot stall2.png shows NO wireframes.

Evidence gathered:
- The plan is "2 blocks" — almost certainly a walk-only 2-block path (east,
  then south) to stand exactly above the target column where
  `dx²+dy²+dz² == 9` satisfies the Near goal.
- Mission-miner gate (bot.rs:1037) is CLOSED at the start:
  `GoalNear::is_in_goal` is Euclidean (src/goals.rs:219-224) and player
  `(-7,63,0)` vs target `(-6,60,1)` = 11 > 9.
- Working hypotheses (pick one, then write a failing unit test):
  a) **Replan loop**: waypoints get consumed by the advance loop
     (bot.rs:980-1012; check `WAYPOINT_REACH` / `LAST_WAYPOINT_REACH` /
     `WAYPOINT_Y_TOLERANCE` values) without actually walking → path
     "complete" but goal unsatisfied → replan → repeat. The ~5 s
     announcement cadence matches the stuck-replan timer.
  b) Snow layer breakability: `hardness::break_ticks("snow")` — check the
     table has an entry for `snow` (bare_name of `snow[layers=1]`); if it's
     COST_INF the step-breaks filter drops it and something downstream
     times out. The "unbreakable terrain" Failed must have come from some
     block — find which.
- Note Java's `GoalNear.isInRange` uses `(range + 0.5)²`; the port uses
  `range²` with no grace. Fixing that alone would open the mission-miner
  gate at the start position (11 ≤ 12.25) and the bot would dig the
  occluder chain straight down — likely the cheapest real fix.

### 2. `/mine diamond_ore` quits instantly

`"mine done — 0 × diamond_ore"` — the scan only finds EXPOSED ore in loaded
chunks; diamond is never exposed on the surface, so it quits. Java baritone
digs toward the ore's likely Y strata instead. Read `MineProcess.java` in
the Java clone (`/tmp/ai-agent/baritone-src`) for the "no known locations →
drop to Y level / strip toward strata" behavior and port enough of it to
dig down to the ore's Y range when the scan is empty.

### 3. Bot pauses when chat opens (user report; root cause FOUND in code)

`mod/mixins/baritone/tick.rs:76`:
`active: game.screen == Screen::Playing && game.sleeping.is_none()` — any
screen (chat!) makes `active=false` and the brain idles. Java baritone
keeps pathing with chat open. NOT yet live-tested. Open questions:
- Does Horizon's motion tick apply `game.keys` while a screen is open?
  (Test: start `/goto` on the Xvfb, `xdotool key t` to open chat, sample
  position/ticks.) If yes, the fix is to keep driving while a screen is
  open. If no, the bot must at least keep the mission/planning state and
  resume; deeper injection would need a stage patch.
- Injected keys while chat is open would leak WASD into the chat box —
  decide how Java-like to make this (baritone bypasses keybindings).
- Esc → pause menu: singleplayer pause likely stops ticks entirely
  (`state.paused`, `pause_menu` flags) — game-level, may be unfixable from
  the mod; verify what exactly pauses.

### 4. Deep-water swim — NOT yet tested

First rig attempt carved the pool inside a mountain (surface was y78 there).
Build the rig at the spawn flat (surface y63): e.g. fill a 24-wide 4-deep
water basin with cobble funnel walls so the only route east is through the
water, then `/goto` past it. Unit test `swims_across_a_deep_pond…` covers
the planner; the live question is the executor in 4-deep water.

## Where things live

- Executor mine/walk flow: `mod/src/bot.rs` ~940-1200 (advance loop 980,
  mission miner 1027-1077, step-breaks 1084-1131, waypoint-mining
  1140-1183, walk branch ~1186+); `mining_tick` 1714-1772 (timeout ×3 →
  Failed); `MINE_APPROACH_RANGE = 3` at bot.rs:100.
- `GoalNear`: `src/goals.rs:211-232`.
- Tick mixin (key injection, `active` gate, `aim_at_block` 7-candidate
  verification, `place_block`): `mod/mixins/baritone/tick.rs`.
- Java reference clone: `/tmp/ai-agent/baritone-src`
  (`MineProcess.java`, `GoalNear.java`).
- Tests: `python3 tools/assemble.py` FIRST (tests compile the flattened
  `mixins/baritone/runtime.rs`), then `cargo test --locked --lib` in
  `mod/` (42 tests). Rebuild: `mod/tools/rebuild.sh --skip-tests`
  (~4 min; the CARGO_TARGET_DIR stale-install bug is fixed as of d45ac9a).
- Verify installs by `strings <bin> | rg "follow players"` / md5, never by
  mtime.

## Suggested order of attack

1. Unit-test the stall: flat world, target 3 under, assert the bot starts
   mining within a few seconds. Watch it fail, then fix (GoalNear
   `(range+0.5)²` first, waypoint-tolerance second).
2. Hardness table: confirm `snow` is priced; the stall's "unbreakable
   terrain" Failed must trace to a concrete block.
3. Port Java's no-targets mine behavior (dig toward strata).
4. Live-test the chat-pause; apply the smallest fix that keeps the bot
   pathing with chat open (or document why it can't).
5. Deep-water swim rig + test.
6. Full suite, rebuild, live re-verify the user's four scenarios, commit
   (Conventional Commits + `Assisted-by: Pi`), push, mv-install, update
   memory (`baritone-horizon-port.md`).
