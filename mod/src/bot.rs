//! Baritone's brain: the mission, the chat commands that set it, and the
//! per-tick state machine that drives the player along a planned path.
//! Pure Rust — no game types — so it tests on its own; the helper files in
//! `mixins/baritone/` hand in what the game knows each tick (the player, a
//! block reader) and apply the [`Inputs`] the brain returns.
//!
//! Driving mirrors baritone's `PathExecutor`: hold forward and face the
//! current waypoint, sprint long stretches, jump steps up and parkour
//! gaps, hop when stuck, and re-plan from wherever the player actually is
//! when the path stops matching the world. Movements the planner priced
//! as minable are cleared first: the executor stops, switches to the
//! fastest tool in the hotbar (baritone's `ToolSet`) and holds attack
//! until the block gives way or a timeout says it will not. `/mine` scans
//! for named blocks and walks to each in turn; `/tunnel` digs a straight
//! corridor.
//!
//! Movement inputs land the same tick: the game reads its key set after
//! the tick hook runs.

use std::sync::{LazyLock, Mutex, PoisonError};

use super::goals::action_costs::COST_INF;
use super::goals::{Goal, GoalBlock, GoalNear, GoalXZ, GoalYLevel};
use super::movement::Moves;
use super::movement::hardness;
use super::movement::is_placeable;
use super::movement::tools::ToolSet;
use super::search::{Path, SearchConfig, path_finder};
use super::world_view::LiveWorld;

pub const ID: &str = "baritone";
pub const VERSION: &str = "1.2.0";
pub const DESCRIPTION: &str =
    "Pathfinding bot: walk, sprint, swim, jump and dig to any goal; /mine ores. Ported from Baritone.";

/// Chat commands the mod claims, without their slash.
const COMMANDS: &[&str] = &[
    "goto", "near", "path", "stop", "cancel", "help", "mine", "tunnel", "status", "eta",
];

/// The same commands for the server's command tree (`NativeCommand`), one
/// usage each in the game's `<name:kind>` shape. `help` and `stop` are
/// vanilla's on the server and left out there — the client-side hook still
/// answers both, and `/cancel` does `/stop`'s work.
const COMMAND_USAGES: &[&str] = &[
    "goto <pos:block_pos>",
    "near [radius:integer]",
    "path",
    "cancel",
    "status",
    "eta",
    "mine <block:word> [radius:integer]",
    "tunnel [length:integer]",
];

/// Node budget per planning run. The search runs on the game's main thread
/// (baritone's runs off it), so this trades reach for frame hitches: in
/// mountain terrain a few thousand nodes covers only a dozen or so
/// blocks, so longer goals get a partial path that is re-planned forward
/// every so often.
const MAX_NODES: i32 = 12_000;
/// Horizontal distance at which a waypoint counts as reached.
const WAYPOINT_REACH: f64 = 0.45;
/// Tighter reach for the waypoint just before the path's end.
const LAST_WAYPOINT_REACH: f64 = 0.3;
/// Vertical tolerance for reaching a waypoint.
const WAYPOINT_Y_TOLERANCE: f64 = 1.1;
/// Ticks without progress before the bot hops to unstick itself.
const STUCK_JUMP_TICKS: u64 = 60;
/// Ticks without progress before a full re-plan from the current position.
const STUCK_REPLAN_TICKS: u64 = 120;
/// Ticks without a waypoint advance before a full re-plan from the
/// current position. Independent of `STUCK_REPLAN_TICKS` because pushing
/// against a wall can keep shaving micrometers off the best distance
/// forever; only advancing a waypoint (or breaking a block) is alive.
const WAYPOINT_STALL_TICKS: u64 = 300;
/// Failed stuck re-plans before the bot gives up and releases the goal.
const MAX_STUCK_REPLANS: u8 = 3;
/// Minimum ticks between any two re-plans (the search costs real frames).
const REPLAN_COOLDOWN_TICKS: u64 = 20;
/// Ticks between routine re-plans, which pick up terrain as it streams in.
const PERIODIC_REPLAN_TICKS: u64 = 100;
/// Chebyshev distance from both the previous and next waypoint beyond
/// which the player has left the path and a re-plan is due.
const OFF_PATH_LIMIT: i32 = 2;
/// Sprint only with at least this many waypoints left ahead.
const SPRINT_MIN_WAYPOINTS: usize = 4;
/// How long a stuck/step/parkour jump holds the jump key.
const JUMP_HOLD_TICKS: u64 = 4;
/// Range of the per-target goal while mining (baritone's `GoalNear` over a
/// block it is mining, Euclidean): standing anywhere this close puts the
/// block's center within ~4.1 blocks of the eye — inside the game's 4.5
/// reach.
const MINE_APPROACH_RANGE: i32 = 3;
/// Default radius of a `/mine` scan, and its cap.
const DEFAULT_MINE_RADIUS: i32 = 16;
const MAX_MINE_RADIUS: i32 = 32;
/// Ticks between periodic `/mine` rescans while targets remain; when the
/// queue has run dry, this shorter cooldown applies instead.
const MINE_RESCAN_TICKS: u64 = 200;
const MINE_RESCAN_COOLDOWN_TICKS: u64 = 20;
/// Default `/tunnel` length, and its cap.
const DEFAULT_TUNNEL_LENGTH: i32 = 16;
const MAX_TUNNEL_LENGTH: i32 = 64;
/// A mining stint is abandoned after this multiple of the block's
/// expected hand-mining time (doubled in water, where breaking is slow),
/// plus a fixed slack for aim settling and restarts.
const MINE_TIMEOUT_FACTOR: f64 = 3.0;
const MINE_TIMEOUT_SLACK: f64 = 60.0;
/// Timed-out stints before a block is given up on.
const MAX_MINE_FAILURES: u32 = 2;
/// A placement (pillar jump, bridge) that has not landed its block within
/// this many ticks re-plans against the world as it is.
const PLACE_GIVE_UP_TICKS: u64 = 60;

/// Where the bot is going, kept in the form the chat command gave it so it
/// can be rebuilt into a planner goal on every (re)plan — the planner's
/// goal boxes are consumed by the search and not clonable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalSpec {
    /// Stand on this block (baritone's `GoalBlock`).
    Block { x: i32, y: i32, z: i32 },
    /// Reach this column, any height (baritone's `GoalXZ`).
    Xz { x: i32, z: i32 },
    /// Reach this Y level, any column (baritone's `GoalYLevel`).
    YLevel { y: i32 },
    /// Get within `range` blocks of a position (baritone's `GoalNear`).
    Near { x: i32, y: i32, z: i32, range: i32 },
}

impl GoalSpec {
    fn to_goal(self) -> Box<dyn Goal> {
        match self {
            GoalSpec::Block { x, y, z } => Box::new(GoalBlock::new(x, y, z)),
            GoalSpec::Xz { x, z } => Box::new(GoalXZ::new(x, z)),
            GoalSpec::YLevel { y } => Box::new(GoalYLevel { y }),
            GoalSpec::Near { x, y, z, range } => Box::new(GoalNear {
                x,
                y,
                z,
                range: f64::from(range),
            }),
        }
    }

    fn is_in_goal(self, pos: (i32, i32, i32)) -> bool {
        self.to_goal().is_in_goal(pos.0, pos.1, pos.2)
    }

    fn describe(self) -> String {
        match self {
            GoalSpec::Block { x, y, z } => format!("({x}, {y}, {z})"),
            GoalSpec::Xz { x, z } => format!("({x}, {z}) — any height"),
            GoalSpec::YLevel { y } => format!("Y={y}"),
            GoalSpec::Near { x, y, z, range } => format!("within {range} of ({x}, {y}, {z})"),
        }
    }
}

/// What the bot is out to do. A goto keeps its fixed goal; a mine keeps a
/// live queue of block positions.
enum Mission {
    Goto(GoalSpec),
    Mine(MineMission),
}

/// The state of a running `/mine`.
struct MineMission {
    /// The names to mine, bare (`oak_log`); a name also matches whole
    /// families by suffix (`log` matches `oak_log`).
    names: Vec<String>,
    /// The names joined for chat lines.
    display: String,
    /// Scan radius around the player.
    radius: i32,
    /// Found positions, nearest first; the walk heads for the first.
    targets: Vec<(i32, i32, i32)>,
    /// How many targets have been broken so far.
    collected: u32,
    /// Positions given up on (unreachable or unbreakable), excluded from
    /// every rescan.
    blacklist: Vec<(i32, i32, i32)>,
    /// Tick of the last scan, which rate-limits rescans.
    last_scan: u64,
}

impl MineMission {
    /// Whether a block name as the game reports it is one of this
    /// mission's names.
    fn matches(&self, block: &str) -> bool {
        names_match(&self.names, block)
    }
}

/// One block being broken right now: a stint survives across ticks while
/// the crosshair works the same block, and its clock drives the timeout.
#[derive(Debug, PartialEq)]
struct MiningStint {
    pos: (i32, i32, i32),
    /// The block's name when the stint started — a change means the block
    /// was replaced (a sand cascade) and the clock restarts.
    name: String,
    started: u64,
}

/// What one call to [`mining_tick`] decided.
#[derive(Clone, Copy, Debug, PartialEq)]
enum MiningOutcome {
    /// Keep breaking the block this tick.
    Hold,
    /// The stint ran past its timeout once — it restarts next tick.
    TimedOut,
    /// Timed out twice — the caller gives up on this block.
    Failed,
    /// Nothing breakable there any more: gone, or never was.
    Nothing,
}

/// What the game knows about the player this tick. The helper fills it in;
/// the brain never touches game types.
#[derive(Clone)]
pub struct PlayerInfo {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub on_ground: bool,
    pub in_water: bool,
    /// In the world and free to move: no chat, menu or sleep screen open.
    pub active: bool,
    pub ticks: u64,
    pub health: f32,
    pub dimension: String,
    /// The player's view yaw, for commands that need a facing (`/tunnel`).
    pub yaw: f32,
    /// Hotbar item names by slot 0–8, None when empty — decides which
    /// tool the bot holds while breaking (baritone's `ToolSet`).
    pub hotbar: [Option<String>; 9],
}

/// The movement the brain wants applied this tick. While `driving` the
/// helper holds the keys; `clear` is a one-tick flag to release them after
/// driving stops — the game's key set also holds the user's own presses,
/// so idle ticks must leave it alone. `mine` asks the helper to aim the
/// view at a block and hold attack on it (only while the crosshair really
/// hits it); `select` switches the hotbar to the best tool for it first.
/// A block to place: `into` is the cell to fill, `supports` the solid
/// neighbours to place against, best first (the game clicks the first
/// whose face the crosshair really hits).
#[derive(Clone, Debug, PartialEq)]
pub struct PlacePlan {
    pub into: (i32, i32, i32),
    pub supports: Vec<(i32, i32, i32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Inputs {
    pub driving: bool,
    pub forward: bool,
    pub jump: bool,
    pub sprint: bool,
    pub yaw: Option<f32>,
    pub mine: Option<(i32, i32, i32)>,
    /// A block to place this tick — a pillar's block beneath the jump or
    /// a bridge's block in the hole ahead. Never set alongside `mine`.
    pub place: Option<PlacePlan>,
    pub select: Option<usize>,
    pub clear: bool,
}

impl Inputs {
    fn idle() -> Self {
        Inputs {
            driving: false,
            forward: false,
            jump: false,
            sprint: false,
            yaw: None,
            mine: None,
            place: None,
            select: None,
            clear: false,
        }
    }
}

/// Everything the bot remembers between ticks.
pub struct BotState {
    mission: Option<Mission>,
    path: Option<Path>,
    /// Index into the path's positions of the waypoint being walked to.
    next_index: usize,
    /// Last tick with measurable progress toward the current waypoint.
    progress_tick: u64,
    /// Last tick a waypoint was advanced or a block was being broken:
    /// the liveness signal the stuck re-plan falls back on when pushing
    /// against a wall keeps refreshing `progress_tick` by micrometers.
    waypoint_tick: u64,
    /// Closest approach to the current waypoint so far.
    best_distance: f64,
    /// Tick the current placement (pillar jump, bridge hole) began: a
    /// placement that does not land its block in time re-plans.
    place_tick: Option<u64>,
    last_plan_tick: u64,
    stuck_replans: u8,
    /// Hold the jump key until this tick.
    jump_until: u64,
    dimension: String,
    was_driving: bool,
    /// The mine target the current path was planned for; a changed target
    /// invalidates the path.
    planned_target: Option<(i32, i32, i32)>,
    /// The block being broken right now, if any.
    mining: Option<MiningStint>,
    /// Timed-out stints for the block in `last_failure`.
    mining_failures: u32,
    last_failure: Option<(i32, i32, i32)>,
    /// The playing client's newest `PlayerInfo`, published every tick: the
    /// server hook answers commands with it, because the server's own view
    /// of the player (the hub's guest entry) lags behind respawns and
    /// other teleports.
    pub last_player: Option<PlayerInfo>,
}

impl BotState {
    pub fn new() -> Self {
        BotState {
            mission: None,
            path: None,
            next_index: 0,
            progress_tick: 0,
            waypoint_tick: 0,
            best_distance: f64::INFINITY,
            place_tick: None,
            last_plan_tick: 0,
            stuck_replans: 0,
            jump_until: 0,
            dimension: String::new(),
            was_driving: false,
            planned_target: None,
            mining: None,
            mining_failures: 0,
            last_failure: None,
            last_player: None,
        }
    }

    /// The goal being walked to — `None` while a `/mine` runs (its goal
    /// moves with the current target) or when idle.
    pub fn goal(&self) -> Option<GoalSpec> {
        match &self.mission {
            Some(Mission::Goto(spec)) => Some(*spec),
            _ => None,
        }
    }

    fn reset_path(&mut self) {
        self.path = None;
        self.next_index = 0;
        self.best_distance = f64::INFINITY;
        self.place_tick = None;
    }
}

impl Default for BotState {
    fn default() -> Self {
        Self::new()
    }
}

/// The bot's state, shared behind a mutex: single player's client tick
/// drives the bot while its server thread answers commands — the server
/// is another `Game` on its own thread, so a thread-local would split the
/// bot in two. A panicked holder leaves the state usable.
static STATE: LazyLock<Mutex<BotState>> = LazyLock::new(|| Mutex::new(BotState::new()));

/// Runs `f` with the bot's state. The helpers' entry points below are the
/// only users.
pub fn with_state<R>(f: impl FnOnce(&mut BotState) -> R) -> R {
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    f(&mut state)
}

/// The helper's per-tick entry point.
pub fn tick_stateful(world: &LiveWorld<'_>, player: &PlayerInfo) -> (Inputs, Vec<String>) {
    with_state(|state| {
        state.last_player = Some(player.clone());
        tick(state, world, player)
    })
}

/// The helper's chat-command entry point: returns whether the line was one
/// of the mod's and what to say about it.
pub fn command_stateful(
    line: &str,
    player: &PlayerInfo,
    world: &LiveWorld<'_>,
) -> (bool, Vec<String>) {
    with_state(|state| handle_command(state, line, player, world))
}

/// The helper's entry point for a closed world (or a disabled mod): drop
/// the mission and, once, the keys.
pub fn world_closed_stateful() -> Inputs {
    with_state(world_closed)
}

/// The world closed while the bot was driving: forget everything, release
/// the keys this once.
pub fn world_closed(state: &mut BotState) -> Inputs {
    state.mission = None;
    state.mining = None;
    state.reset_path();
    finish(state, false, Inputs::idle(), Vec::new()).0
}

/// The names `native_commands` advertises for the chat box.
pub fn command_names() -> &'static [&'static str] {
    COMMANDS
}

/// The usages `native_register_commands` pushes onto the server's command
/// tree.
pub fn command_usages() -> &'static [&'static str] {
    COMMAND_USAGES
}

/// The playing client's newest `PlayerInfo` — the freshest view of the
/// player any thread can get (`None` before the first tick).
pub fn published_player() -> Option<PlayerInfo> {
    with_state(|state| state.last_player.clone())
}

/// One chat command line (sans slash). Claims only the mod's own commands;
/// everything else passes through to the game untouched.
pub fn handle_command(
    state: &mut BotState,
    line: &str,
    player: &PlayerInfo,
    world: &LiveWorld<'_>,
) -> (bool, Vec<String>) {
    let mut messages = Vec::new();
    let Some((name, args)) = split_command(line) else {
        return (false, messages);
    };
    let pos = block_pos(player);
    let mission = match name {
        "goto" => match parse_goto(&args, pos) {
            Some(spec) => Some(Mission::Goto(spec)),
            None => {
                messages.push(
                    "baritone: usage: /goto <x> <y> <z> — or <x> <z>, or <y>; \
                     coordinates accept ~ (relative to you)"
                        .to_string(),
                );
                return (true, messages);
            }
        },
        "near" => {
            let range = args
                .first()
                .and_then(|r| r.parse::<i32>().ok())
                .unwrap_or(8)
                .clamp(1, 64);
            Some(Mission::Goto(GoalSpec::Near {
                x: pos.0,
                y: pos.1,
                z: pos.2,
                range,
            }))
        }
        "mine" => {
            // A trailing number is the scan radius; the rest are block
            // names, bare or `minecraft:`-prefixed (`log` matches every
            // kind of log).
            let (radius, name_args) = match args.last().and_then(|a| a.parse::<i32>().ok()) {
                Some(r) => (r.clamp(1, MAX_MINE_RADIUS), &args[..args.len() - 1]),
                None => (DEFAULT_MINE_RADIUS, &args[..]),
            };
            if name_args.is_empty() {
                messages.push(
                    "baritone: usage: /mine <block…> [radius] — e.g. /mine iron_ore coal_ore 24"
                        .to_string(),
                );
                return (true, messages);
            }
            let names: Vec<String> = name_args.iter().map(|n| bare_name(n).to_string()).collect();
            let display = names.join(", ");
            messages.push(format!("baritone: mining {display} within {radius} — scanning…"));
            Some(Mission::Mine(MineMission {
                names,
                display,
                radius,
                targets: Vec::new(),
                collected: 0,
                blacklist: Vec::new(),
                last_scan: 0,
            }))
        }
        "tunnel" => {
            // The far end of a straight 1×2 corridor at the current feet
            // level, along the facing: the planner digs straight through
            // whatever is in the way (and just walks when there is
            // nothing to dig).
            let length = args
                .first()
                .and_then(|a| a.parse::<i32>().ok())
                .unwrap_or(DEFAULT_TUNNEL_LENGTH)
                .clamp(1, MAX_TUNNEL_LENGTH);
            let (dx, dz, dir) = cardinal(player.yaw);
            messages.push(format!("baritone: tunneling {dir} for {length} blocks"));
            Some(Mission::Goto(GoalSpec::Block {
                x: pos.0 + dx * length,
                y: pos.1,
                z: pos.2 + dz * length,
            }))
        }
        "path" => {
            if state.mission.is_some() {
                state.reset_path();
                state.planned_target = None;
                state.stuck_replans = 0;
                if let Some(goal) = mission_goal(state.mission.as_ref()) {
                    messages.push(format!("baritone: replanning to {}", goal.describe()));
                }
            } else {
                messages.push("baritone: no goal — use /goto first".to_string());
            }
            return (true, messages);
        }
        "stop" | "cancel" => {
            if state.mission.take().is_some() {
                state.mining = None;
                state.reset_path();
                messages.push("baritone: canceled".to_string());
            } else {
                messages.push("baritone: nothing to cancel".to_string());
            }
            return (true, messages);
        }
        "status" => {
            messages.push(status_line(state));
            return (true, messages);
        }
        "eta" => {
            messages.push(eta_line(state, world, player));
            return (true, messages);
        }
        "help" => {
            messages.push(
                "baritone: /goto <x> <y> <z> | <x> <z> | <y> | /near [r] | \
                 /mine <block…> [radius] | /tunnel [len] | /path | /eta | /status | /stop"
                    .to_string(),
            );
            return (true, messages);
        }
        _ => return (false, messages),
    };
    if let Some(mission) = mission {
        state.mission = Some(mission);
        state.reset_path();
        state.planned_target = None;
        state.mining = None;
        state.mining_failures = 0;
        state.last_failure = None;
        state.stuck_replans = 0;
        state.dimension = player.dimension.clone();
    }
    (true, messages)
}

fn split_command(line: &str) -> Option<(&'static str, Vec<&str>)> {
    let mut words = line.split_whitespace();
    let name = words.next()?.to_ascii_lowercase();
    let name = COMMANDS.iter().find(|c| **c == name.as_str())?;
    Some((name, words.collect()))
}

/// `goto` with three coordinates targets a block, two a column, one a Y
/// level; `~` and `~N` are relative to the player's block.
fn parse_goto(args: &[&str], pos: (i32, i32, i32)) -> Option<GoalSpec> {
    match args.len() {
        3 => {
            let x = parse_coord(args[0], pos.0)?;
            let y = parse_coord(args[1], pos.1)?;
            let z = parse_coord(args[2], pos.2)?;
            Some(GoalSpec::Block { x, y, z })
        }
        2 => {
            let x = parse_coord(args[0], pos.0)?;
            let z = parse_coord(args[1], pos.2)?;
            Some(GoalSpec::Xz { x, z })
        }
        1 => {
            let y = parse_coord(args[0], pos.1)?;
            Some(GoalSpec::YLevel { y })
        }
        _ => None,
    }
}

fn parse_coord(arg: &str, current: i32) -> Option<i32> {
    if arg == "~" {
        Some(current)
    } else if let Some(rest) = arg.strip_prefix('~') {
        Some(current + rest.parse::<i32>().ok()?)
    } else {
        arg.parse().ok()
    }
}

/// One line for `/status`.
fn status_line(state: &BotState) -> String {
    match &state.mission {
        None => "baritone: idle — no mission".to_string(),
        Some(Mission::Goto(spec)) => {
            let mut line = format!("baritone: going to {}", spec.describe());
            if let Some(path) = &state.path {
                let left = path.length().saturating_sub(state.next_index);
                line.push_str(&format!(" — {left} of {} blocks to go", path.length()));
            } else {
                line.push_str(" — planning");
            }
            line
        }
        Some(Mission::Mine(m)) => {
            let mut line = format!(
                "baritone: mining {} within {} — {} in queue, {} collected, {} skipped",
                m.display,
                m.radius,
                m.targets.len(),
                m.collected,
                m.blacklist.len()
            );
            if let Some(t) = m.targets.first() {
                line.push_str(&format!("; next ({}, {}, {})", t.0, t.1, t.2));
            }
            if let Some(stint) = &state.mining {
                line.push_str(&format!(
                    "; breaking {} at ({}, {}, {})",
                    stint.name, stint.pos.0, stint.pos.1, stint.pos.2
                ));
            }
            line
        }
    }
}

/// One line for `/eta`: the planned cost of the path's remaining
/// movements, as the planner priced them (with the player's tools).
fn eta_line(state: &BotState, world: &LiveWorld<'_>, player: &PlayerInfo) -> String {
    let Some(path) = &state.path else {
        return if state.mission.is_some() {
            "baritone: no path yet — waiting for a plan".to_string()
        } else {
            "baritone: no mission".to_string()
        };
    };
    let tools = ToolSet::new(player.hotbar.clone());
    let positions = path.positions();
    let start = state.next_index.max(1);
    let mut ticks = 0.0;
    for j in start..positions.len() {
        if let Some(mv) = path.movements().get(j - 1) {
            let (x, y, z) = positions[j - 1];
            ticks += mv.cost(world, &tools, x, y, z).cost;
        }
    }
    let what = match state.mission.as_ref() {
        Some(Mission::Mine(_)) => "to the block being mined",
        _ => "to the goal",
    };
    format!(
        "baritone: ~{:.0}s {what} ({} ticks over {} blocks)",
        ticks / 20.0,
        ticks,
        positions.len() - start
    )
}

/// The player's facing as a cardinal direction: Horizon's yaw is 0 facing
/// south (+Z), 90 west (−X).
fn cardinal(yaw: f32) -> (i32, i32, &'static str) {
    let yaw = yaw.rem_euclid(360.0);
    if yaw < 45.0 || yaw >= 315.0 {
        (0, 1, "south")
    } else if yaw < 135.0 {
        (-1, 0, "west")
    } else if yaw < 225.0 {
        (0, -1, "north")
    } else {
        (1, 0, "east")
    }
}

/// One tick of the bot: guards, mission upkeep, (re)planning, waypoint
/// following, mining, inputs.
pub fn tick(
    state: &mut BotState,
    world: &LiveWorld<'_>,
    player: &PlayerInfo,
) -> (Inputs, Vec<String>) {
    let mut messages = Vec::new();

    // A menu, the chat or sleep: the player may have the keys, the
    // mission waits. (The game does not read the key set while not
    // playing.)
    if !player.active {
        return finish(state, false, Inputs::idle(), messages);
    }

    if player.health <= 0.0 {
        if state.mission.take().is_some() {
            messages.push("baritone: died — mission canceled".to_string());
        }
        state.mining = None;
        state.reset_path();
        return finish(state, false, Inputs::idle(), messages);
    }

    // A dimension change invalidates every planned path.
    if !state.dimension.is_empty()
        && state.dimension != player.dimension
        && state.mission.is_some()
    {
        messages.push("baritone: changed dimension — mission canceled".to_string());
        state.mission = None;
        state.mining = None;
        state.reset_path();
    }
    state.dimension = player.dimension.clone();

    if state.mission.is_none() {
        return finish(state, false, Inputs::idle(), messages);
    }

    // Arrival is checked against the goal, not the path's end: a partial
    // path can finish the goal early, and XZ/Y-level goals are satisfied
    // at whatever block first meets them. Mine missions end per target.
    // A block goal only counts while standing (or floating) in it: the
    // feet position rounds, so a jump's apex reads as the block one above
    // — a pillar goal would "reach" mid-jump and drop the mission before
    // the block beneath was ever placed.
    if let Some(Mission::Goto(spec)) = state.mission
        && (player.on_ground || player.in_water)
        && spec.is_in_goal(block_pos(player))
    {
        messages.push(format!("baritone: reached {}", spec.describe()));
        state.mission = None;
        state.mining = None;
        state.reset_path();
        return finish(state, false, Inputs::idle(), messages);
    }

    // Mine upkeep: collect finished targets, rescan, complete the mission.
    match maintain_mine(state, world, player) {
        MineUpkeep::None => {}
        MineUpkeep::Found => {
            if let Some(Mission::Mine(m)) = &state.mission {
                messages.push(format!(
                    "baritone: {} × {} in range",
                    m.targets.len(),
                    m.display
                ));
            }
        }
        MineUpkeep::Done(message) => {
            state.mission = None;
            state.mining = None;
            state.reset_path();
            state.planned_target = None;
            messages.push(message);
            return finish(state, false, Inputs::idle(), messages);
        }
    }

    // Re-point the walk when the mine's current target changed.
    if let Some(Mission::Mine(m)) = &state.mission {
        let target = m.targets.first().copied();
        if state.planned_target != target {
            state.reset_path();
            state.planned_target = target;
        }
        if target.is_none() {
            // The scan cooldown is running; hold still until it reports.
            return finish(state, false, Inputs::idle(), messages);
        }
    }

    let goal = mission_goal(state.mission.as_ref()).expect("a live mission has a goal");

    if state.path.is_none() {
        if !plan(state, world, player, goal, true, &mut messages) {
            // A mine target the planner cannot reach is skipped; a goal
            // that cannot be reached at all ends the mission. A failure
            // in mid-air is retried after landing instead (the start
            // block is not walkable then, which is not the goal's fault).
            if !player.on_ground {
                return finish(state, false, Inputs::idle(), messages);
            }
            if let Some(Mission::Mine(m)) = state.mission.as_mut()
                && let Some(t) = m.targets.first().copied()
            {
                m.blacklist.push(t);
                m.targets.remove(0);
                state.planned_target = None;
                messages.push(format!(
                    "baritone: can't reach ({}, {}, {}) — skipping it",
                    t.0, t.1, t.2
                ));
                return finish(state, false, Inputs::idle(), messages);
            }
            let (sx, sy, sz) = block_pos(player);
            messages.push(format!(
                "baritone: no path found from ({sx}, {sy}, {sz}) — stopping"
            ));
            state.mission = None;
            state.mining = None;
            return finish(state, false, Inputs::idle(), messages);
        }
        state.progress_tick = player.ticks;
        state.waypoint_tick = player.ticks;
    }

    // Re-plans, most urgent first: stuck, drifted off the path, routine.
    // A failed re-plan keeps the old path and waits for the next trigger —
    // planning can fail for one tick (mid-jump the start block is not
    // walkable) without the goal being unreachable.
    let stalled = player.ticks.saturating_sub(state.progress_tick);
    let no_advance = player.ticks.saturating_sub(state.waypoint_tick);
    let since_plan = player.ticks.saturating_sub(state.last_plan_tick);
    if stalled >= STUCK_REPLAN_TICKS || no_advance >= WAYPOINT_STALL_TICKS {
        if state.stuck_replans >= MAX_STUCK_REPLANS {
            messages.push("baritone: still stuck after 3 replans — giving up".to_string());
            state.mission = None;
            state.mining = None;
            state.reset_path();
            return finish(state, false, Inputs::idle(), messages);
        }
        state.stuck_replans += 1;
        plan(state, world, player, goal, true, &mut messages);
        state.progress_tick = player.ticks;
        state.waypoint_tick = player.ticks;
    } else if (off_path(state, player) && since_plan >= REPLAN_COOLDOWN_TICKS)
        || since_plan >= PERIODIC_REPLAN_TICKS
    {
        plan(state, world, player, goal, false, &mut messages);
    }

    let (positions, movements) = match state.path.as_ref() {
        Some(path) => (path.positions().to_vec(), path.movements().to_vec()),
        None => return finish(state, false, Inputs::idle(), messages),
    };
    if state.next_index >= positions.len() {
        state.next_index = positions.len() - 1;
    }

    // Walk through waypoints as they are reached. The final one is left to
    // the goal check above.
    while state.next_index + 1 < positions.len() {
        // A placement step's waypoint is reached only once we stand on its
        // level: the pillar's sits a jump above us and the bridge's across
        // the hole, and the Y tolerance would wave both through — the walk
        // would strand below them with nothing left to walk to.
        if placement_pending(
            &movements,
            &positions,
            state.next_index,
            world,
            player,
        ) {
            break;
        }
        let center = center_of(positions[state.next_index]);
        let last = state.next_index + 2 == positions.len();
        let reach = if last {
            LAST_WAYPOINT_REACH
        } else {
            WAYPOINT_REACH
        };
        if distance_xz(player, center) < reach
            && (player.y - center.1).abs() < WAYPOINT_Y_TOLERANCE
        {
            state.next_index += 1;
            state.progress_tick = player.ticks;
            state.waypoint_tick = player.ticks;
            state.best_distance = f64::INFINITY;
            state.stuck_replans = 0;
        } else {
            break;
        }
    }

    let aim = center_of(positions[state.next_index]);
    let horizontal = distance_xz(player, aim);
    if horizontal < state.best_distance - 0.01 {
        state.best_distance = horizontal;
        state.progress_tick = player.ticks;
    }

    // The hotbar decides mining speed: the planner priced breaks with the
    // best tool, and the executor holds that slot while breaking.
    let tools = ToolSet::new(player.hotbar.clone());

    // Mining takes the tick: a mine target standing in reach, or the
    // blocks the next step must clear. Both stand still while breaking.
    if player.on_ground || player.in_water {
        if let Some(Mission::Mine(m)) = &state.mission
            && let Some(t) = m.targets.first().copied()
        {
            let near = GoalSpec::Near {
                x: t.0,
                y: t.1,
                z: t.2,
                range: MINE_APPROACH_RANGE,
            };
            if near.is_in_goal(block_pos(player)) {
                // Mine the target when the eye can see it; otherwise dig
                // the first solid block on the way to it — a player
                // tunneling to a buried ore opens the line of sight exactly
                // like this. The stint follows the block actually being
                // broken; when the target itself goes, maintain_mine counts
                // it and retargets.
                let eye = (player.x, player.y + 1.62, player.z);
                let dig = if line_of_sight(world, eye, t) {
                    t
                } else {
                    ray_to(world, eye, t).unwrap_or(t)
                };
                match mining_tick(state, world, player, &tools, dig) {
                    MiningOutcome::Hold => {
                        let mut inputs = Inputs::idle();
                        inputs.mine = Some(dig);
                        inputs.select = tools.best_slot(bare_name(&world.peek(dig.0, dig.1, dig.2)));
                        return finish(state, true, inputs, messages);
                    }
                    MiningOutcome::Failed => {
                        // Two strikes: not breakable from anywhere we can
                        // stand — skip it for good.
                        if let Some(Mission::Mine(m)) = state.mission.as_mut() {
                            m.blacklist.push(t);
                            if let Some(i) = m.targets.iter().position(|p| *p == t) {
                                m.targets.remove(i);
                            }
                        }
                        state.mining = None;
                        state.planned_target = None;
                        state.reset_path();
                        messages.push(format!(
                            "baritone: can't break ({}, {}, {}) — skipping it",
                            t.0, t.1, t.2
                        ));
                        return finish(state, true, Inputs::idle(), messages);
                    }
                    MiningOutcome::TimedOut | MiningOutcome::Nothing => {}
                }
            }
        }

        // Step breaks: the blocks the current movement prices as minable,
        // in the planner's order — clear them before walking through. The
        // first *visible* one is mined (pressing against a wall, its upper
        // block hides the feet block from the eye), else the first solid.
        let eye = (player.x, player.y + 1.62, player.z);
        let break_target = match state.path.as_ref() {
            Some(path) => state.next_index.checked_sub(1).and_then(|i| {
                path.breaks().get(i).and_then(|breaks| {
                    let solid: Vec<(i32, i32, i32)> = breaks
                        .iter()
                        .copied()
                        .filter(|p| {
                            let block = bare_name(&world.peek(p.0, p.1, p.2)).to_string();
                            !is_passable(&block) && hardness::break_ticks(&block) < COST_INF
                        })
                        .collect();
                    solid
                        .iter()
                        .copied()
                        .find(|p| line_of_sight(world, eye, *p))
                        .or_else(|| solid.first().copied())
                })
            }),
            None => None,
        };
        if let Some(b) = break_target {
            match mining_tick(state, world, player, &tools, b) {
                MiningOutcome::Hold => {
                    let mut inputs = Inputs::idle();
                    inputs.mine = Some(b);
                    inputs.select = tools.best_slot(bare_name(&world.peek(b.0, b.1, b.2)));
                    return finish(state, true, inputs, messages);
                }
                MiningOutcome::Failed => {
                    // The path needs a block that will not break: re-plan;
                    // if that keeps happening, the stuck counter ends the
                    // mission.
                    state.stuck_replans += 1;
                    state.mining = None;
                    state.reset_path();
                    if state.stuck_replans >= MAX_STUCK_REPLANS {
                        messages.push(
                            "baritone: path blocked by unbreakable terrain — giving up".to_string(),
                        );
                        state.mission = None;
                        return finish(state, false, Inputs::idle(), messages);
                    }
                    return finish(state, true, Inputs::idle(), messages);
                }
                MiningOutcome::TimedOut | MiningOutcome::Nothing => {}
            }
        }

        // The planner priced the path against a snapshot of the world: a
        // chunk that loaded since can hold a solid block where the plan
        // saw air, and the step's breaks — priced at plan time — say
        // nothing about it, so walking pushes into it forever. Waypoints
        // need their feet and head block passable; mine what the world
        // actually holds, exactly like the step breaks above, once the
        // waypoint is close enough to be in reach.
        if horizontal < 2.0 {
            let wp = positions[state.next_index];
            let solid: Vec<(i32, i32, i32)> = [(wp.0, wp.1, wp.2), (wp.0, wp.1 + 1, wp.2)]
                .into_iter()
                .filter(|p| {
                    let block = bare_name(&world.peek(p.0, p.1, p.2)).to_string();
                    !is_passable(&block) && hardness::break_ticks(&block) < COST_INF
                })
                .collect();
            if let Some(b) = solid
                .iter()
                .copied()
                .find(|p| line_of_sight(world, eye, *p))
                .or_else(|| solid.first().copied())
            {
                match mining_tick(state, world, player, &tools, b) {
                    MiningOutcome::Hold => {
                        let mut inputs = Inputs::idle();
                        inputs.mine = Some(b);
                        inputs.select = tools.best_slot(bare_name(&world.peek(b.0, b.1, b.2)));
                        return finish(state, true, inputs, messages);
                    }
                    MiningOutcome::Failed => {
                        // The planned path crosses a block that will not
                        // break: re-plan against the world as it is; if
                        // that keeps happening, the stuck counter ends
                        // the mission.
                        state.stuck_replans += 1;
                        state.mining = None;
                        state.reset_path();
                        if state.stuck_replans >= MAX_STUCK_REPLANS {
                            messages.push(
                                "baritone: path blocked by unbreakable terrain — giving up"
                                    .to_string(),
                            );
                            state.mission = None;
                            return finish(state, false, Inputs::idle(), messages);
                        }
                        return finish(state, true, Inputs::idle(), messages);
                    }
                    MiningOutcome::TimedOut | MiningOutcome::Nothing => {}
                }
            }
        }
    }

    let mut inputs = Inputs::idle();
    let dx = aim.0 - player.x;
    let dz = aim.2 - player.z;
    if dx.abs() + dz.abs() > 1e-3 {
        inputs.yaw = Some((-dx).atan2(dz).to_degrees() as f32);
    }
    inputs.forward = true;
    let step = state
        .next_index
        .checked_sub(1)
        .and_then(|i| movements.get(i).copied());
    let parkour = matches!(
        step,
        Some(Moves::ParkourNorth | Moves::ParkourSouth | Moves::ParkourEast | Moves::ParkourWest)
    );
    inputs.sprint =
        parkour || (positions.len() - 1 - state.next_index >= SPRINT_MIN_WAYPOINTS && !player.in_water);

    // A step that needs a block placed — a pillar's straight-up jump or a
    // bridge over a hole — takes the tick: hop and place beneath ourselves,
    // or hold this block's center and fill the hole ahead. Its waypoint is
    // not passed until the block is under it, so the walk waits here.
    match placement_step(state, world, player, &positions, &movements, &mut inputs, &mut messages)
    {
        Placement::None => {}
        Placement::Drive => return finish(state, true, inputs, messages),
        Placement::Replan => return finish(state, true, Inputs::idle(), messages),
        Placement::Halt => return finish(state, false, Inputs::idle(), messages),
    }

    if player.on_ground {
        // Step up onto the next block when close to it.
        if aim.1 > player.y + 0.5 && horizontal < 1.6 {
            state.jump_until = player.ticks + JUMP_HOLD_TICKS;
        }
        // A parkour gap: take off near the edge, sprinting.
        if parkour && (2.0..4.6).contains(&horizontal) {
            state.jump_until = player.ticks + JUMP_HOLD_TICKS;
        }
    }
    let stalled = player.ticks.saturating_sub(state.progress_tick);
    if stalled >= STUCK_JUMP_TICKS && player.on_ground {
        state.jump_until = player.ticks + JUMP_HOLD_TICKS;
    }
    inputs.jump = player.ticks < state.jump_until || (player.in_water && aim.1 >= player.y - 0.5);

    finish(state, true, inputs, messages)
}

/// What the current step's placement wants from this tick.
enum Placement {
    /// The step needs no placement (or its block is there) — walk on.
    None,
    /// Placement inputs were set; drive them.
    Drive,
    /// The placement failed; re-plan from here.
    Replan,
    /// The placement kept failing; the mission is over.
    Halt,
}

/// The placing work for the current step, when it is a pillar or bridge
/// move that still misses its block: fills `inputs` with the stand-jump-
/// place behavior (pillar) or the hold-the-edge-and-fill behavior
/// (bridge) and reports [`Placement::Drive`]. Anything else — a walking
/// step, a block already placed, a walk still short of the pillar base —
/// reports [`Placement::None`] and leaves the inputs alone.
fn placement_step(
    state: &mut BotState,
    world: &LiveWorld<'_>,
    player: &PlayerInfo,
    positions: &[(i32, i32, i32)],
    movements: &[Moves],
    inputs: &mut Inputs,
    messages: &mut Vec<String>,
) -> Placement {
    let i = state.next_index;
    let (dx, dz, pillar) = match i.checked_sub(1).and_then(|j| movements.get(j).copied()) {
        Some(Moves::PillarUp) => (0, 0, true),
        Some(Moves::BridgeNorth) => (0, -1, false),
        Some(Moves::BridgeSouth) => (0, 1, false),
        Some(Moves::BridgeEast) => (1, 0, false),
        Some(Moves::BridgeWest) => (-1, 0, false),
        _ => {
            state.place_tick = None;
            return Placement::None;
        }
    };
    let wp = positions[i];
    let into = (wp.0, wp.1 - 1, wp.2);
    let block = bare_name(&world.peek(into.0, into.1, into.2)).to_string();
    if !is_passable(&block) {
        // Placed already — by us or by the world: walk on.
        state.place_tick = None;
        return Placement::None;
    }

    // A pillar waits for nothing while we are still walking to its base.
    if pillar && distance_xz(player, center_of(wp)) > 1.5 {
        state.place_tick = None;
        return Placement::None;
    }

    // Give up on a placement that will not land: the slot may have run dry
    // (the path was priced on blocks no longer held) or the geometry may
    // have changed under the plan. Re-plan against the world as it is; if
    // that keeps happening, the stuck counter ends the mission.
    let started = *state.place_tick.get_or_insert(player.ticks);
    let slot = player
        .hotbar
        .iter()
        .position(|s| s.as_deref().is_some_and(is_placeable));
    if player.ticks.saturating_sub(started) > PLACE_GIVE_UP_TICKS || slot.is_none() {
        state.place_tick = None;
        state.stuck_replans += 1;
        state.reset_path();
        if state.stuck_replans >= MAX_STUCK_REPLANS {
            messages.push("baritone: can't place a block for the path — giving up".to_string());
            state.mission = None;
            return Placement::Halt;
        }
        messages.push(format!(
            "baritone: can't place ({}, {}, {}) — re-planning",
            into.0, into.1, into.2
        ));
        return Placement::Replan;
    }

    // What to place against: the pillar's support is the block below —
    // its top face takes the click while we hang above it. A bridge
    // places against one of the hole's solid horizontal neighbours, the
    // far side first; the block under us never serves (from above, its
    // top face occludes the side face the placement needs). The game
    // clicks the first whose face the crosshair really hits.
    let supports: Vec<(i32, i32, i32)> = if pillar {
        vec![(into.0, into.1 - 1, into.2)]
    } else {
        [
            (into.0 + dx, into.1, into.2 + dz),
            (into.0 + dz, into.1, into.2 + dx),
            (into.0 - dz, into.1, into.2 - dx),
        ]
        .into_iter()
        .filter(|p| {
            let b = bare_name(&world.peek(p.0, p.1, p.2)).to_string();
            !is_passable(&b)
        })
        .collect()
    };

    inputs.forward = false;
    inputs.sprint = false;
    if pillar {
        // Center on the block, hop, and place while high enough that the
        // new block clears our box — feet at or above its top face.
        inputs.forward = distance_xz(player, center_of(wp)) > 0.2;
        if player.on_ground {
            inputs.jump = true;
        } else if player.y >= f64::from(into.1) + 1.0 {
            inputs.place = Some(PlacePlan { into, supports });
            inputs.select = slot;
        }
    } else {
        // Walk up to this block's center, then hold: from there the ray
        // to the far side's face passes over the hole and the click can
        // land. The placement is issued every tick until the hole is
        // filled; the game verifies the geometry before clicking.
        let cur = center_of(positions[i - 1]);
        let along = (player.x - cur.0) * f64::from(dx) + (player.z - cur.2) * f64::from(dz);
        inputs.forward = along < 0.15;
        inputs.place = Some(PlacePlan { into, supports });
        inputs.select = slot;
    }
    Placement::Drive
}

/// Whether the waypoint at `i` still lacks what its placement step needs:
/// a bridge's hole is unfilled, or a pillar's level is not ours yet (the
/// block below its waypoint goes in mid-jump, and the Y tolerance alone
/// would wave the waypoint through while we are still below it).
fn placement_pending(
    movements: &[Moves],
    positions: &[(i32, i32, i32)],
    i: usize,
    world: &LiveWorld<'_>,
    player: &PlayerInfo,
) -> bool {
    let Some(mv) = i.checked_sub(1).and_then(|j| movements.get(j).copied()) else {
        return false;
    };
    let wp = match mv {
        Moves::PillarUp | Moves::BridgeNorth | Moves::BridgeSouth | Moves::BridgeEast
        | Moves::BridgeWest => positions[i],
        _ => return false,
    };
    if matches!(mv, Moves::PillarUp) {
        return block_pos(player).1 != wp.1;
    }
    let block = bare_name(&world.peek(wp.0, wp.1 - 1, wp.2)).to_string();
    is_passable(&block)
}

/// The goal the planner should head for: the mission's own, or the mine's
/// current target.
fn mission_goal(mission: Option<&Mission>) -> Option<GoalSpec> {
    match mission {
        Some(Mission::Goto(spec)) => Some(*spec),
        Some(Mission::Mine(m)) => m.targets.first().map(|t| GoalSpec::Near {
            x: t.0,
            y: t.1,
            z: t.2,
            range: MINE_APPROACH_RANGE,
        }),
        None => None,
    }
}

/// What the mine upkeep decided.
enum MineUpkeep {
    None,
    /// A scan (re)filled the queue.
    Found,
    /// The queue is empty after a scan: the mission is complete.
    Done(String),
}

/// Collects finished mine targets and rescans on schedule. Touches only
/// the mission and mining fields; path-level resets are the caller's.
fn maintain_mine(state: &mut BotState, world: &LiveWorld<'_>, player: &PlayerInfo) -> MineUpkeep {
    let Some(Mission::Mine(m)) = state.mission.as_mut() else {
        return MineUpkeep::None;
    };
    let bp = block_pos(player);

    // The first target may be gone already: we broke it, or the world
    // replaced it. Removing it re-points the walk at the next one.
    if let Some(t0) = m.targets.first().copied() {
        let block = world.peek(t0.0, t0.1, t0.2);
        if is_passable(&block) {
            m.targets.remove(0);
            m.collected += 1;
            state.mining = None;
            state.mining_failures = 0;
        } else if !m.matches(&block) {
            m.targets.remove(0);
            state.mining = None;
        }
    }

    // Rescan when the queue runs dry, and periodically while it lasts, to
    // pick up terrain as it streams in. Rate-limited so a dry world does
    // not spin.
    let since_scan = player.ticks.saturating_sub(m.last_scan);
    let interval = if m.targets.is_empty() {
        MINE_RESCAN_COOLDOWN_TICKS
    } else {
        MINE_RESCAN_TICKS
    };
    if since_scan >= interval {
        m.last_scan = player.ticks;
        let was_empty = m.targets.is_empty();
        m.targets = scan_targets(world, bp, &m.names, m.radius, &m.blacklist);
        if m.targets.is_empty() {
            return MineUpkeep::Done(format!(
                "baritone: mine done — {} × {}",
                m.collected, m.display
            ));
        }
        if was_empty {
            return MineUpkeep::Found;
        }
    }
    MineUpkeep::None
}

/// One tick of breaking `target`: keeps or restarts the stint, watches the
/// timeout, and reports what the caller should do. `tools` prices the
/// timeout — the stint has the hotbar's best tool, not the bare hand.
fn mining_tick(
    state: &mut BotState,
    world: &LiveWorld<'_>,
    player: &PlayerInfo,
    tools: &ToolSet,
    target: (i32, i32, i32),
) -> MiningOutcome {
    let block = bare_name(&world.peek(target.0, target.1, target.2)).to_string();
    if is_passable(&block) || hardness::break_ticks(&block) >= COST_INF {
        state.mining = None;
        return MiningOutcome::Nothing;
    }

    let same_block = state.mining.as_ref().is_some_and(|s| s.pos == target);
    if same_block {
        let stint = state.mining.as_mut().expect("checked above");
        if stint.name != block {
            // The block changed under the stint — a sand cascade dropped
            // new blocks into the space: restart the clock on the new name.
            stint.name = block.clone();
            stint.started = player.ticks;
        }
        let expected = tools.break_ticks_with(&stint.name);
        let factor = if player.in_water {
            MINE_TIMEOUT_FACTOR * 2.0
        } else {
            MINE_TIMEOUT_FACTOR
        };
        let elapsed = player.ticks.saturating_sub(stint.started) as f64;
        if elapsed > expected * factor + MINE_TIMEOUT_SLACK {
            state.mining_failures += 1;
            state.last_failure = Some(target);
            state.mining = None;
            return if state.mining_failures >= MAX_MINE_FAILURES {
                MiningOutcome::Failed
            } else {
                MiningOutcome::TimedOut
            };
        }
        // Breaking is progress: the stuck timers must not fire mid-mine.
        state.progress_tick = player.ticks;
        state.waypoint_tick = player.ticks;
        return MiningOutcome::Hold;
    }

    // (Re)start a stint. The failure counter follows the block it was
    // counting for.
    if state.last_failure.as_ref() != Some(&target) {
        state.mining_failures = 0;
    }
    state.mining = Some(MiningStint {
        pos: target,
        name: block,
        started: player.ticks,
    });
    state.progress_tick = player.ticks;
    state.waypoint_tick = player.ticks;
    MiningOutcome::Hold
}

/// Whether the eye has a clear line to a block's center: a coarse voxel
/// walk of the ray, stopping at the first solid block that is not the
/// target. The game still verifies the final aim with its own raycast —
/// this only decides which of a step's breaks to aim at first (pressing
/// against a wall, its upper block hides the feet block from the eye).
fn line_of_sight(world: &LiveWorld<'_>, eye: (f64, f64, f64), target: (i32, i32, i32)) -> bool {
    ray_to(world, eye, target).is_none()
}

/// The first solid block on the ray from `eye` to the center of `target`,
/// or `None` when the eye can see the target. The mission miner digs this
/// block when the target itself is hidden behind terrain — the same block
/// a player would break to open the line of sight.
fn ray_to(world: &LiveWorld<'_>, eye: (f64, f64, f64), target: (i32, i32, i32)) -> Option<(i32, i32, i32)> {
    let center = (
        f64::from(target.0) + 0.5,
        f64::from(target.1) + 0.5,
        f64::from(target.2) + 0.5,
    );
    let (dx, dy, dz) = (center.0 - eye.0, center.1 - eye.1, center.2 - eye.2);
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    if dist < 1e-3 {
        return None;
    }
    let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
    let mut x = eye.0.floor() as i32;
    let mut y = eye.1.floor() as i32;
    let mut z = eye.2.floor() as i32;
    let (sx, sy, sz) = (ux.signum() as i32, uy.signum() as i32, uz.signum() as i32);
    let next_boundary = |from: f64, at: i32, dir: f64| -> f64 {
        if dir > 0.0 {
            (f64::from(at + 1) - from) / dir
        } else if dir < 0.0 {
            (f64::from(at) - from) / dir
        } else {
            f64::INFINITY
        }
    };
    let mut t_max = (
        next_boundary(eye.0, x, ux),
        next_boundary(eye.1, y, uy),
        next_boundary(eye.2, z, uz),
    );
    let t_delta = (
        if ux == 0.0 { f64::INFINITY } else { ux.abs().recip() },
        if uy == 0.0 { f64::INFINITY } else { uy.abs().recip() },
        if uz == 0.0 { f64::INFINITY } else { uz.abs().recip() },
    );
    let mut t = 0.0;
    for _ in 0..64 {
        if t_max.0 <= t_max.1 && t_max.0 <= t_max.2 {
            t = t_max.0;
            t_max.0 += t_delta.0;
            x += sx;
        } else if t_max.1 <= t_max.2 {
            t = t_max.1;
            t_max.1 += t_delta.1;
            y += sy;
        } else {
            t = t_max.2;
            t_max.2 += t_delta.2;
            z += sz;
        }
        if t > dist || (x, y, z) == target {
            return None;
        }
        if !is_passable(&world.peek(x, y, z)) {
            return Some((x, y, z));
        }
    }
    None
}

/// Every position in the cube of `radius` around `center` whose block is
/// one of `names`, nearest first, minus the blacklisted ones.
fn scan_targets(
    world: &LiveWorld<'_>,
    center: (i32, i32, i32),
    names: &[String],
    radius: i32,
    blacklist: &[(i32, i32, i32)],
) -> Vec<(i32, i32, i32)> {
    let mut found = Vec::new();
    for x in center.0 - radius..=center.0 + radius {
        for y in center.1 - radius..=center.1 + radius {
            for z in center.2 - radius..=center.2 + radius {
                if blacklist.contains(&(x, y, z)) {
                    continue;
                }
                if names_match(names, &world.peek(x, y, z)) {
                    found.push((x, y, z));
                }
            }
        }
    }
    found.sort_by_key(|&(x, y, z)| {
        let (dx, dy, dz) = (x - center.0, y - center.1, z - center.2);
        dx * dx + dy * dy + dz * dz
    });
    found
}

/// Whether a block name (as the game reports it) is one of the wanted
/// names: an exact bare match, or a family suffix (`log` in `oak_log`).
fn names_match(names: &[String], block: &str) -> bool {
    let base = bare_name(block);
    names
        .iter()
        .any(|n| base == n || base.ends_with(&format!("_{n}")))
}

/// A block name without its `minecraft:` prefix.
fn bare_name(block: &str) -> &str {
    block.strip_prefix("minecraft:").unwrap_or(block)
}

/// Blocks a mined-out position may be left filled with and still count as
/// cleared: air, fluids, the plants the planner passes through. Accepts
/// bare and `minecraft:`-prefixed names. A solid block a rescan would not
/// match (gravel fell in) is handled by the target-validity check instead.
pub fn is_passable(block: &str) -> bool {
    let base = bare_name(block);
    base == "air"
        || base.ends_with("_air")
        || matches!(
            base,
            "water"
                | "flowing_water"
                | "lava"
                | "flowing_lava"
                | "short_grass"
                | "tall_grass"
                | "fern"
                | "large_fern"
                | "dead_bush"
                | "snow"
        )
}

/// Plans a fresh path from the player's block. On failure the old path (if
/// any) survives untouched. `announce` reports the new path in chat. A
/// replan is not progress: the stuck timer (`progress_tick`) is only reset
/// by actual movement, by a fresh mission, or by the stuck branch itself.
fn plan(
    state: &mut BotState,
    world: &LiveWorld<'_>,
    player: &PlayerInfo,
    goal: GoalSpec,
    announce: bool,
    messages: &mut Vec<String>,
) -> bool {
    let config = SearchConfig {
        max_nodes: MAX_NODES,
        ..SearchConfig::default()
    };
    let tools = ToolSet::new(player.hotbar.clone());
    let Some(path) = path_finder(world, &tools, goal.to_goal(), block_pos(player), &config) else {
        return false;
    };
    let partial = !goal.is_in_goal(path.dest());
    let length = path.length();
    state.path = Some(path);
    state.next_index = length.saturating_sub(1).min(1);
    state.best_distance = f64::INFINITY;
    state.last_plan_tick = player.ticks;
    if announce {
        messages.push(format!(
            "baritone: pathing to {} — {length} blocks{}",
            goal.describe(),
            if partial {
                " (partial: walking toward it)"
            } else {
                ""
            }
        ));
    }
    true
}

/// Whether the player is farther than [`OFF_PATH_LIMIT`] (chebyshev) from
/// both the previous and the next waypoint — off the path, a re-plan due.
fn off_path(state: &BotState, player: &PlayerInfo) -> bool {
    let Some(path) = &state.path else {
        return false;
    };
    let positions = path.positions();
    let i = state.next_index.min(positions.len() - 1);
    let pos = block_pos(player);
    let far = |b: (i32, i32, i32)| {
        (pos.0 - b.0).abs().max((pos.1 - b.1).abs()).max((pos.2 - b.2).abs()) > OFF_PATH_LIMIT
    };
    let next_too_far = far(positions[i]);
    let prev_too_far = i == 0 || far(positions[i - 1]);
    next_too_far && prev_too_far
}

/// Sets the driving flags on `inputs` and emits the one-tick key release
/// after driving stops.
fn finish(
    state: &mut BotState,
    driving: bool,
    mut inputs: Inputs,
    messages: Vec<String>,
) -> (Inputs, Vec<String>) {
    inputs.driving = driving;
    inputs.clear = !driving && state.was_driving;
    state.was_driving = driving;
    (inputs, messages)
}

/// The player's block: floor of the feet position, with the Y rounded (the
/// game's own goto does the same for its start positions).
fn block_pos(player: &PlayerInfo) -> (i32, i32, i32) {
    (
        player.x.floor() as i32,
        player.y.round() as i32,
        player.z.floor() as i32,
    )
}

/// A waypoint's center in world coordinates.
fn center_of(pos: (i32, i32, i32)) -> (f64, f64, f64) {
    (f64::from(pos.0) + 0.5, f64::from(pos.1), f64::from(pos.2) + 0.5)
}

fn distance_xz(player: &PlayerInfo, to: (f64, f64, f64)) -> f64 {
    ((player.x - to.0).powi(2) + (player.z - to.2).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::super::world_view::LiveWorld;
    use super::*;

    fn player_at(x: f64, y: f64, z: f64, ticks: u64) -> PlayerInfo {
        player_facing(x, y, z, ticks, 0.0)
    }

    fn player_facing(x: f64, y: f64, z: f64, ticks: u64, yaw: f32) -> PlayerInfo {
        PlayerInfo {
            x,
            y,
            z,
            on_ground: true,
            in_water: false,
            active: true,
            ticks,
            health: 20.0,
            dimension: String::from("minecraft:overworld"),
            yaw,
            hotbar: std::array::from_fn(|_| None),
        }
    }

    fn player_with_hotbar(
        x: f64,
        y: f64,
        z: f64,
        ticks: u64,
        hotbar: [Option<String>; 9],
    ) -> PlayerInfo {
        PlayerInfo {
            hotbar,
            ..player_at(x, y, z, ticks)
        }
    }

    fn stone_floor(blocks: &mut HashMap<(i32, i32, i32), &'static str>, extent: i32) {
        for x in -extent..=extent {
            for z in -extent..=extent {
                blocks.insert((x, 64, z), "minecraft:stone");
            }
        }
    }

    fn reader<'a>(
        blocks: &'a HashMap<(i32, i32, i32), &'static str>,
    ) -> impl Fn(i32, i32, i32) -> String + 'a {
        move |x, y, z| {
            blocks
                .get(&(x, y, z))
                .map_or("minecraft:air", |b| *b)
                .to_string()
        }
    }

    /// Walks a mission out on flat ground: slides the player toward the
    /// aimed waypoint while the bot holds forward, deletes the block it
    /// mines, places the held block where the bot places, and returns when
    /// the mission ends. The world is rebuilt from the (mutated) map every
    /// tick, like a real one streaming back. Vertical motion is crudely
    /// simulated so pillar jumps work: a jump rises 0.42 a tick with drag
    /// and gravity, and the feet land on the highest solid block top below
    /// them.
    fn drive(
        blocks: &mut HashMap<(i32, i32, i32), &'static str>,
        state: &mut BotState,
        player: &mut PlayerInfo,
    ) -> Vec<String> {
        let mut messages = Vec::new();
        let mut forward = true;
        let mut jump = false;
        let mut vy = 0.0_f64;
        for _ in 0..4_000 {
            let snapshot = blocks.clone();
            let reader = reader(&snapshot);
            let world = LiveWorld::new(&reader);
            player.ticks += 1;
            if forward {
                if let Some(path) = state.path.as_ref() {
                    let positions = path.positions();
                    let aim = center_of(positions[state.next_index.min(positions.len() - 1)]);
                    let dx = aim.0 - player.x;
                    let dz = aim.2 - player.z;
                    let dist = (dx * dx + dz * dz).sqrt();
                    if dist > 1e-3 {
                        let pace = 0.35_f64.min(dist);
                        player.x += dx / dist * pace;
                        player.z += dz / dist * pace;
                    }
                }
            }
            if jump && player.on_ground {
                player.on_ground = false;
                vy = 0.42;
            }
            if !player.on_ground {
                player.y += vy;
                vy = (vy - 0.08) * 0.98;
                let bx = player.x.floor() as i32;
                let bz = player.z.floor() as i32;
                let feet = player.y.floor() as i32;
                let solid = |y: i32| {
                    blocks
                        .get(&(bx, y, bz))
                        .is_some_and(|b| !super::is_passable(b))
                };
                let top = (feet - 8..=feet)
                    .rev()
                    .find(|y| solid(*y))
                    .map(|y| f64::from(y + 1));
                if vy < 0.0
                    && let Some(top) = top
                    && player.y <= top
                {
                    player.y = top;
                    player.on_ground = true;
                    vy = 0.0;
                }
            }
            let (inputs, tick_messages) = tick(state, &world, player);
            messages = tick_messages;
            forward = inputs.forward;
            jump = inputs.jump;
            if let Some(pos) = inputs.mine {
                blocks.remove(&pos);
            }
            if let Some(plan) = &inputs.place {
                // The game places the held slot's block; the tests give
                // dirt, so the filled cell reads as dirt.
                blocks.insert(plan.into, "minecraft:dirt");
            }
            if state.mission.is_none() {
                break;
            }
        }
        messages
    }

    #[test]
    fn plans_and_drives_on_flat_ground() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 100);

        let (claimed, messages) = handle_command(&mut state, "goto 4 65 4", &player, &world);
        assert!(claimed);
        assert_eq!(state.goal(), Some(GoalSpec::Block { x: 4, y: 65, z: 4 }));
        assert!(messages.is_empty());

        let (inputs, messages) = tick(&mut state, &world, &player);
        assert!(inputs.driving);
        assert!(inputs.forward);
        assert!(messages.iter().any(|m| m.contains("pathing to (4, 65, 4)")));
        // Face the first waypoint, wherever the planner put it; the goal
        // lies ahead in the +x/+z quadrant.
        let first = center_of(state.path.as_ref().unwrap().positions()[1]);
        let expected = (-(first.0 - player.x)).atan2(first.2 - player.z).to_degrees() as f32;
        let yaw = inputs.yaw.expect("driving sets a yaw");
        assert!((yaw - expected).abs() < 0.5, "yaw {yaw} vs {expected}");
        assert!((-90.0..=0.0).contains(&yaw));
        assert_eq!(state.next_index, 1);
    }

    #[test]
    fn advances_waypoints_and_reaches_the_goal() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 100);
        handle_command(&mut state, "goto 2 65 2", &player, &world);
        let (inputs, _) = tick(&mut state, &world, &player);
        assert!(inputs.driving);

        // Walk the path: put the player near the current waypoint each
        // tick, until the goal block itself is underfoot.
        let mut messages = Vec::new();
        for (ticks, step) in (101..).zip(1..=2) {
            player.x = f64::from(step) + 0.4;
            player.z = f64::from(step) + 0.4;
            player.ticks = ticks;
            let (inputs, tick_messages) = tick(&mut state, &world, &player);
            if step < 2 {
                assert!(inputs.driving, "still walking at step {step}");
            }
            messages = tick_messages;
        }
        // Standing in the goal block: arrived, the goal is cleared.
        assert_eq!(state.goal(), None);
        assert!(messages.iter().any(|m| m.contains("reached")));
    }

    #[test]
    fn waypoint_advance_needs_the_reach_threshold() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 100);
        handle_command(&mut state, "goto 4 65 4", &player, &world);
        tick(&mut state, &world, &player);
        let index = state.next_index;
        let center = center_of(state.path.as_ref().unwrap().positions()[1]);

        // 0.7 blocks from the waypoint's center is not reached yet.
        let near = player_at(center.0 + 0.7, 65.0, center.2, 101);
        tick(&mut state, &world, &near);
        assert_eq!(state.next_index, index);

        // 0.4 from it: reached, the next waypoint becomes the aim.
        let on = player_at(center.0 + 0.4, 65.0, center.2, 102);
        let (inputs, _) = tick(&mut state, &world, &on);
        assert!(inputs.driving);
        assert_eq!(state.next_index, index + 1);
    }

    #[test]
    fn relative_and_shorthand_goto() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(10.0, 65.0, -3.0, 5);
        let (claimed, _) = handle_command(&mut state, "goto ~5 65 ~-3", &player, &world);
        assert!(claimed);
        assert_eq!(state.goal(), Some(GoalSpec::Block { x: 15, y: 65, z: -6 }));

        let (claimed, _) = handle_command(&mut state, "goto ~ ~7", &player, &world);
        assert!(claimed);
        assert_eq!(state.goal(), Some(GoalSpec::Xz { x: 10, z: 4 }));

        let (claimed, _) = handle_command(&mut state, "goto 40", &player, &world);
        assert!(claimed);
        assert_eq!(state.goal(), Some(GoalSpec::YLevel { y: 40 }));

        let (claimed, messages) = handle_command(&mut state, "goto", &player, &world);
        assert!(claimed);
        assert!(messages[0].contains("usage"));

        let (claimed, _) = handle_command(&mut state, "goto nonsense", &player, &world);
        assert!(claimed);
    }

    #[test]
    fn near_stops_when_already_close() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 10);
        let (claimed, _) = handle_command(&mut state, "near 3", &player, &world);
        assert!(claimed);
        assert_eq!(
            state.goal(),
            Some(GoalSpec::Near {
                x: 0,
                y: 65,
                z: 0,
                range: 3
            })
        );
        let (inputs, messages) = tick(&mut state, &world, &player);
        assert!(!inputs.driving);
        assert!(messages.iter().any(|m| m.contains("reached")));
    }

    #[test]
    fn stop_and_path_commands() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 5);
        handle_command(&mut state, "goto 4 65 4", &player, &world);
        let (claimed, messages) = handle_command(&mut state, "stop", &player, &world);
        assert!(claimed);
        assert_eq!(state.goal(), None);
        assert!(messages[0].contains("canceled"));

        let (claimed, messages) = handle_command(&mut state, "path", &player, &world);
        assert!(claimed);
        assert!(messages[0].contains("no goal"));

        handle_command(&mut state, "goto 4 65 4", &player, &world);
        let (claimed, messages) = handle_command(&mut state, "path", &player, &world);
        assert!(claimed);
        assert!(messages[0].contains("replanning"));

        let (claimed, _) = handle_command(&mut state, "cancel", &player, &world);
        assert!(claimed);
        assert_eq!(state.goal(), None);

        let (claimed, _) = handle_command(&mut state, "say hi", &player, &world);
        assert!(!claimed, "other mods' commands are left alone");
    }

    #[test]
    fn inactive_ticks_pause_without_releasing_keys_repeatedly() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 10);
        handle_command(&mut state, "goto 4 65 4", &player, &world);
        let (inputs, _) = tick(&mut state, &world, &player);
        assert!(inputs.driving);

        // The chat opens: pause, and release the keys exactly once.
        let paused = PlayerInfo {
            active: false,
            ..player_at(0.0, 65.0, 0.0, 11)
        };
        let (inputs, _) = tick(&mut state, &world, &paused);
        assert!(!inputs.driving);
        assert!(inputs.clear);
        let (inputs, _) = tick(&mut state, &world, &paused);
        assert!(!inputs.clear, "idle ticks leave the key set alone");

        // Back to the game: driving resumes.
        let (inputs, _) = tick(&mut state, &world, &player);
        assert!(inputs.driving);
    }

    #[test]
    fn gives_up_after_repeated_stuck_replans() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 1_000);
        handle_command(&mut state, "goto 30 65 30", &player, &world);
        tick(&mut state, &world, &player);
        assert!(state.path.is_some());

        // Three failed re-plans already; stuck again: give up.
        state.progress_tick = player.ticks - 200;
        state.stuck_replans = MAX_STUCK_REPLANS;
        state.last_plan_tick = player.ticks - 1_000;
        let (inputs, messages) = tick(&mut state, &world, &player);
        assert!(!inputs.driving);
        assert!(state.mission.is_none());
        assert!(messages.iter().any(|m| m.contains("giving up")));
    }

    #[test]
    fn death_and_dimension_change_cancel() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 10);
        handle_command(&mut state, "goto 4 65 4", &player, &world);
        tick(&mut state, &world, &player);

        let dead = PlayerInfo {
            health: 0.0,
            ..player_at(0.0, 65.0, 0.0, 11)
        };
        let (inputs, messages) = tick(&mut state, &world, &dead);
        assert!(!inputs.driving);
        assert!(messages.iter().any(|m| m.contains("died")));

        handle_command(&mut state, "goto 4 65 4", &player, &world);
        tick(&mut state, &world, &player);
        let nether = PlayerInfo {
            dimension: String::from("minecraft:the_nether"),
            ..player_at(0.0, 65.0, 0.0, 12)
        };
        let (inputs, messages) = tick(&mut state, &world, &nether);
        assert!(!inputs.driving);
        assert!(messages.iter().any(|m| m.contains("dimension")));
        assert!(state.mission.is_none());
    }

    #[test]
    fn world_closed_drops_everything() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 10);
        handle_command(&mut state, "goto 4 65 4", &player, &world);
        let (inputs, _) = tick(&mut state, &world, &player);
        assert!(inputs.driving);

        let inputs = world_closed(&mut state);
        assert!(inputs.clear, "the keys are released once");
        assert!(state.mission.is_none());
        let inputs = world_closed(&mut state);
        assert!(!inputs.clear);
    }

    #[test]
    fn plans_a_parkour_path_over_a_gap() {
        // Flat ground, a three-block gap, ground again: the planner needs a
        // 4-block sprint jump across, and the bot sprints and jumps for it.
        let mut blocks = HashMap::new();
        for x in -8..=0 {
            for z in -8..=8 {
                blocks.insert((x, 64, z), "minecraft:stone");
            }
        }
        for x in 4..=12 {
            for z in -8..=8 {
                blocks.insert((x, 64, z), "minecraft:stone");
            }
        }
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 100);
        handle_command(&mut state, "goto 8 65 0", &player, &world);
        let (inputs, messages) = tick(&mut state, &world, &player);
        assert!(inputs.driving, "messages: {messages:?}");
        let has_parkour = state
            .path
            .as_ref()
            .expect("path over the gap")
            .movements()
            .iter()
            .any(|m| {
                matches!(
                    m,
                    Moves::ParkourNorth
                        | Moves::ParkourSouth
                        | Moves::ParkourEast
                        | Moves::ParkourWest
                )
            });
        assert!(has_parkour, "expected a parkour jump across the gap");
    }

    #[test]
    fn mines_a_block_that_changed_under_the_path() {
        // The plan priced a waypoint's block as air (a chunk that had not
        // loaded yet); by execution time the world holds oak_leaves there.
        // Walking into it forever is the failure mode this covers: the
        // executor re-verifies the waypoint's feet/head blocks and mines
        // what actually stands there, then walks on to the goal.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 100);
        let planted = {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            handle_command(&mut state, "goto 12 65 0", &player, &world);
            tick(&mut state, &world, &player);
            let path = state.path.as_ref().expect("planned east");
            path.positions()[3]
        };
        blocks.insert(planted, "minecraft:oak_leaves");
        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(
            messages.iter().any(|m| m.contains("reached")),
            "expected to arrive, got: {messages:?}"
        );
        assert_eq!(
            blocks.get(&planted).copied(),
            None,
            "the world-changed block was never mined"
        );
    }

    #[test]
    fn gives_up_when_no_waypoint_ever_advances() {
        // A big floor (the search runs out of nodes well short of the
        // goal, so every plan is a partial) and the goal sealed inside a
        // bedrock pocket — every neighbour of its cell and everything
        // above it solid, so no placing move can climb in either. Once
        // the partials start ending at the pocket, no waypoint past them
        // can ever be reached and no block on the way is breakable. The
        // liveness timer must end the mission instead of pressing into
        // the wall forever.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 80);
        for x in 59..=61 {
            for y in 65..=80 {
                for z in 59..=61 {
                    if (x, y, z) != (60, 65, 60) {
                        blocks.insert((x, y, z), "minecraft:bedrock");
                    }
                }
            }
        }
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 100);
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            handle_command(&mut state, "goto 60 65 60", &player, &world);
        }
        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(
            messages.iter().any(|m| m.contains("giving up")),
            "expected a stuck give-up, got: {messages:?}"
        );
        assert!(state.mission.is_none());
    }

    #[test]
    fn pillars_up_to_a_goal_in_the_sky() {
        // Flat ground, the goal one block above it: there is no step to
        // climb and nowhere to ascend from, so the pillar jump is the only
        // way up — and it needs a block in the hotbar. The bot hops, places
        // beneath itself while high enough that the block clears its box,
        // and lands on it.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 6);
        let mut state = BotState::new();
        let mut player = player_with_hotbar(
            0.0,
            65.0,
            0.0,
            100,
            std::array::from_fn(|i| (i == 0).then(|| "minecraft:dirt".to_string())),
        );
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            handle_command(&mut state, "goto 0 66 0", &player, &world);
            tick(&mut state, &world, &player);
        }
        assert!(
            state
                .path
                .as_ref()
                .expect("a pillar path")
                .movements()
                .contains(&Moves::PillarUp),
            "expected a pillar jump in the plan"
        );
        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(
            messages.iter().any(|m| m.contains("reached")),
            "messages: {messages:?}"
        );
        assert_eq!(
            blocks.get(&(0, 65, 0)).copied(),
            Some("minecraft:dirt"),
            "the pillar block was never placed"
        );
    }

    #[test]
    fn bridges_a_hole_in_the_floor() {
        // A one-block causeway between two walls, with a four-block gap:
        // too wide for a parkour jump (every landing is the gap itself),
        // bottomless below (no descend), and the walls price any walk-
        // around at a bare-hand mining fortune. The only way over is the
        // bridge, placing against the wall bases beside the gap.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 12);
        for x in 5..=8 {
            blocks.remove(&(x, 64, 0));
        }
        for x in -12..=12 {
            for y in 65..=67 {
                for z in [-1, 1] {
                    blocks.insert((x, y, z), "minecraft:stone");
                }
            }
        }
        let mut state = BotState::new();
        let mut player = player_with_hotbar(
            0.0,
            65.0,
            0.0,
            100,
            std::array::from_fn(|i| (i == 0).then(|| "minecraft:dirt".to_string())),
        );
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            handle_command(&mut state, "goto 8 65 0", &player, &world);
            tick(&mut state, &world, &player);
        }
        assert!(
            state
                .path
                .as_ref()
                .expect("a bridge path")
                .movements()
                .iter()
                .any(|m| matches!(m, Moves::BridgeEast)),
            "expected a bridge in the plan: {:?}",
            state.path.as_ref().map(|p| p.movements().to_vec())
        );
        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(
            messages.iter().any(|m| m.contains("reached")),
            "messages: {messages:?}"
        );
        assert_eq!(
            blocks.get(&(5, 64, 0)).copied(),
            Some("minecraft:dirt"),
            "the hole was never bridged"
        );
    }

    #[test]
    fn prices_placements_as_impassable_without_a_block_to_place() {
        // The same walled causeway and gap, but the hotbar holds only a
        // pickaxe: every placing move is priced impassable. A path still
        // exists — mining through the wall with the pickaxe — but it must
        // not lean on blocks the bot does not have.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 12);
        for x in 5..=8 {
            blocks.remove(&(x, 64, 0));
        }
        for x in -12..=12 {
            for y in 65..=67 {
                for z in [-1, 1] {
                    blocks.insert((x, y, z), "minecraft:stone");
                }
            }
        }
        let mut state = BotState::new();
        let player = player_with_hotbar(
            0.0,
            65.0,
            0.0,
            100,
            std::array::from_fn(|i| (i == 0).then(|| "minecraft:diamond_pickaxe".to_string())),
        );
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            handle_command(&mut state, "goto 8 65 0", &player, &world);
            tick(&mut state, &world, &player);
        }
        let path = state.path.expect("a path still exists");
        assert!(
            path.movements().iter().all(|m| !matches!(
                m,
                Moves::PillarUp
                    | Moves::BridgeNorth
                    | Moves::BridgeSouth
                    | Moves::BridgeEast
                    | Moves::BridgeWest
            )),
            "planned a placement with nothing to place: {:?}",
            path.movements()
        );
    }

    #[test]
    fn mines_through_a_dirt_wall() {
        // Stone floor, a tall dirt wall across the whole map at x = 5, and
        // a goal beyond it: climbing over would mine more than going
        // through, so the bot digs at feet level — and walks on once the
        // way is clear.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        for y in 65..=72 {
            for z in -40..=40 {
                blocks.insert((5, y, z), "minecraft:dirt");
            }
        }
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 100);
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            let (claimed, _) = handle_command(&mut state, "goto 10 65 0", &player, &world);
            assert!(claimed);
            let (_, messages) = tick(&mut state, &world, &player);
            assert!(messages.iter().any(|m| m.contains("pathing")));
        }

        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(state.mission.is_none(), "messages: {messages:?}");
        assert!(messages.iter().any(|m| m.contains("reached")));
        // The wall is where the bot crossed it, and no further.
        assert!(!blocks.contains_key(&(5, 65, 0)), "the feet block is dug");
        assert!(!blocks.contains_key(&(5, 66, 0)), "the head block is dug");
        assert!(blocks.contains_key(&(5, 65, 5)), "the wall stands elsewhere");
    }

    #[test]
    fn mine_mission_finds_and_collects() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        // Two ore blocks sitting on the floor, one either side.
        blocks.insert((6, 65, 0), "minecraft:iron_ore");
        blocks.insert((-6, 65, 0), "minecraft:iron_ore");
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 100);
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            let (claimed, messages) = handle_command(&mut state, "mine iron_ore", &player, &world);
            assert!(claimed);
            assert!(messages[0].contains("mining iron_ore"));
        }

        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(state.mission.is_none(), "messages: {messages:?}");
        assert!(messages.iter().any(|m| m.contains("mine done — 2 × iron_ore")));
        assert!(!blocks.contains_key(&(6, 65, 0)));
        assert!(!blocks.contains_key(&(-6, 65, 0)));
    }

    #[test]
    fn mining_selects_the_best_hotbar_slot() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        // An ore next to the player: in reach on the first tick.
        blocks.insert((2, 65, 0), "minecraft:iron_ore");
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 100);
        player.hotbar[1] = Some(String::from("minecraft:dirt"));
        player.hotbar[5] = Some(String::from("minecraft:stone_pickaxe"));
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            let (claimed, _) = handle_command(&mut state, "mine iron_ore", &player, &world);
            assert!(claimed);
            let (inputs, _) = tick(&mut state, &world, &player);
            assert_eq!(inputs.mine, Some((2, 65, 0)));
            // The stone pickaxe is held for the ore; the dirt in slot 1
            // does nothing for it.
            assert_eq!(inputs.select, Some(5));
        }

        // A tool under the ore's harvest gate is still held when it swings
        // faster: the wooden pickaxe pays the no-harvest divisor but at
        // double the hand's speed (150 ticks vs 300), and drops nothing.
        let mut player = player_at(0.0, 65.0, 0.0, 100);
        player.hotbar[0] = Some(String::from("minecraft:wooden_pickaxe"));
        let mut state = BotState::new();
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            handle_command(&mut state, "mine iron_ore", &player, &world);
            let (inputs, _) = tick(&mut state, &world, &player);
            assert_eq!(inputs.select, Some(0));
        }
    }

    #[test]
    fn powder_snow_is_never_routed_over() {
        use super::super::movement::can_walk_on;

        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        // A drift across the way: the live bot sank in and froze — the
        // planner must treat it as ground it cannot stand on.
        for x in 1..=3 {
            blocks.insert((x, 64, 0), "minecraft:powder_snow");
        }
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        assert!(!can_walk_on(&world, 2, 64, 0));
        assert!(can_walk_on(&world, 0, 64, 0));

        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 100);
        handle_command(&mut state, "goto 6 65 0", &player, &world);
        tick(&mut state, &world, &player);
        let path = state.path.as_ref().expect("a path around the drift");
        for (x, y, z) in path.positions() {
            assert!(
                !(1..=3).contains(x) || *y != 65 || *z != 0,
                "the path stands on the drift at ({x}, {y}, {z})"
            );
        }
    }

    #[test]
    fn mine_matches_family_names_and_radius() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        blocks.insert((3, 65, 0), "minecraft:oak_log");
        blocks.insert((-3, 65, 0), "minecraft:iron_ore");
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 100);
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            // `log` matches oak_log but not iron_ore; a trailing number is
            // the radius.
            let (claimed, _) = handle_command(&mut state, "mine log 8", &player, &world);
            assert!(claimed);
        }
        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(state.mission.is_none(), "messages: {messages:?}");
        assert!(messages.iter().any(|m| m.contains("mine done — 1 × log")));
        assert!(!blocks.contains_key(&(3, 65, 0)), "the log is mined");
        assert!(blocks.contains_key(&(-3, 65, 0)), "the ore is untouched");
    }

    #[test]
    fn tunnel_faces_the_corridor_by_yaw() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        // Solid dirt ahead to the south, taller than the bot: the tunnel
        // has to dig through, not climb over.
        for z in 1..=6 {
            for y in 65..=72 {
                for x in -40..=40 {
                    blocks.insert((x, y, z), "minecraft:dirt");
                }
            }
        }
        let mut state = BotState::new();
        let mut player = player_facing(0.0, 65.0, 0.0, 100, 0.0);
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            let (claimed, messages) = handle_command(&mut state, "tunnel 6", &player, &world);
            assert!(claimed);
            assert!(messages[0].contains("tunneling south"));
            assert_eq!(state.goal(), Some(GoalSpec::Block { x: 0, y: 65, z: 6 }));

            // Yaw 90 faces west: the corridor runs along −X.
            let west = player_facing(0.0, 65.0, 0.0, 100, 90.0);
            handle_command(&mut state, "stop", &player, &world);
            let (claimed, _) = handle_command(&mut state, "tunnel 4", &west, &world);
            assert!(claimed);
            assert_eq!(state.goal(), Some(GoalSpec::Block { x: -4, y: 65, z: 0 }));
            handle_command(&mut state, "stop", &player, &world);
            handle_command(&mut state, "tunnel 6", &player, &world);
        }

        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(state.mission.is_none(), "messages: {messages:?}");
        assert!(messages.iter().any(|m| m.contains("reached")));
        for z in 1..=6 {
            assert!(!blocks.contains_key(&(0, 65, z)), "dug through z={z}");
            assert!(!blocks.contains_key(&(0, 66, z)), "head room at z={z}");
        }
    }

    #[test]
    fn mining_stint_times_out_then_fails() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        blocks.insert((1, 66, 0), "minecraft:stone");
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        // Stone needs 150 ticks bare-handed; the timeout is 3× + slack.
        let player = player_at(0.0, 65.0, 0.0, 1_000);
        let tools = ToolSet::hand();
        assert_eq!(
            mining_tick(&mut state, &world, &player, &tools, (1, 66, 0)),
            MiningOutcome::Hold
        );
        assert_eq!(
            state.mining.as_ref().expect("stint running").name,
            "stone"
        );

        // Well before the timeout: still holding.
        let later = player_at(0.0, 65.0, 0.0, 1_400);
        assert_eq!(
            mining_tick(&mut state, &world, &later, &tools, (1, 66, 0)),
            MiningOutcome::Hold
        );

        // Past it: one strike, the stint is dropped for a restart.
        let late = player_at(0.0, 65.0, 0.0, 2_000);
        assert_eq!(
            mining_tick(&mut state, &world, &late, &tools, (1, 66, 0)),
            MiningOutcome::TimedOut
        );
        assert_eq!(state.mining, None);

        // A fresh stint failing again: two strikes, give up.
        assert_eq!(
            mining_tick(&mut state, &world, &late, &tools, (1, 66, 0)),
            MiningOutcome::Hold
        );
        let later_still = player_at(0.0, 65.0, 0.0, 3_000);
        assert_eq!(
            mining_tick(&mut state, &world, &later_still, &tools, (1, 66, 0)),
            MiningOutcome::Failed
        );
    }

    #[test]
    fn mine_blacklists_an_unbreakable_target() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        blocks.insert((4, 65, 0), "minecraft:iron_ore");
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 1_000);
        handle_command(&mut state, "mine iron_ore", &player, &world);

        // First tick: the scan finds the ore and plans to it.
        let (_, messages) = tick(&mut state, &world, &player);
        assert!(messages.iter().any(|m| m.contains("in range")));

        // In range: mining starts.
        player.x = 2.5;
        player.z = 0.5;
        player.ticks = 1_100;
        let (inputs, _) = tick(&mut state, &world, &player);
        assert_eq!(inputs.mine, Some((4, 65, 0)));

        // Two stints that run triple their expected time: the block is
        // skipped, and the next scan finds nothing left to do.
        for ticks in [1_200u64, 1_300, 1_400, 2_000] {
            player.ticks = ticks;
            if state.mining.is_some() {
                state.mining.as_mut().expect("checked").started = ticks - 1_000;
            }
            let (_, messages) = tick(&mut state, &world, &player);
            if ticks == 1_400 {
                assert!(
                    messages.iter().any(|m| m.contains("can't break")),
                    "messages: {messages:?}"
                );
            }
            if ticks == 2_000 {
                assert!(
                    messages.iter().any(|m| m.contains("mine done — 0 × iron_ore")),
                    "messages: {messages:?}"
                );
            }
        }
        assert!(state.mission.is_none());
    }

    #[test]
    fn line_of_sight_prefers_visible_breaks() {
        // Pressed against a two-tall wall (eye 0.4 from its face): the
        // feet block hides behind the upper one, the upper block itself
        // is clear, and solid ground beyond is occluded.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        for y in 65..=66 {
            blocks.insert((2, y, 0), "minecraft:dirt");
        }
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let eye = (1.6, 66.62, 0.5); // standing in (1, 65, 0)
        assert!(!line_of_sight(&world, eye, (2, 65, 0)), "feet block hidden");
        assert!(line_of_sight(&world, eye, (2, 66, 0)), "upper block clear");
        assert!(!line_of_sight(&world, eye, (3, 65, 0)), "beyond the wall hidden");
        assert!(line_of_sight(&world, eye, (1, 65, 1)), "open ground clear");
        // The occluder ray names the blocks in the way. Rays to lower
        // targets cross the upper wall block's voxel first from this eye
        // height; a level ray hits it head-on.
        assert_eq!(ray_to(&world, eye, (2, 65, 0)), Some((2, 66, 0)));
        assert_eq!(ray_to(&world, eye, (2, 66, 0)), None);
        assert_eq!(ray_to(&world, eye, (3, 65, 0)), Some((2, 66, 0)));
        assert_eq!(ray_to(&world, eye, (3, 66, 0)), Some((2, 66, 0)));
    }

    /// A mine target within approach range but hidden behind a wall: the
    /// bot must break the occluding block (aiming straight at the target
    /// would never verify, and the ore would sit blacklisted forever).
    #[test]
    fn mining_digs_through_an_occluder_to_the_target() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 12);
        // Coal one block behind a stone wall, inside reach of the eye.
        blocks.insert((3, 65, 0), "minecraft:stone");
        blocks.insert((4, 65, 0), "minecraft:coal_ore");
        blocks.insert((4, 66, 0), "minecraft:stone");
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(1.5, 65.0, 0.5, 100);

        let (claimed, _) = handle_command(&mut state, "mine coal_ore", &player, &world);
        assert!(claimed);
        let (inputs, _) = tick(&mut state, &world, &player);
        // (4, 65, 0) is 2.5 blocks from the eye — in approach range, but
        // the stone at (3, 65, 0) hides it: dig the stone, not the ore.
        assert_eq!(inputs.mine, Some((3, 65, 0)), "must dig the occluder");
        assert!(state.mining.is_some(), "stint on the occluder");
    }

    #[test]
    fn status_and_eta_report_the_mission() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 100);
        handle_command(&mut state, "goto 10 65 0", &player, &world);
        tick(&mut state, &world, &player);

        let (claimed, messages) = handle_command(&mut state, "status", &player, &world);
        assert!(claimed);
        assert!(messages[0].contains("going to (10, 65, 0)"), "{:?}", messages[0]);
        assert!(messages[0].contains("blocks to go"));

        let (claimed, messages) = handle_command(&mut state, "eta", &player, &world);
        assert!(claimed);
        assert!(messages[0].contains("~"), "{:?}", messages[0]);
        assert!(messages[0].contains("to the goal"));

        // Idle, both report as much.
        handle_command(&mut state, "stop", &player, &world);
        let (claimed, messages) = handle_command(&mut state, "status", &player, &world);
        assert!(claimed);
        assert!(messages[0].contains("idle"));
        let (claimed, messages) = handle_command(&mut state, "eta", &player, &world);
        assert!(claimed);
        assert!(messages[0].contains("no mission"));

        let (claimed, messages) = handle_command(&mut state, "help", &player, &world);
        assert!(claimed);
        assert!(messages[0].contains("/mine"), "{:?}", messages[0]);
    }

    /// Reproduces the live stone-cliff scene: the bot stands in a pocket in
    /// front of a five-tall cliff whose face is stone, and the goal block is
    /// buried in the cliff one column over. A cheap route exists (step west
    /// around the cliff edge, walk north on the dirt shoulder, then dig two
    /// blocks east into the goal column) — the planner must find something.
    #[test]
    fn plans_out_of_the_cliff_pocket() {
        let mut blocks = HashMap::new();
        // Snow field floor at y=108.
        for x in 8..=32 {
            for z in -6..=12 {
                blocks.insert((x, 108, z), "minecraft:snow_block");
            }
        }
        // Stone cliff: five layers tall, face at z=5 for x=14..=26. The
        // west shoulder (x<14) keeps its snow floor so the bot can walk
        // around the north end of the cliff.
        for x in 14..=26 {
            for z in 0..=5 {
                for y in 108..=112 {
                    blocks.insert((x, y, z), "minecraft:stone");
                }
            }
        }
        let snapshot = blocks.clone();
        let reader = reader(&snapshot);
        let world = LiveWorld::new(&reader);

        let mut state = BotState::new();
        let player = player_at(20.5, 109.0, 6.5, 100);
        let (claimed, _) = handle_command(&mut state, "goto 20 109 0", &player, &world);
        assert!(claimed);

        let (inputs, messages) = tick(&mut state, &world, &player);
        eprintln!("messages: {messages:?}");
        eprintln!("driving: {}", inputs.driving);
        if let Some(path) = state.path.as_ref() {
            eprintln!("path: {:?}", path.positions());
            eprintln!("breaks: {:?}", path.breaks());
        } else {
            eprintln!("NO PATH PLANNED");
        }
        assert!(
            state.path.is_some(),
            "planner must find a way out of the pocket, got: {messages:?}"
        );
    }

    /// Replays the exact live terrain around the failed goto (dumped from
    /// the game via `blocks`, one `x y z name` line per non-air block) and
    /// requires a plan from the same pocket.
    #[test]
    fn plans_out_of_the_live_terrain() {
        let Ok(data) = std::fs::read_to_string("/home/hakergeniusz/.cache/tmp-build/live-region.txt")
        else {
            eprintln!("live-region.txt not present; skipping");
            return;
        };
        let blocks: HashMap<(i32, i32, i32), String> = data
            .lines()
            .filter_map(|line| {
                let mut parts = line.split_whitespace();
                let x = parts.next()?.parse().ok()?;
                let y = parts.next()?.parse().ok()?;
                let z = parts.next()?.parse().ok()?;
                let name = parts.next()?.to_string();
                Some(((x, y, z), name))
            })
            .collect();
        let reader = move |x: i32, y: i32, z: i32| {
            blocks
                .get(&(x, y, z))
                .cloned()
                .unwrap_or_else(|| "minecraft:air".to_string())
        };
        let world = LiveWorld::new(&reader);

        let mut state = BotState::new();
        // The player carries the tools a mining trip implies: without them
        // the pocket's stone walls price at hand speed (150 ticks/block)
        // and no path fits the node budget — baritone makes the same
        // assumption (its ToolSet is always given the real inventory).
        let mut player = player_at(20.5, 109.0, 6.5, 100);
        player.hotbar[0] = Some(String::from("minecraft:diamond_pickaxe"));
        let (claimed, _) = handle_command(&mut state, "goto 20 109 0", &player, &world);
        assert!(claimed);

        let (inputs, messages) = tick(&mut state, &world, &player);
        eprintln!("messages: {messages:?}");
        eprintln!("driving: {}", inputs.driving);
        if let Some(path) = state.path.as_ref() {
            eprintln!("path: {:?}", path.positions());
            eprintln!("breaks: {:?}", path.breaks());
        } else {
            eprintln!("NO PATH PLANNED");
        }
        assert!(
            state.path.is_some(),
            "planner must find a way out of the live pocket, got: {messages:?}"
        );
    }
}
