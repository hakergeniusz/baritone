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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock, PoisonError};

use super::goals::action_costs::COST_INF;
use super::goals::{Goal, GoalBlock, GoalNear, GoalXZ, GoalYLevel};
use super::movement::Moves;
use super::movement::hardness;
use super::movement::is_placeable;
use super::movement::tools::ToolSet;
use super::search::{Path, SearchConfig, path_finder};
use super::world_view::LiveWorld;

pub const ID: &str = "baritone";
pub const VERSION: &str = "1.3.0";
pub const DESCRIPTION: &str =
    "Pathfinding bot: walk, sprint, swim, jump and dig to any goal; /mine ores. Ported from Baritone.";

/// Chat commands the mod claims, without their slash.
const COMMANDS: &[&str] = &[
    "goto", "near", "path", "stop", "cancel", "help", "mine", "tunnel", "status", "eta", "follow",
];

/// The same commands for the server's command tree (`NativeCommand`), one
/// usage each in the game's `<name:kind>` shape. `help` and `stop` are
/// vanilla's on the server and left out there — the client-side hook still
/// answers both, and `/cancel` does `/stop`'s work. `follow`'s selectors
/// are separate literal branches, not one `<players|entities|...>` word —
/// the pipe-bearing node name never made it into the server's grammar.
const COMMAND_USAGES: &[&str] = &[
    "goto <pos:block_pos>",
    "near [radius:integer]",
    "path",
    "cancel",
    "status",
    "eta",
    "mine <block:word> [radius:integer]",
    "tunnel [length:integer]",
    "follow players",
    "follow entities",
    "follow player <name:word>",
    "follow entity <kind:word>",
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
/// How far ahead the strip-mining goal sits while hunting a block no
/// scan exposes (baritone's `exploreForBlocks`): tunnel outward at the
/// ore's Y level until a rescan finds some.
const MINE_STRIP_LENGTH: i32 = 12;
/// How far the hunt's descent goal sits below (or above) the player:
/// the strata can sit far below what the planner's node budget spans,
/// so the dig goes to it in bounded segments.
const MINE_DESCEND_SEGMENT: i32 = 12;
/// How far each hunt escape walks out along the strip direction before
/// trying to descend again: a dead-end column (frozen lake, sealed pocket)
/// must not pin the descent forever.
const MINE_ESCAPE_DIST: i32 = 32;
/// Dead ends the hunt escapes before it admits defeat: each one walks
/// `MINE_ESCAPE_DIST` further out, so the third is 96 blocks from where
/// the command ran. Without a cap a hopeless world wanders forever.
const MAX_HUNT_ESCAPES: u32 = 3;
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
/// Ticks of an unmoving y that count as standing: Horizon's support query
/// misses snow-layer tops (the player rests at the collision top while
/// on_ground stays false), so the ground gates read the rest as airborne.
const RESTING_TICKS: u64 = 10;

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
/// live queue of block positions; a follow re-targets a nearby entity
/// every tick (baritone's `FollowProcess`).
enum Mission {
    Goto(GoalSpec),
    Mine(MineMission),
    Follow(FollowSelector),
}

/// Which entities `/follow` heads for (baritone's `FollowCommand` groups).
#[derive(Clone, Debug, PartialEq)]
pub enum FollowSelector {
    /// Any other player.
    Players,
    /// Anything alive that is not us.
    Entities,
    /// A player by nametag.
    Player(String),
    /// An entity type (`zombie`, `cow`, `minecraft:zombie`).
    EntityKind(String),
}

impl FollowSelector {
    fn describe(&self) -> String {
        match self {
            FollowSelector::Players => "players".to_string(),
            FollowSelector::Entities => "entities".to_string(),
            FollowSelector::Player(name) => format!("player {name}"),
            FollowSelector::EntityKind(kind) => format!("{kind}"),
        }
    }
}

/// One nearby entity as the game reports it, for `/follow`.
#[derive(Clone, Debug)]
pub struct EntityInfo {
    pub id: u32,
    /// The entity type id without the `minecraft:` prefix (`player`,
    /// `zombie`).
    pub kind: String,
    /// The nametag, when the entity wears one (players do).
    pub name: Option<String>,
    /// Feet position in the world.
    pub pos: (f64, f64, f64),
}

/// How close `/follow` stays to its target (baritone's `followRadius`).
const FOLLOW_RADIUS: i32 = 3;

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
    /// Y level to hunt at while no scan exposes any target (baritone's
    /// `exploreForBlocks`): dig down to the ore's strata, then strip
    /// outward until a rescan finds some. `None` until the first dry
    /// scan; a mission that already collected something never hunts —
    /// it is done once the queue runs dry.
    strata: Option<i32>,
    /// The strip's horizontal direction, fixed from the facing when the
    /// command ran.
    strip_dir: (i32, i32),
    /// Whether the hunt has been announced in chat.
    hunt_announced: bool,
    /// How many times a previous hunt attempt gave up on its column (a
    /// dead end: flooded ice, a sealed pocket). Each one walks the next
    /// descent `MINE_ESCAPE_DIST` further out along the strip direction,
    /// so a frozen lake does not pin the bot forever.
    hunt_stalls: u32,
}

/// What `/mine` asked for, kept so a hunt that gives up on a dead-end
/// column can revive with its escape counter raised instead of ending.
#[derive(Clone)]
struct MineSpec {
    names: Vec<String>,
    display: String,
    radius: i32,
    strip_dir: (i32, i32),
    /// Targets given up on, preserved across revivals so a revived hunt
    /// never re-tries a block it already blacklisted.
    blacklist: Vec<(i32, i32, i32)>,
}

/// Copies the live mine's blacklist into the standing spec, at the points
/// where the mission is about to be dropped.
fn preserve_blacklist(state: &mut BotState) {
    if let Some(Mission::Mine(m)) = state.mission.as_ref()
        && let Some(spec) = state.mine_spec.as_mut()
    {
        spec.blacklist.clone_from(&m.blacklist);
    }
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
    /// Where the player stood at the last progress tick: progress is
    /// horizontal displacement from this anchor, not distance to the aim
    /// (a plan flip swaps the aim and the old distance check read the
    /// switch itself as progress — see the walk branch).
    progress_anchor: (f64, f64),
    /// Tick the current placement (pillar jump, bridge hole) began: a
    /// placement that does not land its block in time re-plans.
    place_tick: Option<u64>,
    last_plan_tick: u64,
    stuck_replans: u8,
    /// Set when a mine hunt gave up on its column; the next dry scan
    /// carries it into the fresh mission as [`MineMission::hunt_stalls`].
    hunt_stalled: bool,
    /// Ticks the player's y has held still while off ground and dry —
    /// the resting heuristic (see the tick top).
    still_ticks: u64,
    /// The y the still-tick count refers to.
    last_y: f64,
    /// The live hunt's dead-end count across missions.
    hunt_stall_count: u32,
    /// The standing `/mine` request, for hunt revival after a dead end.
    mine_spec: Option<MineSpec>,
    /// Hold the jump key until this tick.
    jump_until: u64,
    dimension: String,
    was_driving: bool,
    /// The driving inputs of the last tick, for the game's screen gates
    /// (`screen_inputs`): `(forward, jump, sprint, mining)`.
    screen_drive: Option<(bool, bool, bool, bool)>,
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
    /// The entity currently followed (`/follow`), by the game's entity id.
    follow_target: Option<u32>,
    /// The followed entity's last known feet position — the renderer's
    /// goal box tracks it.
    follow_pos: Option<(f64, f64, f64)>,
    /// The follow goal the current path was planned for; a moved (or
    /// changed) target invalidates it.
    follow_goal: Option<GoalSpec>,
}

impl BotState {
    pub fn new() -> Self {
        BotState {
            mission: None,
            path: None,
            next_index: 0,
            progress_tick: 0,
            progress_anchor: (0.0, 0.0),
            waypoint_tick: 0,
            place_tick: None,
            last_plan_tick: 0,
            stuck_replans: 0,
            hunt_stalled: false,
            hunt_stall_count: 0,
            still_ticks: 0,
            last_y: f64::NAN,
            mine_spec: None,
            jump_until: 0,
            dimension: String::new(),
            was_driving: false,
            screen_drive: None,
            planned_target: None,
            mining: None,
            mining_failures: 0,
            last_failure: None,
            last_player: None,
            follow_target: None,
            follow_pos: None,
            follow_goal: None,
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

    /// Records a progress tick and re-anchors the displacement check:
    /// every site that used to refresh `progress_tick` on its own.
    fn mark_progress(&mut self, player: &PlayerInfo) {
        self.progress_tick = player.ticks;
        self.progress_anchor = (player.x, player.z);
    }

    fn reset_path(&mut self) {
        self.path = None;
        self.next_index = 0;
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
/// The scene as of the last brain tick, for the render stage patch. The
/// patch runs on the render thread while that thread holds the game's
/// own locks — taking [`STATE`] there too (the brain holds it while it
/// reads the world) makes lock-order inversion with the game thread
/// possible, and every so often the two threads find it: the client
/// seizes up with the world still ticking underneath. The cache breaks
/// the cycle: the render thread only ever takes this one.
static RENDER_CACHE: LazyLock<Mutex<Scene>> = LazyLock::new(|| Mutex::new(Scene::default()));

// ---- freeze watchdog ----------------------------------------------------
// The client freezes intermittently with the world apparently still alive:
// the main thread parks on a futex and stops serving the control socket and
// the frame loop. This watchdog answers, from the game's stderr alone,
// three questions at a glance: is the process alive (heartbeats keep
// coming), is the brain still ticking (`tick` advances), and is a mod lock
// stuck (state/cache held or waited on for >5 s). It is diagnostics for
// the freeze hunt, not behavior: it never touches game types and only
// writes to stderr.
static STATE_WAIT_MS: AtomicU64 = AtomicU64::new(0);
static STATE_HELD_MS: AtomicU64 = AtomicU64::new(0);
static STATE_HOLDER: Mutex<String> = Mutex::new(String::new());
static STATE_WAITER: Mutex<String> = Mutex::new(String::new());
static STATE_PHASE: AtomicU64 = AtomicU64::new(0);
static LAST_TICK: AtomicU64 = AtomicU64::new(0);
/// The executor's last `/status` line, refreshed every tick when the
/// watchdog is on — the heartbeat prints it, so a stuck bot reports what
/// it was doing even while the control path is dead.
static LAST_STATUS: Mutex<String> = Mutex::new(String::new());
static WATCHDOG_ON: OnceLock<bool> = OnceLock::new();
static RC_HELD_MS: AtomicU64 = AtomicU64::new(0);
static WATCHDOG: OnceLock<()> = OnceLock::new();

const PHASE_TICK: u64 = 1;
const PHASE_COMMAND: u64 = 2;

fn phase_name(p: u64) -> &'static str {
    match p {
        PHASE_TICK => "tick",
        PHASE_COMMAND => "command",
        _ => "other",
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Spawns (once) the thread that reports the brain's liveness every 5 s
/// and names a mod lock stuck for >5 s. Off by default — the heartbeat
/// is freeze-hunt diagnostics, not something every game should pay for —
/// set `BARITONE_WATCHDOG=1` in the environment to enable it.
fn ensure_watchdog() {
    if !WATCHDOG_ON.get_or_init(|| {
        std::env::var("BARITONE_WATCHDOG").as_deref() == Ok("1")
    }) {
        return;
    }
    WATCHDOG.get_or_init(|| {
        let _ = std::thread::Builder::new()
            .name("bot-watchdog".into())
            .spawn(|| loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let now = now_ms();
                let wait = STATE_WAIT_MS.load(Ordering::SeqCst);
                let held = STATE_HELD_MS.load(Ordering::SeqCst);
                let rc_held = RC_HELD_MS.load(Ordering::SeqCst);
                let phase = phase_name(STATE_PHASE.load(Ordering::SeqCst));
                let lock_note = if wait != 0 && held == 0 && now.saturating_sub(wait) > 5000 {
                    let waiter =
                        STATE_WAITER.lock().map(|w| w.clone()).unwrap_or_default();
                    let holder =
                        STATE_HOLDER.lock().map(|h| h.clone()).unwrap_or_default();
                    format!(
                        " STATE stuck: '{waiter}' waiting {}ms (holder '{holder}')",
                        now - wait
                    )
                } else if held != 0 && now.saturating_sub(held) > 5000 {
                    let holder =
                        STATE_HOLDER.lock().map(|h| h.clone()).unwrap_or_default();
                    format!(
                        " STATE stuck: '{holder}' held {}ms (phase {phase})",
                        now - held
                    )
                } else if rc_held != 0 && now.saturating_sub(rc_held) > 5000 {
                    format!(" RENDER_CACHE stuck: held {}ms", now - rc_held)
                } else {
                    String::new()
                };
                let status = LAST_STATUS
                    .lock()
                    .map(|s| s.clone())
                    .unwrap_or_default();
                eprintln!(
                    "bot-watchdog: alive tick={}{} | {}",
                    LAST_TICK.load(Ordering::SeqCst),
                    lock_note,
                    status
                );
            });
    });
}

/// Runs `f` with the bot's state. The helpers' entry points below are the
/// only users.
pub fn with_state<R>(f: impl FnOnce(&mut BotState) -> R) -> R {
    ensure_watchdog();
    let me = std::thread::current().name().unwrap_or("?").to_string();
    *STATE_WAITER.lock().unwrap_or_else(PoisonError::into_inner) = me;
    STATE_WAIT_MS.store(now_ms(), Ordering::SeqCst);
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    *STATE_HOLDER.lock().unwrap_or_else(PoisonError::into_inner) =
        std::thread::current().name().unwrap_or("?").to_string();
    STATE_HELD_MS.store(now_ms(), Ordering::SeqCst);
    STATE_WAIT_MS.store(0, Ordering::SeqCst);
    let out = f(&mut state);
    STATE_HELD_MS.store(0, Ordering::SeqCst);
    STATE_HOLDER
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
    out
}

/// The bot's driving inputs for the game's screen gates: while a screen
/// (the chat, an inventory) is open the game ignores the key set, so the
/// motion stage patch drives the movement keys from these instead — the
/// user's own presses behind a screen stay inert. `None` while the bot
/// is not driving. `(forward, jump, sprint, mining)`.
pub fn screen_inputs() -> Option<(bool, bool, bool, bool)> {
    with_state(|state| state.screen_drive)
}

/// The helper's per-tick entry point.
pub fn tick_stateful(
    world: &LiveWorld<'_>,
    player: &PlayerInfo,
    entities: &[EntityInfo],
) -> (Inputs, Vec<String>) {
    STATE_PHASE.store(PHASE_TICK, Ordering::SeqCst);
    let (inputs, messages) = with_state(|state| {
        state.last_player = Some(player.clone());
        let out = tick(state, world, player, entities);
        if *WATCHDOG_ON.get_or_init(|| {
            std::env::var("BARITONE_WATCHDOG").as_deref() == Ok("1")
        }) {
            *LAST_STATUS.lock().unwrap_or_else(PoisonError::into_inner) =
                status_line(state, world);
        }
        out
    });
    LAST_TICK.store(player.ticks, Ordering::SeqCst);
    STATE_PHASE.store(0, Ordering::SeqCst);
    // Publish the wireframe scene for the render thread (see the cache
    // note above): rendered here, where the bot-state lock is already
    // dropped and no game lock is held.
    let scene = with_state(|state| render_scene(state));
    if let Ok(mut cache) = RENDER_CACHE.lock() {
        *cache = scene;
    }
    (inputs, messages)
}

/// The wireframe scene as of the last brain tick. The render thread's
/// only entry into the mod — see the [`RENDER_CACHE`] note for why it
/// must not go through [`with_state`].
pub fn cached_scene() -> Scene {
    ensure_watchdog();
    let mut cache = match RENDER_CACHE.lock() {
        Ok(cache) => cache,
        Err(poisoned) => poisoned.into_inner(),
    };
    RC_HELD_MS.store(now_ms(), Ordering::SeqCst);
    let scene = cache.clone();
    RC_HELD_MS.store(0, Ordering::SeqCst);
    scene
}

/// The helper's chat-command entry point: returns whether the line was one
/// of the mod's and what to say about it.
pub fn command_stateful(
    line: &str,
    player: &PlayerInfo,
    world: &LiveWorld<'_>,
) -> (bool, Vec<String>) {
    STATE_PHASE.store(PHASE_COMMAND, Ordering::SeqCst);
    let out = with_state(|state| handle_command(state, line, player, world));
    STATE_PHASE.store(0, Ordering::SeqCst);
    out
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
            let (sdx, sdz, _) = cardinal(player.yaw);
            state.mine_spec = Some(MineSpec {
                names: names.clone(),
                display: display.clone(),
                radius,
                strip_dir: (sdx, sdz),
                blacklist: Vec::new(),
            });
            Some(Mission::Mine(MineMission {
                names,
                display,
                radius,
                targets: Vec::new(),
                collected: 0,
                blacklist: Vec::new(),
                last_scan: 0,
                strata: None,
                strip_dir: (sdx, sdz),
                hunt_announced: false,
                hunt_stalls: 0,
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
        "follow" => {
            // Baritone's FollowCommand groups: `players`, `entities`, a
            // player by nametag (`player herobrine`), or an entity type
            // (`entity zombie`, bare or `minecraft:`-prefixed).
            let selector = match args.first().copied() {
                Some("players") if args.len() == 1 => Some(FollowSelector::Players),
                Some("entities") if args.len() == 1 => Some(FollowSelector::Entities),
                Some("player") if args.len() == 2 => {
                    Some(FollowSelector::Player(args[1].to_string()))
                }
                Some("entity") if args.len() == 2 => {
                    Some(FollowSelector::EntityKind(bare_name(args[1]).to_string()))
                }
                _ => None,
            };
            match selector {
                Some(selector) => {
                    messages.push(format!("baritone: following {}", selector.describe()));
                    Some(Mission::Follow(selector))
                }
                None => {
                    messages.push(
                        "baritone: usage: /follow players | entities | \
                         player <name> | entity <kind>"
                            .to_string(),
                    );
                    return (true, messages);
                }
            }
        }
        "path" => {
            if state.mission.is_some() {
                state.reset_path();
                state.planned_target = None;
                state.stuck_replans = 0;
                if let Some(goal) = mission_goal(state, player) {
                    messages.push(format!("baritone: replanning to {}", goal.describe()));
                }
            } else {
                messages.push("baritone: no goal — use /goto first".to_string());
            }
            return (true, messages);
        }
        "stop" | "cancel" => {
            state.mine_spec = None;
            state.hunt_stalled = false;
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
            messages.push(status_line(state, world));
            return (true, messages);
        }
        "eta" => {
            messages.push(eta_line(state, world, player));
            return (true, messages);
        }
        "help" => {
            messages.push(
                "baritone: /goto <x> <y> <z> | <x> <z> | <y> | /near [r] | \
                 /mine <block…> [radius] | /tunnel [len] | /follow players|entities|\
                 player <name>|entity <kind> | /path | /eta | /status | /stop"
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
        state.follow_target = None;
        state.follow_pos = None;
        state.follow_goal = None;
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

/// One line for `/status` — the mission plus the executor's live internals
/// (path step, the current move's break list, stuck counters, stint age),
/// which is what a stuck bot needs to be interrogated about.
fn status_line(state: &BotState, world: &LiveWorld<'_>) -> String {
    let mut line = match &state.mission {
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
            line
        }
        Some(Mission::Follow(selector)) => {
            let mut line = format!("baritone: following {}", selector.describe());
            if let (Some(pos), Some(path)) = (&state.follow_pos, &state.path) {
                let left = path.length().saturating_sub(state.next_index);
                line.push_str(&format!(
                    " at ({:.0}, {:.0}, {:.0}) — {left} of {} blocks to go",
                    pos.0,
                    pos.1,
                    pos.2,
                    path.length()
                ));
            } else {
                line.push_str(" — planning");
            }
            line
        }
    };

    if let Some(path) = &state.path {
        let i = state.next_index.saturating_sub(1);
        let breaks = path
            .breaks()
            .get(i)
            .map(|bs| {
                bs.iter()
                    .map(|b| {
                        let name = bare_name(&world.peek(b.0, b.1, b.2)).to_string();
                        format!("({},{},{}) {}", b.0, b.1, b.2, name)
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_else(|| "-".to_string());
        let kind = path
            .movements()
            .get(i)
            .map(|m| format!("{m:?}"))
            .unwrap_or_else(|| "-".to_string());
        let seg = |j: usize| {
            path.positions()
                .get(j)
                .map(|p| format!("({},{},{})", p.0, p.1, p.2))
                .unwrap_or_else(|| "-".to_string())
        };
        line.push_str(&format!(
            "; move {i} {} {}->{} breaks [{}] ; step {}/{} stuck {}/{} fails {}",
            kind,
            seg(i),
            seg(i + 1),
            breaks,
            state.next_index,
            path.length(),
            state.stuck_replans,
            MAX_STUCK_REPLANS,
            state.mining_failures,
        ));
    }
    if let Some(stint) = &state.mining {
        let now = state
            .last_player
            .as_ref()
            .map(|p| p.ticks)
            .unwrap_or(stint.started);
        line.push_str(&format!(
            "; breaking {} at ({}, {}, {}) {}t old",
            stint.name, stint.pos.0, stint.pos.1, stint.pos.2, now - stint.started,
        ));
    }
    line
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
    entities: &[EntityInfo],
) -> (Inputs, Vec<String>) {
    let mut messages = Vec::new();

    // A screen (the chat, an inventory) does not pause the brain: the
    // game ignores the key set behind a screen, so the motion stage
    // patch drives the player from the brain's own inputs instead — the
    // user can type while the bot keeps pathing, and nothing the user
    // types leaks into the motion (Java baritone paths with any screen
    // open). Esc still pauses: singleplayer stops ticking entirely.

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

    // Resting: a y that holds still long enough while off ground and dry
    // is standing — snow-layer tops report airborne forever (see
    // RESTING_TICKS), and the gates below must not.
    let still = !player.on_ground
        && !player.in_water
        && (player.y - state.last_y).abs() < 0.01;
    state.still_ticks = if still {
        state.still_ticks.saturating_add(1)
    } else {
        0
    };
    state.last_y = player.y;
    let resting = still && state.still_ticks >= RESTING_TICKS;

    // A hunt that gave up on a dead-end column revives one walk further
    // out along the strip direction: /mine keeps hunting until it finds
    // some or the user stops it (baritone's exploreForBlocks persists
    // too). maintain_mine carries the raised escape counter into the
    // fresh mission.
    if state.mission.is_none()
        && state.hunt_stalled
        && let Some(spec) = state.mine_spec.clone()
    {
        messages.push(format!(
            "baritone: no path at this column — escaping {} blocks out and hunting on",
            MINE_ESCAPE_DIST
        ));
        // hunt_stalled stays set: maintain_mine consumes it on the first
        // dry scan, raising the escape counter for the fresh mission.
        state.mission = Some(Mission::Mine(MineMission {
            names: spec.names,
            display: spec.display,
            radius: spec.radius,
            targets: Vec::new(),
            collected: 0,
            blacklist: spec.blacklist.clone(),
            last_scan: 0,
            strata: None,
            strip_dir: spec.strip_dir,
            hunt_announced: false,
            hunt_stalls: 0,
        }));
    }

    // The hotbar decides mining speed: the planner priced breaks with the
    // best tool, and the executor holds that slot while breaking.
    let tools = ToolSet::new(player.hotbar.clone());

    // Entombed: a block went solid inside the player's own box — a gravel
    // cascade burying a descending dig is the usual way. Plans fail from a
    // solid start block, the mining gate below waits for solid ground, and
    // the mid-air plan retry would idle inside the stone forever: dig the
    // cocoon first — sides of the box before its own cells, so the bot
    // steps free instead of chewing the floor — and let the normal tick
    // resume once a side opens.
    let (bx, _, bz) = block_pos(player);
    let solid_at = |cell: (i32, i32, i32)| {
        let b = bare_name(&world.peek(cell.0, cell.1, cell.2)).to_string();
        !is_passable(&b)
    };
    let box_bottom = player.y.floor() as i32;
    let box_top = (player.y + 1.8).floor() as i32;
    if (box_bottom..=box_top).any(|y| solid_at((bx, y, bz))) {
        if state.mining.is_none() {
            messages.push("baritone: buried — digging out".to_string());
        }
        let mut ring: Vec<(i32, i32, i32)> = Vec::new();
        for y in box_bottom..=box_top {
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                ring.push((bx + dx, y, bz + dz));
            }
        }
        ring.push((bx, box_top + 1, bz));
        for y in box_bottom..=box_top {
            ring.push((bx, y, bz));
        }
        let breakable: Vec<(i32, i32, i32)> = ring
            .into_iter()
            .filter(|cell| {
                let b = bare_name(&world.peek(cell.0, cell.1, cell.2)).to_string();
                !is_passable(&b) && hardness::break_ticks(&b) < COST_INF
            })
            .collect();
        let eye = (player.x, player.y + 1.62, player.z);
        if let Some(b) = breakable
            .iter()
            .copied()
            .find(|cell| line_of_sight(world, eye, *cell))
            .or_else(|| breakable.first().copied())
        {
            match mining_tick(state, world, player, &tools, b) {
                MiningOutcome::Hold => {
                    // A stint in progress is progress (same as the miner
                    // below): keep the stall timers cold while digging
                    // out, or the stuck counter ends the mission mid-escape.
                    state.mark_progress(player);
                    state.waypoint_tick = player.ticks;
                    let mut inputs = Inputs::idle();
                    inputs.mine = Some(b);
                    inputs.select =
                        tools.best_slot(bare_name(&world.peek(b.0, b.1, b.2)));
                    if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
                        eprintln!(
                            "baritone-digout{} at=({},{},{}) mine={:?}",
                            player.ticks,
                            b.0,
                            b.1,
                            b.2,
                            inputs.mine
                        );
                    }
                    return finish(state, true, inputs, messages);
                }
                MiningOutcome::Failed | MiningOutcome::TimedOut
                | MiningOutcome::Nothing => {}
            }
        }
    }

    if state.mission.is_none() {
        if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
            eprintln!("baritone-idle{} no-mission", player.ticks);
        }
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
        && (player.on_ground || player.in_water || resting)
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
        MineUpkeep::Hunt => {
            if let Some(Mission::Mine(m)) = state.mission.as_mut()
                && !m.hunt_announced
            {
                m.hunt_announced = true;
                let y = m.strata.unwrap_or(11);
                messages.push(format!(
                    "baritone: no {} in range — digging to Y {y} to hunt",
                    m.display
                ));
            }
        }
        MineUpkeep::Done(message) => {
            state.mission = None;
            state.mining = None;
            state.reset_path();
            state.planned_target = None;
            state.mine_spec = None;
            messages.push(message);
            return finish(state, false, Inputs::idle(), messages);
        }
    }

    // Re-point the walk when the mine's current target changed.
    if let Some(Mission::Mine(m)) = &state.mission {
        let target = m.targets.first().copied();
        let hunting = m.strata.is_some();
        if state.planned_target != target {
            state.reset_path();
            state.planned_target = target;
        }
        if target.is_none() && !hunting {
            // The scan cooldown is running; hold still until it reports.
            if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
                eprintln!("baritone-idle{} scan-cooldown", player.ticks);
            }
            return finish(state, false, Inputs::idle(), messages);
        }
    }

    // Follow upkeep: pick (or re-pick) the entity to head for, and drop
    // the mission when nothing matches. Runs before the goal is derived —
    // the follow goal moves with the target.
    if !maintain_follow(state, entities, player, &mut messages) {
        return finish(state, false, Inputs::idle(), messages);
    }

    let goal = mission_goal(state, player).expect("a live mission has a goal");

    if state.path.is_none() {
        if !plan(state, world, player, goal, true, &mut messages) {
            // A mine target the planner cannot reach is skipped; a goal
            // that cannot be reached at all ends the mission. A failure
            // in mid-air is retried after landing instead (the start
            // block is not walkable then, which is not the goal's fault).
            if !player.on_ground && !resting {
                if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
                    eprintln!("baritone-idle{} plan-failed-airborne", player.ticks);
                }
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
            // Any mine mission revives: a scan-fed one (strata never
            // armed) would otherwise end silently, idle forever, with
            // /mine still nominally running.
            state.hunt_stalled = matches!(&state.mission, Some(Mission::Mine(_)));
            preserve_blacklist(state);
            state.mission = None;
            state.mining = None;
            return finish(state, false, Inputs::idle(), messages);
        }
        state.mark_progress(player);
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
            let hunting = matches!(&state.mission, Some(Mission::Mine(_)));
            messages.push("baritone: still stuck after 3 replans — giving up".to_string());
            preserve_blacklist(state);
            state.mission = None;
            state.mining = None;
            state.reset_path();
            state.hunt_stalled = hunting;
            return finish(state, false, Inputs::idle(), messages);
        }
        state.stuck_replans += 1;
        plan(state, world, player, goal, true, &mut messages);
        state.mark_progress(player);
        state.waypoint_tick = player.ticks;
    } else if (off_path(state, player) && since_plan >= REPLAN_COOLDOWN_TICKS)
        || since_plan >= PERIODIC_REPLAN_TICKS
    {
        plan(state, world, player, goal, false, &mut messages);
    }

    let (mut positions, mut movements) = match state.path.as_ref() {
        Some(path) => (path.positions().to_vec(), path.movements().to_vec()),
        None => {
            if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
                eprintln!("baritone-idle{} no-path", player.ticks);
            }
            return finish(state, false, Inputs::idle(), messages);
        }
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
        // A waypoint the path lands in from above (a descend) counts as
        // reached only at/below its own level: the Y tolerance (1.1)
        // otherwise waves the player through while it still stands a
        // block above — the walk then aims down a corridor that assumes
        // the drop happened, and the head grinds into terrain the plan
        // never priced, hopping in place forever.
        let lands_lower = state.next_index > 0
            && positions[state.next_index].1 < positions[state.next_index - 1].1;
        let y_ok = if lands_lower {
            player.y - center.1 <= 0.55
        } else {
            (player.y - center.1).abs() < WAYPOINT_Y_TOLERANCE
        };
        if distance_xz(player, center) < reach && y_ok {
            state.next_index += 1;
            state.mark_progress(player);
            state.waypoint_tick = player.ticks;
            state.stuck_replans = 0;
        } else {
            break;
        }
    }

    // Standing at a partial path's destination without the goal means the
    // plan ran out: a fresh plan from here is due NOW — the stale path aims
    // at the bot's own block, and waiting for the stuck timers adds idle
    // seconds to every hunt cycle. Still counts toward the give-up, so a
    // genuinely hopeless column ends the mission and the hunt can escape.
    // The fresh path drives on below: walking its first move is progress,
    // not a stop.
    if state.next_index + 1 >= positions.len() {
        let dest = center_of(positions[state.next_index]);
        let dh = distance_xz(player, dest);
        if dh < LAST_WAYPOINT_REACH
            && (player.y - dest.1).abs() < f64::from(WAYPOINT_Y_TOLERANCE)
            && !goal.is_in_goal(block_pos(player))
        {
            if state.stuck_replans >= MAX_STUCK_REPLANS {
                let hunting =
                    matches!(&state.mission, Some(Mission::Mine(_)));
                messages.push(
                    "baritone: still stuck after 3 replans — giving up".to_string(),
                );
                preserve_blacklist(state);
                state.mission = None;
                state.mining = None;
                state.reset_path();
                state.hunt_stalled = hunting;
                return finish(state, false, Inputs::idle(), messages);
            }
            state.stuck_replans += 1;
            let planned = plan(state, world, player, goal, true, &mut messages);
            if !planned {
                let (sx, sy, sz) = block_pos(player);
                messages.push(format!(
                    "baritone: re-plan from ({sx}, {sy}, {sz}) failed — retrying"
                ));
            }
            match state.path.as_ref() {
                // The fresh plan actually goes somewhere: this was progress
                // toward the goal, not a stall — the counter is for plans
                // that lead back to the bot's own block.
                Some(p) if p.length() > 1 && p.dest() != block_pos(player) => {
                    state.stuck_replans = 0;
                }
                _ => {}
            }
            let _ = planned;
            state.mark_progress(player);
            state.waypoint_tick = player.ticks;
            match state.path.as_ref() {
                Some(p) => {
                    positions = p.positions().to_vec();
                    movements = p.movements().to_vec();
                }
                None => return finish(state, false, Inputs::idle(), messages),
            }
        }
    }

    let aim = center_of(positions[state.next_index]);
    let horizontal = distance_xz(player, aim);
    // Progress is horizontal displacement from the anchor: a plan flip
    // swaps the aim point, and the old distance-to-aim check read the
    // switch itself as progress — a blocked walk then hops in place
    // forever, re-planning every period, never stalling.
    if distance_xz(player, (state.progress_anchor.0, 0.0, state.progress_anchor.1)) > 0.5 {
        state.mark_progress(player);
    }


    // Mining takes the tick: a mine target standing in reach, or the
    // blocks the next step must clear. Both stand still while breaking.
    if player.on_ground || player.in_water || resting {
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
                        if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
                            eprintln!(
                                "baritone-miner{} target={:?} dig={:?}",
                                player.ticks, t, dig
                            );
                        }
                        // Breaking a block is progress: the stint's own
                        // timeout guards a hopeless block, so the stall
                        // timers must not run while a stint works — a
                        // buried target needs several blocks dug and the
                        // stuck counter would end the mission mid-dig,
                        // before the target ever came into view.
                        state.mark_progress(player);
                        state.waypoint_tick = player.ticks;
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
                            // Snow reads passable here, but its collision
                            // blocks the walk — the planner prices it as a
                            // break, so it is one.
                            (!is_passable(&block) || block == "snow")
                                && hardness::break_ticks(&block) < COST_INF
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
        // The plan priced its breaks against the world as planned: blocks
        // it missed (a scene race, or a waypoint skipped one height too
        // early) stand in the corridor the walk pushes into, and the
        // step's break list says nothing about them. Java baritone breaks
        // what blocks it: probe straight ahead of the player, at feet and
        // head height, and mine the first solid one when the plan's own
        // breaks offer nothing.
        let break_target = break_target.or_else(|| {
            let dx = aim.0 - player.x;
            let dz = aim.2 - player.z;
            if dx.abs() + dz.abs() <= 0.3 {
                // The aim straight underfoot (the player stands over a
                // descend's landing column): the step's breaks say
                // nothing and the plan may have priced the drop against
                // air that is now solid — probe the column below
                // instead of pressing the rim. The section gate keeps
                // this off while falling.
                if aim.1 < player.y - 0.5 {
                    let (px, py, pz) = block_pos(player);
                    // The feet cell too: a thin cap rests the player just
                    // above its own cell, so the feet round INTO it.
                    let below = [
                        (px, py, pz),
                        (px, py - 1, pz),
                        (px, py - 2, pz),
                    ];
                    let solid: Vec<(i32, i32, i32)> = below
                        .into_iter()
                        .filter(|p| {
                            let block =
                                bare_name(&world.peek(p.0, p.1, p.2)).to_string();
                            // Snow reads passable to the planner, but the
                            // layer's collision rests the player on top —
                            // a cap over the drop is in the way, break it.
                            (!is_passable(&block) || block == "snow")
                                && hardness::break_ticks(&block) < COST_INF
                        })
                        .collect();
                    return solid
                        .iter()
                        .copied()
                        .find(|p| line_of_sight(world, eye, *p))
                        .or_else(|| solid.first().copied());
                }
                return None;
            }
            let mut fx = 0;
            let mut fz = 0;
            if dx.abs() > 0.3 {
                fx = dx.signum() as i32;
            }
            if dz.abs() > 0.3 {
                fz = dz.signum() as i32;
            }
            let (px, py, pz) = block_pos(player);
            let front = [(px + fx, py, pz + fz), (px + fx, py + 1, pz + fz)];
            let solid: Vec<(i32, i32, i32)> = front
                .into_iter()
                .filter(|p| {
                    let block = bare_name(&world.peek(p.0, p.1, p.2)).to_string();
                    !is_passable(&block) && hardness::break_ticks(&block) < COST_INF
                })
                .collect();
            let axis = solid
                .iter()
                .copied()
                .find(|p| line_of_sight(world, eye, *p))
                .or_else(|| solid.first().copied());
            if axis.is_some() {
                return axis;
            }
            // Last resort: the first breakable block on the ray from the
            // eye to the aim's center — the blocker the axis probes miss
            // (a diagonal push into a tree, a partial-height rest over
            // the drop). A clear ray mines nothing.
            let aim_block = (
                aim.0.floor() as i32,
                aim.1.round() as i32,
                aim.2.floor() as i32,
            );
            let blocker = ray_to(world, eye, aim_block)?;
            let block = bare_name(&world.peek(blocker.0, blocker.1, blocker.2)).to_string();
            if (!is_passable(&block) || block == "snow")
                && hardness::break_ticks(&block) < COST_INF
            {
                Some(blocker)
            } else {
                None
            }
        });
        if let Some(b) = break_target {
            match mining_tick(state, world, player, &tools, b) {
                MiningOutcome::Hold => {
                    // Same as the miner above: a stint in progress is
                    // progress, keep the stall timers cold.
                    state.mark_progress(player);
                    state.waypoint_tick = player.ticks;
                    let mut inputs = Inputs::idle();
                    inputs.mine = Some(b);
                    inputs.select = tools.best_slot(bare_name(&world.peek(b.0, b.1, b.2)));
                    if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
                        eprintln!("baritone-break{} block={:?}", player.ticks, b);
                    }
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
                        let hunting =
                            matches!(&state.mission, Some(Mission::Mine(_)));
                        messages.push(
                            "baritone: path blocked by unbreakable terrain — giving up".to_string(),
                        );
                        preserve_blacklist(state);
                        state.mission = None;
                        state.hunt_stalled = hunting;
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
                    (!is_passable(&block) || block == "snow")
                        && hardness::break_ticks(&block) < COST_INF
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
                        // Same as the miner above: a stint in progress is
                        // progress, keep the stall timers cold.
                        state.mark_progress(player);
                        state.waypoint_tick = player.ticks;
                        let mut inputs = Inputs::idle();
                        inputs.mine = Some(b);
                        inputs.select = tools.best_slot(bare_name(&world.peek(b.0, b.1, b.2)));
                        if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
                            eprintln!("baritone-break{} block={:?}", player.ticks, b);
                        }
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

    // A swim-up rises straight out of the water: walking forward would
    // drift wherever the player happens to face, so hold still (re-center
    // first, exactly like a pillar) and let the held jump key rise.
    if matches!(step, Some(Moves::SwimUp)) {
        inputs.forward = distance_xz(player, aim) > 0.2;
    }

    if player.on_ground || resting {
        // Step up onto the next block when close to it. The threshold is
        // the player's support top, not feet + 0.5: a snow rest sits the
        // feet ~0.8 above the floor, and the step then reads as +0.2 —
        // the old threshold called it flat and never jumped.
        if aim.1 > player.y + 0.05 && horizontal < 1.6 {
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

    if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
        eprintln!(
            "baritone-tick{} pos=({:.2},{:.2},{:.2}) yaw={:?} fwd={} jump={} sprint={} ground={} mine={:?} step={}/{} aim=({:.1},{:.1},{:.1})",
            player.ticks,
            player.x,
            player.y,
            player.z,
            inputs.yaw,
            inputs.forward,
            inputs.jump,
            inputs.sprint,
            player.on_ground,
            inputs.mine,
            state.next_index,
            positions.len(),
            aim.0,
            aim.1,
            aim.2,
        );
    }
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
fn mission_goal(state: &BotState, player: &PlayerInfo) -> Option<GoalSpec> {
    match &state.mission {
        Some(Mission::Goto(spec)) => Some(*spec),
        Some(Mission::Mine(m)) => match m.targets.first() {
            Some(t) => Some(GoalSpec::Near {
                x: t.0,
                y: t.1,
                z: t.2,
                range: MINE_APPROACH_RANGE,
            }),
            // The hunt: reach the ore's Y strata, then strip outward —
            // the goal sits a full strip ahead of wherever the player
            // stands, so the tunnel never runs out (baritone's
            // GoalRunAway branch point with isInGoal=false). The strata
            // can sit far below what the node budget spans, so the
            // descent goes to it in bounded segments.
            None => m.strata.map(|y| {
                let (bx, by, bz) = block_pos(player);
                // A dead-end column walks the escape first: head out along
                // the strip direction and descend there instead. Near the
                // new column the descent resumes as usual.
                if m.hunt_stalls > 0 {
                    let dist = MINE_ESCAPE_DIST * m.hunt_stalls as i32;
                    let (gx, gz) = (
                        bx + m.strip_dir.0 * dist,
                        bz + m.strip_dir.1 * dist,
                    );
                    if (gx - bx).abs() > 2 || (gz - bz).abs() > 2 {
                        return GoalSpec::Near {
                            x: gx,
                            y: by,
                            z: gz,
                            range: 2,
                        };
                    }
                }
                if by == y {
                    GoalSpec::Block {
                        x: bx + m.strip_dir.0 * MINE_STRIP_LENGTH,
                        y,
                        z: bz + m.strip_dir.1 * MINE_STRIP_LENGTH,
                    }
                } else if by > y {
                    GoalSpec::YLevel {
                        y: y.max(by - MINE_DESCEND_SEGMENT),
                    }
                } else {
                    GoalSpec::YLevel {
                        y: y.min(by + MINE_DESCEND_SEGMENT),
                    }
                }
            }),
        },
        Some(Mission::Follow(_)) => state.follow_goal,
        None => None,
    }
}

/// Whether an entity with this kind and nametag satisfies a follow
/// selector. Kinds are bare (`zombie`); names compare case-insensitively.
fn follow_matches(selector: &FollowSelector, kind: &str, name: Option<&str>) -> bool {
    let kind = kind.strip_prefix("minecraft:").unwrap_or(kind);
    match selector {
        FollowSelector::Entities => true,
        FollowSelector::Players => kind == "player",
        FollowSelector::Player(want) => {
            kind == "player" && name.is_some_and(|n| n.eq_ignore_ascii_case(want))
        }
        FollowSelector::EntityKind(want) => {
            let want = want.strip_prefix("minecraft:").unwrap_or(want);
            kind == want
        }
    }
}

/// One tick of follow upkeep: keep the nearest matching entity as the
/// target, moving the goal (and the path) as it goes. Reports whether the
/// mission is still alive — nothing matching ends it.
fn maintain_follow(
    state: &mut BotState,
    entities: &[EntityInfo],
    player: &PlayerInfo,
    messages: &mut Vec<String>,
) -> bool {
    let Some(Mission::Follow(selector)) = &state.mission else {
        return true;
    };
    let describe = selector.describe();
    let target = entities
        .iter()
        .filter(|e| follow_matches(selector, &e.kind, e.name.as_deref()))
        .min_by(|a, b| {
            let da = (a.pos.0 - player.x).powi(2)
                + (a.pos.1 - player.y).powi(2)
                + (a.pos.2 - player.z).powi(2);
            let db = (b.pos.0 - player.x).powi(2)
                + (b.pos.1 - player.y).powi(2)
                + (b.pos.2 - player.z).powi(2);
            da.total_cmp(&db)
        });
    let Some(e) = target else {
        messages.push("baritone: nothing to follow — stopping".to_string());
        state.mission = None;
        state.follow_target = None;
        state.follow_pos = None;
        state.follow_goal = None;
        state.reset_path();
        return false;
    };
    let block = (e.pos.0.floor() as i32, e.pos.1.floor() as i32, e.pos.2.floor() as i32);
    let goal = GoalSpec::Near {
        x: block.0,
        y: block.1,
        z: block.2,
        range: FOLLOW_RADIUS,
    };
    if state.follow_target != Some(e.id) {
        let first = state.follow_target.is_none();
        state.follow_target = Some(e.id);
        state.follow_goal = Some(goal);
        state.reset_path();
        if first {
            messages.push(format!("baritone: following {describe}"));
        }
    } else if state.follow_goal != Some(goal) {
        // Same entity, new block: the goal moves with it.
        state.follow_goal = Some(goal);
        state.reset_path();
    }
    state.follow_pos = Some(e.pos);
    true
}

/// Baritone's render colors, its `Settings` defaults: the goal box green,
/// the current path red, blocks to break red, to place green, the block
/// walked into magenta.
pub const COLOR_GOAL: u32 = 0xFF_00_FF_00;
pub const COLOR_PATH: u32 = 0xFF_FF_00_00;
pub const COLOR_TO_BREAK: u32 = 0xFF_FF_00_00;
pub const COLOR_TO_PLACE: u32 = 0xFF_00_FF_00;
pub const COLOR_TO_WALK_INTO: u32 = 0xFF_FF_00_FF;

/// One wireframe primitive the client should draw this frame: a box as
/// (min, max, color) or a line segment as (start, end, color), world
/// coordinates, ARGB colors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub boxes: Vec<([f64; 3], [f64; 3], u32)>,
    pub segs: Vec<([f64; 3], [f64; 3], u32)>,
}

/// What the client should draw for the bot this frame — baritone's
/// `PathRenderer`: the goal as a bobbing two-tall green box (a full
/// column for an XZ goal), the path as red lines through its blocks, the
/// block being broken red, the block to place green, the next waypoint
/// magenta.
pub fn render_scene(state: &BotState) -> Scene {
    let mut scene = Scene::default();
    let ticks = state.last_player.as_ref().map_or(0, |p| p.ticks);

    // The goal: a goto's fixed spec, or the follow target's moving one.
    if let Some(spec) = match &state.mission {
        Some(Mission::Goto(spec)) => Some(*spec),
        Some(Mission::Follow(_)) => state.follow_goal,
        _ => None,
    } {
        draw_goal(&mut scene, spec, state, ticks);
    }

    // The path's remaining blocks, nearest waypoint first.
    if let Some(path) = &state.path {
        let positions = path.positions();
        let from = state.next_index.min(positions.len().saturating_sub(1));
        for pair in positions[from..].windows(2) {
            let a = center_of_f(pair[0]);
            let b = center_of_f(pair[1]);
            scene.segs.push((a, b, COLOR_PATH));
        }
        // The block walked into right now.
        let wp = positions[from];
        scene
            .boxes
            .push(box_of(wp, COLOR_TO_WALK_INTO));
    }

    // The block being broken right now.
    if let Some(stint) = &state.mining {
        scene.boxes.push(box_of(stint.pos, COLOR_TO_BREAK));
    }

    // The cell a pending placement fills.
    if state.place_tick.is_some()
        && let Some(path) = &state.path
        && let Some(mv) = state.next_index.checked_sub(1).and_then(|i| path.movements().get(i))
        && matches!(
            mv,
            Moves::PillarUp
                | Moves::BridgeNorth
                | Moves::BridgeSouth
                | Moves::BridgeEast
                | Moves::BridgeWest
        )
        && let Some(&wp) = path.positions().get(state.next_index)
    {
        scene.boxes.push(box_of((wp.0, wp.1 - 1, wp.2), COLOR_TO_PLACE));
    }

    scene
}

/// The goal's wireframe: a two-tall box around the goal block with two
/// bobbing rings through it (baritone's `drawGoal`), a full column for an
/// XZ goal, a slab for a Y level. `ticks` drives the bob (a two-second
/// period, like baritone's).
fn draw_goal(scene: &mut Scene, spec: GoalSpec, state: &BotState, ticks: u64) {
    match spec {
        GoalSpec::Block { x, y, z } | GoalSpec::Near { x, y, z, .. } => {
            let bob = bob_of(ticks);
            let (x, y, z) = (f64::from(x), f64::from(y), f64::from(z));
            scene.boxes.push((
                [x + 0.002, y, z + 0.002],
                [x + 0.998, y + 2.0, z + 0.998],
                COLOR_GOAL,
            ));
            let (px, pz) = (x + 0.5, z + 0.5);
            for ring_y in [y + 1.0 + bob, y + 1.0 - bob] {
                ring(scene, px, ring_y, pz, COLOR_GOAL);
            }
        }
        GoalSpec::Xz { x, z } => {
            // Baritone spans the world's full height; the dimension's
            // build limits are not known here, so -64..320 covers the
            // overworld.
            scene.boxes.push((
                [f64::from(x) + 0.002, -64.0, f64::from(z) + 0.002],
                [f64::from(x) + 0.998, 320.0, f64::from(z) + 0.998],
                COLOR_GOAL,
            ));
        }
        GoalSpec::YLevel { y } => {
            // A plane at the level, centered on the player (baritone
            // draws the whole XZ plane; a radius keeps the line budget
            // sane).
            if let Some(p) = &state.last_player {
                let (px, pz) = (p.x, p.z);
                scene.boxes.push((
                    [px - 16.0, f64::from(y), pz - 16.0],
                    [px + 16.0, f64::from(y) + 1.0, pz + 16.0],
                    COLOR_GOAL,
                ));
            }
        }
    }
}

/// The goal box's ring bob: `cos` over a two-second period, ±0.5 blocks.
fn bob_of(ticks: u64) -> f64 {
    0.5 * ((f64::from((ticks % 40) as u32) / 40.0) * std::f64::consts::TAU).cos()
}

/// One horizontal square ring around (x, y, z).
fn ring(scene: &mut Scene, x: f64, y: f64, z: f64, color: u32) {
    let half = 0.498;
    let corners = [
        ([x - half, y, z - half], [x + half, y, z - half]),
        ([x + half, y, z - half], [x + half, y, z + half]),
        ([x + half, y, z + half], [x - half, y, z + half]),
        ([x - half, y, z + half], [x - half, y, z - half]),
    ];
    for (a, b) in corners {
        scene.segs.push((a, b, color));
    }
}

/// A one-block selection box around a cell.
fn box_of(pos: (i32, i32, i32), color: u32) -> ([f64; 3], [f64; 3], u32) {
    (
        [
            f64::from(pos.0),
            f64::from(pos.1),
            f64::from(pos.2),
        ],
        [
            f64::from(pos.0) + 1.0,
            f64::from(pos.1) + 1.0,
            f64::from(pos.2) + 1.0,
        ],
        color,
    )
}

/// A block's center, as f64 coordinates.
fn center_of_f(pos: (i32, i32, i32)) -> [f64; 3] {
    [
        f64::from(pos.0) + 0.5,
        f64::from(pos.1) + 0.5,
        f64::from(pos.2) + 0.5,
    ]
}

/// What the mine upkeep decided.
enum MineUpkeep {
    None,
    /// A scan (re)filled the queue.
    Found,
    /// The queue is empty and nothing has been collected yet: hunt at
    /// the strata instead of quitting.
    Hunt,
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
            if m.collected == 0 && state.hunt_stall_count >= MAX_HUNT_ESCAPES {
                return MineUpkeep::Done(format!(
                    "baritone: hunt gave up after {} dead ends — use /mine to try again",
                    state.hunt_stall_count
                ));
            }
            if m.collected == 0 {
                // Nothing exposed anywhere in range: hunt instead of
                // quitting (baritone's exploreForBlocks) — dig to the
                // block's Y strata and strip outward until a rescan
                // finds some. The mission ends when a vein has been
                // mined out, or when the user stops it.
                if m.strata.is_none() {
                    m.strata = Some(m.names.iter().map(|n| strata_y(n)).min().unwrap_or(11));
                    // A previous hunt that gave up walks its escape before descending
                    // again; reaching the strata clears the escapes.
                    if let Some(y) = m.strata {
                        let (_, by, _) = block_pos(player);
                        if by <= y {
                            state.hunt_stall_count = 0;
                            m.hunt_stalls = 0;
                        } else if state.hunt_stalled {
                            state.hunt_stalled = false;
                            state.hunt_stall_count += 1;
                        }
                        m.hunt_stalls = state.hunt_stall_count;
                    }
                }
                return MineUpkeep::Hunt;
            }
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
    // Snow reads passable (the planner walks through it), but the layer's
    // collision rests the player on top — a stint can be asked to break
    // the cap, and it goes in a few ticks.
    if (is_passable(&block) && block != "snow")
        || hardness::break_ticks(&block) >= COST_INF
    {
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
        state.mark_progress(player);
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
    state.mark_progress(player);
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
/// The Y level to hunt a block at when no scan exposes one (baritone's
/// `legitMineYLevel`, per ore family — where vanilla generation peaks).
fn strata_y(name: &str) -> i32 {
    match name {
        "diamond_ore" | "deepslate_diamond_ore" | "redstone_ore"
        | "deepslate_redstone_ore" => -59,
        "gold_ore" | "deepslate_gold_ore" => -16,
        "lapis_ore" | "deepslate_lapis_ore" => -1,
        "iron_ore" | "deepslate_iron_ore" => 15,
        "copper_ore" | "deepslate_copper_ore" => 48,
        "coal_ore" | "deepslate_coal_ore" => 45,
        "emerald_ore" | "deepslate_emerald_ore" => 90,
        // Java baritone's legitMineYLevel default.
        _ => 11,
    }
}

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
    state.last_plan_tick = player.ticks;
    state.progress_anchor = (player.x, player.z);
    if announce {
        let (sx, sy, sz) = block_pos(player);
        messages.push(format!(
            "baritone: pathing to {} — {length} blocks from ({sx}, {sy}, {sz}){}",
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
    state.screen_drive = driving.then(|| {
        (inputs.forward, inputs.jump, inputs.sprint, inputs.mine.is_some())
    });
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
    /// them. In water the jump key rises gently instead, idling sinks, and
    /// the player's own `in_water` flag follows the feet cell.
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
            let bx = player.x.floor() as i32;
            let bz = player.z.floor() as i32;
            let feet = player.y.floor() as i32;
            // Support check: the executor mines the landing column of a
            // descend before stepping off, so the ground under the bot can
            // vanish while it stands there — the game drops it, and so
            // must the sim, or descend moves hang forever.
            if player.on_ground
                && !blocks
                    .get(&(bx, feet - 1, bz))
                    .is_some_and(|b| !super::is_passable(b))
            {
                player.on_ground = false;
                vy = 0.0;
            }
            let water_at = |y: i32| blocks.get(&(bx, y, bz)) == Some(&"minecraft:water");
            let in_water = water_at(feet) || water_at(feet + 1);
            player.in_water = in_water;
            if in_water {
                // Swimming: the jump key rises, idling sinks, and the
                // bottom stops the sink.
                vy = if jump { 0.12 } else { (vy - 0.02).max(-0.08) };
                player.y += vy;
                player.on_ground = false;
                if vy < 0.0 {
                    let solid = |y: i32| {
                        blocks
                            .get(&(bx, y, bz))
                            .is_some_and(|b| !super::is_passable(b))
                    };
                    if let Some(top) = (feet - 6..=feet)
                        .rev()
                        .find(|y| solid(*y))
                        .map(|y| f64::from(y + 1))
                        && player.y <= top
                    {
                        player.y = top;
                        player.on_ground = true;
                        vy = 0.0;
                    }
                }
            } else {
                if jump && player.on_ground {
                    player.on_ground = false;
                    vy = 0.42;
                }
                if !player.on_ground {
                    player.y += vy;
                    vy = (vy - 0.08) * 0.98;
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
            }
            let (inputs, tick_messages) = tick(state, &world, player, &[]);
            // Keep every tick lines: tests assert on announcements made
            // long before the mission ends.
            messages.extend(tick_messages);
            forward = inputs.forward;
            jump = inputs.jump;
            if std::env::var("BARITONE_DEBUG").as_deref() == Ok("1") {
                let aim_dbg = state.path.as_ref().and_then(|pth| {
                    pth.positions()
                        .get(state.next_index.min(pth.length() - 1))
                        .copied()
                });
                let step = state
                    .next_index
                    .checked_sub(1)
                    .and_then(|i| state.path.as_ref().and_then(|pth| pth.movements().get(i)));
                eprintln!(
                    "tick{} pos=({:.2},{:.2},{:.2}) fwd={} jump={} mine={:?} yaw={:?} ground={} step={}/{} aim={:?} mv={:?}",
                    player.ticks,
                    player.x,
                    player.y,
                    player.z,
                    forward,
                    jump,
                    inputs.mine,
                    inputs.yaw,
                    player.on_ground,
                    state.next_index,
                    state.path.as_ref().map(|pth| pth.length()).unwrap_or(0),
                    aim_dbg,
                    step,
                );
            }
            if let Some(pos) = inputs.mine {
                blocks.remove(&pos);
            }
            if let Some(plan) = &inputs.place {
                // The game places the held slot's block; the tests give
                // dirt, so the filled cell reads as dirt.
                blocks.insert(plan.into, "minecraft:dirt");
            }
            // A hunt that gave up revives on the next tick (see the
            // revival in tick): keep simulating while a revival is pending.
            if state.mission.is_none() && !state.hunt_stalled {
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

        let (inputs, messages) = tick(&mut state, &world, &player, &[]);
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
        let (inputs, _) = tick(&mut state, &world, &player, &[]);
        assert!(inputs.driving);

        // Walk the path: put the player near the current waypoint each
        // tick, until the goal block itself is underfoot.
        let mut messages = Vec::new();
        for (ticks, step) in (101..).zip(1..=2) {
            player.x = f64::from(step) + 0.4;
            player.z = f64::from(step) + 0.4;
            player.ticks = ticks;
            let (inputs, tick_messages) = tick(&mut state, &world, &player, &[]);
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
        tick(&mut state, &world, &player, &[]);
        let index = state.next_index;
        let center = center_of(state.path.as_ref().unwrap().positions()[1]);

        // 0.7 blocks from the waypoint's center is not reached yet.
        let near = player_at(center.0 + 0.7, 65.0, center.2, 101);
        tick(&mut state, &world, &near, &[]);
        assert_eq!(state.next_index, index);

        // 0.4 from it: reached, the next waypoint becomes the aim.
        let on = player_at(center.0 + 0.4, 65.0, center.2, 102);
        let (inputs, _) = tick(&mut state, &world, &on, &[]);
        assert!(inputs.driving);
        assert_eq!(state.next_index, index + 1);
    }

    #[test]
    fn descend_waypoint_needs_the_actual_drop() {
        // A rim with a one-block pit east of x=4: the plan walks east and
        // drops into the pit. Hovering over the pit one block above the
        // landing waypoint is not there (the old Y tolerance read 1.0 as
        // reached) — the path must keep aiming at the landing, because
        // the next move's corridor assumes the drop happened.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 12);
        for x in 5..=11 {
            for z in -3..=3 {
                blocks.insert((x, 64, z), "minecraft:air");
                blocks.insert((x, 63, z), "minecraft:stone");
            }
        }
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(4.6, 65.0, 0.5, 100);
        handle_command(&mut state, "goto 10 64 0", &player, &world);
        tick(&mut state, &world, &player, &[]);
        let index = state.next_index;
        let center = center_of(state.path.as_ref().unwrap().positions()[1]);
        assert_eq!(center.1, 64.0, "the second waypoint is the pit landing");

        let above = player_at(center.0 - 0.1, 65.0, center.2, 101);
        tick(&mut state, &world, &above, &[]);
        assert_eq!(state.next_index, index);

        let dropped = player_at(center.0 - 0.1, 64.2, center.2, 102);
        let (inputs, _) = tick(&mut state, &world, &dropped, &[]);
        assert!(inputs.driving);
        assert_eq!(state.next_index, index + 1);
    }

    #[test]
    fn mines_a_solid_block_the_plan_missed_in_front() {
        // The plan walks east on open floor; then the player stands one
        // block higher than the path expects with a solid straight ahead
        // at head height — terrain the plan never priced (a scene race,
        // or a waypoint skipped one height too early). The waypoint guard
        // checks the waypoint's own two blocks and sees nothing; the
        // front probe must mine the blocker instead of pushing into it.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 12);
        let reader1 = reader(&blocks);
        let world = LiveWorld::new(&reader1);
        let mut state = BotState::new();
        let player = player_at(0.5, 65.0, 0.5, 100);
        handle_command(&mut state, "goto 6 65 0", &player, &world);
        tick(&mut state, &world, &player, &[]);

        let mut revised = blocks.clone();
        revised.insert((1, 67, 0), "minecraft:stone");
        let reader2 = reader(&revised);
        let world = LiveWorld::new(&reader2);
        let raised = player_at(0.5, 66.0, 0.5, 101);
        let (inputs, _) = tick(&mut state, &world, &raised, &[]);
        assert_eq!(inputs.mine, Some((1, 67, 0)));
    }

    #[test]
    fn mining_stint_pauses_the_stall_timers() {
        // In reach of a target, the miner holds: the stint works the
        // block for its whole expected time. The stall timers would read
        // that hold as stuck after STUCK_REPLAN_TICKS and re-plan, then
        // end the mission after three — a buried target needs several
        // blocks dug first, more than the give-up budget. While a stint
        // holds, nothing may accumulate.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        blocks.insert((4, 65, 0), "minecraft:iron_ore");
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 1_000);
        handle_command(&mut state, "mine iron_ore", &player, &world);
        let (_, messages) = tick(&mut state, &world, &player, &[]);
        assert!(messages.iter().any(|m| m.contains("in range")));

        player.x = 2.5;
        player.z = 0.5;
        player.ticks = 1_100;
        let (inputs, _) = tick(&mut state, &world, &player, &[]);
        assert_eq!(inputs.mine, Some((4, 65, 0)), "mining did not start");

        // 200 held ticks: 120 of them would trip a stuck re-plan without
        // the pause, and the stint never breaks stride here.
        for t in 1_101..=1_300u64 {
            player.ticks = t;
            let (inputs, messages) = tick(&mut state, &world, &player, &[]);
            assert_eq!(inputs.mine, Some((4, 65, 0)), "stint broke at tick {t}");
            assert!(
                state.mission.is_some(),
                "mission died at tick {t}: {messages:?}"
            );
            assert_eq!(
                state.stuck_replans, 0,
                "stuck re-plans during a live stint at tick {t}"
            );
        }
    }

    #[test]
    fn a_scan_fed_hunt_gives_up_alive_and_revives() {
        // A mission whose targets came from a scan (strata never armed)
        // that gives up must not end /mine silently: the revival in tick
        // rebuilds the mission from the standing spec and escapes, so the
        // hunt carries on until it finds some or the user stops it.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        blocks.insert((12, 65, 0), "minecraft:iron_ore");
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 1_000);
        handle_command(&mut state, "mine iron_ore", &player, &world);
        let (_, messages) = tick(&mut state, &world, &player, &[]);
        assert!(messages.iter().any(|m| m.contains("in range")));

        // Walk the stuck counter to the give-up directly: the mission is
        // scan-fed, so strata is None and the old give-up left the hunt
        // dead with no revival.
        state.stuck_replans = MAX_STUCK_REPLANS;
        state.progress_tick = player.ticks - STUCK_REPLAN_TICKS;
        let (_, messages) = tick(&mut state, &world, &player, &[]);
        assert!(
            messages.iter().any(|m| m.contains("giving up")),
            "no give-up: {messages:?}"
        );
        assert!(state.mission.is_none());

        let (_, messages) = tick(&mut state, &world, &player, &[]);
        assert!(
            messages.iter().any(|m| m.contains("escaping")),
            "no revival: {messages:?}"
        );
        assert!(state.mission.is_some(), "the hunt did not revive");
    }

    #[test]
    fn digs_out_when_a_cascade_buries_the_bot() {
        // A gravel cascade fills the dig shaft: the box's own cells read
        // solid, plans fail from a solid start block, mining is gated on
        // standing, and the mid-air plan retry would idle inside the
        // stone forever. The cocoon dig mines the nearest breakable
        // block around the box — a side first — mission or not.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        // The box of a player at y 41.3 spans 41..=43; bury all of it.
        for y in 41..=43 {
            blocks.insert((2, y, 0), "minecraft:gravel");
        }
        blocks.insert((3, 41, 0), "minecraft:stone");
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let mut player = player_at(2.5, 41.3, 0.5, 1_000);
        player.on_ground = false;
        let (inputs, messages) = tick(&mut state, &world, &player, &[]);
        assert_eq!(
            inputs.mine,
            Some((3, 41, 0)),
            "no dig-out, messages: {messages:?}"
        );
        assert!(messages.iter().any(|m| m.contains("digging out")));
        assert!(state.mission.is_none(), "a dig-out needs no mission");
    }

    #[test]
    fn resting_on_snow_still_opens_the_ground_gates() {
        // Standing on a snow layer rests the player on the layer's
        // collision top while on_ground stays false — Horizon's support
        // query misses the layer. The old gates then blocked every mine
        // and step-up on snowy flats (the original /mine stone stall);
        // RESTING_TICKS of an unmoving y count as standing.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        blocks.insert((4, 65, 0), "minecraft:iron_ore");
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let mut player = player_at(2.5, 65.2, 0.5, 1_000);
        player.on_ground = false;
        handle_command(&mut state, "mine iron_ore", &player, &world);
        let (_, messages) = tick(&mut state, &world, &player, &[]);
        assert!(messages.iter().any(|m| m.contains("in range")));

        // The y never moves; after RESTING_TICKS the mining gate opens.
        let mut mine = None;
        for t in 1_001..=1_050u64 {
            player.ticks = t;
            let (inputs, _) = tick(&mut state, &world, &player, &[]);
            mine = inputs.mine;
        }
        assert_eq!(mine, Some((4, 65, 0)), "mining never started on snow");
    }

    #[test]
    fn mines_the_solid_column_below_a_descend_aim() {
        // The plan drops into a shaft, then the shaft filled in (a
        // race): the player stands over the landing column, the step's
        // breaks are empty, and the Y gate rightly refuses to wave the
        // player through. The straight-down probe mines the column
        // underfoot instead of pressing the rim forever.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 12);
        // A shaft at the origin: open at 64 and 63, support at 62.
        blocks.insert((0, 64, 0), "minecraft:air");
        blocks.insert((0, 63, 0), "minecraft:air");
        blocks.insert((0, 62, 0), "minecraft:stone");
        let reader1 = reader(&blocks);
        let world = LiveWorld::new(&reader1);
        let mut state = BotState::new();
        let player = player_at(0.5, 65.0, 0.5, 100);
        handle_command(&mut state, "goto 0 63 0", &player, &world);
        tick(&mut state, &world, &player, &[]);
        let center = center_of(state.path.as_ref().unwrap().positions()[1]);
        assert_eq!(center.0, 0.5, "on the shaft column: {center:?}");
        assert_eq!(center.2, 0.5, "on the shaft column: {center:?}");
        assert!(center.1 < 65.0, "the aim sits below: {center:?}");

        // The race: the shaft's throat is solid again, the player
        // hovers over it — the plan priced the drop against air.
        let mut blocks = blocks.clone();
        blocks.insert((0, 63, 0), "minecraft:stone");
        let reader2 = reader(&blocks);
        let world = LiveWorld::new(&reader2);
        let (inputs, _) = tick(&mut state, &world, &player, &[]);
        assert_eq!(
            inputs.mine,
            Some((0, 63, 0)),
            "the column below was not probed"
        );
    }

    #[test]
    fn mines_the_snow_cap_over_a_descend() {
        // The plan drops through an open shaft; then a snow cap settles
        // over the mouth and the player rests on the layer's collision
        // top (0.84 above the floor). The cap reads passable — the plan
        // priced no break for it — but its collision is the only thing
        // keeping the player above the drop: the down-probe mines it.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 12);
        blocks.insert((0, 64, 0), "minecraft:air");
        blocks.insert((0, 63, 0), "minecraft:air");
        blocks.insert((0, 62, 0), "minecraft:air");
        blocks.insert((0, 61, 0), "minecraft:stone");
        let reader1 = reader(&blocks);
        let world = LiveWorld::new(&reader1);
        let mut state = BotState::new();
        let player = player_at(0.5, 65.0, 0.5, 100);
        handle_command(&mut state, "goto 0 62 0", &player, &world);
        tick(&mut state, &world, &player, &[]);
        let center = center_of(state.path.as_ref().unwrap().positions()[1]);
        assert_eq!(center.0, 0.5, "over the shaft column: {center:?}");
        assert_eq!(center.2, 0.5, "over the shaft column: {center:?}");
        assert!(center.1 < 65.0, "the aim sits below: {center:?}");

        // The race: the cap settles, the player rests on it.
        let mut revised = blocks.clone();
        revised.insert((0, 64, 0), "minecraft:snow");
        let reader2 = reader(&revised);
        let world = LiveWorld::new(&reader2);
        let resting = player_at(0.5, 64.84, 0.5, 101);
        let (inputs, _) = tick(&mut state, &world, &resting, &[]);
        assert_eq!(
            inputs.mine,
            Some((0, 64, 0)),
            "the snow cap over the drop was not mined"
        );
    }

    #[test]
    fn mines_a_thin_snow_cap_at_the_feet_cell() {
        // A one-layer cap tops out ~0.2 above its cell's floor: the feet
        // rest there and round INTO the cap's cell — the old two-cell
        // probe (py-1, py-2) skipped the cap itself. The probe now
        // starts at the feet cell.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 12);
        blocks.insert((0, 64, 0), "minecraft:air");
        blocks.insert((0, 63, 0), "minecraft:air");
        blocks.insert((0, 62, 0), "minecraft:air");
        blocks.insert((0, 61, 0), "minecraft:stone");
        let reader1 = reader(&blocks);
        let world = LiveWorld::new(&reader1);
        let mut state = BotState::new();
        let player = player_at(0.5, 65.0, 0.5, 100);
        handle_command(&mut state, "goto 0 62 0", &player, &world);
        tick(&mut state, &world, &player, &[]);

        // The thin cap settles; the feet rest at its top.
        let mut revised = blocks.clone();
        revised.insert((0, 64, 0), "minecraft:snow");
        let reader2 = reader(&revised);
        let world = LiveWorld::new(&reader2);
        let rested = player_at(0.5, 64.2, 0.5, 101);
        let (inputs, _) = tick(&mut state, &world, &rested, &[]);
        assert_eq!(
            inputs.mine,
            Some((0, 64, 0)),
            "the thin cap at the feet cell was not mined"
        );
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
        let (inputs, messages) = tick(&mut state, &world, &player, &[]);
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
    fn inactive_ticks_keep_the_brain_driving() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 10);
        handle_command(&mut state, "goto 4 65 4", &player, &world);
        let (inputs, _) = tick(&mut state, &world, &player, &[]);
        assert!(inputs.driving);

        // A screen opens (the chat): the brain keeps pathing, Java
        // baritone style — the game applies the bot inputs through the
        // motion stage patch while the screen holds the key set, and
        // nothing the user types leaks into the motion.
        let paused = PlayerInfo {
            active: false,
            ..player_at(0.0, 65.0, 0.0, 11)
        };
        let (inputs, _) = tick(&mut state, &world, &paused, &[]);
        assert!(inputs.driving);
        assert!(!inputs.clear, "no release while the bot keeps driving");
        let (inputs, _) = tick(&mut state, &world, &paused, &[]);
        assert!(inputs.driving);

        // Back to the game: driving continues as before.
        let (inputs, _) = tick(&mut state, &world, &player, &[]);
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
        tick(&mut state, &world, &player, &[]);
        assert!(state.path.is_some());

        // Three failed re-plans already; stuck again: give up.
        state.progress_tick = player.ticks - 200;
        state.stuck_replans = MAX_STUCK_REPLANS;
        state.last_plan_tick = player.ticks - 1_000;
        let (inputs, messages) = tick(&mut state, &world, &player, &[]);
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
        tick(&mut state, &world, &player, &[]);

        let dead = PlayerInfo {
            health: 0.0,
            ..player_at(0.0, 65.0, 0.0, 11)
        };
        let (inputs, messages) = tick(&mut state, &world, &dead, &[]);
        assert!(!inputs.driving);
        assert!(messages.iter().any(|m| m.contains("died")));

        handle_command(&mut state, "goto 4 65 4", &player, &world);
        tick(&mut state, &world, &player, &[]);
        let nether = PlayerInfo {
            dimension: String::from("minecraft:the_nether"),
            ..player_at(0.0, 65.0, 0.0, 12)
        };
        let (inputs, messages) = tick(&mut state, &world, &nether, &[]);
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
        let (inputs, _) = tick(&mut state, &world, &player, &[]);
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
        let (inputs, messages) = tick(&mut state, &world, &player, &[]);
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
            tick(&mut state, &world, &player, &[]);
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
            tick(&mut state, &world, &player, &[]);
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
            tick(&mut state, &world, &player, &[]);
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
            tick(&mut state, &world, &player, &[]);
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
            let (_, messages) = tick(&mut state, &world, &player, &[]);
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
            let (inputs, _) = tick(&mut state, &world, &player, &[]);
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
            let (inputs, _) = tick(&mut state, &world, &player, &[]);
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
        tick(&mut state, &world, &player, &[]);
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
        let (_, messages) = tick(&mut state, &world, &player, &[]);
        assert!(messages.iter().any(|m| m.contains("in range")));

        // In range: mining starts.
        player.x = 2.5;
        player.z = 0.5;
        player.ticks = 1_100;
        let (inputs, _) = tick(&mut state, &world, &player, &[]);
        assert_eq!(inputs.mine, Some((4, 65, 0)));

        // Two stints that run triple their expected time: the block is
        // skipped, and the next scan finds nothing left to do — the
        // mission goes hunting at the ore's strata instead of quitting
        // (baritone's exploreForBlocks). The hunt outlives the dead
        // target: it is blacklisted for every rescan and the mission
        // carries on, escape-walking when a column dead-ends.
        let mut seen = Vec::new();
        for ticks in [1_200u64, 1_300, 1_400, 1_500, 2_000] {
            player.ticks = ticks;
            if state.mining.is_some() {
                state.mining.as_mut().expect("checked").started = ticks - 1_000;
            }
            let (_, messages) = tick(&mut state, &world, &player, &[]);
            if ticks == 1_400 {
                assert!(
                    messages.iter().any(|m| m.contains("can't break")),
                    "messages: {messages:?}"
                );
            }
            seen.extend(messages);
        }
        assert!(
            seen.iter().any(|m| m.contains("digging to Y 15 to hunt")),
            "messages: {seen:?}"
        );
        match &state.mission {
            Some(Mission::Mine(m)) => {
                assert!(
                    m.blacklist.contains(&(4, 65, 0)),
                    "blacklist: {:?}",
                    m.blacklist
                );
                assert!(m.targets.is_empty(), "queue: {:?}", m.targets);
                assert!(m.strata.is_some(), "no strata - the hunt did not arm");
            }
            None => panic!("expected a live hunt, got none"),
            Some(Mission::Goto(_)) | Some(Mission::Follow(_)) => {
                panic!("expected a live mine hunt, got another mission")
            }
        }
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
        let (inputs, _) = tick(&mut state, &world, &player, &[]);
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
        tick(&mut state, &world, &player, &[]);

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

        let (inputs, messages) = tick(&mut state, &world, &player, &[]);
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

        let (inputs, messages) = tick(&mut state, &world, &player, &[]);
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

    #[test]
    fn swims_up_out_of_a_pool() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        // A 3×3 pool two deep: the floor blocks at y64 become water over a
        // stone bottom at y62, so wading on the bottom cannot reach the
        // shore — only swimming up and out does.
        for x in -1..=1 {
            for z in -1..=1 {
                blocks.insert((x, 64, z), "minecraft:water");
                blocks.insert((x, 63, z), "minecraft:water");
                blocks.insert((x, 62, z), "minecraft:stone");
            }
        }
        let mut state = BotState::new();
        // Floating in the pool's center, feet at the top water cell.
        let mut player = player_at(0.5, 64.5, 0.5, 100);
        player.on_ground = false;

        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            let (claimed, _) = handle_command(&mut state, "goto 4 65 4", &player, &world);
            assert!(claimed);
        }
        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(
            messages.iter().any(|m| m.contains("reached")),
            "must swim out and reach the goal: {messages:?}"
        );
        assert!(player.on_ground, "ends standing on the shore");
    }

    #[test]
    fn swims_across_a_deep_pond_without_blocks_to_place() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        // A 3-wide pond with no bottom within reach: swimming is the only
        // way across, and the hotbar holds nothing placeable to bridge
        // with.
        for x in 1..=3 {
            for z in -8..=8 {
                blocks.insert((x, 64, z), "minecraft:water");
                blocks.insert((x, 63, z), "minecraft:water");
                blocks.insert((x, 62, z), "minecraft:water");
                blocks.insert((x, 61, z), "minecraft:stone");
            }
        }
        let mut state = BotState::new();
        let mut player = player_at(0.5, 65.0, 0.0, 100);

        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            let (claimed, _) = handle_command(&mut state, "goto 6 65 0", &player, &world);
            assert!(claimed);
        }
        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(
            messages.iter().any(|m| m.contains("reached")),
            "must swim across and reach the goal: {messages:?}"
        );
        assert!(player.x > f64::from(4), "crossed to the far shore");
    }

    #[test]
    fn plans_from_a_water_column() {
        // A plan can start mid-water: the bot re-plans while swimming
        // (the path ran out, a stuck replan fired) and the search must
        // seed from the water column — the old gate read the water below
        // the start as unwalkable ground and every re-plan died
        // silently, stranding the swim.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        for x in 1..=3 {
            for z in -8..=8 {
                blocks.insert((x, 64, z), "minecraft:water");
                blocks.insert((x, 63, z), "minecraft:water");
                blocks.insert((x, 62, z), "minecraft:water");
                blocks.insert((x, 61, z), "minecraft:stone");
            }
        }
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let mut player = player_at(2.5, 63.2, 0.0, 100);
        player.in_water = true;
        player.on_ground = false;
        handle_command(&mut state, "goto 6 65 0", &player, &world);
        let (_, messages) = tick(&mut state, &world, &player, &[]);
        assert!(
            state.path.is_some(),
            "no plan from the water column: {messages:?}"
        );
    }

    #[test]
    fn jumps_a_step_up_from_a_snow_rest() {
        // Planned while grounded, the walk then rests on a snow layer
        // (the feet sit ~0.84 above the floor) with the next waypoint
        // one block up: the old step-up threshold (aim > feet + 0.5)
        // read the step as flat and never jumped — the walk pressed the
        // step face until the hunt burned its escapes.
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 12);
        blocks.insert((0, 64, 0), "minecraft:snow");
        let reader1 = reader(&blocks);
        let world = LiveWorld::new(&reader1);
        let mut state = BotState::new();
        let player = player_at(-0.5, 65.0, 0.5, 100);
        handle_command(&mut state, "goto 1 65 0", &player, &world);
        tick(&mut state, &world, &player, &[]);
        assert_eq!(
            center_of(state.path.as_ref().unwrap().positions()[1]),
            (0.5, 65.0, 0.5),
            "the second waypoint is the snow column"
        );

        // The walk rests on the layer over the column.
        let rested = player_at(0.45, 64.84, 0.5, 101);
        let (inputs, _) = tick(&mut state, &world, &rested, &[]);
        assert!(inputs.jump, "no step-up jump from the snow rest");
    }

    #[test]
    fn follows_the_nearest_matching_entity() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 40);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 100);

        let (claimed, _) = handle_command(&mut state, "follow players", &player, &world);
        assert!(claimed);
        let entities = vec![
            EntityInfo {
                id: 1,
                kind: String::from("player"),
                name: Some(String::from("far")),
                pos: (5.0, 65.0, 5.0),
            },
            EntityInfo {
                id: 2,
                kind: String::from("player"),
                name: Some(String::from("near")),
                pos: (2.0, 65.0, 2.0),
            },
        ];
        let (inputs, messages) = tick(&mut state, &world, &player, &entities);
        assert!(messages.iter().any(|m| m.contains("following players")));
        assert_eq!(state.follow_target, Some(2), "the nearest one");
        assert_eq!(
            state.follow_goal,
            Some(GoalSpec::Near {
                x: 2,
                y: 65,
                z: 2,
                range: 3
            })
        );
        assert!(inputs.driving, "already walking toward the target");

        // The target wanders off and the other player is now nearer: the
        // goal moves with it, silently.
        let moved = vec![EntityInfo {
            id: 2,
            kind: String::from("player"),
            name: Some(String::from("near")),
            pos: (3.0, 65.0, 2.0),
        }];
        let (_, messages) = tick(&mut state, &world, &player, &moved);
        assert!(!messages.iter().any(|m| m.contains("following")), "no repeat announce");
        assert_eq!(
            state.follow_goal,
            Some(GoalSpec::Near {
                x: 3,
                y: 65,
                z: 2,
                range: 3
            })
        );

        // Nothing matching any more: the mission ends.
        let (_, messages) = tick(&mut state, &world, &player, &[]);
        assert!(messages.iter().any(|m| m.contains("nothing to follow")));
        assert!(state.mission.is_none());
    }

    #[test]
    fn follow_selectors_match_kinds_and_names() {
        assert!(follow_matches(&FollowSelector::Players, "player", None));
        assert!(!follow_matches(&FollowSelector::Players, "zombie", None));
        assert!(follow_matches(
            &FollowSelector::Player("Herobrine".into()),
            "minecraft:player",
            Some("herobrine")
        ));
        assert!(!follow_matches(
            &FollowSelector::Player("Herobrine".into()),
            "player",
            Some("notch")
        ));
        assert!(follow_matches(
            &FollowSelector::EntityKind("cow".into()),
            "minecraft:cow",
            None
        ));
        assert!(follow_matches(&FollowSelector::Entities, "enderman", None));
    }

    #[test]
    fn renders_goal_path_and_work_blocks() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 100);
        handle_command(&mut state, "goto 4 65 4", &player, &world);
        tick(&mut state, &world, &player, &[]);

        let scene = render_scene(&state);
        // The goal: a green box at the goal cell, two tall, inset a
        // little like baritone's.
        assert!(
            scene.boxes.iter().any(|(min, max, c)| {
                *c == COLOR_GOAL
                    && min[0] == 4.002
                    && min[1] == 65.0
                    && min[2] == 4.002
                    && max[0] == 4.998
                    && max[1] == 67.0
                    && max[2] == 4.998
            }),
            "green goal box missing: {:?}",
            scene.boxes
        );
        // The path: red segments joining its block centers, starting at
        // the next waypoint.
        let positions = state.path.as_ref().unwrap().positions();
        let first = center_of_f(positions[state.next_index]);
        let second = center_of_f(positions[state.next_index + 1]);
        assert!(
            scene.segs.iter().any(|(a, b, c)| {
                *c == COLOR_PATH
                    && a[0] == first[0]
                    && a[1] == first[1]
                    && a[2] == first[2]
                    && b[0] == second[0]
                    && b[1] == second[1]
                    && b[2] == second[2]
            }),
            "red path segment missing: {:?}",
            scene.segs
        );
        // The block walked into right now: magenta.
        let wp = positions[state.next_index];
        assert!(scene.boxes.iter().any(|(min, max, c)| {
            *c == COLOR_TO_WALK_INTO
                && min[0] == f64::from(wp.0)
                && min[1] == f64::from(wp.1)
                && min[2] == f64::from(wp.2)
                && max[0] == f64::from(wp.0) + 1.0
        }));
        // The block being broken right now: red.
        state.mining = Some(MiningStint {
            pos: (1, 65, 1),
            name: String::from("minecraft:stone"),
            started: 99,
        });
        let scene = render_scene(&state);
        assert!(scene.boxes.iter().any(|(min, _, c)| {
            *c == COLOR_TO_BREAK && min[0] == 1.0 && min[1] == 65.0 && min[2] == 1.0
        }));
    }

    #[test]
    fn goal_near_keeps_java_half_block_grace() {
        // GoalNear.isInRange: (range + 0.5)² — the feet block's center
        // may sit half a block outside the bare radius. The mine gate
        // leans on this: a target three under the player stands at
        // Euclidean 11 from the feet block, inside range 3 only with
        // the grace.
        let near = GoalNear {
            x: 0,
            y: 0,
            z: 0,
            range: 3.0,
        };
        assert!(near.is_in_goal(1, 3, 1), "11 must be in range 3");
        assert!(!near.is_in_goal(2, 3, 0), "13 must be out of range 3");
    }

    #[test]
    fn mines_straight_down_on_snowy_flat_ground() {
        // The live stall: snowy flat, the target stone three under the
        // surface, the mine gate closed at the start (Euclidean 11 > 9
        // before the half-block grace). The bot must start breaking on
        // the first tick instead of planning a descend-walk it then
        // freezes on — the step-break block hides under a snow layer
        // the crosshair keeps hitting.
        let mut blocks = HashMap::new();
        for x in -24..=24 {
            for z in -24..=24 {
                blocks.insert((x, 60, z), "minecraft:dirt");
                blocks.insert((x, 61, z), "minecraft:dirt");
                blocks.insert((x, 62, z), "minecraft:dirt");
                blocks.insert((x, 63, z), "minecraft:snow");
            }
        }
        blocks.insert((-6, 60, 1), "minecraft:stone");
        let mut state = BotState::new();
        let mut player = player_at(-7.0, 63.0, 0.6, 100);
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            let (claimed, _) = handle_command(&mut state, "mine stone", &player, &world);
            assert!(claimed);
        }
        // The gate is open where the player stands (the half-block
        // grace): the first tick already breaks toward the target.
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            player.ticks = 101;
            let (inputs, _) = tick(&mut state, &world, &player, &[]);
            assert!(inputs.driving);
            assert!(inputs.mine.is_some(), "no break started on the first tick");
        }
        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(state.mission.is_none(), "messages: {messages:?}");
        assert!(!blocks.contains_key(&(-6, 60, 1)), "the stone was never mined");
    }

    #[test]
    fn hunts_at_strata_when_nothing_is_exposed() {
        // `/mine iron_ore` with nothing exposed: the bot digs down to
        // the ore's Y strata instead of quitting "0 × iron_ore", and
        // the rescans along the way pick the ore up once it is in
        // range (baritone's exploreForBlocks).
        let mut blocks = HashMap::new();
        for x in -14..=14 {
            for z in -14..=14 {
                for y in 10..=64 {
                    blocks.insert((x, y, z), "minecraft:stone");
                }
            }
        }
        // One iron ore straight below the start, below the initial
        // scan's vertical reach (radius 16 from the surface): the hunt
        // must find it on the way down, before any stripping starts.
        blocks.insert((0, 15, 0), "minecraft:iron_ore");
        let mut state = BotState::new();
        let mut player = player_at(0.0, 65.0, 0.0, 100);
        {
            let reader = reader(&blocks);
            let world = LiveWorld::new(&reader);
            let (claimed, messages) =
                handle_command(&mut state, "mine iron_ore", &player, &world);
            assert!(claimed);
            assert!(messages[0].contains("mining iron_ore"));
        }
        let messages = drive(&mut blocks, &mut state, &mut player);
        assert!(state.mission.is_none(), "messages: {messages:?}");
        assert!(
            messages.iter().any(|m| m.contains("mine done — 1 × iron_ore")),
            "messages: {messages:?}"
        );
        assert_ne!(
            blocks.get(&(0, 15, 0)),
            Some(&"minecraft:iron_ore"),
            "the ore was never mined"
        );
        assert!(
            messages.iter().any(|m| m.contains("digging to Y 15 to hunt")),
            "the hunt was not announced: {messages:?}"
        );
    }

    #[test]
    fn strata_y_matches_ore_generation() {
        assert_eq!(strata_y("diamond_ore"), -59);
        assert_eq!(strata_y("deepslate_diamond_ore"), -59);
        assert_eq!(strata_y("iron_ore"), 15);
        assert_eq!(strata_y("coal_ore"), 45);
        assert_eq!(strata_y("something_else"), 11);
    }

    #[test]
    fn screen_inputs_expose_the_driving_keys() {
        let mut blocks = HashMap::new();
        stone_floor(&mut blocks, 8);
        let reader = reader(&blocks);
        let world = LiveWorld::new(&reader);
        let mut state = BotState::new();
        let player = player_at(0.0, 65.0, 0.0, 100);
        handle_command(&mut state, "goto 4 65 4", &player, &world);
        let (inputs, _) = tick(&mut state, &world, &player, &[]);
        assert!(inputs.driving);
        // The screen gates read this through the global state in the
        // game; here the local state is the one under test.
        assert_eq!(
            state.screen_drive,
            Some((true, false, false, false)),
            "a driving tick exposes its inputs"
        );
    }
}
