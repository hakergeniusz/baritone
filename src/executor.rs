//! Path execution — sends movement commands to Horizon via its control socket.
//!
//! Ported conceptually from baritone's `PathExecutor.java`, adapted to
//! Horizon's control-socket protocol. Baritone's executor drives the game's
//! input system directly and can mine or place blocks mid-path; this port
//! instead delegates micro-steering to Horizon's own `goto` command, which
//! walks but cannot mine or place. The planner reflects that: movements
//! that would need mining or placing are impassable (`COST_INF`), so every
//! planned path is walk-only.
//!
//! Each run of simple steps (walk or 1-block ascend/descend) becomes one
//! `goto X Y Z --near R` command; non-simple steps (e.g. parkour jumps) get
//! one goto aimed at their landing block. After each segment the player's
//! actual position is checked against the intended one, so drift (Horizon
//! taking a different route, or stopping early) fails loudly instead of
//! cascading through the rest of the path.

use anyhow::{Context, Result, anyhow};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path as SocketPath;

use crate::pathing::Path;
use crate::world::LiveWorld;

/// Maximum allowed block-coordinate error after a goto segment.
const DRIFT_TOLERANCE: i32 = 2;

/// Execute a path by sending commands to the Horizon control socket.
pub fn execute_path(world: &LiveWorld, path: &Path) -> Result<()> {
    if path.positions().is_empty() {
        return Ok(());
    }

    println!("Executing path of {} steps...", path.positions().len());

    let positions = path.positions();
    let socket = world.socket_path();

    let mut idx = 0;
    while idx < positions.len().saturating_sub(1) {
        let target = positions[idx];
        let next = positions[idx + 1];
        let (dx, dy, dz) = (next.0 - target.0, next.1 - target.1, next.2 - target.2);

        // For a run of simple steps, send one goto for the whole run;
        // otherwise aim at the single non-simple step's landing block.
        let (dest, next_idx) = if is_simple_step(dx, dy, dz) {
            let mut run_end = idx + 1;
            while run_end < positions.len() - 1 {
                let p0 = positions[run_end];
                let p1 = positions[run_end + 1];
                if !is_simple_step(p1.0 - p0.0, p1.1 - p0.1, p1.2 - p0.2) {
                    break;
                }
                run_end += 1;
            }
            (positions[run_end], run_end + 1)
        } else {
            (next, idx + 1)
        };
        // Ascents need to land precisely on top of the block, so keep the
        // tolerance tight there; walks and drops can stop a bit short.
        let near = if is_simple_step(dx, dy, dz) && dy > 0 {
            1.0
        } else {
            1.5
        };

        println!("  goto ({}, {}, {}) near {}", dest.0, dest.1, dest.2, near);
        let resp = ctl(
            socket,
            &[
                "goto",
                &dest.0.to_string(),
                &dest.1.to_string(),
                &dest.2.to_string(),
                "--near",
                &near.to_string(),
            ],
        )
        .with_context(|| format!("goto to ({}, {}, {}) failed", dest.0, dest.1, dest.2))?;
        println!("  response: {}", resp.trim());

        // Verify we actually arrived: Horizon's goto may report success via
        // a very different route, or stop short of the target.
        let (px, py, pz) = world
            .player_pos()
            .context("failed to read player position after goto segment")?;
        let drift = (px - dest.0)
            .abs()
            .max((py - dest.1).abs().max((pz - dest.2).abs()));
        if drift > DRIFT_TOLERANCE {
            return Err(anyhow!(
                "path execution aborted: after goto to ({}, {}, {}) the player is at \
                 ({}, {}, {}) — drifted {} blocks (max {})",
                dest.0,
                dest.1,
                dest.2,
                px,
                py,
                pz,
                drift,
                DRIFT_TOLERANCE
            ));
        }

        idx = next_idx;
    }

    println!("Path execution complete.");
    Ok(())
}

/// Whether a movement delta is a simple step (walk or 1-block ascend/descend).
fn is_simple_step(dx: i32, dy: i32, dz: i32) -> bool {
    let adx = dx.abs();
    let ady = dy.abs();
    let adz = dz.abs();
    ady <= 1 && (adx + adz) <= 1
}

/// Send a single command-line command to the horizon-ctl control socket
/// and return the text response.
fn ctl(socket_path: &SocketPath, args: &[&str]) -> Result<String> {
    let sock = UnixStream::connect(socket_path)?;
    sock.set_read_timeout(Some(std::time::Duration::from_secs(120)))?;

    let cmd = args.join(" ");
    let mut sock = sock;
    writeln!(sock, "{}", cmd)?;
    let _ = sock.shutdown(std::net::Shutdown::Write);

    let mut buf = Vec::new();
    sock.read_to_end(&mut buf)?;

    let text = String::from_utf8_lossy(&buf);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("empty response from horizon-ctl"));
    }
    let parsed: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| anyhow!("failed to parse JSON: {e}\nraw: {trimmed}"))?;
    if let Some(err) = parsed.get("error") {
        return Err(anyhow!("horizon-ctl error: {}", err));
    }
    Ok(trimmed.to_string())
}
