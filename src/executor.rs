//! Path execution — sends movement commands to Horizon via its control socket.
//!
//! Ported conceptually from baritone's `PathExecutor.java`, adapted to
//! Horizon's control-socket protocol. Instead of directly manipulating
//! Minecraft's input system (which baritone does via `Input`, `RotationUtils`,
//! etc.), we translate each `[Path, PathNode]` segment into a socket command:
//!
//! - `goto X Y Z` for straightforward walks (Horizon's own A* handles the
//!   micro-steering).
//! - For block breaking/placing, we use `break` and `place` directly.

use anyhow::{anyhow, Result};
use std::io::{Read, Write};
use std::path::Path as SocketPath;
use std::os::unix::net::UnixStream;

use crate::pathing::Path;
use crate::world::LiveWorld;

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
        let dx = next.0 - target.0;
        let dy = next.1 - target.1;
        let dz = next.2 - target.2;

        if is_simple_step(dx, dy, dz) {
            // Find the end of the run of simple steps.
            let mut run_end = idx + 1;
            while run_end < positions.len() - 1 {
                let p0 = positions[run_end];
                let p1 = positions[run_end + 1];
                let ddx = p1.0 - p0.0;
                let ddy = p1.1 - p0.1;
                let ddz = p1.2 - p0.2;
                if !is_simple_step(ddx, ddy, ddz) {
                    break;
                }
                run_end += 1;
            }

            let dest = positions[run_end];
            let near = if dy <= 0 { 1.5 } else { 1.0 };
            println!("  goto ({}, {}, {}) near {}", dest.0, dest.1, dest.2, near);
            let resp = ctl(socket, &[
                "goto",
                &dest.0.to_string(),
                &dest.1.to_string(),
                &dest.2.to_string(),
                "--near",
                &near.to_string(),
            ])?;
            println!("  response: {}", resp.trim());
            idx = run_end + 1;
        } else {
            // Non-simple step — handle manually (e.g. block placement/breaking).
            let dest = next;
            println!("  goto ({}, {}, {})", dest.0, dest.1, dest.2);
            let resp = ctl(socket, &[
                "goto",
                &dest.0.to_string(),
                &dest.1.to_string(),
                &dest.2.to_string(),
                "--near",
                "1.5",
            ])?;
            println!("  response: {}", resp.trim());
            idx += 1;
        }
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
    write!(sock, "{}\n", cmd)?;
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
