//! World model for the Baritone pathfinder.
//!
//! This module wraps Horizon's control socket protocol (`horizon-ctl`) to
//! provide block queries and player state to the pathfinder.
//!
//! Ported from baritone's `BlockStateInterface` and `CalculationContext`,
//! adapted to Horizon's JSON-over-unix-socket RPC protocol.

use std::env;
use std::path::PathBuf;

use anyhow::{Result, anyhow};

/// Default control socket path (from horizon-ctl's `--socket` option or
/// `$XDG_RUNTIME_DIR/horizon-control.sock`).
fn default_socket() -> PathBuf {
    if let Ok(p) = env::var("HORIZON_CONTROL") {
        return PathBuf::from(p);
    }
    if let Ok(xdg) = env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join("horizon-control.sock");
    }
    PathBuf::from("/run/user/1000/horizon-control.sock")
}

/// A handle to the Horizon game instance via its control socket.
/// All queries are synchronous JSON requests sent over the Unix domain socket.
pub struct World {
    socket_path: PathBuf,
}

impl World {
    pub fn new() -> Result<Self> {
        Ok(World {
            socket_path: default_socket(),
        })
    }

    pub fn with_socket(path: impl Into<PathBuf>) -> Result<Self> {
        let w = World {
            socket_path: path.into(),
        };
        w.check_alive()?;
        Ok(w)
    }

    pub fn socket_path(&self) -> &PathBuf {
        &self.socket_path
    }

    /// Verify the game is reachable and running.
    fn check_alive(&self) -> Result<()> {
        let _ = self.send_command(&["status"])?;
        Ok(())
    }

    /// Send a command to the horizon-ctl socket and parse the JSON reply.
    fn send_command(&self, args: &[&str]) -> Result<serde_json::Value> {
        use std::io::Write;
        use std::os::unix::net::UnixStream;

        let sock = UnixStream::connect(&self.socket_path)?;
        sock.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;

        let cmd = args.join(" ");
        let request = format!("{}\n", cmd);
        let mut sock = sock;
        sock.write_all(request.as_bytes())?;
        // Shut down the write half so the server knows we're done sending.
        let _ = sock.shutdown(std::net::Shutdown::Write);

        use std::io::Read;
        let mut buf = Vec::new();
        let mut reader = &sock;
        reader.read_to_end(&mut buf)?;

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
        Ok(parsed)
    }

    /// Query the block state at a single position.
    /// Equivalent to `horizon-ctl block X Y Z`.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> String {
        let args = ["block", &x.to_string(), &y.to_string(), &z.to_string()];
        match self.send_command(&args) {
            Ok(v) => v
                .get("block")
                .and_then(|b| b.as_str())
                .unwrap_or("minecraft:air")
                .to_string(),
            Err(_) => "minecraft:air".to_string(),
        }
    }

    /// Query a box of non-air blocks defined by two opposite corners.
    /// Uses `horizon-ctl blocks X1 Y1 Z1 X2 Y2 Z2`.
    /// Returns `Vec<(x, y, z, block_name)>`.
    pub fn get_blocks_box(
        &self,
        x1: i32,
        y1: i32,
        z1: i32,
        x2: i32,
        y2: i32,
        z2: i32,
    ) -> Result<Vec<(i32, i32, i32, String)>> {
        let args = [
            "blocks",
            &x1.to_string(),
            &y1.to_string(),
            &z1.to_string(),
            &x2.to_string(),
            &y2.to_string(),
            &z2.to_string(),
        ];
        let resp = self.send_command(&args)?;

        let mut results = Vec::new();
        if let Some(arr) = resp.get("blocks").and_then(|b| b.as_array()) {
            for entry in arr {
                if let Some(arr) = entry.as_array()
                    && arr.len() >= 4
                {
                    let x = arr[0].as_i64().unwrap_or(0) as i32;
                    let y = arr[1].as_i64().unwrap_or(0) as i32;
                    let z = arr[2].as_i64().unwrap_or(0) as i32;
                    let name = arr[3].as_str().unwrap_or("minecraft:air").to_string();
                    results.push((x, y, z, name));
                }
            }
        }
        Ok(results)
    }

    /// Get the player's current position.
    /// Uses `horizon-ctl state`.
    pub fn player_pos(&self) -> Result<(i32, i32, i32)> {
        let resp = self.send_command(&["state"])?;
        let player = resp
            .get("player")
            .or_else(|| resp.get("server"))
            .ok_or_else(|| anyhow!("no player in state"))?;
        let block_pos = player
            .get("block_pos")
            .or_else(|| player.get("position"))
            .and_then(|p| p.as_array())
            .ok_or_else(|| anyhow!("no block_pos in player state"))?;
        if block_pos.len() >= 3 {
            let x = block_pos[0].as_i64().unwrap_or(0) as i32;
            let y = block_pos[1].as_i64().unwrap_or(0) as i32;
            let z = block_pos[2].as_i64().unwrap_or(0) as i32;
            return Ok((x, y, z));
        }
        Err(anyhow!("invalid block_pos: {:?}", block_pos))
    }
}

/// A cache of block states queried from the game.
#[derive(Default)]
pub struct BlockCache {
    blocks: std::collections::HashMap<(i32, i32, i32), String>,
}

impl BlockCache {
    pub fn insert(&mut self, x: i32, y: i32, z: i32, block: String) {
        self.blocks.insert((x, y, z), block);
    }

    pub fn get(&self, x: i32, y: i32, z: i32) -> Option<&str> {
        self.blocks.get(&(x, y, z)).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// A world wrapper that fetches a region of blocks into a cache on demand
/// and serves cached lookups to the pathfinder. Uncached positions are
/// treated as air (the pathfinder only explores loaded chunks).
pub struct LiveWorld {
    pub(crate) inner: World,
    cache: BlockCache,
    /// Half-size of the cubic region to prefetch around the player.
    pub cache_radius: i32,
}

impl LiveWorld {
    pub fn new() -> Result<Self> {
        let inner = World::new()?;
        let mut world = LiveWorld {
            inner,
            cache: BlockCache::default(),
            cache_radius: 24, // matches Horizon's default render distance
        };
        world.prefetch()?;
        Ok(world)
    }

    /// Prefetch a cubic region around the player into the cache.
    fn prefetch(&mut self) -> Result<()> {
        let (px, py, pz) = self.inner.player_pos()?;
        let r = self.cache_radius;
        let size = (r * 2 + 1) as usize;

        // Horizon limits to 32768 blocks per request.
        // We cap the box to fit (32^3 = 32768).
        let max_side: i32 = 31; // safe limit
        let total = (size as i64) * (size as i64) * (size as i64);
        log::debug!(
            "prefetching {} blocks around ({}, {}, {})",
            total,
            px,
            py,
            pz
        );

        let x1 = px - r;
        let y1 = (py - r).max(0);
        let z1 = pz - r;
        let x2 = px + r;
        let y2 = py + r;
        let z2 = pz + r;

        // Fetch in sub-chunks to stay under the 32768 limit.
        let step = max_side;
        for bx in (x1..=x2).step_by(step as usize) {
            let bx2 = (bx + step).min(x2);
            for by in (y1..=y2).step_by(step as usize) {
                let by2 = (by + step).min(y2);
                for bz in (z1..=z2).step_by(step as usize) {
                    let bz2 = (bz + step).min(z2);
                    let result = self.inner.get_blocks_box(bx, by, bz, bx2, by2, bz2)?;
                    for (x, y, z, block) in result {
                        self.cache.insert(x, y, z, block);
                    }
                }
            }
        }
        log::debug!("cached {} non-air blocks", self.cache.len());
        Ok(())
    }

    pub fn socket_path(&self) -> &PathBuf {
        self.inner.socket_path()
    }

    pub fn player_pos(&self) -> Result<(i32, i32, i32)> {
        self.inner.player_pos()
    }

    /// Get block at a position from the cache.
    /// Uncached positions default to "minecraft:air" — the pathfinder only
    /// explores loaded/cached chunks, so uncached = not loaded = air.
    /// This avoids per-block socket round-trips during search.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> String {
        self.cache
            .get(x, y, z)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "minecraft:air".to_string())
    }
}
