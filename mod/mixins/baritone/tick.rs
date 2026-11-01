// Once a tick: the brain plans and drives; this file is its game-side
// half. It builds the player info the brain reads (position, ground,
// water, yaw), hands it a block reader over the game's own chunks, and
// applies the returned inputs to the game's key set — the game reads the
// same set right after this hook for its motion tick, so steering lands
// this tick. While the bot drives, the key set is the bot's; when it
// stops, the keys are released exactly once (the user's own presses are
// left alone). A `mine` input aims the view at a block and holds attack
// only while a ray from the eye along the aimed view really hits it.
//
// One helper file, one function: the aiming helpers live inside.

fn baritone_tick(game: &mut crate::game::Game<'_>) {
    use crate::game::baritone::bot;

    // Disabled in Options > Mods, or the world closed: drop the mission
    // and, once, the keys. Runs even while the world is gone.
    if !game.native_on(bot::ID) || game.player.is_none() {
        let inputs = bot::world_closed_stateful();
        if inputs.clear {
            game.keys.remove(&KeyCode::KeyW);
            game.keys.remove(&KeyCode::Space);
            game.keys.remove(&KeyCode::ControlLeft);
            game.left = false;
        }
        return;
    }

    let (px, py, pz, on_ground, health) = {
        let p = game.player.as_ref().expect("checked above");
        (
            p.position.x,
            p.position.y,
            p.position.z,
            p.ground.on_ground,
            p.health.health(),
        )
    };
    let in_water = {
        let b = [px.floor() as i32, py.floor() as i32, pz.floor() as i32];
        game.res.facts().get(game.world.block(b[0], b[1], b[2])).fluid
            != horizon_client::blocks::Fluid::None
    };
    let player = bot::PlayerInfo {
        x: px,
        y: py,
        z: pz,
        on_ground,
        in_water,
        active: game.screen == Screen::Playing && game.sleeping.is_none(),
        ticks: u64::from(game.ticks),
        health,
        dimension: String::from(game.dim.id()),
        yaw: game.yaw,
    };

    let (inputs, messages) = {
        let reader = |x: i32, y: i32, z: i32| game.native_block([x, y, z]);
        let world = crate::game::baritone::world_view::LiveWorld::new(&reader);
        bot::tick_stateful(&world, &player)
    };
    for message in messages {
        game.native_chat(&message);
    }
    if inputs.driving {
        if inputs.forward {
            game.keys.insert(KeyCode::KeyW);
        } else {
            game.keys.remove(&KeyCode::KeyW);
        }
        if inputs.jump {
            game.keys.insert(KeyCode::Space);
        } else {
            game.keys.remove(&KeyCode::Space);
        }
        if inputs.sprint {
            game.keys.insert(KeyCode::ControlLeft);
        } else {
            game.keys.remove(&KeyCode::ControlLeft);
        }
        if let Some(yaw) = inputs.yaw {
            game.yaw = yaw;
        }
        if let Some(target) = inputs.mine {
            game.left = aim_at_block(game, [target.0, target.1, target.2]);
        } else {
            game.left = false;
        }
    } else if inputs.clear {
        game.keys.remove(&KeyCode::KeyW);
        game.keys.remove(&KeyCode::Space);
        game.keys.remove(&KeyCode::ControlLeft);
        game.left = false;
    }

    /// The unit vector a (yaw, pitch) view points along, as the game's own
    /// `view_vector` (`-sin(yaw)cos(pitch), -sin(pitch), cos(yaw)cos(pitch)`).
    fn view(yaw: f32, pitch: f32) -> [f64; 3] {
        let (y, p) = (f64::from(yaw).to_radians(), f64::from(pitch).to_radians());
        [-y.sin() * p.cos(), -p.sin(), y.cos() * p.cos()]
    }

    /// Aims the view at `target` and reports whether the crosshair now
    /// really hits it: a ray from the eye along the aimed view is clipped
    /// against the world first, and attack is held only on a verified hit
    /// so the bot never punches air or the wrong block. The aim point is
    /// the block's center, or a point pulled 0.45 into one of its faces
    /// when the center ray is blocked by a neighbour's corner.
    fn aim_at_block(game: &mut crate::game::Game<'_>, target: [i32; 3]) -> bool {
        let Some((origin, _)) = game.aim() else {
            return false;
        };
        let center = [
            f64::from(target[0]) + 0.5,
            f64::from(target[1]) + 0.5,
            f64::from(target[2]) + 0.5,
        ];
        let mut candidates = vec![center];
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut point = center;
                point[axis] += sign * 0.45;
                candidates.push(point);
            }
        }
        let reach = game.reach();
        for point in candidates {
            let (dx, dy, dz) = (point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]);
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < 0.05 || dist > reach {
                continue;
            }
            let yaw = (-dx).atan2(dz).to_degrees() as f32;
            let pitch = (-(dy / dist)).asin().to_degrees() as f32;
            if let Some(hit) = game.clip_ray(origin, view(yaw, pitch), reach)
                && hit.pos == target
            {
                game.yaw = yaw;
                game.pitch = pitch.clamp(-90.0, 90.0);
                return true;
            }
        }
        // Nothing verified exactly — corner-on geometry (the eye level with
        // a block boundary) can put every candidate ray on a voxel edge the
        // game's raycast resolves against a neighbour. Face the target's
        // center and swing anyway when the crosshair lands on any solid
        // block along the way: whatever the ray hits first stands between
        // the eye and the target, and breaking it is progress toward the
        // target either way.
        let (dx, dy, dz) = (
            center[0] - origin[0],
            center[1] - origin[1],
            center[2] - origin[2],
        );
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist > 0.05 {
            game.yaw = (-dx).atan2(dz).to_degrees() as f32;
            game.pitch = (-(dy / dist)).asin().to_degrees() as f32;
        }
        if dist > reach {
            return false;
        }
        match game.clip_ray(origin, view(game.yaw, game.pitch), reach) {
            Some(hit) => {
                let block = game.native_block(hit.pos);
                !bot::is_passable(&block)
            }
            None => false,
        }
    }
}
