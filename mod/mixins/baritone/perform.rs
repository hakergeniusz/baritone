// Server commands: a player's line for a command the mod registered
// (register.rs) lands here with that player's state swapped in. Only the
// mod's commands are claimed; the answers go out through the server's
// feedback, which reaches the sender's chat (and the console) — this is
// the path horizon-ctl's `chat` and other players' chat take, where the
// client-side hook in command.rs never runs.

fn baritone_perform(
    game: &mut crate::game::Game<'_>,
    player: &NativeServerPlayer,
    command: &str,
    ran: &mut bool,
) {
    use crate::game::baritone::bot;

    let name = command
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if !bot::command_names().contains(&name.as_str()) || !game.native_on(bot::ID) {
        return;
    }
    *ran = true;
    // The playing client publishes a fresh PlayerInfo every tick (the
    // server's own view of the player — the hub's guest entry — lags
    // behind respawns and other teleports); the swapped-in player is only
    // the fallback before the first client tick.
    let info = bot::published_player().unwrap_or(bot::PlayerInfo {
        x: player.position[0],
        y: player.position[1],
        z: player.position[2],
        on_ground: false,
        in_water: false,
        active: true,
        ticks: u64::from(game.ticks),
        health: player.health,
        dimension: player.dimension.clone(),
        yaw: 0.0,
        hotbar: std::array::from_fn(|_| None),
    });
    let (_, messages) = {
        let reader = |x: i32, y: i32, z: i32| game.native_block([x, y, z]);
        let world = crate::game::baritone::world_view::LiveWorld::new(&reader);
        bot::command_stateful(command, &info, &world)
    };
    for message in messages {
        game.native_feedback(Text::literal(&message));
    }
}
