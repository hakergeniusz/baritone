// Chat commands: the client asks this hook before running a command of
// its own, so setting `ran` claims the line. Only the mod's commands are
// claimed, and only while the mod is enabled; everything else — including
// server commands — passes through untouched.

fn baritone_command(game: &mut crate::game::Game<'_>, command: &str, ran: &mut bool) {
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
    let pos = game
        .player
        .as_ref()
        .map_or([0.0; 3], |p| [p.position.x, p.position.y, p.position.z]);
    let player = bot::PlayerInfo {
        x: pos[0],
        y: pos[1],
        z: pos[2],
        on_ground: false,
        in_water: false,
        active: true,
        ticks: u64::from(game.ticks),
        health: game.player.as_ref().map_or(20.0, |p| p.health.health()),
        dimension: String::from(game.dim.id()),
        yaw: game.yaw,
        hotbar: std::array::from_fn(|i| {
            game.inventory
                .get(i)
                .and_then(|s| s.as_ref())
                .map(|s| String::from(s.item.as_ref()))
        }),
    };
    let (_, messages) = {
        let reader = |x: i32, y: i32, z: i32| game.native_block([x, y, z]);
        let world = crate::game::baritone::world_view::LiveWorld::new(&reader);
        bot::command_stateful(command, &player, &world)
    };
    for message in messages {
        game.native_chat(&message);
    }
}
