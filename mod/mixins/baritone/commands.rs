// The command names the chat box completes and colours as the game's own,
// advertised only while the mod is enabled.

fn baritone_commands(game: &mut crate::game::Game<'_>, commands: &mut Vec<&'static str>) {
    if game.native_on(crate::game::baritone::bot::ID) {
        commands.extend_from_slice(crate::game::baritone::bot::command_names());
    }
}
