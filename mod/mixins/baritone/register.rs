// The commands the mod adds to the server's command tree, asked once when
// the server opens: a player's line for one of these reaches perform.rs
// at all, and every player's chat box completes them as vanilla's.
// `help` and `stop` are vanilla's on the server, so they stay out of the
// tree (the client-side hook answers both, and /cancel does /stop's work).

fn baritone_register(game: &mut crate::game::Game<'_>, commands: &mut Vec<NativeCommand>) {
    use crate::game::baritone::bot;

    if !game.native_on(bot::ID) {
        return;
    }
    for usage in bot::command_usages() {
        commands.push(NativeCommand::new(usage));
    }
}
