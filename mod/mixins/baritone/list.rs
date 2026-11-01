// Baritone's row in Options > Mods.

fn baritone_listed(mods: &mut Vec<crate::game::native_hooks::NativeMod>) {
    use crate::game::baritone::bot;
    mods.push(crate::game::native_hooks::NativeMod {
        id: bot::ID.into(),
        title: format!("Baritone {}", bot::VERSION),
        description: bot::DESCRIPTION.into(),
        keys: vec![],
    });
}
