#!/usr/bin/env python3
"""Stage patch: wire the baritone mod's rendering into the game frame.

The mixin tool refuses every injection whose target function body uses
closures, and the frame's whole world-line path is closure-heavy
(`Game::render`, `debug_gizmos`, `fishing_line`, `camera_relative`), so
the mod's wireframes cannot be hooked in as a mixin. The rendering
primitives live in the mod (`bot::render_scene`, pure and tested); this
patch — applied to the STAGED source after `horizon-mixins apply` — adds
the one call the mod cannot make itself: appending the scene's boxes and
segments to the frame's debug gizmos, right where the game gathers its
own debug lines. Depth-tested, fogged, drawn like the game's F3
renderers.

Run: python3 tools/stage_patch.py <stage-dir>
Fails loudly (nonzero exit) if the anchor is missing — a Horizon change
needs a re-check, exactly like a mixin target change does.
"""

import sys
from pathlib import Path

ANCHOR = """        let mut lines = self.fishing_line(alpha, size);
        let g = self.debug_gizmos(alpha as f32, camera.to_array());
        lines.extend(g.lines);
"""

REPLACEMENT = """        let mut lines = self.fishing_line(alpha, size);
        let mut g = self.debug_gizmos(alpha as f32, camera.to_array());
        // baritone mod: draw the bot's wireframes — goal box, path lines,
        // the blocks it breaks, places and walks into — with the game's
        // own debug gizmos. Primitives come from the mod's tested
        // render_scene; the scene is empty while the bot is idle.
        {
            let scene = crate::game::baritone::bot::with_state(
                |state| crate::game::baritone::bot::render_scene(state),
            );
            for (min, max, color) in &scene.boxes {
                g.cuboid(*min, *max, *color, horizon_client::gizmos::STROKE_WIDTH);
            }
            for (start, end, color) in &scene.segs {
                g.line(*start, *end, *color, 5.0);
            }
        }
        lines.extend(g.lines);
"""


def main() -> None:
    stage = Path(sys.argv[1])
    game = stage / "crates/horizon-client/src/game.rs"
    text = game.read_text()
    count = text.count(ANCHOR)
    if count != 1:
        print(
            f"stage_patch: expected exactly 1 anchor in game.rs, found {count} — "
            "the client source changed; re-check the render patch"
        )
        sys.exit(1)
    if "baritone::bot::render_scene" in text:
        print("stage_patch: already applied")
        return
    game.write_text(text.replace(ANCHOR, REPLACEMENT))
    print("stage_patch: render hook wired into the frame")


if __name__ == "__main__":
    main()
