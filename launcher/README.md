# Leonida launcher

A small desktop window over the repository's own commands. It checks your copy of the game with
`make_game.py`, builds the port, and starts the loader as [docs/boot.md](../docs/boot.md) (section 2)
and the [README](../README.md) ("How to build", "Running the loader on Linux") describe. It
draws nothing from the game, reads no game file itself, and never goes online: no telemetry, no
update check.

## Build and run

Rust 1.88 or newer, from this folder:

```sh
cargo build --release
cargo run --release                # finds the repository above itself or the current folder
cargo run --release -- --repo PATH # or names it: the folder that holds make_game.py
```

The release build is a window without a console on Windows. It needs a GPU with Vulkan, DX12
(Windows) or, failing those, draws on the CPU (iced's tiny-skia renderer; `ICED_BACKEND=tiny-skia`
forces it).

## The window

A sidebar with five pages, and the light and dark pill in the header bar: click a side, or drag
its highlight across, which follows the pointer and slides into place, the mode changing as it's
let go; the window then crossfades into the other mode. Buttons' glows fade in and out with the
pointer.

- **Play**: the logo, the launcher's version and the game copy's version once a check has run,
  one status line, and one big main button whose meaning follows what exists:
  **Check** (`make_game.py --game <package> --check`), **Build** (`make_game.py --game <package>
  --out <output> [--work <work>]` and the options below), **Play** (the loader, below), and
  **Cancel** while something runs. Play is enabled only when the output folder holds the built
  loader and the package holds the game's executable; today's package has no executable, so
  "Not bootable yet" is the normal state after a build. While a command runs, the loader turns,
  the bar follows the build's stages (from make_game.py's `HH:MM:SS  <stage>: …` lines; it also
  reads `@@stage`, `@@progress`, `@@eta`, `@@check`, `@@done` and `@@error` lines if a later
  make_game.py prints them), and the output's last lines show below. A failure shows its exit
  code and last words.
- **Game files**: the game package, output and work folders (a field and a folder picker each),
  and the full check.
- **Build**: the stages and the WAV audio.
- **Performance**: the GPU filter, the Vulkan validation layer, wave64 translation and jobs.
- **Diagnostics**: the loader's D3D12 trace, what it does with unimplemented imports, and where
  the game's writes go.

| Control | Page | What it drives |
|---|---|---|
| Game package | Game files | `--game` (Check, Build); `--run`, `--game-dir` (Play) |
| Output folder | Game files | `--out` (Build); where the loader is looked for (Play) |
| Work folder | Game files | `--work` (Build); left out when empty |
| Compare every file's SHA-256 | Game files | `--full-check` (Check) |
| Shaders, Assets, Runtime, Pipelines | Build | `--stages` (Build); left out when all are on; none is `--stages check` |
| Also decode audio to WAV | Build | `--wav` (Build) |
| GPU | Performance | `LEONIDA_VK_DEVICE` (Build, for make_game.py's GPU probe, and Play) |
| Vulkan validation | Performance | `LEONIDA_VK_VALIDATION=1` (Play) |
| Choose wave64 translation automatically; Emulate or Native | Performance | `--wave64-emulate auto`, `on` or `off` (Build); `auto` is left out, as it's the default |
| Jobs | Performance | `--jobs N` (Build); left out when empty |
| Trace D3D12 calls | Diagnostics | `gta6rt --trace-d3d12` (Play) |
| Unimplemented imports: Log, Fail, Stop | Diagnostics | `LEONIDA_UNIMPLEMENTED=fail` or `stop` (Play); `log`, the default, is left out |
| Writes folder | Diagnostics | `gta6rt --overlay DIR` (Play; Wine's `Z:` path on Linux); left out when empty |

On/off settings are switches; a three-way one is a choice.

make_game.py runs as `python -I -u make_game.py …` (`python3` off Windows) in the repository,
with no input, so a question it would ask ends it instead of waiting. Play runs
`gta6rt.exe --run <package>\game_scarlett_master.exe --game-dir <package>` from the output folder
with `LEONIDA_D3D12=vulkan`; on Linux the output folder's `gta6rt.sh` (Wine) runs in the package
folder, `--run ./game_scarlett_master.exe --game-dir Z:\<package>`. The executable's name is
the version manifest's `expected_exe` (`game/*.json`).

Settings are saved to `launcher.json` in `%APPDATA%\Leonida` (Windows) or `~/.config/leonida`
(Linux), with a version; a missing, broken or newer file gives the defaults.

## Keys

| Keys | |
|---|---|
| Tab, Shift+Tab | Move focus through the header bar, the sidebar and the page |
| Enter, Space | Press the focused button; Space toggles a switch or checkbox |
| Up, Down | Choose a page in the focused sidebar |
| Left, Right | Choose in the focused light and dark pill (or drag across it) |
| Ctrl+Page Down, Ctrl+Page Up | The next and previous page |
| Ctrl+L | Light and dark |

The window draws its own decorations: drag the header bar to move it, double-click it to
maximize or restore, drag inside its edges to resize. Narrower than 640 pixels, the sidebar folds
and the header bar shows a back button.

`LEONIDA_WINDOW_FRAME`, read at start, says who draws the frame: unset or `1`, the launcher (see
below); `0`, nobody: an opaque, square window, for a desktop that draws its own or a driver that
misbehaves with a transparent window; anything else is as unset, with one line on stderr.

- **Linux (Wayland, X11)**: a transparent window, its corners rounded (14 px) and a 1 px line
  drawn by the launcher while it floats, square when maximized.
- **Windows**: stock iced can't present a transparent window there (iced_wgpu makes its DX12
  swapchain from the window's HWND, which presents opaque only, and tiny-skia's softbuffer has no
  alpha), so the window is opaque and asks Windows 11 for its own rounded corners
  (`CornerPreference::Round`), and the launcher draws its 1 px line at their 8 px radius. **Not
  tested on a real Windows machine**; it builds with `cargo check --target x86_64-pc-windows-gnu`.
  Windows 10 shows square corners.

## Tests

```sh
cargo test                                   # everything; no game files, no GPU, no window
LEONIDA_LAUNCHER_SCREENSHOTS=/tmp/shots cargo test --test screenshots
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Unit tests cover the command lines on Windows and Linux, the state machine, the settings file,
finding the repository, reading the output, the window's frame, the Play page's scene, and every
palette's contrast. The process tests run a stand-in shell script. `tests/keyboard.rs` drives the
window by keys alone, `tests/backdrop.rs` counts the scene's generations, and
`tests/screenshots.rs` draws every page offscreen on tiny-skia (on the CPU), writing PNGs to
`$LEONIDA_LAUNCHER_SCREENSHOTS` or `target/tmp/screenshots`.
`cargo test --release --lib scene::tests::timing -- --ignored --nocapture` times the scene.

## The Play page's scene

Behind the Play page: a dusk (dark) or a morning (light) over water, with a shore and two palms
in silhouette, generated by the launcher (`src/scene.rs`) from the theme alone: its gradient's six
stops, its background, its mode, and its outline and shadow colors. There are no image files.
The scene is one RGBA picture of the page's size in physical pixels, its soft layers computed per
pixel and the silhouettes filled as anti-aliased paths (tiny-skia), in a 16:10 frame that covers
the page from its bottom right (horizon at 0.66 of the frame's height, sun at 0.805 of its width).

- Over the text column on the left (its padding, its 420-pixel content and 24 pixels), the
  page's background lies at 0.97 opacity, then fades out across at most 40% of the page; the
  tests check that the `muted` ink keeps 4.5:1 on every pixel there, in both modes, at three
  sizes. Text sits on the page only in that column; the progress bar and the output lie on an
  opaque panel. However narrow the window, folded too, the scene stays: the column's background
  covers what lies under the text, and the fade takes at least 48 pixels beside it.
- It's generated off the interface's thread, once per page size, scale, theme, mode and column,
  after the size has held for 150 ms, and cached: drawing a frame only shows the cached picture,
  and while a resize is under way the last picture shows, scaled to cover. Both modes' pictures
  stay cached for the current size.
- About 30–34 ms for a 1040×754 page and 123–132 ms for 2320×1394 in a release build on the
  machine it was written on.

## The design system

`src/ui/` is the launcher's UI kit: the design language's tokens (colors, type, spacing, radii,
motion, glow), the theme, text roles, fonts and focus model, and the widgets the window shows,
each with its states, focus and keys: button, text field and its editing rules, checkbox, choice,
switch, toggle group (the header bar's pill), loader, progress bar, status chip, scroll area,
sidebar list, settings groups, form, the window's own decorations and the folded window.

- **Palettes**: Sunset dark and light, from the logo's pixels. Its light orange and dark purple
  are the two colors: in light, orange is the main one (the main button's fill, selection, the
  brand) and purple the secondary (focus, switches, every glow); in dark, the other way round,
  except that the main button stays orange, with an orange glow, to stand out on the purple
  page. It is filled with the logo's orange and its shadow as ink; as text or a line, each
  color keeps its hue but is darkened (light) or lightened (dark) only as far as 4.5:1 needs.
  Information is the logo's blue, errors its coral, warnings a gold. The logo's near-black
  purple shadow is dark's background and light's text, its deep purple outline dark's lines. A
  running progress bar goes from the purple to the orange. The tests check every ink on every
  surface in both.
- **Stock iced 0.14** from crates.io, with no patched crates. iced reads the system's fonts at
  start; text names only the launcher's own, so that changes nothing drawn.
- **Rounded corners**: `shape.rs` draws the window's corners and the close button's cross with
  one custom primitive through iced_wgpu's public primitive API. On the tiny-skia renderer, which
  can't draw one, the frame's and header bar's fills are rounded quads, the line a rounded quad's
  border, and the cross canvas strokes. Windows: the frame described above.
- **Fonts**: the four faces the text roles use (Chakra Petch SemiBold and Bold, IBM Plex Sans
  Regular, JetBrains Mono Regular), compiled in from `fonts/`.
- `LEONIDA_REDUCE_MOTION=1` stills what moves (the loader, a busy bar, the theme's crossfade);
  `LEONIDA_BATTERY_SAVER=1` halves animations and softens glow.
- One test is ignored: the text field's right-to-left caret positions, as with stock iced a
  system Hebrew font shapes the line and an unbounded right-to-left line lays out at infinity.

## Licence

The launcher is under the repository's licence, GPLv3 or later; the fonts in `fonts/` are under
the SIL Open Font License 1.1, with their licence files beside them.

## Not done

- Run in a real window on Linux only (Wayland, an RX 7900 XTX), not yet on Windows or macOS, and
  not through a complete build or a real launch: the tests use a stand-in script and offscreen
  drawing.
- The translator's fallback flags (`--no-float-controls`, `--bary-fallback`,
  `--texel-offset-fallback`, `--f16-interp-fallback`, `--mesh-compute`, `--pc-tail-ubo`) and
  `--runtime-build` have no controls yet; they are real options and could go on a Compatibility
  page.
- No translations; no high-contrast palette.
