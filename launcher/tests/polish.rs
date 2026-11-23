// SPDX-License-Identifier: GPL-3.0-or-later

//! Every page and the Play page's states, in both modes, at the smallest window, the default and
//! 1920x1080, drawn offscreen for a look over the whole launcher: written as PNGs to
//! `$LEONIDA_LAUNCHER_SCREENSHOTS/polish/<size>/<name>-<mode>.png`. Each is checked not to be
//! blank, and its middle opaque. Everything shown is made up.

mod common;

use std::path::PathBuf;

use iced::Size;

use leonida_launcher::app::{LEAST, Launcher, Page, SIZE};
use leonida_launcher::config::{Checked, Shade};
use leonida_launcher::state::{Kind, Outcome};

use common::{Session, launcher, write_png};

fn set_up(launcher: &mut Launcher, state: &str) {
    let configured = |launcher: &mut Launcher| {
        launcher.settings.package = "/games/made-up-package".to_owned();
        launcher.settings.output = "/home/player/Leonida".to_owned();
        launcher.disk.package = true;
        launcher.settings.checked = Some(Checked {
            package: launcher.settings.package.clone(),
            game: "Xbox Series package 1.0.2.1".to_owned(),
        });
    };
    match state {
        "play-idle" => {}
        "play-not-bootable" => {
            configured(launcher);
            launcher.disk.loader = true;
        }
        "play-failed" => {
            configured(launcher);
            let mut tracker = leonida_launcher::output::Tracker::build(None);
            tracker.feed("ERROR: made-up failure with a long message that has to wrap inside the text column, as a real one might");
            launcher.last = Some((
                Outcome {
                    kind: Kind::Build,
                    code: Some(1),
                    cancelled: false,
                    words: tracker.last_words(),
                },
                leonida_launcher::commands::Command {
                    program: "python3".to_owned(),
                    args: vec!["-I".to_owned(), "-u".to_owned(), "make_game.py".to_owned()],
                    env: Vec::new(),
                    cwd: String::new(),
                },
                tracker,
            ));
        }
        "game-files" => {
            configured(launcher);
            launcher.settings.work =
                "/a/very/long/path/to/a/work/folder/on/another/drive/leonida-work".to_owned();
            launcher.page = Page::GameFiles;
        }
        "build" => launcher.page = Page::Build,
        "performance" => launcher.page = Page::Performance,
        "diagnostics" => launcher.page = Page::Diagnostics,
        _ => unreachable!(),
    }
}

#[test]
fn every_page_at_three_sizes() {
    let folder = std::env::var_os("LEONIDA_LAUNCHER_SCREENSHOTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screenshots"))
        .join("polish");
    let sizes = [
        ("smallest", LEAST),
        ("default", SIZE),
        ("large", Size::new(1920.0, 1080.0)),
    ];
    let states = [
        "play-idle",
        "play-not-bootable",
        "play-failed",
        "game-files",
        "build",
        "performance",
        "diagnostics",
    ];
    for (label, size) in sizes {
        for state in states {
            for shade in [Shade::Dark, Shade::Light] {
                let mut app = launcher();
                app.settings.shade = shade;
                set_up(&mut app, state);
                let mut session = Session::new(app, size);
                session.settle_scene(1.0);
                let rgba = session.render(1.0);
                let (width, height) = (size.width as u32, size.height as u32);
                let middle = ((height / 2 * width + width / 2) * 4 + 3) as usize;
                assert_eq!(rgba[middle], 255, "{label} {state}");
                let mode = match shade {
                    Shade::Dark => "dark",
                    Shade::Light => "light",
                };
                write_png(
                    &folder.join(label).join(format!("{state}-{mode}.png")),
                    (width, height),
                    &rgba,
                );
            }
        }
    }
}
