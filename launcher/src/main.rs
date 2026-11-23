// SPDX-License-Identifier: GPL-3.0-or-later

// A window, not a console, on Windows (in release builds; a debug build keeps its console for
// what it prints).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! `leonida-launcher [--repo PATH]`: the launcher's window. Without `--repo`, the repository is
//! the first folder holding `make_game.py` above the launcher's executable, or above the current
//! folder.

use std::path::PathBuf;

use leonida_launcher::app::{LEAST, Launcher, SIZE};
use leonida_launcher::config;
use leonida_launcher::repo;
use leonida_launcher::ui::font;
use leonida_launcher::ui::window::{FrameSetting, settings};

const USAGE: &str = "usage: leonida-launcher [--repo PATH]

  --repo PATH   the Leonida repository (the folder that holds make_game.py); found by walking up
                from the launcher, then from the current folder, when left out

Keys: Tab and Shift+Tab move focus; Enter and Space press; Ctrl+Page Down and Ctrl+Page Up go to
the next and previous page; Ctrl+L switches light and dark.
Environment: LEONIDA_WINDOW_FRAME=0 for an opaque, square window.";

fn main() -> iced::Result {
    let mut given: Option<PathBuf> = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--repo") => match args.next() {
                Some(path) => given = Some(PathBuf::from(path)),
                None => usage_error("--repo needs a folder"),
            },
            Some("-h" | "--help") => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => usage_error(&format!("unknown argument {}", arg.to_string_lossy())),
        }
    }
    let exe = std::env::current_exe().ok();
    let cwd = std::env::current_dir().ok();
    let repo = repo::find(given.as_deref(), exe.as_deref(), cwd.as_deref());
    if let Err(why) = &repo {
        eprintln!("leonida-launcher: {why}");
    }
    let frame = FrameSetting::from_env();
    font::load();
    let boot = move || Launcher::boot(repo.clone(), config::path(), frame);
    let mut application = iced::application(boot, Launcher::update, Launcher::view)
        .title("Leonida")
        .subscription(Launcher::subscription)
        .theme(Launcher::theme)
        .style(Launcher::style)
        .window(settings(SIZE, LEAST, frame))
        .default_font(font::DEFAULT);
    for bytes in font::SHIPPED {
        application = application.font(bytes);
    }
    application.run()
}

fn usage_error(why: &str) -> ! {
    eprintln!("leonida-launcher: {why}\n\n{USAGE}");
    std::process::exit(2)
}
