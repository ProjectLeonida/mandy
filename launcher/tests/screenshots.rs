// SPDX-License-Identifier: GPL-3.0-or-later

//! Screenshots of the launcher's window, drawn offscreen on tiny-skia (no GPU, no display): every
//! page in Sunset dark and light, the Play page's states, and the folded sidebar. Each is written
//! as a PNG to `$LEONIDA_LAUNCHER_SCREENSHOTS`, or to `target/tmp/screenshots`, to look at; the
//! test checks what can be checked of a picture: its size, the window's clear corners, and that
//! it isn't blank. Everything shown is made up: no game files are read.

mod common;

use std::path::PathBuf;

use iced::Size;

use leonida_launcher::app::{Launcher, Page, Running};
use leonida_launcher::commands::{self, Platform};
use leonida_launcher::config::{Checked, Shade};
use leonida_launcher::output::Tracker;
use leonida_launcher::state::{Kind, Outcome};

use common::{Session, launcher, write_png};

const WIDE: Size = Size::new(980.0, 680.0);
const NARROW: Size = Size::new(520.0, 680.0);
/// The reference sketch's window, its page 1040 by 754 as the scene's budget is measured.
const LARGE: Size = Size::new(1282.0, 802.0);

fn folder() -> PathBuf {
    std::env::var_os("LEONIDA_LAUNCHER_SCREENSHOTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screenshots"))
}

/// Made-up folders.
fn configured(launcher: &mut Launcher) {
    launcher.settings.package = "/games/made-up-package".to_owned();
    launcher.settings.output = "/home/player/Leonida".to_owned();
    launcher.settings.work = "/scratch/leonida-work".to_owned();
    launcher.disk.package = true;
}

fn checked(launcher: &mut Launcher) {
    configured(launcher);
    launcher.settings.checked = Some(Checked {
        package: launcher.settings.package.clone(),
        game: "Xbox Series package 1.0.2.1".to_owned(),
    });
}

/// A build part of the way, as made-up make_game.py lines say.
fn building(launcher: &mut Launcher) {
    checked(launcher);
    let command = commands::build(Platform::current(), "/src/Leonida", &launcher.settings).unwrap();
    let mut tracker = Tracker::build(None);
    for line in [
        "12:00:00  check: Xbox Series package 1.0.2.1, 1234 files",
        "12:00:01  check: all files present. Executable found.",
        "12:00:02  building into /home/player/Leonida (work files in /scratch/leonida-work)",
        "12:00:03  shaders: carving shaders.rpf",
        "12:00:09  shaders: unpacking native code",
        "12:01:30  shaders: disassembling (gfx1013)",
        "12:04:12  shaders: translating RDNA2 to SPIR-V on 8 processes (the longest step)",
        "  [made-up progress line from a tool]",
    ] {
        tracker.feed(line);
    }
    launcher.running = Some(Running {
        kind: Kind::Build,
        command,
        pid: None,
        tracker,
        cancelled: false,
    });
}

fn not_bootable(launcher: &mut Launcher) {
    checked(launcher);
    launcher.disk.loader = true;
    launcher.disk.exe = false;
}

fn failed(launcher: &mut Launcher) {
    checked(launcher);
    let command = commands::build(Platform::current(), "/src/Leonida", &launcher.settings).unwrap();
    let mut tracker = Tracker::build(None);
    for line in [
        "12:00:00  check: Xbox Series package 1.0.2.1, 1234 files",
        "12:00:02  building into /home/player/Leonida (work files in /scratch/leonida-work)",
        "12:00:03  shaders: carving shaders.rpf",
        "",
        "ERROR: llvm-mc wasn't found. Set LLVM_MC (made-up failure for the screenshot)",
    ] {
        tracker.feed(line);
    }
    launcher.last = Some((
        Outcome {
            kind: Kind::Build,
            code: Some(1),
            cancelled: false,
            words: tracker.last_words(),
        },
        command,
        tracker,
    ));
}

/// Draws one screenshot, checks it, and writes it.
fn shot(name: &str, shade: Shade, size: Size, set_up: impl Fn(&mut Launcher)) {
    let mut launcher = launcher();
    launcher.settings.shade = shade;
    set_up(&mut launcher);
    let mut session = Session::new(launcher, size);
    session.settle_scene(1.0);
    let rgba = session.render(1.0);
    let (width, height) = (size.width as u32, size.height as u32);
    assert_eq!(rgba.len(), (width * height * 4) as usize, "{name}");
    let alpha = |x: u32, y: u32| rgba[((y * width + x) * 4 + 3) as usize];
    // The frame's corners are clear on a transparent window (Linux); the middle is opaque.
    if session
        .launcher
        .style(&session.launcher.theme())
        .background_color
        .a
        == 0.0
    {
        assert_eq!(alpha(0, 0), 0, "{name}: top left corner");
        assert_eq!(
            alpha(width - 1, height - 1),
            0,
            "{name}: bottom right corner"
        );
    }
    assert_eq!(alpha(width / 2, height / 2), 255, "{name}: the middle");
    // Not one color: something's drawn.
    let first = &rgba[(width * height * 2) as usize..(width * height * 2 + 4) as usize];
    assert!(rgba.chunks(4).any(|pixel| pixel != first), "{name}: blank");
    let mode = match shade {
        Shade::Dark => "dark",
        Shade::Light => "light",
    };
    write_png(
        &folder().join(format!("{name}-{mode}.png")),
        (width, height),
        &rgba,
    );
}

fn both(name: &str, size: Size, set_up: impl Fn(&mut Launcher)) {
    for shade in [Shade::Dark, Shade::Light] {
        shot(name, shade, size, &set_up);
    }
}

#[test]
fn play_idle_with_nothing_configured() {
    both("play-idle", WIDE, |_| {});
}

#[test]
fn play_build_running() {
    both("play-building", WIDE, building);
}

#[test]
fn play_not_bootable_yet() {
    both("play-not-bootable", WIDE, not_bootable);
}

#[test]
fn play_failed_command() {
    both("play-failed", WIDE, failed);
}

#[test]
fn settings_pages() {
    both("game-files", WIDE, |launcher| {
        configured(launcher);
        launcher.page = Page::GameFiles;
    });
    both("build", WIDE, |launcher| {
        launcher.page = Page::Build;
        launcher.settings.stages.retain(|stage| stage != "assets");
    });
    both("diagnostics", WIDE, |launcher| {
        launcher.page = Page::Diagnostics;
        launcher.settings.trace_d3d12 = true;
    });
}

#[test]
fn performance_page() {
    both("performance", WIDE, |launcher| {
        launcher.page = Page::Performance;
        launcher.settings.gpu = "radeon".to_owned();
        launcher.settings.jobs = "8".to_owned();
    });
}

#[test]
fn play_scene_at_the_sketchs_size() {
    both("play-scene-large", LARGE, not_bootable);
    both("play-scene-large-building", LARGE, building);
}

#[test]
fn folded_sidebar() {
    both("folded-sidebar", NARROW, |_| {});
    both("folded-page", NARROW, |launcher| {
        not_bootable(launcher);
        launcher.chrome.show_page();
    });
}
