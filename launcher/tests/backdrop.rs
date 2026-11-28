// SPDX-License-Identifier: GPL-3.0-or-later

//! The Play page's scene is generated once per final key and cached: redraws, the pointer, focus,
//! page switches and a burst of resizes generate nothing more, and switching the mode back and
//! forth costs one picture per mode. Counted with a stand-in generator.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use iced::keyboard::{Key, Modifiers, key::Named};
use iced::{Event, Point, Size, mouse, window};

use leonida_launcher::app::{Message, Page, SIZE};
use leonida_launcher::backdrop::DEBOUNCE;
use leonida_launcher::scene::{Input, Picture};
use leonida_launcher::ui::window::ChromeMessage;

use common::{Session, launcher};

static GENERATED: AtomicUsize = AtomicUsize::new(0);

/// A stand-in generator: counts, and makes a blank picture of the size asked.
fn counting(input: &Input) -> Option<Picture> {
    GENERATED.fetch_add(1, Ordering::SeqCst);
    Some(Picture {
        width: input.width,
        height: input.height,
        rgba: vec![0; input.width as usize * input.height as usize * 4],
    })
}

fn settle(session: &mut Session) {
    let later = Instant::now() + DEBOUNCE * 2;
    let _ = session.launcher.update(Message::SceneTick(later));
}

#[test]
fn one_generation_per_distinct_final_key() {
    let mut app = launcher();
    app.synchronous = true;
    app.backdrop.generator = counting;
    let mut session = Session::new(app, SIZE);
    let generated = || GENERATED.load(Ordering::SeqCst);

    // At start: both modes' pictures, at once, with no wait.
    let _ = session.launcher.update(Message::SceneTick(Instant::now()));
    assert_eq!(generated(), 2, "both modes, at once");

    // Many frames, the pointer moving, focus moving: nothing.
    for at in 0..30 {
        session.send(&[Event::Mouse(mouse::Event::CursorMoved {
            position: Point::new(300.0 + at as f32 * 10.0, 200.0),
        })]);
        let _ = session.render(1.0);
        session.press(Key::Named(Named::Tab), Modifiers::empty());
        settle(&mut session);
    }
    assert_eq!(generated(), 2);

    // Pages switched and back: nothing.
    for page in [
        Page::GameFiles,
        Page::Performance,
        Page::Play,
        Page::GameFiles,
        Page::Play,
    ] {
        let _ = session.launcher.update(Message::Show(page));
        let _ = session.render(1.0);
        settle(&mut session);
    }
    assert_eq!(generated(), 2);

    // A resize drag: many sizes, none held long enough, then the last one holds: both modes'
    // pictures for it.
    let id = window::Id::unique();
    for step in 0..25 {
        let size = Size::new(
            SIZE.width + step as f32 * 8.0,
            SIZE.height + step as f32 * 3.0,
        );
        let _ = session
            .launcher
            .update(Message::Chrome(ChromeMessage::Resized(id, size)));
        let _ = session.launcher.update(Message::SceneTick(Instant::now()));
        let _ = session.render(1.0);
    }
    assert_eq!(generated(), 2, "nothing while the size keeps changing");
    settle(&mut session);
    assert_eq!(generated(), 4, "both modes for the final size");

    // Light, dark, light: both are made already, nothing more.
    for _ in 0..3 {
        let _ = session.launcher.update(Message::ToggleShade);
        settle(&mut session);
        let _ = session.render(1.0);
        assert_eq!(generated(), 4);
    }
    assert_eq!(session.launcher.backdrop.generations, 4);
}

/// The RGBA pixel at `x`, `y` of a frame `width` wide.
fn pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 3] {
    let at = ((y * width + x) * 4) as usize;
    [rgba[at], rgba[at + 1], rgba[at + 2]]
}

#[test]
fn while_resizing_the_last_scene_stays_until_the_new_one_is_made() {
    let mut session = Session::new(launcher(), SIZE);
    session.settle_scene(1.0);
    let bg = session
        .launcher
        .theme()
        .palette()
        .surface(leonida_launcher::ui::Surface::Bg)
        .rgb8();
    let seen = |session: &mut Session, size: Size| {
        let rgba = session.render(1.0);
        // Low on the right, where the shore and the water are: never the page's background.
        let (x, y) = (size.width as u32 - 60, size.height as u32 - 40);
        pixel(&rgba, size.width as u32, x, y)
    };
    assert_ne!(seen(&mut session, SIZE), bg, "the scene shows");
    // A drag: sizes come one after another, none held long enough to make a new picture.
    for step in 1..6 {
        let size = Size::new(
            SIZE.width + step as f32 * 30.0,
            SIZE.height + step as f32 * 10.0,
        );
        session.resize_to(size);
        assert_ne!(
            seen(&mut session, size),
            bg,
            "the last scene, while resizing to {size:?}"
        );
    }
}
