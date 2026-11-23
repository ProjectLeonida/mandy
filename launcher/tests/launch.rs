// SPDX-License-Identifier: GPL-3.0-or-later

//! A launch, as `main` boots the launcher: only what a real window delivers (its size once it
//! opens, its scale factor), the tasks the launcher returns run as the runtime runs them, on
//! threads, with the real generator. The Play page's scene must be made, both modes once, and
//! shown, then fade in and stop: at the default size and scale, on another page, folded, and with
//! the frame off.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use iced::futures::StreamExt;
use iced::futures::executor::block_on;
use iced::{Size, Task, window};
use iced_runtime::Action;

use leonida_launcher::app::{Launcher, Message, Page, SIZE};
use leonida_launcher::ui::tokens::motion;
use leonida_launcher::ui::window::{ChromeMessage, FrameSetting};

/// Runs a task's stream on a thread, sending what it outputs. A task that waits on the window
/// (none answers here) just never sends.
fn run(task: Task<Message>, sender: &mpsc::Sender<Message>) {
    if let Some(stream) = iced_runtime::task::into_stream(task) {
        let sender = sender.clone();
        std::thread::spawn(move || {
            block_on(stream.for_each(|action| {
                if let Action::Output(message) = action {
                    let _ = sender.send(message);
                }
                std::future::ready(())
            }));
        });
    }
}

fn launch(frame: FrameSetting, size: Size, page: Page) {
    let repo = PathBuf::from("/src/Leonida");
    let (mut launcher, boot) = Launcher::boot(Ok(repo), None, frame);
    launcher.page = page;
    let (sender, receiver) = mpsc::channel();
    run(boot, &sender);
    // What the window says as it opens: its size, and its scale.
    let id = window::Id::unique();
    run(
        launcher.update(Message::Chrome(ChromeMessage::Resized(id, size))),
        &sender,
    );
    run(launcher.update(Message::Scale(1.0)), &sender);
    let deadline = Instant::now() + Duration::from_secs(120);
    while launcher.backdrop.layers().1.is_none() || launcher.backdrop.generations < 2 {
        let left = deadline.saturating_duration_since(Instant::now());
        let message = receiver
            .recv_timeout(left)
            .unwrap_or_else(|_| panic!("{frame:?} {size:?} {page:?}: no picture"));
        run(launcher.update(message), &sender);
    }
    // Both modes, once each; the current one showing, fading in.
    assert_eq!(
        launcher.backdrop.generations, 2,
        "{frame:?} {size:?} {page:?}"
    );
    let (_, shown) = launcher.backdrop.layers();
    let (shown, _) = shown.unwrap();
    assert!(shown.handle.is_some(), "a picture, not an empty one");
    assert!(launcher.backdrop.is_fading(), "it fades in once it's ready");
    // The fade's frames, until it's whole: then it stops.
    let start = Instant::now();
    let _ = launcher.update(Message::SceneFrame(start));
    let _ = launcher.update(Message::SceneFrame(start + motion::SCENE_IN));
    assert!(!launcher.backdrop.is_fading());
    assert_eq!(launcher.backdrop.layers().1.unwrap().1, 1.0);
    // Nothing more is wanted, and nothing more is made.
    let _ = launcher.update(Message::Show(Page::Play));
    assert_eq!(launcher.backdrop.generations, 2);
}

#[test]
fn a_launch_at_the_default_size_and_scale_shows_the_scene() {
    launch(FrameSetting::Drawn, SIZE, Page::Play);
}

#[test]
fn a_launch_on_another_page_has_the_scene_ready_for_play() {
    launch(FrameSetting::Drawn, SIZE, Page::Build);
}

#[test]
fn a_launch_folded_and_without_a_frame_shows_the_scene() {
    launch(FrameSetting::Drawn, Size::new(520.0, 680.0), Page::Play);
    launch(FrameSetting::Off, SIZE, Page::Play);
}
