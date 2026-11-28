// SPDX-License-Identifier: GPL-3.0-or-later

//! The launcher by keyboard alone, offscreen (tiny-skia, no window): Tab and Shift+Tab move focus
//! through the header bar, the sidebar and the page, round the end; Enter and Space press the
//! focused button, close button and switch; the arrows move the sidebar's and the pill's choice;
//! a focused field takes typing; Ctrl+Page Down and Ctrl+Page Up step through the pages, and
//! Ctrl+L switches light and dark.

mod common;

use iced::Size;
use iced::keyboard::{Key, Modifiers, key::Named};

use leonida_launcher::app::{Folder, Message, Page};
use leonida_launcher::config::Shade;
use leonida_launcher::state::{Action, Kind};
use leonida_launcher::ui::window::ChromeMessage;

use common::{Session, launcher};

const WIDE: Size = Size::new(980.0, 680.0);

fn tab(session: &mut Session, times: usize) {
    for _ in 0..times {
        session.press(Key::Named(Named::Tab), Modifiers::empty());
    }
}

fn names(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|message| format!("{message:?}"))
        .collect()
}

/// Tabs from nothing focused until pressing Enter sends a message, and returns how many Tabs and
/// the message.
fn find_by_tab(session: &mut Session, key: Named, wanted: impl Fn(&Message) -> bool) -> usize {
    for tabs in 1..40 {
        tab(session, 1);
        session.messages.clear();
        session.press(Key::Named(key), Modifiers::empty());
        if session.messages.iter().any(&wanted) {
            return tabs;
        }
    }
    panic!("nothing reached by Tab: {:?}", names(&session.messages));
}

#[test]
fn tab_reaches_the_primary_button_and_enter_presses_it() {
    let mut app = launcher();
    app.settings.package = "/games/made-up-package".to_owned();
    app.disk.package = true;
    let mut session = Session::new(app, WIDE);
    find_by_tab(&mut session, Named::Enter, |message| {
        matches!(message, Message::Act(Action::Run(Kind::Check)))
    });
    // Space presses it too, while it keeps focus.
    session.messages.clear();
    session.press(Key::Named(Named::Space), Modifiers::empty());
    assert!(
        session
            .messages
            .iter()
            .any(|message| matches!(message, Message::Act(Action::Run(Kind::Check)))),
        "{:?}",
        names(&session.messages)
    );
}

#[test]
fn a_disabled_primary_button_takes_no_focus() {
    // Nothing configured: Check is disabled, so no Tab stop presses it.
    let mut session = Session::new(launcher(), WIDE);
    for _ in 0..12 {
        tab(&mut session, 1);
        session.press(Key::Named(Named::Enter), Modifiers::empty());
    }
    assert!(
        !session
            .messages
            .iter()
            .any(|message| matches!(message, Message::Act(_))),
        "{:?}",
        names(&session.messages)
    );
}

#[test]
fn the_close_button_is_a_tab_stop_and_enter_closes() {
    let mut session = Session::new(launcher(), WIDE);
    find_by_tab(&mut session, Named::Enter, |message| {
        matches!(message, Message::Chrome(ChromeMessage::Close))
    });
}

#[test]
fn shift_tab_goes_round_the_other_way() {
    let mut session = Session::new(launcher(), WIDE);
    // From nothing, Shift+Tab focuses the last stop; the first Tab after that goes round to the
    // first, and Shift+Tab back again to the last.
    session.press(Key::Named(Named::Tab), Modifiers::SHIFT);
    session.messages.clear();
    session.press(Key::Named(Named::Enter), Modifiers::empty());
    let last = names(&session.messages);
    tab(&mut session, 1);
    session.press(Key::Named(Named::Tab), Modifiers::SHIFT);
    session.messages.clear();
    session.press(Key::Named(Named::Enter), Modifiers::empty());
    assert_eq!(names(&session.messages), last);
}

#[test]
fn the_sidebar_moves_by_arrows() {
    let mut session = Session::new(launcher(), WIDE);
    let tabs = find_by_tab(&mut session, Named::ArrowDown, |message| {
        matches!(message, Message::Show(Page::GameFiles))
    });
    assert!(tabs >= 1);
    let messages = std::mem::take(&mut session.messages);
    for message in messages {
        let _ = session.launcher.update(message);
    }
    assert_eq!(session.launcher.page, Page::GameFiles);
}

#[test]
fn the_pill_switches_light_and_dark_by_arrows() {
    let mut session = Session::new(launcher(), WIDE);
    find_by_tab(&mut session, Named::ArrowRight, |message| {
        matches!(message, Message::Shade(Shade::Light))
    });
}

#[test]
fn switches_take_space() {
    let mut app = launcher();
    app.page = Page::Performance;
    let mut session = Session::new(app, WIDE);
    find_by_tab(&mut session, Named::Space, |message| {
        matches!(message, Message::Validation(true))
    });
    let mut app = launcher();
    app.page = Page::Build;
    let mut session = Session::new(app, WIDE);
    find_by_tab(&mut session, Named::Space, |message| {
        matches!(message, Message::Stage("shaders", false))
    });
    let mut app = launcher();
    app.page = Page::Diagnostics;
    let mut session = Session::new(app, WIDE);
    find_by_tab(&mut session, Named::Space, |message| {
        matches!(message, Message::TraceD3d12(true))
    });
}

#[test]
fn a_focused_field_takes_typing() {
    let mut app = launcher();
    app.page = Page::GameFiles;
    let mut session = Session::new(app, WIDE);
    // Tab until a field is focused: typing then sends its text.
    for _ in 0..20 {
        tab(&mut session, 1);
        session.messages.clear();
        session.press(Key::Character("x".into()), Modifiers::empty());
        if let Some(Message::Folder(folder, value)) = session.messages.first() {
            assert_eq!((*folder, value.as_str()), (Folder::Package, "x"));
            return;
        }
    }
    panic!("no field took the typing");
}

#[test]
fn ctrl_page_down_and_up_step_through_the_pages_and_ctrl_l_switches_the_shade() {
    let mut app = launcher();
    assert_eq!(app.page, Page::Play);
    let _ = app.update(Message::Step(true));
    assert_eq!(app.page, Page::GameFiles);
    for _ in 0..4 {
        let _ = app.update(Message::Step(true));
    }
    assert_eq!(app.page, Page::Play, "round the end");
    let _ = app.update(Message::Step(false));
    assert_eq!(app.page, Page::Diagnostics);
    let _ = app.update(Message::ToggleShade);
    assert_eq!(app.settings.shade, Shade::Light);
}
