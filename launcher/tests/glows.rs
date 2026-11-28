// SPDX-License-Identifier: GPL-3.0-or-later

//! Every glow comes and goes on the buttons' timing: a button and a text field under the
//! pointer, and a switch turned on, are part lit halfway through `motion::GLOW_IN` and whole at
//! its end. Drawn offscreen, the pictures compared.

mod common;

use std::time::Instant;

use iced::advanced::renderer::{Headless, Style};
use iced::advanced::{clipboard, mouse};
use iced::{Event, Point, Size, window};
use iced_runtime::user_interface::{Cache, UserInterface};

use leonida_launcher::app::Message;
use leonida_launcher::ui::tokens::{Palette, motion};
use leonida_launcher::ui::widget::container;
use leonida_launcher::ui::{Element, Theme, button, switch, text_field};

const SIZE: Size = Size::new(240.0, 120.0);

/// One control, centred, with room for its glow.
fn view(which: &str, on: bool) -> Element<'static, Message> {
    let control: Element<'static, Message> = match which {
        "button" => button("Check").on_press(Message::Scale(1.0)).into(),
        "field" => text_field("A folder", "")
            .on_input(|_| Message::Scale(1.0))
            .into(),
        "switch" => switch(on).on_toggle(|_| Message::Scale(1.0)).into(),
        _ => unreachable!(),
    };
    container(control)
        .width(iced::Fill)
        .height(iced::Fill)
        .center_x(iced::Fill)
        .center_y(iced::Fill)
        .padding(32)
        .into()
}

struct Harness {
    which: &'static str,
    on: bool,
    renderer: iced::Renderer,
    cache: Cache,
    cursor: mouse::Cursor,
    theme: Theme,
}

impl Harness {
    fn send(&mut self, events: &[Event]) {
        let cache = std::mem::take(&mut self.cache);
        let mut ui =
            UserInterface::build(view(self.which, self.on), SIZE, cache, &mut self.renderer);
        let mut messages = Vec::new();
        let _ = ui.update(
            events,
            self.cursor,
            &mut self.renderer,
            &mut clipboard::Null,
            &mut messages,
        );
        self.cache = ui.into_cache();
    }

    /// A frame at `now`, then its picture.
    fn render_at(&mut self, now: Instant) -> Vec<u8> {
        self.send(&[Event::Window(window::Event::RedrawRequested(now))]);
        let cache = std::mem::take(&mut self.cache);
        let mut ui =
            UserInterface::build(view(self.which, self.on), SIZE, cache, &mut self.renderer);
        let style = Style {
            text_color: iced::theme::Base::base(&self.theme).text_color,
        };
        ui.draw(&mut self.renderer, &self.theme, &style, self.cursor);
        self.cache = ui.into_cache();
        let background = iced::theme::Base::base(&self.theme).background_color;
        self.renderer.screenshot(
            Size::new(SIZE.width as u32, SIZE.height as u32),
            1.0,
            background,
        )
    }
}

fn difference(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| u64::from(a.abs_diff(*b)))
        .sum()
}

#[test]
fn every_glow_fades_in_over_the_buttons_time() {
    let centre = Point::new(SIZE.width / 2.0, SIZE.height / 2.0);
    for palette in Palette::ALL {
        for which in ["button", "field", "switch"] {
            let mut harness = Harness {
                which,
                on: false,
                renderer: common::renderer(),
                cache: Cache::default(),
                cursor: mouse::Cursor::Unavailable,
                theme: Theme::new(palette),
            };
            let start = Instant::now();
            let before = harness.render_at(start);
            // Lit: the pointer comes over the button and the field; the switch turns on, the
            // pointer kept away so that only its being on lights it.
            if which == "switch" {
                // Its new value is seen on the next event, as a window's next one would be.
                harness.on = true;
                harness.send(&[Event::Mouse(mouse::Event::CursorLeft)]);
            } else {
                harness.cursor = mouse::Cursor::Available(centre);
                harness.send(&[Event::Mouse(mouse::Event::CursorMoved { position: centre })]);
            }
            harness.send(&[Event::Window(window::Event::RedrawRequested(start))]);
            let half = harness.render_at(start + motion::GLOW_IN / 2);
            let whole = harness.render_at(start + motion::GLOW_IN);
            let later = harness.render_at(start + motion::GLOW_IN * 2);
            let name = format!("{} {which}", palette.name);
            assert!(difference(&before, &half) > 0, "{name}: halfway, some glow");
            assert!(
                difference(&half, &whole) > 0,
                "{name}: halfway, not yet whole"
            );
            assert_eq!(
                difference(&whole, &later),
                0,
                "{name}: whole at the end of GLOW_IN"
            );
        }
    }
}
