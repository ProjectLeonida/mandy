// SPDX-License-Identifier: GPL-3.0-or-later

//! The Play page's scene, generated once per key and cached: never in `view`, never per frame.
//!
//! A key is a page (its size in physical pixels, the scale, the text column, the window's corner)
//! and a theme with its mode. For the page in use, both modes' pictures are made, the current
//! mode's first, each off the interface's thread, so switching the mode never waits. At start,
//! with nothing shown yet, they're made at once; after a resize, only once the size has held for
//! [`DEBOUNCE`], while the last picture shows, scaled to cover.
//!
//! A picture fades in when it's ready ([`motion::SCENE_IN`]), crossfades with the one it
//! replaces, and then nothing moves. Whenever a wanted picture isn't made, isn't being made and
//! no tick is due, [`Backdrop::schedule`] says when to tick, so none is ever stranded.

use std::time::{Duration, Instant};

use iced::widget::image;

use crate::scene::{self, Input, Picture};
use crate::ui::tokens::motion;
use crate::ui::transition::Transition;

/// How long a resized page's size must hold before its pictures are made.
pub const DEBOUNCE: Duration = Duration::from_millis(150);

/// A page, as its pictures depend on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Page {
    pub width: u32,
    pub height: u32,
    scale: u32,
    column: u32,
    corner: Option<u32>,
}

/// What a picture depends on: its page and its theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    pub page: Page,
    /// The palette's name, which says its theme and mode.
    pub theme: &'static str,
}

impl Key {
    pub fn of(input: &Input, theme: &'static str) -> Self {
        Self {
            page: Page {
                width: input.width,
                height: input.height,
                scale: input.scale.to_bits(),
                column: input.column.to_bits(),
                corner: input.corner.map(f32::to_bits),
            },
            theme,
        }
    }
}

/// The generator: [`scene::generate`], or a stand-in in the tests.
pub type Generator = fn(&Input) -> Option<Picture>;

/// A generated picture, ready to show, or none where it couldn't be made.
#[derive(Debug, Clone)]
pub struct Shown {
    pub handle: Option<image::Handle>,
    /// Its size, in logical pixels.
    pub size: iced::Size,
}

/// The pictures made, the ones wanted, and the one showing.
pub struct Backdrop {
    made: Vec<(Key, Shown)>,
    /// The pictures wanted, the current mode's first, and since when their page has held.
    wanted: Vec<(Key, Input)>,
    since: Option<Instant>,
    in_flight: Option<Key>,
    tick_pending: bool,
    /// The picture showing, the one it replaces while they crossfade, and how far the fade is.
    shown: Option<Key>,
    previous: Option<Key>,
    fade: Option<Transition>,
    fade_at: f32,
    /// How many pictures have been generated: what the tests count.
    pub generations: usize,
    pub generator: Generator,
}

impl Default for Backdrop {
    fn default() -> Self {
        Self {
            made: Vec::new(),
            wanted: Vec::new(),
            since: None,
            in_flight: None,
            tick_pending: false,
            shown: None,
            previous: None,
            fade: None,
            fade_at: 1.0,
            generations: 0,
            generator: scene::generate,
        }
    }
}

impl Backdrop {
    /// The pictures the page wants now, the current mode's first, noted at `now`. A new page
    /// starts its wait; a current picture that's made and isn't showing crossfades in.
    pub fn want(&mut self, wanted: Vec<(Key, Input)>, now: Instant) {
        let page = |list: &[(Key, Input)]| list.first().map(|(key, _)| key.page);
        if page(&wanted) != page(&self.wanted) {
            // A resize, with a picture showing, waits for the size to hold; at start, nothing does.
            self.since = self.shown.map(|_| now);
        }
        self.wanted = wanted;
        if let Some(current) = self.current()
            && self.is_made(&current)
            && self.shown != Some(current)
        {
            let duration = if self.shown.is_some() {
                motion::SLOW
            } else {
                motion::SCENE_IN
            };
            self.show(current, duration);
        }
    }

    /// The key the page should show: the current mode's for the page in use.
    fn current(&self) -> Option<Key> {
        self.wanted.first().map(|(key, _)| *key)
    }

    fn is_made(&self, key: &Key) -> bool {
        self.made.iter().any(|(made, _)| made == key)
    }

    /// Whether a wanted picture isn't made yet.
    fn missing(&self) -> bool {
        self.wanted.iter().any(|(key, _)| !self.is_made(key))
    }

    /// If a wanted picture isn't made, isn't being made and no tick is due, how long from `now`
    /// a tick should come: at once while nothing shows (at start), else when the page's size has
    /// held for [`DEBOUNCE`]. The caller schedules it; until it comes, no other is asked for.
    pub fn schedule(&mut self, now: Instant) -> Option<Duration> {
        if !self.missing() || self.in_flight.is_some() || self.tick_pending {
            return None;
        }
        self.tick_pending = true;
        Some(self.wait_left(now))
    }

    /// What's left of the page's wait: none while nothing shows.
    fn wait_left(&self, now: Instant) -> Duration {
        match (self.shown, self.since) {
            (Some(_), Some(since)) => DEBOUNCE.saturating_sub(now.saturating_duration_since(since)),
            _ => Duration::ZERO,
        }
    }

    /// A tick at `now`: the picture to generate, if one is wanted, its wait is over and nothing
    /// is being made. Counted as a generation.
    pub fn due(&mut self, now: Instant) -> Option<(Key, Input)> {
        self.tick_pending = false;
        if self.in_flight.is_some() || !self.wait_left(now).is_zero() {
            return None;
        }
        let (key, input) = *self.wanted.iter().find(|(key, _)| !self.is_made(key))?;
        self.in_flight = Some(key);
        self.generations += 1;
        Some((key, input))
    }

    /// A picture was generated for `key`. Pictures of another page go, but for what's showing;
    /// the current one fades in, or crossfades with what showed.
    pub fn finished(&mut self, key: Key, picture: Option<Picture>) {
        if self.in_flight == Some(key) {
            self.in_flight = None;
        }
        let scale = f32::from_bits(key.page.scale);
        let shown = match picture {
            Some(picture) => Shown {
                size: iced::Size::new(picture.width as f32 / scale, picture.height as f32 / scale),
                handle: Some(image::Handle::from_rgba(
                    picture.width,
                    picture.height,
                    picture.rgba,
                )),
            },
            None => Shown {
                handle: None,
                size: iced::Size::ZERO,
            },
        };
        let page = self.current().map(|current| current.page);
        let keep = [self.shown, self.previous];
        self.made.retain(|(made, _)| {
            *made != key && (Some(made.page) == page || keep.contains(&Some(*made)))
        });
        self.made.push((key, shown));
        if Some(key) == self.current() && self.shown != Some(key) {
            let duration = if self.shown.is_some() {
                motion::NORMAL
            } else {
                motion::SCENE_IN
            };
            self.show(key, duration);
        }
    }

    /// `key` comes in over `duration`, from nothing or over what showed.
    fn show(&mut self, key: Key, duration: Duration) {
        self.previous = self.shown.take();
        self.shown = Some(key);
        self.fade = Transition::start(0.0, 1.0, duration);
        self.fade_at = if self.fade.is_some() { 0.0 } else { 1.0 };
        if self.fade.is_none() {
            self.previous = None;
        }
    }

    /// A frame at `now`, while a picture fades in: once it's whole, what it replaced goes.
    pub fn frame(&mut self, now: Instant) {
        if let Some(fade) = &mut self.fade {
            let going = fade.frame(now);
            self.fade_at = fade.at();
            if !going {
                self.fade = None;
                self.fade_at = 1.0;
                self.previous = None;
            }
        }
    }

    /// Whether a picture is fading in: it wants every frame until it's whole, and none after.
    pub fn is_fading(&self) -> bool {
        self.fade.is_some()
    }

    fn get(&self, key: Option<Key>) -> Option<&Shown> {
        let key = key?;
        self.made
            .iter()
            .find(|(made, _)| *made == key)
            .map(|(_, shown)| shown)
    }

    /// What to draw: the picture it replaces, whole, and the picture showing, at how far it has
    /// faded in.
    pub fn layers(&self) -> (Option<&Shown>, Option<(&Shown, f32)>) {
        (
            self.get(self.previous),
            self.get(self.shown).map(|shown| (shown, self.fade_at)),
        )
    }
}

/// The size that covers `page` with a picture of `picture`'s, keeping its shape.
pub fn cover(picture: iced::Size, page: iced::Size) -> iced::Size {
    if picture.width <= 0.0 || picture.height <= 0.0 {
        return page;
    }
    let k = (page.width / picture.width).max(page.height / picture.height);
    iced::Size::new(picture.width * k, picture.height * k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Colors;
    use crate::ui::tokens::Palette;

    fn wanted(width: u32, dark_first: bool) -> Vec<(Key, Input)> {
        let input = |palette: &Palette| Input {
            colors: Colors::of(palette),
            width,
            height: 600,
            scale: 1.0,
            column: 468.0,
            corner: None,
        };
        let dark = input(&Palette::SUNSET_DARK);
        let light = input(&Palette::SUNSET_LIGHT);
        let pair = [
            (Key::of(&dark, "sunset-dark"), dark),
            (Key::of(&light, "sunset-light"), light),
        ];
        if dark_first {
            pair.to_vec()
        } else {
            vec![pair[1], pair[0]]
        }
    }

    fn tiny(input: &Input) -> Option<Picture> {
        Some(Picture {
            width: input.width,
            height: input.height,
            rgba: vec![0; 4],
        })
    }

    /// Ticks and makes pictures as the window would, at `now`, until nothing more is due.
    fn make_all(backdrop: &mut Backdrop, now: Instant) {
        while backdrop.schedule(now).is_some_and(|wait| wait.is_zero()) {
            if let Some((key, input)) = backdrop.due(now) {
                backdrop.finished(key, tiny(&input));
            }
        }
    }

    #[test]
    fn at_start_both_modes_are_made_at_once_and_the_current_fades_in() {
        let mut backdrop = Backdrop::default();
        let start = Instant::now();
        backdrop.want(wanted(900, true), start);
        assert_eq!(
            backdrop.schedule(start),
            Some(Duration::ZERO),
            "no wait at start"
        );
        assert_eq!(backdrop.schedule(start), None, "one tick at a time");
        let (key, input) = backdrop.due(start).unwrap();
        assert_eq!(key.theme, "sunset-dark", "the current mode first");
        assert!(
            !backdrop.is_fading(),
            "nothing fades before a picture is ready"
        );
        backdrop.finished(key, tiny(&input));
        assert!(backdrop.is_fading());
        assert_eq!(backdrop.layers().1.unwrap().1, 0.0);
        make_all(&mut backdrop, start);
        assert_eq!(backdrop.generations, 2);
        // The fade ends, and then nothing moves.
        backdrop.frame(start);
        backdrop.frame(start + motion::SCENE_IN);
        assert!(!backdrop.is_fading());
        assert_eq!(backdrop.layers().1.unwrap().1, 1.0);
        assert_eq!(backdrop.schedule(start + motion::SCENE_IN), None);
    }

    #[test]
    fn a_mode_switch_crossfades_without_making_anything() {
        let mut backdrop = Backdrop::default();
        let start = Instant::now();
        backdrop.want(wanted(900, true), start);
        make_all(&mut backdrop, start);
        backdrop.frame(start + motion::SLOW * 4);
        backdrop.want(wanted(900, false), start);
        assert!(backdrop.is_fading());
        assert_eq!(backdrop.schedule(start), None);
        assert_eq!(backdrop.generations, 2);
    }

    #[test]
    fn a_resize_waits_shows_the_last_picture_then_makes_both() {
        let mut backdrop = Backdrop::default();
        let start = Instant::now();
        backdrop.want(wanted(900, true), start);
        make_all(&mut backdrop, start);
        backdrop.want(wanted(1000, true), start);
        let wait = backdrop.schedule(start).unwrap();
        assert_eq!(wait, DEBOUNCE, "a resize waits");
        assert_eq!(backdrop.due(start), None, "too soon");
        assert_eq!(
            backdrop.layers().1.unwrap().0.size.width,
            900.0,
            "the last picture"
        );
        let later = start + DEBOUNCE;
        assert!(backdrop.schedule(later).is_some());
        let (key, input) = backdrop.due(later).unwrap();
        backdrop.finished(key, tiny(&input));
        make_all(&mut backdrop, later);
        assert_eq!(backdrop.generations, 4);
        assert_eq!(backdrop.layers().1.unwrap().0.size.width, 1000.0);
        let covering = cover(
            iced::Size::new(900.0, 600.0),
            iced::Size::new(1000.0, 600.0),
        );
        assert!((covering.width - 1000.0).abs() < 0.01 && covering.height >= 600.0);
    }
}
