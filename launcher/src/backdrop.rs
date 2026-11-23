// SPDX-License-Identifier: GPL-3.0-or-later

//! The Play page's scene, generated once per key and cached: never in `view`, never per frame.
//!
//! The key is what the picture depends on: the page's size in physical pixels, the scale, the
//! theme and its mode, the text column's width and the window's corner. A change of key starts a
//! wait of [`DEBOUNCE`]; only a key that has held that long is generated, off the interface's
//! thread, so a resize drag generates once, for its final size, and shows the last picture, scaled
//! to cover, until then. Pictures of the current size stay cached for every theme and mode, so
//! switching back and forth generates nothing.

use std::time::{Duration, Instant};

use iced::widget::image;

use crate::scene::{self, Input, Picture};

/// How long a key must hold before its picture is generated.
pub const DEBOUNCE: Duration = Duration::from_millis(150);

/// What a picture depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    pub width: u32,
    pub height: u32,
    scale: u32,
    /// The palette's name, which says its theme and mode.
    pub theme: &'static str,
    column: u32,
    corner: Option<u32>,
}

impl Key {
    pub fn of(input: &Input, theme: &'static str) -> Self {
        Self {
            width: input.width,
            height: input.height,
            scale: input.scale.to_bits(),
            theme,
            column: input.column.to_bits(),
            corner: input.corner.map(f32::to_bits),
        }
    }

    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale)
    }

    /// Whether two keys are for the same page, whatever their theme.
    fn same_page(&self, other: &Key) -> bool {
        Key {
            theme: other.theme,
            ..*self
        } == *other
    }
}

/// The generator: [`scene::generate`], or a stand-in in the tests.
pub type Generator = fn(&Input) -> Option<Picture>;

/// A generated picture, ready to show, or none where the page has no scene.
#[derive(Debug, Clone)]
pub struct Shown {
    pub handle: Option<image::Handle>,
    /// Its size, in logical pixels.
    pub size: iced::Size,
}

/// The pictures made, and the one wanted.
pub struct Backdrop {
    made: Vec<(Key, Shown)>,
    wanted: Option<(Key, Input, Instant)>,
    in_flight: Option<Key>,
    /// How many pictures have been generated: what the tests count.
    pub generations: usize,
    pub generator: Generator,
}

impl Default for Backdrop {
    fn default() -> Self {
        Self {
            made: Vec::new(),
            wanted: None,
            in_flight: None,
            generations: 0,
            generator: scene::generate,
        }
    }
}

impl Backdrop {
    /// The picture the page wants now (none: no scene), noted at `now`. Whether it's a new key
    /// that isn't made yet, so a wait should start.
    pub fn want(&mut self, wanted: Option<(Key, Input)>, now: Instant) -> bool {
        let Some((key, input)) = wanted else {
            self.wanted = None;
            return false;
        };
        if self.wanted.as_ref().is_some_and(|(held, ..)| *held == key) {
            return false;
        }
        self.wanted = Some((key, input, now));
        !self.is_made(&key)
    }

    fn is_made(&self, key: &Key) -> bool {
        self.made.iter().any(|(made, _)| made == key)
    }

    /// At `now`, the picture to generate, if the wanted key has held for [`DEBOUNCE`], isn't made
    /// and isn't being made. Counted as a generation.
    pub fn due(&mut self, now: Instant) -> Option<(Key, Input)> {
        let (key, input, since) = self.wanted?;
        if self.is_made(&key)
            || self.in_flight.is_some()
            || now.saturating_duration_since(since) < DEBOUNCE
        {
            return None;
        }
        self.in_flight = Some(key);
        self.generations += 1;
        Some((key, input))
    }

    /// A picture was generated for `key`. Pictures of another page size go; every theme's for
    /// this size stays.
    pub fn finished(&mut self, key: Key, picture: Option<Picture>) {
        if self.in_flight == Some(key) {
            self.in_flight = None;
        }
        let scale = key.scale();
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
        self.made
            .retain(|(made, _)| made.same_page(&key) && *made != key);
        self.made.push((key, shown));
        // Another key may have come while this one was made: its wait has passed by now if
        // nothing changed since, and the next tick makes it.
    }

    /// Whether a picture is wanted that isn't made: the window should tick until it is.
    pub fn waiting(&self) -> bool {
        self.wanted
            .as_ref()
            .is_some_and(|(key, ..)| !self.is_made(key))
    }

    /// What to show: the wanted picture if it's made; while it isn't (a resize under way), the
    /// last made for the same theme, to scale to cover; nothing otherwise.
    pub fn current(&self) -> Option<&Shown> {
        let (key, ..) = self.wanted.as_ref()?;
        if let Some((_, shown)) = self.made.iter().find(|(made, _)| made == key) {
            return Some(shown);
        }
        self.made
            .iter()
            .rev()
            .find(|(made, shown)| made.theme == key.theme && shown.handle.is_some())
            .map(|(_, shown)| shown)
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

    fn input(width: u32) -> Input {
        Input {
            colors: Colors::of(&Palette::SUNSET_DARK),
            width,
            height: 600,
            scale: 1.0,
            column: 468.0,
            corner: None,
        }
    }

    fn tiny(input: &Input) -> Option<Picture> {
        Some(Picture {
            width: input.width,
            height: input.height,
            rgba: vec![0; 4],
        })
    }

    #[test]
    fn a_key_is_made_once_after_it_holds() {
        let mut backdrop = Backdrop {
            generator: tiny,
            ..Backdrop::default()
        };
        let start = Instant::now();
        let key = Key::of(&input(900), "sunset-dark");
        assert!(backdrop.want(Some((key, input(900))), start));
        assert!(
            !backdrop.want(Some((key, input(900))), start),
            "the same key"
        );
        assert_eq!(backdrop.due(start + DEBOUNCE / 2), None, "too soon");
        let (due, made) = backdrop.due(start + DEBOUNCE).unwrap();
        assert_eq!(due, key);
        assert_eq!(backdrop.due(start + DEBOUNCE * 2), None, "in flight");
        backdrop.finished(due, tiny(&made));
        assert!(!backdrop.waiting());
        assert_eq!(backdrop.due(start + DEBOUNCE * 3), None, "made");
        assert_eq!(backdrop.generations, 1);
    }

    #[test]
    fn while_resizing_the_last_picture_shows_and_other_sizes_go() {
        let mut backdrop = Backdrop::default();
        let start = Instant::now();
        let first = Key::of(&input(900), "sunset-dark");
        backdrop.want(Some((first, input(900))), start);
        backdrop.due(start + DEBOUNCE);
        backdrop.finished(first, tiny(&input(900)));
        let light = Key {
            theme: "sunset-light",
            ..first
        };
        backdrop.want(Some((light, input(900))), start);
        backdrop.due(start + DEBOUNCE);
        backdrop.finished(light, tiny(&input(900)));
        // A new size: the last dark picture shows until the new one's made.
        let second = Key::of(&input(1000), "sunset-dark");
        backdrop.want(Some((second, input(1000))), start);
        assert_eq!(backdrop.current().unwrap().size.width, 900.0);
        backdrop.due(start + DEBOUNCE);
        backdrop.finished(second, tiny(&input(1000)));
        assert_eq!(backdrop.current().unwrap().size.width, 1000.0);
        assert_eq!(backdrop.made.len(), 1, "only this size's");
        let covering = cover(
            iced::Size::new(900.0, 600.0),
            iced::Size::new(1000.0, 600.0),
        );
        assert!((covering.width - 1000.0).abs() < 0.01 && covering.height >= 600.0);
    }
}
