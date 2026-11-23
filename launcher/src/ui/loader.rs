// SPDX-License-Identifier: GPL-3.0-or-later

//! A loader: a ring that pulses while something is under way and can't say how far, such as
//! signing in or looking for networks. A busy progress bar ([`crate::ui::progress::busy`]) says the
//! same along a line; a loader says it in a spot.

use std::time::Duration;

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, renderer};
use iced::time::Instant;
use iced::{Event, Length, Rectangle, Size, window};

use crate::ui::tokens::{motion, size};

use crate::ui::theme::Look;
use crate::ui::transition::{phase, wants_frames};
use crate::ui::{Element, Renderer, Theme, paint};

/// A loader, its ring [`size::LOADER`] across. With motion reduced, the ring is still.
pub fn loader() -> Loader {
    Loader {
        size: size::LOADER,
        preview: None,
    }
}

/// A loader ([`loader`]).
pub struct Loader {
    size: f32,
    /// Drawn still this far into a pulse, for the gallery.
    preview: Option<f32>,
}

impl Loader {
    /// How wide the ring is across: [`size::LOADER`] unless set.
    pub fn size(self, size: f32) -> Self {
        Self { size, ..self }
    }

    /// Drawn still as it is `at` this far into a pulse, from 0 to 1, asking for no frames: for
    /// the gallery.
    pub fn preview(self, at: f32) -> Self {
        Self {
            preview: Some(at.clamp(0.0, 1.0)),
            ..self
        }
    }
}

/// How long a pulse takes: the ring brightens and dims once, and a ripple goes out from it.
const PULSE: Duration = Duration::from_millis(1400);
/// How far a ripple goes out past the ring, as a share of its size.
const RIPPLE: f32 = 0.5;

/// A loader's pulse: when it started, and the frame it's at.
#[derive(Default)]
struct Pulse {
    start: Option<Instant>,
    now: Option<Instant>,
}

impl Pulse {
    /// How far into a pulse it is, from 0 to 1; with motion reduced, none.
    fn at(&self) -> Option<f32> {
        if crate::ui::system::reduced_motion() {
            return None;
        }
        Some(phase(self.start?, self.now?, PULSE))
    }
}

impl<Message> Widget<Message, Theme, Renderer> for Loader {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<Pulse>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(Pulse::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fixed(self.size), Length::Fixed(self.size))
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &Renderer,
        _limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(Size::new(self.size, self.size))
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _renderer: &Renderer,
        _clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        // It pulses: a frame after every frame while it's in sight, unless motion's reduced or
        // it's a still.
        if self.preview.is_none()
            && let Event::Window(window::Event::RedrawRequested(now)) = event
        {
            let pulse = tree.state.downcast_mut::<Pulse>();
            pulse.start.get_or_insert(*now);
            pulse.now = Some(*now);
            if wants_frames(layout.bounds(), viewport) {
                shell.request_redraw();
            }
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let look = theme.loader();
        let ring = size::LOADER_RING;
        let Some(at) = self
            .preview
            .or_else(|| tree.state.downcast_ref::<Pulse>().at())
        else {
            paint::framed(renderer, bounds, &look, ring);
            return;
        };
        // The ring breathes, brightest half way through a pulse.
        let breath = 0.5 - 0.5 * (std::f32::consts::TAU * at).cos();
        paint::framed(renderer, bounds, &look.faded(0.55 + 0.45 * breath), ring);
        // A ripple goes out from it, thinning and fading as it goes.
        let out = motion::EASE.at(at);
        let ripple = bounds.expand(self.size * RIPPLE * out / 2.0);
        let ripple_look = Look { glow: None, ..look }.faded(0.6 * (1.0 - out));
        paint::framed(renderer, ripple, &ripple_look, ring * (1.0 - 0.5 * out));
    }
}

impl<'a, Message: 'a> From<Loader> for Element<'a, Message> {
    fn from(loader: Loader) -> Self {
        Element::new(loader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pulse_goes_round_and_round() {
        let start = Instant::now();
        let at = |ms| {
            Pulse {
                start: Some(start),
                now: Some(start + Duration::from_millis(ms)),
            }
            .at()
        };
        if crate::ui::system::reduced_motion() {
            return;
        }
        assert_eq!(at(0), Some(0.0));
        let half = at(700).unwrap();
        assert!((half - 0.5).abs() < 1e-3);
        // A pulse later, it's where it began.
        assert!(at(1400).unwrap() < 1e-3);
        assert_eq!(Pulse::default().at(), None, "not drawn yet");
    }
}
