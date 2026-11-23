// SPDX-License-Identifier: GPL-3.0-or-later

//! A progress bar: how far a long step is, such as writing the system image; or, not knowing how
//! far, busy ([`busy`]).

use std::time::Duration;

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, renderer};
use iced::time::Instant;
use iced::{Event, Length, Point, Rectangle, Size, window};

use crate::ui::tokens::{motion, size};

use crate::ui::theme::Progress as State;
use crate::ui::transition::{phase, wants_frames};
use crate::ui::{Element, Renderer, Theme, paint};

/// A bar this far along, from 0 to 1, running. It never animates: it moves when the value does.
pub fn progress(done: f32) -> Progress {
    Progress {
        done: if done.is_nan() {
            0.0
        } else {
            done.clamp(0.0, 1.0)
        },
        state: State::Running,
        busy: false,
        preview: None,
    }
}

/// A bar for a step that's under way but can't say how far: a stretch of it grows out of its
/// start, runs across and shrinks into its end, over and over, never past the track; with motion
/// reduced, it rests in the middle.
pub fn busy() -> Progress {
    Progress {
        busy: true,
        ..progress(0.0)
    }
}

/// A progress bar ([`progress`], [`busy`]).
pub struct Progress {
    done: f32,
    state: State,
    busy: bool,
    /// Busy, drawn still this far into its sweep, for the gallery.
    preview: Option<f32>,
}

/// How long a busy bar's stretch takes to cross it, and how far its tail starts behind its head,
/// as a share of that.
const SWEEP: Duration = Duration::from_millis(1600);
const TAIL: f32 = 0.3;

/// A busy bar's sweep: when it started, and the frame it's at.
#[derive(Default)]
struct Sweep {
    start: Option<Instant>,
    now: Option<Instant>,
}

impl Sweep {
    /// Where the stretch's tail and head are, as shares of the bar's width: the head runs from
    /// the start to the end, eased, the tail after it, so the stretch grows out of the start and
    /// shrinks into the end; with motion reduced, a still stretch in the middle.
    fn stretch(&self) -> (f32, f32) {
        if crate::ui::system::reduced_motion() {
            return (0.35, 0.65);
        }
        let (Some(start), Some(now)) = (self.start, self.now) else {
            return (0.0, 0.0);
        };
        // A third of the way in at first, so its first frame shows it.
        stretch_at((phase(start, now, SWEEP) + 1.0 / 3.0).fract())
    }
}

/// Where the stretch's tail and head are this far into a sweep, from 0 to 1.
fn stretch_at(into: f32) -> (f32, f32) {
    let eased = |from: f32| motion::EASE.at(((into - from) / (1.0 - TAIL)).clamp(0.0, 1.0));
    (eased(TAIL), eased(0.0))
}

impl Progress {
    /// Finished, in mint: the whole bar, however far it said it was.
    pub fn done(self) -> Self {
        Self {
            done: 1.0,
            state: State::Done,
            busy: false,
            preview: None,
        }
    }

    /// Busy, drawn still as it is `at` this far into its sweep, from 0 to 1, asking for no
    /// frames: for the gallery.
    pub fn preview(self, at: f32) -> Self {
        Self {
            preview: Some(at.clamp(0.0, 1.0)),
            ..self
        }
    }

    /// Stopped by an error, in red: as far as it got, or, busy, the whole bar, still.
    pub fn failed(self) -> Self {
        Self {
            state: State::Failed,
            done: if self.busy { 1.0 } else { self.done },
            busy: false,
            ..self
        }
    }
}

impl<Message> Widget<Message, Theme, Renderer> for Progress {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<Sweep>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(Sweep::default())
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
        // Busy, it sweeps: a frame after every frame while it's in sight, unless motion's reduced
        // or it's a still.
        if self.busy
            && self.preview.is_none()
            && let Event::Window(window::Event::RedrawRequested(now)) = event
        {
            let sweep = tree.state.downcast_mut::<Sweep>();
            sweep.start.get_or_insert(*now);
            sweep.now = Some(*now);
            if wants_frames(layout.bounds(), viewport) {
                shell.request_redraw();
            }
        }
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Shrink)
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.resolve(Length::Fill, Length::Shrink, Size::new(0.0, size::PROGRESS));
        layout::Node::new(Size::new(size.width, size::PROGRESS))
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
        let look = theme.progress(self.state);
        paint::boxed(renderer, bounds, &look);
        if self.busy {
            // The stretch, always inside the track: shorter than the bar is tall, as it comes out
            // of the start or goes into the end, it fades with its length, its glow too.
            let (tail, head) = match self.preview {
                Some(at) => stretch_at(at),
                None => tree.state.downcast_ref::<Sweep>().stretch(),
            };
            let length = (head - tail) * bounds.width;
            if length > 0.0 {
                let shown = Rectangle::new(
                    Point::new(bounds.x + tail * bounds.width, bounds.y),
                    Size::new(length.max(bounds.height.min(length * 2.0)), bounds.height),
                );
                let shown = Rectangle {
                    x: shown.x.min(bounds.x + bounds.width - shown.width),
                    ..shown
                };
                let look = theme
                    .progress_bar(self.state)
                    .faded((length / bounds.height).min(1.0));
                paint::boxed(renderer, shown, &look);
            }
            return;
        }
        if self.done > 0.0 {
            let done = Rectangle::new(
                Point::new(bounds.x, bounds.y),
                Size::new((bounds.width * self.done).max(bounds.height), bounds.height),
            );
            let look = theme.progress_bar(self.state);
            if self.state == State::Running {
                sunset(renderer, theme, done, &look);
            } else {
                paint::boxed(renderer, done, &look);
            }
        }
    }
}

/// A running bar's done part in the launcher's palette: the logo's gradient across it, blue at
/// its start to orange at its end, with the bar's glow.
fn sunset(renderer: &mut Renderer, theme: &Theme, done: Rectangle, look: &crate::ui::theme::Look) {
    use iced::advanced::Renderer as _;
    use iced::gradient::Linear;
    let stops = theme.progress_gradient();
    let last = (stops.len() - 1).max(1) as f32;
    let gradient = stops
        .iter()
        .enumerate()
        .fold(Linear::new(iced::Degrees(90.0)), |gradient, (at, stop)| {
            gradient.add_stop(at as f32 / last, crate::ui::theme::color(*stop))
        });
    let radius = look.radius.min(done.height / 2.0);
    renderer.fill_quad(
        renderer::Quad {
            bounds: done,
            border: iced::Border {
                radius: radius.into(),
                ..iced::Border::default()
            },
            shadow: look
                .glow
                .map_or_else(iced::Shadow::default, |glow| iced::Shadow {
                    color: crate::ui::theme::color(glow.color),
                    offset: iced::Vector::new(0.0, glow.offset),
                    blur_radius: glow.blur,
                }),
            snap: true,
        },
        iced::Background::Gradient(iced::Gradient::Linear(gradient)),
    );
}

impl<'a, Message: 'a> From<Progress> for Element<'a, Message> {
    fn from(progress: Progress) -> Self {
        Element::new(progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn done_is_the_whole_bar_and_failed_keeps_how_far() {
        assert_eq!(progress(0.4).done().done, 1.0);
        assert_eq!(progress(0.7).failed().done, 0.7);
        // Busy, failed stops it: the whole bar, still, asking for no frames.
        let failed = busy().failed();
        assert!(!failed.busy && failed.done == 1.0 && failed.state == State::Failed);
    }

    #[test]
    fn progress_stays_between_none_and_all() {
        assert_eq!(progress(-1.0).done, 0.0);
        assert_eq!(progress(2.0).done, 1.0);
        assert_eq!(progress(f32::NAN).done, 0.0);
        assert_eq!(progress(0.5).state, State::Running);
        assert_eq!(progress(1.0).done().state, State::Done);
        assert_eq!(progress(0.3).failed().state, State::Failed);
    }
}
