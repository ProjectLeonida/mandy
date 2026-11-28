// SPDX-License-Identifier: GPL-3.0-or-later

//! A selection list: one of many rows, such as the language, the keyboard layout or the Wi-Fi
//! network. It scrolls itself, keeping the chosen row in sight.

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, renderer};
use iced::keyboard::{self, Key, key::Named};

use iced::{Event, Length, Point, Rectangle, Size, touch, window};

use crate::ui::tokens::{motion, size, space, typography};

use crate::ui::focus::Focus;
use crate::ui::input;
use crate::ui::paint::{self, Plain};
use crate::ui::theme::{Interaction, color};
use crate::ui::transition::Transition;
use crate::ui::{Element, Renderer, Theme};

/// A list of rows, showing this many at once, with the chosen one's index. Without a message to
/// send it is disabled.
pub fn list<'a, Message>(
    rows: impl IntoIterator<Item = impl Into<String>>,
    chosen: Option<usize>,
) -> List<'a, Message> {
    List {
        rows: rows.into_iter().map(Into::into).collect(),
        chosen,
        shown: 6,
        on_choose: None,
        preview: None,
        sidebar: false,
    }
}

/// A sidebar's places: every row shown, taller than a list's, with no
/// box round them; the chosen one marked. The keys and the pointer work as in a [`list`].
pub fn navigation<'a, Message>(
    rows: impl IntoIterator<Item = impl Into<String>>,
    chosen: Option<usize>,
) -> List<'a, Message> {
    let list = list(rows, chosen);
    List {
        shown: list.rows.len().max(1),
        sidebar: true,
        ..list
    }
}

/// A selection list. The arrow keys, Page Up and Page Down, Home and End move the choice; a
/// letter moves it to the next row that starts with it. The wheel scrolls it.
pub struct List<'a, Message> {
    rows: Vec<String>,
    chosen: Option<usize>,
    shown: usize,
    on_choose: Option<Box<dyn Fn(usize) -> Message + 'a>>,
    preview: Option<(usize, Interaction, bool)>,
    /// Drawn as a sidebar's places ([`navigation`]).
    sidebar: bool,
}

impl<'a, Message> List<'a, Message> {
    /// How many rows show at once.
    pub fn shown(self, rows: usize) -> Self {
        Self {
            shown: rows.max(1),
            ..self
        }
    }

    pub fn on_choose(self, on_choose: impl Fn(usize) -> Message + 'a) -> Self {
        Self {
            on_choose: Some(Box::new(on_choose)),
            ..self
        }
    }

    /// Draws one row as if in this state, and the focus outline, for the gallery.
    pub fn preview(self, row: usize, interaction: Interaction, focused: bool) -> Self {
        Self {
            preview: Some((row, interaction, focused)),
            ..self
        }
    }

    fn choose(&mut self, row: usize, shell: &mut Shell<'_, Message>) {
        if let Some(on_choose) = &self.on_choose
            && Some(row) != self.chosen
            && row < self.rows.len()
        {
            shell.publish(on_choose(row));
            // Taken at once, so a second key in the same batch of events starts from it, as the
            // app will.
            self.chosen = Some(row);
        }
    }
}

/// The row a key moves the choice to, among `rows` with `page` shown at once.
pub(crate) fn step(
    rows: &[String],
    chosen: Option<usize>,
    page: usize,
    key: &Key,
) -> Option<usize> {
    let last = rows.len().checked_sub(1)?;
    // A choice past the end, as the app's rows shrank under it (a rescan found fewer networks),
    // is no choice.
    let chosen = chosen.filter(|&at| at <= last);
    let at = chosen.unwrap_or(0);
    match key {
        Key::Named(Named::ArrowDown) => Some(chosen.map_or(0, |at| (at + 1).min(last))),
        Key::Named(Named::ArrowUp) => Some(chosen.map_or(last, |at| at.saturating_sub(1))),
        Key::Named(Named::PageDown) => Some((at + page.saturating_sub(1).max(1)).min(last)),
        Key::Named(Named::PageUp) => Some(at.saturating_sub(page.saturating_sub(1).max(1))),
        Key::Named(Named::Home) => Some(0),
        Key::Named(Named::End) => Some(last),
        Key::Character(letter) => {
            let letter = letter.to_lowercase();
            let starts = |row: &usize| rows[*row].to_lowercase().starts_with(&letter);
            let from = chosen.map_or(0, |at| at + 1);
            (from..rows.len()).chain(0..from).find(starts)
        }
        _ => None,
    }
}

/// The first row shown, so `row` is in sight with as little scrolling as can be.
pub(crate) fn scroll_to(first: usize, row: usize, shown: usize) -> usize {
    if row < first {
        row
    } else if row >= first + shown {
        row + 1 - shown
    } else {
        first
    }
}

/// What it last drew: the row under the pointer, the pressed one, the outline, the first row
/// shown, the chosen one.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Drawn {
    over: Option<usize>,
    pressed: Option<usize>,
    outline: bool,
    first: usize,
    chosen: Option<usize>,
}

/// A sidebar's mark moving: the row it leaves, where it shrinks away, and the row it comes to,
/// where it grows, both at once, from their middles; each from the height it had, so a choice
/// changed again part of the way doesn't jump.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Mark {
    from: Option<usize>,
    from_height: f32,
    to: Option<usize>,
    to_height: f32,
    /// How far along it is, from 0 to 1.
    moving: Transition,
}

impl Mark {
    /// The mark leaving `from` for `to`, from the heights they have now under `was`, the mark
    /// moving before, if any; none with motion reduced.
    fn new(was: Option<Mark>, from: Option<usize>, to: Option<usize>) -> Option<Self> {
        let height = |row: Option<usize>| {
            row.map_or(0.0, |row| {
                was.map_or(f32::from(u8::from(Some(row) == from)), |was| {
                    was.height(row)
                })
            })
        };
        Some(Self {
            from,
            from_height: height(from),
            to,
            to_height: height(to),
            moving: Transition::start(0.0, 1.0, motion::SLOW)?,
        })
    }

    /// How tall a row's mark is, a share of its whole height.
    fn height(&self, row: usize) -> f32 {
        let along = self.moving.at();
        if Some(row) == self.to {
            self.to_height + (1.0 - self.to_height) * along
        } else if Some(row) == self.from {
            self.from_height * (1.0 - along)
        } else {
            0.0
        }
    }
}

#[derive(Default)]
struct State {
    /// The chosen row when last seen, and the mark moving from it.
    marked: Option<Option<usize>>,
    mark: Option<Mark>,
    focus: Focus,
    first: usize,
    /// The chosen row when last laid out, to bring a new choice into sight.
    laid_out: Option<usize>,
    /// The rows' words, shaped: again only when they change.
    labels: Vec<Plain>,
    pressed: Option<usize>,
    /// What a touchpad's wheel has scrolled short of a whole row, in rows.
    wheeled: f32,
    /// A finger's drag: where it went down and the first row then, and whether it's moved further
    /// than a slip, so it scrolls the rows rather than choosing one.
    drag: Option<Swipe>,
    drawn: Option<Drawn>,
    /// Its focus ring, fading in and out with keyboard focus on the glows' own timing.
    focus_lit: crate::ui::transition::Lit,
}

#[derive(Debug, Clone, Copy)]
struct Swipe {
    from: f32,
    first: usize,
    moved: bool,
}

/// The room between two rows.
const GAP: f32 = space::XS;

impl<Message> List<'_, Message> {
    /// From one row's top to the next's: a row and the gap after it.
    fn pitch(&self) -> f32 {
        self.row_height() + GAP
    }

    fn row_height(&self) -> f32 {
        if self.sidebar {
            size::SIDEBAR_ROW
        } else {
            size::ROW
        }
    }

    /// Where the rows are: inside the box's padding, or, in a sidebar, edge to edge.
    fn inner(&self, bounds: Rectangle) -> Rectangle {
        if self.sidebar {
            bounds
        } else {
            bounds.shrink(space::XS)
        }
    }

    fn row_at(&self, state: &State, layout: Layout<'_>, cursor: mouse::Cursor) -> Option<usize> {
        let inner = self.inner(layout.bounds());
        let point = cursor.position_over(inner)?;
        let down = point.y - inner.y;
        // In the gap between two rows, neither.
        if down % self.pitch() >= self.row_height() {
            return None;
        }
        let row = state.first + (down / self.pitch()) as usize;
        (row < self.rows.len()).then_some(row)
    }

    /// What the focus outline goes round: the list, or a sidebar's chosen row, or its first, as
    /// a sidebar's list has no box to round.
    fn outlined(&self, bounds: Rectangle, first: usize) -> Rectangle {
        if !self.sidebar {
            return bounds;
        }
        let inner = self.inner(bounds);
        let slot = self.chosen.unwrap_or(0).saturating_sub(first) as f32;
        Rectangle::new(
            Point::new(inner.x, inner.y + slot * self.pitch()),
            Size::new(inner.width, self.row_height()),
        )
    }

    fn max_first(&self) -> usize {
        self.rows.len().saturating_sub(self.shown)
    }
}

impl<Message> Widget<Message, Theme, Renderer> for List<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Shrink)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let state = tree.state.downcast_mut::<State>();
        state.labels.resize_with(self.rows.len(), Plain::default);
        for (plain, row) in state.labels.iter_mut().zip(&self.rows) {
            paint::shape(plain, &typography::INTERFACE, row);
        }
        if self.chosen != state.laid_out {
            if let Some(row) = self.chosen {
                state.first = scroll_to(state.first, row, self.shown);
            }
            state.laid_out = self.chosen;
        }
        state.first = state.first.min(self.max_first());
        let inset = if self.sidebar { 0.0 } else { 2.0 * space::XS };
        let rows = self.shown.max(1) as f32;
        let height = rows * self.row_height() + (rows - 1.0) * GAP + inset;
        let size = limits.resolve(Length::Fill, Length::Shrink, Size::new(0.0, height));
        layout::Node::new(Size::new(size.width, height))
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        _renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        if self.on_choose.is_some() && self.preview.is_none() && !self.rows.is_empty() {
            let state = tree.state.downcast_mut::<State>();
            // Where the outline goes, so focus scrolls that into view: a sidebar's long list
            // in a scroll area would show its end, and not the row it outlines.
            let outlined = self.outlined(layout.bounds(), state.first);
            operation.focusable(None, outlined, &mut state.focus);
        }
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        _clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        // A preview is a picture of a state: it takes no input.
        if self.preview.is_some() {
            return;
        }
        let state = tree.state.downcast_mut::<State>();
        state
            .focus
            .keep_if(self.on_choose.is_some() && !self.rows.is_empty());
        if self.sidebar {
            // The chosen place changed: the mark leaves one row and comes to another.
            if let Some(was) = state.marked
                && was != self.chosen
            {
                state.mark = Mark::new(state.mark, was, self.chosen);
                shell.request_redraw();
            }
            state.marked = Some(self.chosen);
            if let Event::Window(window::Event::RedrawRequested(now)) = event
                && let Some(mark) = &mut state.mark
            {
                if mark.moving.frame(*now) {
                    shell.request_redraw();
                } else {
                    state.mark = None;
                }
            }
        }
        let over = self.row_at(state, layout, cursor);
        let scrolls = self.rows.len() > self.shown;
        if let Event::Mouse(mouse::Event::WheelScrolled { delta }) = event
            && cursor.is_over(layout.bounds())
            && scrolls
        {
            // A wheel's line is a row; a touchpad's pixels add up, a row at a time.
            state.wheeled -= match delta {
                mouse::ScrollDelta::Lines { y, .. } => *y,
                mouse::ScrollDelta::Pixels { y, .. } => y / self.pitch(),
            };
            let rows = state.wheeled.trunc();
            state.wheeled -= rows;
            let first = (state.first as f32 + rows).clamp(0.0, self.max_first() as f32) as usize;
            // At an end, nothing's kept towards a row past it.
            if (first == 0 && state.wheeled < 0.0)
                || (first == self.max_first() && state.wheeled > 0.0)
            {
                state.wheeled = 0.0;
            }
            state.first = first;
            shell.capture_event();
        }
        // A finger dragged up or down further than a slip scrolls the rows, and lets go of the
        // one it pressed; the drag is the list's, not a scroll area's round it.
        match event {
            Event::Touch(touch::Event::FingerPressed { position, .. })
                if scrolls && cursor.is_over(layout.bounds()) =>
            {
                state.drag = Some(Swipe {
                    from: position.y,
                    first: state.first,
                    moved: false,
                });
            }
            Event::Touch(touch::Event::FingerMoved { position, .. }) => {
                if let Some(drag) = &mut state.drag {
                    let by = drag.from - position.y;
                    drag.moved |= by.abs() > size::SLIP;
                    if drag.moved {
                        state.pressed = None;
                        let rows = (by / self.pitch()).round();
                        let first = (drag.first as f32 + rows).clamp(0.0, self.max_first() as f32);
                        state.first = first as usize;
                        shell.capture_event();
                    }
                }
            }
            Event::Touch(touch::Event::FingerLifted { .. } | touch::Event::FingerLost { .. }) => {
                let dragged = state.drag.take().is_some_and(|drag| drag.moved);
                if dragged {
                    shell.capture_event();
                }
            }
            _ => {}
        }
        if self.on_choose.is_some() {
            match event {
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerPressed { .. })
                    if over.is_some() =>
                {
                    state.pressed = over;
                    state.focus.click();
                    shell.capture_event();
                }
                Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerLifted { .. }) => {
                    if let Some(row) = state.pressed.take() {
                        if over == Some(row) {
                            self.choose(row, shell);
                        }
                        shell.capture_event();
                    }
                }
                Event::Touch(touch::Event::FingerLost { .. }) => state.pressed = None,
                Event::Keyboard(keyboard::Event::KeyPressed { key, modifiers, .. })
                    if state.focus.is_focused() && input::plain(*modifiers) =>
                {
                    if let Some(row) = step(&self.rows, self.chosen, self.shown, key) {
                        state.focus.key();
                        state.first = scroll_to(state.first, row, self.shown);
                        self.choose(row, shell);
                        shell.capture_event();
                    }
                }
                _ => {}
            }
        } else {
            state.pressed = None;
        }
        state.focus_lit.frame(event, shell);
        state.focus_lit.aim(state.focus.is_shown(), shell);
        let now = Drawn {
            over,
            pressed: state.pressed,
            outline: state.focus.is_shown(),
            first: state.first,
            chosen: self.chosen,
        };
        input::redraw_on_change(&mut state.drawn, now, event, shell);
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_ref::<State>();
        let bounds = layout.bounds();
        let disabled = self.on_choose.is_none();
        let frame = theme.list(match self.preview {
            _ if disabled => Interaction::Disabled,
            Some((_, interaction, _)) => interaction,
            None if cursor.is_over(bounds) => Interaction::Hovered,
            None => Interaction::Idle,
        });
        if !self.sidebar {
            paint::boxed(renderer, bounds, &frame);
        }
        let inner = self.inner(bounds);
        let height = self.row_height();
        let over = self.row_at(state, layout, cursor);
        // No scrollbar: the wheel, the keys and the chosen row brought into sight scroll it.
        let width = inner.width;
        let shown = state
            .labels
            .iter()
            .enumerate()
            .skip(state.first)
            .take(self.shown);
        for (slot, (row, label)) in shown.enumerate() {
            let bounds = Rectangle::new(
                Point::new(inner.x, inner.y + slot as f32 * self.pitch()),
                Size::new(width, height),
            );
            // A row out of sight, as a sidebar's long list scrolled, isn't drawn.
            if !bounds.intersects(viewport) {
                continue;
            }
            let interaction = match self.preview {
                Some((shown, interaction, _)) if shown == row => interaction,
                _ if disabled => Interaction::Disabled,
                _ if state.pressed == Some(row) => Interaction::Pressed,
                _ if over == Some(row) => Interaction::Hovered,
                _ => Interaction::Idle,
            };
            let chosen = self.chosen == Some(row);
            let look = if self.sidebar {
                theme.sidebar_row(chosen, interaction)
            } else {
                theme.row(chosen, interaction)
            };
            paint::boxed(renderer, bounds, &look);
            // The mark's height: whole on the chosen row, or growing and shrinking as it moves.
            let marked = match (state.mark, self.preview) {
                (Some(mark), None) => mark.height(row),
                _ => f32::from(u8::from(chosen)),
            };
            if self.sidebar && marked > 0.0 && interaction != Interaction::Disabled {
                // The mark: a pill at the row's start, set in as far from its start as from its
                // top and bottom, from its middle as it grows.
                let inset = size::SIDEBAR_MARK_INSET;
                let tall = (height - 2.0 * inset) * marked;
                let mark = Rectangle::new(
                    Point::new(bounds.x + inset, bounds.center_y() - tall / 2.0),
                    Size::new(size::SIDEBAR_MARK, tall),
                );
                // A small light: it sparks, softly, and its glow grows and shrinks with it.
                let ink = theme.sidebar_mark();
                let look = crate::ui::theme::Look::solid(ink, size::SIDEBAR_MARK / 2.0)
                    .glowing(theme.spark(ink).small());
                paint::boxed(renderer, mark, &look);
            }
            // A sidebar's words start past the mark's room, chosen or not, so they line up; long
            // words end as far from the row's end, cut there.
            let start = if self.sidebar {
                2.0 * size::SIDEBAR_MARK_INSET + size::SIDEBAR_MARK
            } else {
                space::M
            };
            let words = typography::INTERFACE.size * typography::INTERFACE.line_height;
            let at = Point::new(bounds.x + start, bounds.center_y() - words / 2.0);
            let room = Rectangle::new(
                Point::new(at.x, bounds.y),
                Size::new((bounds.width - start - space::M).max(0.0), bounds.height),
            );
            paint::fill(renderer, label, at, color(look.ink), room);
        }
        let focused = match self.preview {
            Some((_, _, focused)) => focused,
            None => state.focus.is_shown(),
        };
        // The focus ring fades in and out with keyboard focus, drawn while it fades out too.
        let ring = if self.preview.is_some() {
            f32::from(u8::from(focused))
        } else {
            state.focus_lit.share()
        };
        if ring > 0.0 {
            let outlined = self.outlined(bounds, state.first);
            let radius = if self.sidebar {
                crate::ui::tokens::radius::CHIP
            } else {
                frame.radius
            };
            paint::outline_lit(renderer, theme, outlined, radius, ring);
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        let state = tree.state.downcast_ref::<State>();
        if self.on_choose.is_some() && self.row_at(state, layout, cursor).is_some() {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, Message: 'a> From<List<'a, Message>> for Element<'a, Message> {
    fn from(list: List<'a, Message>) -> Self {
        Element::new(list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn the_mark_shrinks_from_one_row_as_it_grows_in_another() {
        let start = Instant::now();
        let mut mark = Mark::new(None, Some(1), Some(4)).unwrap();
        mark.moving.frame(start);
        assert_eq!(
            (mark.height(1), mark.height(4), mark.height(2)),
            (1.0, 0.0, 0.0)
        );
        mark.moving.frame(start + motion::SLOW / 4);
        let (leaving, coming) = (mark.height(1), mark.height(4));
        assert!(leaving < 0.5 && coming > 0.5 && (leaving + coming - 1.0).abs() < 1e-6);
        // Chosen again part of the way, back to row 1: each goes on from the height it has.
        let back = Mark::new(Some(mark), Some(4), Some(1)).unwrap();
        assert_eq!((back.height(4), back.height(1)), (coming, leaving));
        mark.moving.frame(start + motion::SLOW);
        assert_eq!((mark.height(1), mark.height(4)), (0.0, 1.0));
        // From nothing chosen, it only grows.
        let mut first = Mark::new(None, None, Some(0)).unwrap();
        first.moving.frame(start);
        first.moving.frame(start + motion::SLOW);
        assert_eq!(first.height(0), 1.0);
    }

    fn rows() -> Vec<String> {
        [
            "English",
            "Español",
            "Deutsch",
            "Ελληνικά",
            "Eesti",
            "Suomi",
        ]
        .map(String::from)
        .to_vec()
    }

    #[test]
    fn keys_move_the_choice() {
        let rows = rows();
        let named = |at, key| step(&rows, at, 3, &Key::Named(key));
        assert_eq!(named(None, Named::ArrowDown), Some(0));
        assert_eq!(named(None, Named::ArrowUp), Some(5));
        assert_eq!(named(Some(5), Named::ArrowDown), Some(5));
        assert_eq!(named(Some(0), Named::ArrowUp), Some(0));
        assert_eq!(named(Some(0), Named::PageDown), Some(2));
        assert_eq!(named(Some(5), Named::PageUp), Some(3));
        assert_eq!(named(Some(3), Named::Home), Some(0));
        assert_eq!(named(Some(0), Named::End), Some(5));
        assert_eq!(named(Some(0), Named::Enter), None);
        // A choice past the rows' end, as they shrank under it: no panic, taken as none.
        let rows: Vec<String> = ["Home", "Office", "Cafe"].map(String::from).into();
        let letter = |key: &str| Key::Character(key.into());
        assert_eq!(step(&rows, Some(7), 3, &letter("z")), None);
        assert_eq!(step(&rows, Some(7), 3, &letter("c")), Some(2));
        assert_eq!(
            step(&rows, Some(7), 3, &Key::Named(Named::ArrowDown)),
            Some(0)
        );
        assert_eq!(step(&[], None, 3, &Key::Named(Named::ArrowDown)), None);
    }

    #[test]
    fn a_letter_finds_the_next_row_starting_with_it() {
        let rows = rows();
        let letter = |at, letter: &str| step(&rows, at, 3, &Key::Character(letter.into()));
        assert_eq!(letter(None, "e"), Some(0));
        assert_eq!(letter(Some(0), "E"), Some(1));
        assert_eq!(letter(Some(1), "e"), Some(4));
        assert_eq!(letter(Some(4), "e"), Some(0));
        assert_eq!(letter(Some(0), "ε"), Some(3));
        assert_eq!(letter(Some(0), "x"), None);
    }

    #[test]
    fn the_chosen_row_stays_in_sight() {
        assert_eq!(scroll_to(0, 2, 3), 0);
        assert_eq!(scroll_to(0, 3, 3), 1);
        assert_eq!(scroll_to(4, 2, 3), 2);
        assert_eq!(scroll_to(2, 4, 3), 2);
    }
}
