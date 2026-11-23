// SPDX-License-Identifier: GPL-3.0-or-later

//! Text fields: [`TextField`], single-line, with a caret that blinks only for a few seconds after
//! the last input, and selection by Shift and the arrow keys or by dragging. A text field's value
//! is the app's, as iced's are.
//!
//! Input methods: none are asked for yet. A text field takes their committed text when the
//! desktop turns them on.

mod text;

pub use text::{TextField, text_field};

use std::time::Duration;

use iced::advanced::Renderer as _;
use iced::advanced::Shell;
use iced::keyboard::{Key, Modifiers, key::Named};
use iced::time::Instant;
use iced::{Point, Rectangle, Size, mouse};

use crate::ui::tokens::{motion, size, space, typography};

use crate::ui::edit::{Edit, Motion, Selection};
use crate::ui::focus::Focus;
use crate::ui::paint::{self, Plain};
use crate::ui::theme::{Interaction, Look, color};
use crate::ui::{Renderer, Theme};

/// The padding inside a field.
const PAD: f32 = space::L;

/// The caret's blink: for a few seconds after the last input, shown, fading out, hidden and
/// fading in again, then shown and still.
#[derive(Debug, Clone, Copy, Default)]
struct Blink {
    since: Option<Instant>,
    /// How much it shows, from 0 to 1.
    opacity: f32,
}

/// How long the caret takes to fade out, or in: most of each half of its blink, slowly.
const CARET_FADE: Duration = motion::SLOW;
/// A frame, while the caret fades: the next one's asked for this soon, 30 a second. Each frame
/// draws the whole window again, and a fade this slow looks as smooth at 30 as at 60.
const FRAME: Duration = Duration::from_millis(33);

impl Blink {
    fn restart(&mut self) {
        self.start(Instant::now());
    }

    /// Shows the caret, blinking from `now`.
    fn start(&mut self, now: Instant) {
        self.since = Some(now);
        self.opacity = 1.0;
    }

    /// At a redraw: how much the caret shows now, and when to draw next while it blinks: every
    /// frame while it fades, and otherwise when it next starts to.
    fn at(&mut self, now: Instant) -> Option<Instant> {
        let since = self.since?;
        let elapsed = now.saturating_duration_since(since);
        if elapsed >= motion::CARET_BLINKS_FOR {
            self.opacity = 1.0;
            return None;
        }
        let blink = motion::CARET_BLINK;
        let fade = crate::ui::system::duration(CARET_FADE).min(blink);
        let period = blink * 2;
        let periods = u32::try_from(elapsed.as_nanos() / period.as_nanos()).unwrap_or(u32::MAX);
        let start = since + period * periods;
        let into = now.saturating_duration_since(start);
        let faded = |from: Duration| {
            motion::EASE.at(into.saturating_sub(from).as_secs_f32() / fade.as_secs_f32())
        };
        // Shown, fading out by half way; hidden, fading in by the end.
        let (opacity, next) = if into < blink - fade {
            (1.0, start + blink - fade)
        } else if into < blink {
            (1.0 - faded(blink - fade), now + FRAME)
        } else if into < period - fade {
            (0.0, start + period - fade)
        } else {
            (faded(period - fade), now + FRAME)
        };
        self.opacity = opacity;
        Some(next.min(since + motion::CARET_BLINKS_FOR))
    }
}

/// What both fields keep of their line: focus, the selection, the scroll, the caret, and where
/// each letter boundary is, worked out once a layout.
#[derive(Default)]
struct Line {
    focus: Focus,
    selection: Selection,
    scroll: f32,
    blink: Blink,
    /// Whether it had focus at the last redraw: gaining it starts the caret.
    was_focused: bool,
    dragging: bool,
    modifiers: Modifiers,
    /// Every letter boundary's byte offset and x, from the text's start.
    xs: Vec<(usize, f32)>,
    placeholder: Plain,
    /// How wide the text area was at the last layout.
    width: f32,
}

impl Line {
    /// How far the text reaches: its rightmost boundary, the end of left-to-right text, the
    /// start of right-to-left.
    fn end(&self) -> f32 {
        self.xs.iter().map(|(_, x)| *x).fold(0.0, f32::max)
    }

    fn x_of(&self, at: usize) -> f32 {
        self.xs
            .iter()
            .find(|(offset, _)| *offset >= at)
            .or(self.xs.last())
            .map_or(0.0, |(_, x)| *x)
    }

    /// The boundary nearest an x from the text's start.
    fn hit(&self, x: f32) -> usize {
        self.xs
            .iter()
            .min_by(|(_, a), (_, b)| (a - x).abs().total_cmp(&(b - x).abs()))
            .map_or(0, |(at, _)| *at)
    }

    /// At a layout: the text area is this wide, and the caret stays in sight in it.
    fn laid_out(&mut self, width: f32) {
        self.width = width;
        self.follow_caret(width);
    }

    /// The caret or the selection moved, and the text didn't change: the caret stays in sight,
    /// with no new layout.
    fn caret_moved<Message>(&mut self, shell: &mut Shell<'_, Message>) {
        self.follow_caret(self.width);
        shell.request_redraw();
    }

    /// Scrolls so the caret stays in sight in an area this wide.
    fn follow_caret(&mut self, width: f32) {
        let caret = self.x_of(self.selection.head);
        if caret - self.scroll > width - size::CARET {
            self.scroll = caret - width + size::CARET;
        } else if caret < self.scroll {
            self.scroll = caret;
        }
        let end = self.end();
        self.scroll = self
            .scroll
            .min((end - width + size::CARET).max(0.0))
            .max(0.0);
    }

    /// A press in the text: the caret goes there, or the selection reaches there with Shift. A
    /// focus scope lets go of focus as the press reaches it, so it counts as focused if it was at
    /// the last redraw.
    fn press(&mut self, x: f32) {
        let at = self.hit(x + self.scroll);
        let focused = self.focus.is_focused() || self.was_focused;
        self.selection = if self.modifiers.shift() && focused {
            Selection {
                anchor: self.selection.anchor,
                head: at,
            }
        } else {
            Selection::caret(at)
        };
        self.focus.click();
        self.dragging = true;
        self.blink.restart();
    }

    /// The pointer moved while dragging: whether the selection changed.
    fn drag(&mut self, x: f32) -> bool {
        let at = self.hit(x + self.scroll);
        if at == self.selection.head {
            return false;
        }
        self.selection.head = at;
        self.blink.restart();
        true
    }

    /// A redraw: the caret's blink asks for the next. Focus just come, by Tab or a click, the
    /// caret shows at once, blinking.
    fn redraw<Message>(&mut self, now: Instant, shell: &mut Shell<'_, Message>) {
        let focused = self.focus.is_focused();
        if focused && !self.was_focused {
            self.blink.start(now);
        }
        self.was_focused = focused;
        if focused && let Some(next) = self.blink.at(now) {
            shell.request_redraw_at(next);
        }
    }
}

/// The text area inside a field's bounds, left of a toggle this wide, if any.
fn text_area(bounds: Rectangle, toggle: Option<f32>) -> Rectangle {
    let inner = bounds.shrink([0.0, PAD]);
    match toggle {
        None => inner,
        Some(width) => Rectangle {
            width: (bounds.width - PAD - space::XS - width - space::S).max(0.0),
            ..inner
        },
    }
}

fn interaction(enabled: bool, over: bool) -> Interaction {
    if !enabled {
        Interaction::Disabled
    } else if over {
        Interaction::Hovered
    } else {
        Interaction::Idle
    }
}

/// The edit a key asks for, if any. `latin` is the key's letter on a Latin layout, by where it
/// is: Ctrl+A selects all whatever layout's letter is on the A key.
fn key_edit<'k>(
    key: &Key,
    latin: Option<char>,
    text: Option<&'k str>,
    modifiers: Modifiers,
) -> Option<Edit<'k>> {
    let word = modifiers.command();
    let select = modifiers.shift();
    let go = |motion| Some(Edit::Move { motion, select });
    match key {
        Key::Named(Named::Backspace) => Some(Edit::DeleteBack { word }),
        Key::Named(Named::Delete) => Some(Edit::DeleteForward { word }),
        Key::Named(Named::ArrowLeft) if word => go(Motion::WordLeft),
        Key::Named(Named::ArrowRight) if word => go(Motion::WordRight),
        Key::Named(Named::ArrowLeft) => go(Motion::Left),
        Key::Named(Named::ArrowRight) => go(Motion::Right),
        Key::Named(Named::Home) => go(Motion::Home),
        Key::Named(Named::End) => go(Motion::End),
        Key::Character(_) if modifiers.command() && latin == Some('a') => Some(Edit::SelectAll),
        _ if modifiers.command() || modifiers.alt() => None,
        _ => text
            .filter(|text| !text.chars().any(char::is_control) && !text.is_empty())
            .map(Edit::Insert),
    }
}

/// Draws a field's box, and inside its text area the selection, the content and the caret.
#[allow(clippy::too_many_arguments)]
fn draw_line(
    renderer: &mut Renderer,
    theme: &Theme,
    line: &Line,
    bounds: Rectangle,
    area: Rectangle,
    look: &Look,
    focused: bool,
    preview: bool,
    empty: bool,
    content: impl FnOnce(&mut Renderer, f32, f32, Rectangle),
) {
    paint::boxed(renderer, bounds, look);
    let height = typography::INTERFACE.size * typography::INTERFACE.line_height;
    let top = bounds.center_y() - height / 2.0;
    let origin = area.x - line.scroll;
    let clip = Rectangle {
        x: area.x - size::CARET,
        width: area.width + 2.0 * size::CARET,
        ..bounds
    };
    // How much the caret shows: a preview's still at the end.
    let shown = |focused: bool, opacity: f32| if focused { opacity } else { 0.0 };
    let (caret_shown, selection) = if preview {
        let end = line.xs.last().map_or(0, |(at, _)| *at);
        (shown(focused, 1.0), Selection::caret(end))
    } else {
        (shown(focused, line.blink.opacity), line.selection)
    };
    // Kept to the text area, in a layer clipped to it, only when there's more than fits: each
    // layer costs a frame a text preparation of its own. What fits is inside without one, and
    // text is clipped where it's drawn.
    let overflows = !empty && (line.scroll > 0.0 || line.end() > area.width);
    let inside = |renderer: &mut Renderer| {
        if !selection.is_empty() && focused {
            // From the lesser x to the greater: right to left, the selection's start is right
            // of its end.
            let (a, b) = (
                line.x_of(selection.range().start),
                line.x_of(selection.range().end),
            );
            let (from, to) = (a.min(b), a.max(b));
            let marked = Look::solid(theme.selection(), 0.0);
            let bounds =
                Rectangle::new(Point::new(origin + from, top), Size::new(to - from, height));
            paint::boxed(renderer, bounds, &marked);
        }
        if empty {
            let ink = color(theme.placeholder());
            paint::fill(
                renderer,
                &line.placeholder,
                Point::new(area.x, top),
                ink,
                clip,
            );
        } else {
            content(renderer, origin, top, clip);
        }
        if caret_shown > 0.0 && selection.is_empty() {
            let x = origin + line.x_of(selection.head);
            let ink = theme.caret();
            // Round at both ends: its corners half its width.
            let caret = Look::solid(ink.opacity(ink.alpha() * caret_shown), size::CARET / 2.0);
            let bounds = Rectangle::new(Point::new(x, top), Size::new(size::CARET, height));
            paint::boxed(renderer, bounds, &caret);
        }
    };
    if overflows {
        renderer.with_layer(clip, inside);
    } else {
        inside(renderer);
    }
    if focused {
        paint::outline(renderer, theme, bounds, look.radius);
    }
}

/// Whether a press is outside a field: it lets go of focus, as a click elsewhere does, without
/// waiting for a focus scope.
pub(crate) fn pressed_outside(
    event: &iced::Event,
    cursor: mouse::Cursor,
    bounds: Rectangle,
) -> bool {
    matches!(
        event,
        iced::Event::Mouse(mouse::Event::ButtonPressed(_))
            | iced::Event::Touch(iced::touch::Event::FingerPressed { .. })
    ) && !cursor.is_over(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_caret_fades_out_and_in_then_stays() {
        let mut blink = Blink::default();
        assert_eq!(blink.at(Instant::now()), None, "no input, no blinking");
        blink.restart();
        let since = blink.since.unwrap();
        let (half, fade) = (motion::CARET_BLINK, CARET_FADE);
        // Shown, and nothing to draw till it starts to fade.
        assert_eq!(blink.at(since), Some(since + half - fade));
        assert_eq!(blink.opacity, 1.0);
        // Fading out: partly shown, and the next frame wanted.
        let fading = since + half - fade / 2;
        assert_eq!(blink.at(fading), Some(fading + FRAME));
        assert!(blink.opacity > 0.0 && blink.opacity < 1.0);
        // Hidden, till it starts to fade back in.
        assert_eq!(blink.at(since + half), Some(since + 2 * half - fade));
        assert_eq!(blink.opacity, 0.0);
        // The last redraw it asks for is when it stops, shown.
        let late = since + motion::CARET_BLINKS_FOR - Duration::from_millis(1);
        assert_eq!(blink.at(late), Some(since + motion::CARET_BLINKS_FOR));
        assert_eq!(blink.at(since + motion::CARET_BLINKS_FOR), None);
        assert_eq!(blink.opacity, 1.0);
    }

    #[test]
    fn focus_coming_shows_the_caret_blinking() {
        let mut line = Line::default();
        let mut messages: Vec<()> = Vec::new();
        let mut shell = Shell::new(&mut messages);
        let now = Instant::now();
        line.focus.click();
        line.redraw(now, &mut shell);
        assert_eq!(line.blink.opacity, 1.0, "focused, the caret shows at once");
        assert_eq!(line.blink.since, Some(now));
    }

    #[test]
    fn ctrl_a_selects_all_on_any_layout() {
        // A Russian layout's key where A is types ф: by where it is, it's still Ctrl+A.
        let key = Key::Character("ф".into());
        assert_eq!(
            key_edit(&key, Some('a'), None, Modifiers::CTRL),
            Some(Edit::SelectAll)
        );
        assert_eq!(key_edit(&key, None, None, Modifiers::CTRL), None);
    }

    #[test]
    fn keys_ask_for_edits() {
        let none = Modifiers::empty();
        let edit = |key: Key, text: Option<&'static str>, modifiers| {
            let latin = match &key {
                Key::Character(c) => c.chars().next(),
                Key::Named(_) | Key::Unidentified => None,
            };
            key_edit(&key, latin, text, modifiers)
        };
        assert_eq!(
            edit(Key::Named(Named::Backspace), None, Modifiers::CTRL),
            Some(Edit::DeleteBack { word: true })
        );
        assert_eq!(
            edit(Key::Named(Named::ArrowLeft), None, Modifiers::SHIFT),
            Some(Edit::Move {
                motion: Motion::Left,
                select: true
            })
        );
        assert_eq!(
            edit(Key::Named(Named::ArrowRight), None, Modifiers::CTRL),
            Some(Edit::Move {
                motion: Motion::WordRight,
                select: false
            })
        );
        assert_eq!(
            edit(Key::Character("a".into()), Some("a"), Modifiers::CTRL),
            Some(Edit::SelectAll)
        );
        assert_eq!(
            edit(Key::Character("é".into()), Some("é"), none),
            Some(Edit::Insert("é"))
        );
        assert_eq!(
            edit(Key::Character("q".into()), Some("q"), Modifiers::ALT),
            None
        );
        assert_eq!(
            edit(Key::Character("c".into()), Some("c"), Modifiers::CTRL),
            None
        );
        assert_eq!(edit(Key::Named(Named::Escape), Some("\u{1b}"), none), None);
        assert_eq!(edit(Key::Named(Named::F1), None, none), None);
    }

    #[test]
    fn a_line_finds_and_follows_its_caret() {
        let mut line = Line {
            xs: vec![(0, 0.0), (1, 10.0), (2, 20.0), (3, 30.0)],
            ..Line::default()
        };
        assert_eq!(line.x_of(2), 20.0);
        assert_eq!(line.x_of(9), 30.0);
        assert_eq!(line.hit(14.0), 1);
        assert_eq!(line.hit(16.0), 2);
        line.selection = Selection::caret(3);
        line.follow_caret(12.0);
        assert_eq!(line.scroll, 20.0);
        line.selection = Selection::caret(0);
        line.follow_caret(12.0);
        assert_eq!(line.scroll, 0.0);
        line.press(25.0);
        assert_eq!(line.selection, Selection::caret(2));
        assert!(line.focus.is_focused() && line.dragging);
        assert!(line.drag(5.0));
        assert!(!line.drag(4.0));
        assert_eq!(line.selection, Selection { anchor: 2, head: 0 });
    }

    #[test]
    fn areas_leave_room_for_the_toggle() {
        let bounds = Rectangle::new(Point::ORIGIN, Size::new(200.0, 36.0));
        assert_eq!(text_area(bounds, None).width, 200.0 - 2.0 * PAD);
        let area = text_area(bounds, Some(50.0));
        assert_eq!(area.width, 200.0 - PAD - 50.0 - space::XS - space::S);
        assert_eq!(interaction(false, true), Interaction::Disabled);
        assert_eq!(interaction(true, true), Interaction::Hovered);
        assert_eq!(interaction(true, false), Interaction::Idle);
    }
}
