// SPDX-License-Identifier: GPL-3.0-or-later

//! The text field: a name, a host name, a list's filter. Its value is the app's.

use std::borrow::Cow;

use iced::advanced::layout::{self, Layout};
use iced::advanced::text::Renderer as _;
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, clipboard, input_method, mouse, renderer};
use iced::keyboard::{self, Key, key::Named};
use iced::{Event, Length, Point, Rectangle, Size, touch, window};
use unicode_segmentation::UnicodeSegmentation;

use crate::ui::tokens::{size, typography};

use super::{Line, draw_line, interaction, key_edit, pressed_outside, text_area};
use crate::ui::edit::{self, Buffer, Edit};
use crate::ui::input;
use crate::ui::paint::{self, Plain};
use crate::ui::theme::{Interaction, color};
use crate::ui::{Element, Renderer, Theme};

/// The longest value a text field takes by default, in bytes: a name, a host name.
pub const MAX_LEN: usize = 256;

/// A single-line text field showing `value`. Without a message for input it is disabled.
pub fn text_field<'a, Message>(
    placeholder: impl Into<String>,
    value: &'a str,
) -> TextField<'a, Message> {
    TextField {
        placeholder: placeholder.into(),
        value: Cow::Borrowed(value),
        on_input: None,
        on_submit: None,
        max_len: MAX_LEN,
        invalid: false,
        width: Length::Fill,
        preview: None,
    }
}

/// A text field. Ctrl+C, X and V copy, cut and paste.
pub struct TextField<'a, Message> {
    placeholder: String,
    /// The app's value, and after an edit the new one, so a second key in the same batch of
    /// events edits that, as the app will.
    value: Cow<'a, str>,
    on_input: Option<Box<dyn Fn(String) -> Message + 'a>>,
    on_submit: Option<Message>,
    max_len: usize,
    invalid: bool,
    width: Length,
    preview: Option<(Interaction, bool)>,
}

impl<'a, Message> TextField<'a, Message> {
    /// The message for each change, with the new value.
    pub fn on_input(self, on_input: impl Fn(String) -> Message + 'a) -> Self {
        Self {
            on_input: Some(Box::new(on_input)),
            ..self
        }
    }

    /// The message Enter sends.
    pub fn on_submit(self, message: Message) -> Self {
        Self {
            on_submit: Some(message),
            ..self
        }
    }

    /// The longest value it takes, in bytes: [`MAX_LEN`] unless set. A paste that would make it
    /// longer is cut to what fits, at a letter's end; a value already longer, from the app, can
    /// still be shortened.
    pub fn max_len(self, max_len: usize) -> Self {
        Self { max_len, ..self }
    }

    /// Shows the value as not acceptable, with a red line. Say why next to it.
    pub fn invalid(self, invalid: bool) -> Self {
        Self { invalid, ..self }
    }

    pub fn width(self, width: impl Into<Length>) -> Self {
        Self {
            width: width.into(),
            ..self
        }
    }

    /// Draws it as if in this state, for the gallery; focused, with a still caret at the end.
    pub fn preview(self, interaction: Interaction, focused: bool) -> Self {
        Self {
            preview: Some((interaction, focused)),
            ..self
        }
    }

    fn enabled(&self) -> bool {
        self.on_input.is_some()
    }

    /// Applies an edit, telling the app the new value.
    fn edit(&mut self, line: &mut Line, edit: Edit<'_>, shell: &mut Shell<'_, Message>) {
        let mut value = Limited::new(self.value.clone().into_owned(), self.max_len);
        if edit::apply(&mut value, &mut line.selection, edit, true) {
            if let Some(on_input) = &self.on_input {
                shell.publish(on_input(value.text.clone()));
            }
            self.value = Cow::Owned(value.text);
            // Laid out again, the new text's shaped before another key in the same batch.
            shell.invalidate_layout();
            shell.request_redraw();
        } else {
            line.caret_moved(shell);
        }
        line.blink.restart();
    }

    /// Ctrl+C, X, V and A: by where the key is, on any layout.
    fn clipboard(
        &mut self,
        line: &mut Line,
        key: &str,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
    ) -> bool {
        let selected = self
            .value
            .get(line.selection.range())
            .unwrap_or_default()
            .to_owned();
        match key {
            "c" if !selected.is_empty() => {
                clipboard.write(clipboard::Kind::Standard, selected);
                true
            }
            "x" if !selected.is_empty() => {
                clipboard.write(clipboard::Kind::Standard, selected);
                self.edit(line, Edit::DeleteBack { word: false }, shell);
                true
            }
            "v" => {
                if let Some(pasted) = clipboard.read(clipboard::Kind::Standard) {
                    self.edit(line, Edit::Insert(&pasted), shell);
                }
                true
            }
            "a" => {
                self.edit(line, Edit::SelectAll, shell);
                true
            }
            _ => false,
        }
    }
}

/// A value under edit, which grows no longer than its limit. One already longer, from the app,
/// can shrink but not grow: within an edit it may go back up to the length it had.
struct Limited {
    text: String,
    max: usize,
}

impl Limited {
    fn new(text: String, limit: usize) -> Self {
        let max = limit.max(text.len());
        Self { text, max }
    }
}

impl Buffer for Limited {
    fn text(&self) -> &str {
        &self.text
    }

    fn replace(&mut self, range: std::ops::Range<usize>, with: &str) -> bool {
        if self.text.len() - range.len() + with.len() > self.max {
            return false;
        }
        self.text.replace_range(range, with);
        true
    }

    fn room(&self) -> usize {
        self.max.saturating_sub(self.text.len())
    }
}

/// Every letter boundary's offset and x in a shaped line. A boundary is where its letter's glyph
/// begins: the glyph's left edge, or, right to left, as Hebrew and Arabic run, its right edge;
/// inside a glyph that covers several letters, part of the way along it; at the text's end, or a
/// letter with no glyph of its own, where the glyph before it ends. Glyphs come in the order
/// they're drawn, which right to left reverses, so each byte is first matched to its glyph.
pub(super) fn boundaries(paragraph: &Plain, text: &str) -> Vec<(usize, f32)> {
    let glyphs: Vec<(usize, usize, f32, f32, bool)> = paragraph
        .raw()
        .buffer()
        .layout_runs()
        .next()
        .map(|run| {
            run.glyphs
                .iter()
                .map(|g| (g.start, g.end, g.x, g.w, g.level.is_rtl()))
                .collect()
        })
        .unwrap_or_default();
    // Which glyph each byte is in.
    let mut owner: Vec<Option<usize>> = vec![None; text.len() + 1];
    for (at, &(start, end, ..)) in glyphs.iter().enumerate() {
        for byte in owner.iter_mut().take(end.min(text.len())).skip(start) {
            *byte = Some(at);
        }
    }
    let edge = |at: usize| match owner.get(at).copied().flatten() {
        Some(glyph) => {
            let (start, end, x, w, rtl) = glyphs[glyph];
            let along = w * (at - start) as f32 / (end - start).max(1) as f32;
            if rtl { x + w - along } else { x + along }
        }
        None => owner[..at.min(owner.len())]
            .iter()
            .rev()
            .flatten()
            .next()
            .map_or(0.0, |&glyph| {
                let (_, _, x, w, rtl) = glyphs[glyph];
                if rtl { x } else { x + w }
            }),
    };
    text.grapheme_indices(true)
        .map(|(at, _)| at)
        .chain(std::iter::once(text.len()))
        .map(|at| (at, edge(at)))
        .collect()
}

#[derive(Default)]
struct TextState {
    line: Line,
    value: Plain,
    /// What it showed when last drawn: how it looks, focus, how much the caret shows, the
    /// selection.
    drawn: Option<(Interaction, bool, u8, crate::ui::edit::Selection)>,
}

impl<Message: Clone> Widget<Message, Theme, Renderer> for TextField<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<TextState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(TextState::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(self.width, Length::Shrink)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let state = tree.state.downcast_mut::<TextState>();
        let role = &typography::INTERFACE;
        paint::shape(&mut state.line.placeholder, role, &self.placeholder);
        // The letter boundaries are found again only when the text is shaped again.
        if paint::shape(&mut state.value, role, &self.value) || state.line.xs.is_empty() {
            state.line.xs = boundaries(&state.value, &self.value);
        }
        state.line.selection = state.line.selection.clamp(&self.value);
        let size = limits.resolve(self.width, Length::Shrink, Size::new(0.0, size::CONTROL));
        let size = Size::new(size.width, size::CONTROL);
        let area = text_area(Rectangle::with_size(size), None);
        state.line.laid_out(area.width);
        layout::Node::new(size)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        _renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        if self.enabled() && self.preview.is_none() {
            let state = tree.state.downcast_mut::<TextState>();
            let area = text_area(layout.bounds(), None);
            operation.focusable(None, area, &mut state.line.focus);
        }
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        // A preview is a picture of a state: it takes no input.
        if self.preview.is_some() {
            return;
        }
        let state = tree.state.downcast_mut::<TextState>();
        // Disabled, it lets go of focus: no outline, keys no longer its, and no second focus
        // once it's enabled again.
        if !self.enabled() && state.line.focus.is_focused() {
            state.line.focus.unfocus();
            shell.request_redraw();
        }
        let line = &mut state.line;
        let bounds = layout.bounds();
        let area = text_area(bounds, None);
        if let Event::Keyboard(keyboard::Event::ModifiersChanged(modifiers)) = event {
            line.modifiers = *modifiers;
        }
        if let Event::Window(window::Event::RedrawRequested(now)) = event {
            line.redraw(*now, shell);
        }
        if pressed_outside(event, cursor, bounds) && line.focus.is_focused() {
            line.focus.unfocus();
            shell.request_redraw();
        }
        if self.enabled() {
            match event {
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerPressed { .. })
                    if cursor.is_over(bounds) =>
                {
                    line.press(cursor.position().map_or(0.0, |p| p.x) - area.x);
                    line.caret_moved(shell);
                    shell.capture_event();
                }
                Event::Mouse(mouse::Event::CursorMoved { position })
                | Event::Touch(touch::Event::FingerMoved { position, .. })
                    if line.dragging =>
                {
                    if line.drag(position.x - area.x) {
                        line.caret_moved(shell);
                    }
                }
                Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
                | Event::Touch(
                    touch::Event::FingerLifted { .. } | touch::Event::FingerLost { .. },
                ) => line.dragging = false,
                Event::Keyboard(keyboard::Event::KeyPressed {
                    key,
                    text,
                    modifiers,
                    physical_key,
                    repeat,
                    ..
                }) if line.focus.is_focused() => {
                    let handled = match key {
                        // Enter is the field's only with something to send, and sends it once a
                        // press; without, it's the form's or the app's.
                        Key::Named(Named::Enter) => match &self.on_submit {
                            Some(message) => {
                                if !repeat {
                                    shell.publish(message.clone());
                                }
                                true
                            }
                            None => false,
                        },
                        // By where the key is, not what it types: Ctrl+C copies on any layout.
                        Key::Character(_) if modifiers.command() => {
                            let latin = key.to_latin(*physical_key).map(String::from);
                            self.clipboard(line, latin.as_deref().unwrap_or(""), clipboard, shell)
                        }
                        _ => match key_edit(
                            key,
                            key.to_latin(*physical_key),
                            text.as_deref(),
                            *modifiers,
                        ) {
                            Some(edit) => {
                                self.edit(line, edit, shell);
                                true
                            }
                            None => false,
                        },
                    };
                    if handled {
                        line.focus.key();
                        shell.capture_event();
                    }
                }
                Event::InputMethod(input_method::Event::Commit(text))
                    if line.focus.is_focused() =>
                {
                    self.edit(line, Edit::Insert(text), shell);
                    shell.capture_event();
                }
                _ => {}
            }
        }
        let now = (
            interaction(self.enabled(), cursor.is_over(bounds)),
            line.focus.is_focused(),
            (line.blink.opacity * 255.0).round() as u8,
            line.selection,
        );
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
        _viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_ref::<TextState>();
        let bounds = layout.bounds();
        let (interaction, focused) = self.preview.unwrap_or((
            interaction(self.enabled(), cursor.is_over(bounds)),
            state.line.focus.is_focused(),
        ));
        let look = theme.field(interaction, focused, self.invalid);
        let area = text_area(bounds, None);
        let value = &state.value;
        draw_line(
            renderer,
            theme,
            &state.line,
            bounds,
            area,
            &look,
            focused,
            self.preview.is_some(),
            self.value.is_empty(),
            |renderer, origin, top, clip| {
                renderer.fill_paragraph(
                    value.raw(),
                    Point::new(origin, top),
                    color(look.ink),
                    clip,
                );
            },
        );
    }

    fn mouse_interaction(
        &self,
        _tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        if self.enabled() && cursor.is_over(layout.bounds()) {
            mouse::Interaction::Text
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, Message: Clone + 'a> From<TextField<'a, Message>> for Element<'a, Message> {
    fn from(field: TextField<'a, Message>) -> Self {
        Element::new(field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::edit::Selection;

    #[test]
    fn boundaries_come_in_one_pass_and_grow() {
        let text = "Ae\u{301} ffi 🙂";
        let mut plain = Plain::default();
        paint::shape(&mut plain, &typography::INTERFACE, text);
        let xs = boundaries(&plain, text);
        let offsets: Vec<usize> = xs.iter().map(|(at, _)| *at).collect();
        let mut expected: Vec<usize> = text.grapheme_indices(true).map(|(at, _)| at).collect();
        expected.push(text.len());
        assert_eq!(offsets, expected);
        assert_eq!(xs[0].1, 0.0);
        assert!(xs.windows(2).all(|pair| pair[0].1 <= pair[1].1), "{xs:?}");
        assert!((xs.last().unwrap().1 - plain.min_width()).abs() < 1.0);
        assert_eq!(boundaries(&Plain::default(), ""), [(0, 0.0)]);
    }

    #[test]
    #[ignore = "with stock iced the system's Hebrew font shapes it, and an unbounded right-to-left \
                line lays out at infinity; the original runs it with only its own fonts (an \
                iced_graphics fork)"]
    fn right_to_left_the_first_letter_is_at_the_right() {
        // Hebrew runs right to left: its first letter's boundary is the line's right edge, and
        // its end the left.
        crate::ui::font::load();
        let text = "\u{5d0}\u{5d1}\u{5d2}";
        let mut plain = Plain::default();
        paint::shape(&mut plain, &typography::INTERFACE, text);
        let xs = boundaries(&plain, text);
        assert_eq!(
            xs.iter().map(|(at, _)| *at).collect::<Vec<_>>(),
            [0, 2, 4, 6]
        );
        assert!(
            xs.windows(2).all(|pair| pair[0].1 > pair[1].1),
            "right to left: {xs:?}"
        );
        assert!(
            xs[3].1.abs() < 1e-3 && (xs[0].1 - plain.min_width()).abs() < 1.0,
            "{xs:?}"
        );
    }

    #[test]
    fn a_value_over_the_limit_can_shrink_but_not_grow() {
        // Each edit starts from the app's value, as the field's do.
        let edit = |text: &str, mut selection: Selection, edit: Edit<'_>| {
            let mut value = Limited::new(text.to_owned(), 5);
            let changed = edit::apply(&mut value, &mut selection, edit, true);
            (changed, value.text)
        };
        let long = "a".repeat(10);
        let back = Edit::DeleteBack { word: false };
        assert_eq!(
            edit(&long, Selection::caret(10), back),
            (true, "a".repeat(9))
        );
        let grown = edit(&long, Selection::caret(10), Edit::Insert("b"));
        assert_eq!(grown, (false, long.clone()));
        // Two letters for one shortens it, so it's taken.
        let pair = Selection { anchor: 0, head: 2 };
        let shorter = edit(&long, pair, Edit::Insert("b"));
        assert_eq!(shorter, (true, format!("b{}", "a".repeat(8))));
    }

    #[test]
    fn a_limited_value_takes_what_fits() {
        let mut value = Limited::new("abc".into(), 5);
        let mut selection = Selection::caret(3);
        assert!(edit::apply(
            &mut value,
            &mut selection,
            Edit::Insert("def"),
            true
        ));
        assert_eq!(value.text, "abcde", "cut to what fits");
        assert_eq!(value.room(), 0);
        assert!(
            !edit::apply(&mut value, &mut selection, Edit::Insert("x"), true),
            "full, nothing more"
        );
        selection = Selection { anchor: 0, head: 5 };
        assert!(edit::apply(
            &mut value,
            &mut selection,
            Edit::Insert("xyz"),
            true
        ));
        assert_eq!(value.text, "xyz");
    }
}
