// SPDX-License-Identifier: GPL-3.0-or-later

//! Checkboxes: a yes or no, such as swap or "I've saved my recovery key". Checked or not, the box
//! and its mark fade from the one to the other.

use iced::advanced::layout::{self, Layout};
use iced::advanced::text::{self, Renderer as _};
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, renderer};
use iced::keyboard::{self, Key, key::Named};

use iced::{Event, Length, Pixels, Point, Rectangle, Size, touch, window};

use crate::ui::tokens::{motion, size, space, typography};

use crate::ui::focus::Focus;
use crate::ui::input;
use crate::ui::paint::{self, Plain};
use crate::ui::theme::{Interaction, color};
use crate::ui::transition::Transition;
use crate::ui::{Element, Renderer, Theme};

/// A checkbox with a label. Without a message to send it is disabled.
pub fn checkbox<'a, Message>(label: impl Into<String>, checked: bool) -> Checkbox<'a, Message> {
    Checkbox {
        label: label.into(),
        checked,
        on_toggle: None,
        preview: None,
    }
}

/// A checkbox: a click on it or its label, or Space when it has focus, toggles it.
pub struct Checkbox<'a, Message> {
    label: String,
    checked: bool,
    on_toggle: Option<Box<dyn Fn(bool) -> Message + 'a>>,
    preview: Option<(Interaction, bool)>,
}

impl<'a, Message> Checkbox<'a, Message> {
    /// The message it sends with the new value.
    pub fn on_toggle(self, on_toggle: impl Fn(bool) -> Message + 'a) -> Self {
        Self {
            on_toggle: Some(Box::new(on_toggle)),
            ..self
        }
    }

    /// Draws it as if in this state, for the gallery.
    pub fn preview(self, interaction: Interaction, focused: bool) -> Self {
        Self {
            preview: Some((interaction, focused)),
            ..self
        }
    }

    fn interaction(&self, state: &State, over: bool) -> Interaction {
        match (&self.on_toggle, state.pressed, over) {
            (None, _, _) => Interaction::Disabled,
            (Some(_), true, _) => Interaction::Pressed,
            (Some(_), false, true) => Interaction::Hovered,
            (Some(_), false, false) => Interaction::Idle,
        }
    }

    fn toggle(&mut self, shell: &mut Shell<'_, Message>) {
        if let Some(on_toggle) = &self.on_toggle {
            shell.publish(on_toggle(!self.checked));
            // Taken at once, so a second key in the same batch of events starts from it, as the
            // app will.
            self.checked = !self.checked;
        }
    }
}

#[derive(Default)]
struct State {
    focus: Focus,
    pressed: bool,
    label: Plain,
    mark: Plain,
    drawn: Option<(Interaction, bool, bool)>,
    /// The value when last seen, and the fade to a new one under way: how checked it looks,
    /// from 0 to 1.
    seen: Option<bool>,
    fade: Option<Transition>,
}

impl<Message> Widget<Message, Theme, Renderer> for Checkbox<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Shrink, Length::Shrink)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let state = tree.state.downcast_mut::<State>();
        paint::shape(&mut state.label, &typography::INTERFACE, &self.label);
        // The check, iced's icon, shaped once rather than at every frame it shows.
        let mut glyph = [0; 4];
        state.mark.update(text::Text {
            content: Renderer::CHECKMARK_ICON.encode_utf8(&mut glyph),
            bounds: Size::new(size::CHECK, size::CHECK),
            size: Pixels(size::CHECK * 0.7),
            line_height: text::LineHeight::default(),
            font: Renderer::ICON_FONT,
            align_x: text::Alignment::Center,
            align_y: iced::alignment::Vertical::Center,
            shaping: text::Shaping::Basic,
            wrapping: text::Wrapping::None,
        });
        let width = size::CHECK + space::M + state.label.min_width().ceil();
        let size = limits.resolve(Length::Shrink, Length::Shrink, Size::new(width, size::ROW));
        layout::Node::new(size)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        _renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        if self.on_toggle.is_some() && self.preview.is_none() {
            let state = tree.state.downcast_mut::<State>();
            operation.focusable(None, layout.bounds(), &mut state.focus);
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
        state.focus.keep_if(self.on_toggle.is_some());
        // A new value, whatever set it: the box fades to it, from as far as a fade got.
        if let Some(seen) = state.seen
            && seen != self.checked
        {
            let (was, to) = (f32::from(u8::from(seen)), f32::from(u8::from(self.checked)));
            Transition::turn(&mut state.fade, was, to, motion::FAST, shell);
        }
        state.seen = Some(self.checked);
        if let Event::Window(window::Event::RedrawRequested(now)) = event {
            Transition::advance(&mut state.fade, *now, shell);
        }
        let over = cursor.is_over(layout.bounds());
        if self.on_toggle.is_some() {
            match event {
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerPressed { .. })
                    if over =>
                {
                    state.pressed = true;
                    state.focus.click();
                    shell.capture_event();
                }
                Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerLifted { .. })
                    if state.pressed =>
                {
                    state.pressed = false;
                    if over {
                        self.toggle(shell);
                    }
                    shell.capture_event();
                }
                Event::Touch(touch::Event::FingerLost { .. }) => state.pressed = false,
                // Space, not Enter: Enter is the form's, such as to go on. Once a press: held
                // down, it doesn't flicker.
                Event::Keyboard(keyboard::Event::KeyPressed {
                    key: Key::Named(Named::Space),
                    modifiers,
                    repeat: false,
                    ..
                }) if state.focus.is_focused() && input::plain(*modifiers) => {
                    state.focus.key();
                    self.toggle(shell);
                    shell.capture_event();
                }
                _ => {}
            }
        } else {
            state.pressed = false;
        }
        let now = (
            self.interaction(state, over),
            state.focus.is_shown(),
            self.checked,
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
        viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_ref::<State>();
        let bounds = layout.bounds();
        let (interaction, focused) = self.preview.unwrap_or((
            self.interaction(state, cursor.is_over(bounds)),
            state.focus.is_shown(),
        ));
        // How checked it looks: fading, part of the way; else its value.
        let checked = match (self.preview, state.fade) {
            (None, Some(fade)) => fade.at(),
            _ => f32::from(u8::from(self.checked)),
        };
        let (off, on) = (
            theme.check(false, interaction),
            theme.check(true, interaction),
        );
        let check = Rectangle::new(
            Point::new(bounds.x, bounds.center_y() - size::CHECK / 2.0),
            Size::new(size::CHECK, size::CHECK),
        );
        paint::boxed(renderer, check, &off.faded(1.0 - checked));
        paint::boxed(renderer, check, &on.faded(checked));
        let look = if checked >= 0.5 { on } else { off };
        let mark = on.ink.opacity(on.ink.alpha() * checked);
        if checked > 0.0 {
            // A paragraph's drawn from its top left: its middle at the box's, as text aligned to
            // the middle of a point is.
            let shaped = state.mark.min_bounds();
            let at = Point::new(
                check.center_x() - shaped.width / 2.0,
                check.center_y() - shaped.height / 2.0,
            );
            renderer.fill_paragraph(state.mark.raw(), at, color(mark), *viewport);
        }
        let label = state.label.min_bounds();
        let at = Point::new(
            check.x + size::CHECK + space::M,
            bounds.center_y() - label.height / 2.0,
        );
        let ink = theme.label(interaction);
        paint::fill(renderer, &state.label, at, color(ink), *viewport);
        if focused {
            paint::outline(renderer, theme, check, look.radius);
        }
    }

    fn mouse_interaction(
        &self,
        _tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        if self.on_toggle.is_some() && cursor.is_over(layout.bounds()) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, Message: 'a> From<Checkbox<'a, Message>> for Element<'a, Message> {
    fn from(checkbox: Checkbox<'a, Message>) -> Self {
        Element::new(checkbox)
    }
}
