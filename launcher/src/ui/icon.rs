// SPDX-License-Identifier: GPL-3.0-or-later

//! The window's own buttons, close and back, and the header bar's menu button: an icon in a
//! small square button. An icon is straight strokes in a 16 by 16 box, drawn at the icon stroke's
//! width with round caps.

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer;
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse};
use iced::keyboard::{self, Key, key::Named};
use iced::{Event, Length, Rectangle, Size, touch};

use crate::ui::tokens::{size, window};

use crate::ui::focus::Focus;
use crate::ui::input;
use crate::ui::paint;
use crate::ui::shape;
use crate::ui::theme::{Interaction, color};
use crate::ui::{Element, Renderer, Theme};

/// An icon: straight strokes, each from (x0, y0) to (x1, y1) in a 16 by 16 box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Icon {
    strokes: &'static [[f32; 4]],
}

impl Icon {
    /// A cross: close.
    pub const CLOSE: Icon = Icon {
        strokes: &window::CLOSE_CROSS,
    };
    /// A chevron pointing left: back.
    pub const BACK: Icon = Icon {
        strokes: &window::BACK_CHEVRON,
    };

    pub(crate) fn lines(&self) -> &[[f32; 4]] {
        self.strokes
    }
}

/// A window button showing an icon, which sends `on_press` when it's clicked, or when it has
/// focus and Enter or Space is pressed. The keys and the pointer work as a button's.
pub(crate) fn window_button<Message>(icon: Icon, on_press: Message) -> WindowButton<Message> {
    WindowButton {
        icon,
        on_press,
        danger: false,
        preview: None,
    }
}

/// A window button, close or back ([`close_button`](crate::ui::chrome::close_button),
/// [`back_button`](crate::ui::chrome::back_button)).
pub struct WindowButton<Message> {
    icon: Icon,
    on_press: Message,
    danger: bool,
    preview: Option<(Interaction, bool)>,
}

impl<Message> WindowButton<Message> {
    /// For what can't be undone, such as closing: red under the pointer.
    pub(crate) fn danger(self) -> Self {
        Self {
            danger: true,
            ..self
        }
    }

    /// Draws it as if in this state, whatever the pointer and focus do: for the gallery.
    pub fn preview(self, interaction: Interaction, focused: bool) -> Self {
        Self {
            preview: Some((interaction, focused)),
            ..self
        }
    }
}

#[derive(Default)]
struct Pressing {
    focus: Focus,
    pressed: bool,
    drawn: Option<(Interaction, bool)>,
    /// Its glow under the pointer, fading in and out.
    lit: crate::ui::transition::Lit,
}

impl Pressing {
    fn interaction(&self, over: bool) -> Interaction {
        if self.pressed {
            Interaction::Pressed
        } else if over {
            Interaction::Hovered
        } else {
            Interaction::Idle
        }
    }
}

impl<Message: Clone> Widget<Message, Theme, Renderer> for WindowButton<Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<Pressing>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(Pressing::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(
            Length::Fixed(window::BUTTON_SIZE),
            Length::Fixed(window::BUTTON_SIZE),
        )
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &Renderer,
        _limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(Size::new(window::BUTTON_SIZE, window::BUTTON_SIZE))
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        _renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        if self.preview.is_none() {
            let state = tree.state.downcast_mut::<Pressing>();
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
        if self.preview.is_some() {
            return;
        }
        let state = tree.state.downcast_mut::<Pressing>();
        let over = cursor.is_over(layout.bounds());
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
                    shell.publish(self.on_press.clone());
                }
                shell.capture_event();
            }
            Event::Touch(touch::Event::FingerLost { .. }) => state.pressed = false,
            // Once a press: a key held down doesn't close a window and then the next.
            Event::Keyboard(keyboard::Event::KeyPressed {
                key: Key::Named(Named::Enter | Named::Space),
                modifiers,
                repeat: false,
                ..
            }) if state.focus.is_focused() && input::plain(*modifiers) => {
                state.focus.key();
                shell.publish(self.on_press.clone());
                shell.capture_event();
            }
            _ => {}
        }
        let interaction = state.interaction(over);
        state.lit.frame(event, shell);
        state.lit.aim(interaction != Interaction::Idle, shell);
        let now = (interaction, state.focus.is_shown());
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
        let state = tree.state.downcast_ref::<Pressing>();
        let bounds = layout.bounds();
        let (interaction, focused) = self.preview.unwrap_or((
            state.interaction(cursor.is_over(bounds)),
            state.focus.is_shown(),
        ));
        let looks = |interaction| {
            if self.danger {
                theme.close_button(interaction)
            } else {
                theme.window_button(interaction)
            }
        };
        let mut look = looks(interaction);
        if self.preview.is_none() {
            // The glow fades in and out with the pointer.
            let lit = if interaction == Interaction::Pressed {
                Interaction::Pressed
            } else {
                Interaction::Hovered
            };
            look = look.lit(looks(lit).glow, state.lit.share());
        }
        paint::boxed(renderer, bounds, &look);
        shape::icon(
            renderer,
            bounds,
            self.icon.lines(),
            size::ICON_STROKE,
            color(look.ink),
        );
        if focused {
            let glow = theme.focus_glow().small();
            paint::outline_glowing(renderer, theme, bounds, look.radius, glow);
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
        if self.preview.is_none() && cursor.is_over(layout.bounds()) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, Message: Clone + 'a> From<WindowButton<Message>> for Element<'a, Message> {
    fn from(button: WindowButton<Message>) -> Self {
        Element::new(button)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_icons_are_the_window_buttons_glyphs() {
        assert_eq!(Icon::CLOSE.lines(), window::CLOSE_CROSS);
    }
}
