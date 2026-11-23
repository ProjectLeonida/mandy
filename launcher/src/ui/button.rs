// SPDX-License-Identifier: GPL-3.0-or-later

//! Buttons: Next and Back, Install, Erase the disk.

use std::time::Instant;

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, renderer};
use iced::keyboard::{self, Key, key::Named};
use iced::{Event, Length, Point, Rectangle, Size, touch};

use crate::ui::tokens::{size, space, typography};

use crate::ui::focus::Focus;
use crate::ui::input;
use crate::ui::paint::{self, Plain};
use crate::ui::theme::{Emphasis, Interaction, color};
use crate::ui::{Element, Renderer, Theme};

/// A button with a label. Without a message to send it is disabled.
pub fn button<Message>(label: impl Into<String>) -> Button<Message> {
    Button {
        label: label.into(),
        emphasis: Emphasis::Secondary,
        on_press: None,
        width: Length::Shrink,
        height: size::CONTROL,
        role: &typography::INTERFACE,
        least: 0.0,
        preview: None,
    }
}

/// A button: Enter or Space presses it when it has focus, as a click does.
pub struct Button<Message> {
    label: String,
    emphasis: Emphasis,
    on_press: Option<Message>,
    width: Length,
    height: f32,
    /// Its label's type role.
    role: &'static typography::Role,
    /// The least it's wide, in logical pixels.
    least: f32,
    preview: Option<(Interaction, bool)>,
}

/// A large button's height, and the least it's wide.
pub const LARGE_HEIGHT: f32 = 56.0;
pub const LARGE_LEAST: f32 = 200.0;

impl<Message> Button<Message> {
    /// Large, for the one thing a page is for, such as Play: taller and wider, its label a
    /// heading's.
    pub fn large(self) -> Self {
        Self {
            height: LARGE_HEIGHT,
            role: &typography::HEADING,
            least: LARGE_LEAST,
            ..self
        }
    }

    /// The message it sends when pressed.
    pub fn on_press(self, message: Message) -> Self {
        Self {
            on_press: Some(message),
            ..self
        }
    }

    /// The message it sends, if any; none disables it.
    pub fn on_press_maybe(self, message: Option<Message>) -> Self {
        Self {
            on_press: message,
            ..self
        }
    }

    /// How much it stands out: secondary unless set ([`Button::primary`], [`Button::danger`]).
    pub fn emphasis(self, emphasis: Emphasis) -> Self {
        Self { emphasis, ..self }
    }

    /// The step forward: Next, Install.
    pub fn primary(self) -> Self {
        self.emphasis(Emphasis::Primary)
    }

    /// What can't be undone.
    pub fn danger(self) -> Self {
        self.emphasis(Emphasis::Danger)
    }

    /// Its width: as wide as its words unless set.
    pub fn width(self, width: impl Into<Length>) -> Self {
        Self {
            width: width.into(),
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
struct State {
    focus: Focus,
    pressed: bool,
    label: Plain,
    drawn: Option<(Interaction, bool)>,
    /// Which button the state is: iced keeps a widget's state for whatever widget of its kind
    /// comes where it was, such as the Erase a wizard's next page puts where Next was.
    whose: Option<(Emphasis, String)>,
    /// When it came, as this button, or last fired, if it's a danger button.
    came: Option<Instant>,
    /// It fired, and the view hasn't been built again since: a second press in the same batch of
    /// events (a double-click while a frame was slow) doesn't fire it again.
    fired: bool,
}

/// How long a danger button takes no press from the pointer after it comes: a double-click on
/// Next, the second click landing on the Erase the next page puts there, presses nothing.
const DANGER_SETTLES: std::time::Duration = std::time::Duration::from_millis(500);

impl<Message: Clone> Button<Message> {
    /// It fired: not again until the view is built again; and a danger button lets go of focus
    /// and settles again, so a second Enter or click can't do what can't be undone twice, as
    /// when the next dialog asks the same of another disk.
    fn fired(&self, state: &mut State) {
        state.fired = true;
        // A press under way when Enter or Space fired it doesn't fire it again as it's released.
        state.pressed = false;
        if self.emphasis == Emphasis::Danger {
            state.focus.unfocus();
            state.came = Some(Instant::now());
        }
    }

    fn interaction(&self, state: &State, over: bool) -> Interaction {
        if self.on_press.is_none() {
            Interaction::Disabled
        } else if state.pressed {
            Interaction::Pressed
        } else if over {
            Interaction::Hovered
        } else {
            Interaction::Idle
        }
    }
}

impl<Message: Clone> Widget<Message, Theme, Renderer> for Button<Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    /// Made, it knows whose state it is, and when it came: iced makes a widget's state without
    /// calling [`diff`](Widget::diff), and builds the view again only after a message.
    fn state(&self) -> tree::State {
        tree::State::new(State {
            whose: Some((self.emphasis, self.label.clone())),
            came: Some(Instant::now()),
            ..State::default()
        })
    }

    /// A danger button where another button was doesn't get its focus or a press it had under
    /// way, and takes no pointer press for a moment, even where another danger button was: the
    /// Enter or the click that pressed what was there can't press it too. Any other button keeps
    /// them as its words change, as Connect becoming Connecting does.
    fn diff(&self, tree: &mut Tree) {
        let state = tree.state.downcast_mut::<State>();
        state.fired = false;
        let same = state
            .whose
            .as_ref()
            .is_some_and(|(emphasis, label)| *emphasis == self.emphasis && *label == self.label);
        if same {
            return;
        }
        if self.emphasis == Emphasis::Danger {
            state.focus.unfocus();
            state.pressed = false;
            state.came = Some(Instant::now());
        }
        state.whose = Some((self.emphasis, self.label.clone()));
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
        let state = tree.state.downcast_mut::<State>();
        paint::shape(&mut state.label, self.role, &self.label);
        let width = (state.label.min_width().ceil() + 2.0 * space::XL).max(self.least);
        let size = limits.resolve(self.width, Length::Shrink, Size::new(width, self.height));
        layout::Node::new(Size::new(size.width, self.height))
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        _renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        if self.on_press.is_some() && self.preview.is_none() {
            let state = tree.state.downcast_mut::<State>();
            if self.emphasis == Emphasis::Danger {
                operation.custom(None, layout.bounds(), &mut crate::ui::focus::Danger);
            }
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
        state.focus.keep_if(self.on_press.is_some());
        let over = cursor.is_over(layout.bounds());
        if let Some(message) = &self.on_press {
            match event {
                // A press elsewhere ends one this had under way whose release it never saw, as
                // when a dialog coming or going took it: a later release over it isn't a click.
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerPressed { .. })
                    if !over =>
                {
                    state.pressed = false;
                }
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerPressed { .. })
                    if over =>
                {
                    // A danger button just come takes the press, and does nothing with it.
                    let settling = self.emphasis == Emphasis::Danger
                        && state
                            .came
                            .is_some_and(|came| came.elapsed() < DANGER_SETTLES);
                    // A press it had under way and never saw end, as when a dialog coming or going
                    // took the release, doesn't count either.
                    state.pressed = !settling && !state.fired;
                    if state.pressed {
                        state.focus.click();
                    }
                    shell.capture_event();
                }
                Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerLifted { .. })
                    if state.pressed =>
                {
                    state.pressed = false;
                    if over && !state.fired {
                        shell.publish(message.clone());
                        self.fired(state);
                    }
                    shell.capture_event();
                }
                Event::Touch(touch::Event::FingerLost { .. }) => state.pressed = false,
                // Once a press: a key held down doesn't press it again and again.
                Event::Keyboard(keyboard::Event::KeyPressed {
                    key: Key::Named(Named::Enter | Named::Space),
                    modifiers,
                    repeat: false,
                    ..
                }) if state.focus.is_focused() && input::plain(*modifiers) => {
                    if !state.fired {
                        state.focus.key();
                        shell.publish(message.clone());
                        self.fired(state);
                    }
                    shell.capture_event();
                }
                _ => {}
            }
        } else {
            state.pressed = false;
        }
        let now = (self.interaction(state, over), state.focus.is_shown());
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
        let look = theme.button(self.emphasis, interaction);
        paint::boxed(renderer, bounds, &look);
        let label = state.label.min_bounds();
        let at = Point::new(
            bounds.center_x() - label.width / 2.0,
            bounds.center_y() - label.height / 2.0,
        );
        // Kept to the button: a label wider than a set width doesn't spill onto its neighbours.
        let clip = bounds.intersection(viewport).unwrap_or(bounds);
        paint::fill(renderer, &state.label, at, color(look.ink), clip);
        if focused {
            paint::outline(renderer, theme, bounds, look.radius);
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
        if self.on_press.is_some() && self.preview.is_none() && cursor.is_over(layout.bounds()) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, Message: Clone + 'a> From<Button<Message>> for Element<'a, Message> {
    fn from(button: Button<Message>) -> Self {
        Element::new(button)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::advanced::widget::operation::Focusable;

    fn element(button: Button<()>) -> Element<'static, ()> {
        button.into()
    }

    fn state(tree: &mut Tree) -> &mut State {
        tree.state.downcast_mut::<State>()
    }

    #[test]
    fn a_danger_button_made_where_another_was_gets_neither_its_focus_nor_its_press() {
        // As iced_winit has it: the dialog's buttons made, never diffed, then Continue focused
        // and pressed, and the app swaps Erase into its place.
        let mut tree = Tree::new(element(button("Continue").primary().on_press(())));
        Focusable::focus(&mut state(&mut tree).focus);
        state(&mut tree).pressed = true;
        tree.diff(element(button("Erase").danger().on_press(())));
        let erase = state(&mut tree);
        assert!(!erase.focus.is_focused());
        assert!(!erase.pressed);
        assert!(
            erase
                .came
                .is_some_and(|came| came.elapsed() < DANGER_SETTLES)
        );
    }

    #[test]
    fn a_danger_button_made_fresh_settles_before_a_click() {
        let mut tree = Tree::new(element(button("Erase").danger().on_press(())));
        assert!(state(&mut tree).came.is_some());
        assert!(state(&mut tree).whose.is_some());
    }

    #[test]
    fn a_danger_button_where_another_was_settles_again() {
        let mut tree = Tree::new(element(button("Erase sda").danger().on_press(())));
        state(&mut tree).came = Some(Instant::now() - DANGER_SETTLES * 2);
        tree.diff(element(button("Erase sdb").danger().on_press(())));
        assert!(
            state(&mut tree)
                .came
                .is_some_and(|came| came.elapsed() < DANGER_SETTLES)
        );
    }

    #[test]
    fn a_button_whose_words_change_keeps_its_focus() {
        let mut tree = Tree::new(element(button("Connect").primary().on_press(())));
        Focusable::focus(&mut state(&mut tree).focus);
        tree.diff(element(button("Connecting…").primary().on_press(())));
        assert!(state(&mut tree).focus.is_focused());
    }
}
