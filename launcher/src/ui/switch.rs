// SPDX-License-Identifier: GPL-3.0-or-later

//! Switches: on or off, as a setting's row ends with one, such as "Scroll on output". The knob
//! slides from one end to the other, and can be dragged there.

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, renderer};
use iced::keyboard::{self, Key, key::Named};
use iced::{Event, Length, Point, Rectangle, Size, touch, window};

use crate::ui::tokens::{motion, size};

use crate::ui::focus::Focus;
use crate::ui::input;
use crate::ui::paint;
use crate::ui::theme::{Interaction, Look};
use crate::ui::transition::Transition;
use crate::ui::{Element, Renderer, Theme};

/// Room round the knob inside the track.
const INSET: f32 = 3.0;

/// A switch, on or off. Without a message to send it is disabled.
pub fn switch<'a, Message>(on: bool) -> Switch<'a, Message> {
    Switch {
        on,
        on_toggle: None,
        preview: None,
    }
}

/// A switch: a click on it, or Space when it has focus, turns it over, its knob sliding
/// across; or the knob is dragged across, by the pointer or a finger, and let go nearer one end.
pub struct Switch<'a, Message> {
    on: bool,
    on_toggle: Option<Box<dyn Fn(bool) -> Message + 'a>>,
    preview: Option<(Interaction, bool)>,
}

impl<'a, Message> Switch<'a, Message> {
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
        match (&self.on_toggle, state.drag.is_some(), over) {
            (None, _, _) => Interaction::Disabled,
            (Some(_), true, _) => Interaction::Pressed,
            (Some(_), false, true) => Interaction::Hovered,
            (Some(_), false, false) => Interaction::Idle,
        }
    }

    /// Turns it to `on`, if that's a change, and sends it.
    fn set(&mut self, on: bool, shell: &mut Shell<'_, Message>) {
        if let Some(on_toggle) = &self.on_toggle
            && on != self.on
        {
            shell.publish(on_toggle(on));
            self.on = on;
        }
    }
}

/// Where the knob is in a track, `at` the share of the way from its start, off, to its end, on.
pub(crate) fn knob(track: Rectangle, at: f32) -> Rectangle {
    let diameter = track.height - 2.0 * INSET;
    let travel = track.width - 2.0 * INSET - diameter;
    Rectangle::new(
        Point::new(
            track.x + INSET + travel * at.clamp(0.0, 1.0),
            track.y + INSET,
        ),
        Size::new(diameter, diameter),
    )
}

/// How far the knob travels across a track.
fn travel(track: Rectangle) -> f32 {
    track.width - 2.0 * INSET - (track.height - 2.0 * INSET)
}

/// A press on the knob's track: where it began, where the knob was, and whether it has moved far
/// enough to be a drag.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Drag {
    from_x: f32,
    from_at: f32,
    dragged: bool,
}

#[derive(Default)]
struct State {
    focus: Focus,
    /// Where the knob is drawn, from 0, off, to 1, on; and the value it was last drawn for.
    at: f32,
    seen: Option<bool>,
    slide: Option<Transition>,
    drag: Option<Drag>,
    drawn: Option<(Interaction, bool, bool)>,
}

impl State {
    /// The knob slides to `on`'s end from where it is, or with motion reduced, jumps there.
    fn slide_to(&mut self, on: bool) {
        let to = f32::from(u8::from(on));
        self.slide = Transition::start(self.at, to, motion::FAST);
        if self.slide.is_none() {
            self.at = to;
        }
    }
}

/// Where an event puts the pointer or the finger across, if it says.
fn across(event: &Event, cursor: mouse::Cursor) -> Option<f32> {
    match event {
        Event::Touch(
            touch::Event::FingerPressed { position, .. }
            | touch::Event::FingerMoved { position, .. },
        ) => Some(position.x),
        Event::Mouse(mouse::Event::CursorMoved { position }) => Some(position.x),
        _ => cursor.position().map(|position| position.x),
    }
}

impl<Message> Widget<Message, Theme, Renderer> for Switch<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(
            Length::Fixed(size::SWITCH_WIDTH),
            Length::Fixed(size::SWITCH_HEIGHT),
        )
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &Renderer,
        _limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(Size::new(size::SWITCH_WIDTH, size::SWITCH_HEIGHT))
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
        if self.preview.is_some() {
            return;
        }
        let track = layout.bounds();
        let state = tree.state.downcast_mut::<State>();
        state.focus.keep_if(self.on_toggle.is_some());
        // The value changed from outside, as a row's click turns it: the knob slides there.
        match state.seen {
            None => state.at = f32::from(u8::from(self.on)),
            Some(seen) if seen != self.on && state.drag.is_none() => state.slide_to(self.on),
            Some(_) => {}
        }
        state.seen = Some(self.on);
        let over = cursor.is_over(track);
        if self.on_toggle.is_some() {
            match event {
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerPressed { .. })
                    if over =>
                {
                    if let Some(x) = across(event, cursor) {
                        state.drag = Some(Drag {
                            from_x: x,
                            from_at: state.at,
                            dragged: false,
                        });
                    }
                    state.focus.click();
                    shell.capture_event();
                }
                Event::Mouse(mouse::Event::CursorMoved { .. })
                | Event::Touch(touch::Event::FingerMoved { .. }) => {
                    if let Some(drag) = &mut state.drag
                        && let Some(x) = across(event, cursor)
                    {
                        // Further than a slip, it drags the knob rather than clicks; the drag is
                        // the switch's, not a scroll area's round it.
                        let moved = x - drag.from_x;
                        drag.dragged |= moved.abs() > size::SLIP;
                        if drag.dragged {
                            state.slide = None;
                            state.at = (drag.from_at + moved / travel(track)).clamp(0.0, 1.0);
                            shell.request_redraw();
                            shell.capture_event();
                        }
                    }
                }
                Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerLifted { .. }) => {
                    if let Some(drag) = state.drag.take() {
                        // Dragged, it goes to the nearer end; clicked, it turns over.
                        let on = if drag.dragged {
                            state.at >= 0.5
                        } else if over {
                            !self.on
                        } else {
                            self.on
                        };
                        state.slide_to(on);
                        self.set(on, shell);
                        let state = tree.state.downcast_mut::<State>();
                        state.seen = Some(self.on);
                        shell.request_redraw();
                        shell.capture_event();
                    }
                }
                Event::Touch(touch::Event::FingerLost { .. }) => {
                    if state.drag.take().is_some() {
                        state.slide_to(self.on);
                        shell.request_redraw();
                    }
                }
                // Space, not Enter, as a checkbox: Enter is the form's, such as to go on. Once a
                // press: held down, it doesn't flicker.
                Event::Keyboard(keyboard::Event::KeyPressed {
                    key: Key::Named(Named::Space),
                    modifiers,
                    repeat: false,
                    ..
                }) if state.focus.is_focused() && input::plain(*modifiers) => {
                    state.focus.key();
                    state.slide_to(!self.on);
                    self.set(!self.on, shell);
                    let state = tree.state.downcast_mut::<State>();
                    state.seen = Some(self.on);
                    shell.request_redraw();
                    shell.capture_event();
                }
                _ => {}
            }
        } else {
            state.drag = None;
        }
        let state = tree.state.downcast_mut::<State>();
        if let Event::Window(window::Event::RedrawRequested(now)) = event
            && let Some(slide) = &mut state.slide
        {
            let moving = slide.frame(*now);
            state.at = slide.at();
            if moving {
                shell.request_redraw();
            } else {
                state.slide = None;
            }
        }
        let now = (
            self.interaction(state, over),
            state.focus.is_shown(),
            self.on,
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
        let state = tree.state.downcast_ref::<State>();
        let track = layout.bounds();
        let (interaction, focused) = self.preview.unwrap_or((
            self.interaction(state, cursor.is_over(track)),
            state.focus.is_shown(),
        ));
        // Where the knob is: where it slides or is dragged to, or, before it's seen a change,
        // where its value puts it.
        let at = match (self.preview, state.seen) {
            (None, Some(_)) => state.at,
            _ => f32::from(u8::from(self.on)),
        };
        // The off and on looks crossfade as the knob crosses: at either end, one alone.
        let (off, on) = (
            theme.switch(false, interaction),
            theme.switch(true, interaction),
        );
        paint::boxed(renderer, track, &off.track.faded(1.0 - at));
        paint::boxed(renderer, track, &on.track.faded(at));
        let knob = knob(track, at);
        let radius = knob.height / 2.0;
        paint::boxed(renderer, knob, &Look::solid(off.knob, radius));
        paint::boxed(
            renderer,
            knob,
            &Look::solid(on.knob.opacity(on.knob.alpha() * at), radius),
        );
        if focused {
            paint::outline(renderer, theme, track, off.track.radius);
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
        if self.on_toggle.is_none() || self.preview.is_some() {
            mouse::Interaction::default()
        } else if state.drag.is_some_and(|drag| drag.dragged) {
            mouse::Interaction::Grabbing
        } else if cursor.is_over(layout.bounds()) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, Message: 'a> From<Switch<'a, Message>> for Element<'a, Message> {
    fn from(switch: Switch<'a, Message>) -> Self {
        Element::new(switch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_sits_at_the_start_off_and_the_end_on() {
        let track = Rectangle::new(Point::new(0.0, 0.0), Size::new(40.0, 22.0));
        let off = knob(track, 0.0);
        let on = knob(track, 1.0);
        assert_eq!(
            off,
            Rectangle::new(Point::new(3.0, 3.0), Size::new(16.0, 16.0))
        );
        assert_eq!(on.x, 21.0);
        assert_eq!((on.y, on.width), (3.0, 16.0));
        // Halfway, halfway across; never past either end.
        assert_eq!(knob(track, 0.5).x, 12.0);
        assert_eq!(knob(track, 2.0), on);
        assert_eq!(travel(track), 18.0);
    }

    #[test]
    fn a_faded_look_keeps_its_shape_and_loses_its_color() {
        let cyan = crate::ui::tokens::Palette::SUNSET_DARK.accent(crate::ui::tokens::Accent::Cyan);
        let look = Look::solid(cyan, 4.0);
        let half = look.faded(0.5);
        assert_eq!(half.radius, 4.0);
        assert_eq!(half.fill.map(crate::ui::tokens::Color::alpha), Some(0.5));
        assert_eq!(look.faded(1.0), look);
    }
}
