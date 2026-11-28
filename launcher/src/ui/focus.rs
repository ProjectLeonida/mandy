// SPDX-License-Identifier: GPL-3.0-or-later

//! The focus model. Every widget that takes input can hold keyboard focus,
//! and the focused one takes the keys and shows the focus outline. A [`scope`] around a window's
//! content moves focus: Tab to the next widget and Shift+Tab to the previous, in the order they're
//! laid out, round the end; a click focuses what it hits and nothing else. Focus moved by the
//! keyboard into a widget scrolled out of sight scrolls it into view.
//!
//! The scope also remembers which keys' presses a widget took, and takes their releases too,
//! wherever focus has gone by then: a key held while a secret field hands its secret over and the
//! page changes never reaches the app's keyboard subscriptions. Nor do its repeats, should focus
//! have moved somewhere that ignores them. A release can go astray (the window loses focus while
//! the key is down), so a fresh press nothing takes hands that key back to the app.
//!
//! And it keeps touches in order: one finger at a time, so a second finger's lift can't finish
//! the first's press; and once a finger lifts, the content sees no pointer until a mouse moves
//! it, so what was tapped isn't left hovered.

use std::cell::{Cell, RefCell};

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::operation::{Focusable, Operation, Scrollable};
use iced::advanced::widget::{Id, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, overlay, renderer};
use iced::keyboard::{self, Key, key::Named, key::Physical};
use iced::widget::scrollable::AbsoluteOffset;
use iced::{Event, Length, Rectangle, Size, Vector, touch};

use crate::ui::{Element, Renderer, Theme};

/// Whether a widget holds keyboard focus: part of every focusable widget's state. Focus from the
/// keyboard shows the outline at once; focus from a click shows it once a key is used, as a
/// browser's `:focus-visible` does. Text fields show it whenever they're focused.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Focus {
    focused: bool,
    shown: bool,
}

impl Focus {
    pub fn is_focused(&self) -> bool {
        self.focused
    }

    /// Whether the outline shows.
    pub fn is_shown(&self) -> bool {
        self.focused && self.shown
    }

    /// Focus from a click.
    pub(crate) fn click(&mut self) {
        self.focused = true;
        self.shown = false;
    }

    /// A key was used on the focused widget.
    pub(crate) fn key(&mut self) {
        self.shown = self.focused;
    }

    /// Lets go of focus, as a click elsewhere does.
    pub(crate) fn unfocus(&mut self) {
        *self = Self::default();
    }

    /// Lets go of focus unless the widget takes it now: one disabled while it has focus lets go,
    /// as nothing can move focus off a widget that no longer counts, and it would be a second
    /// focus once it's enabled again.
    pub(crate) fn keep_if(&mut self, takes: bool) {
        if !takes {
            self.unfocus();
        }
    }
}

impl Focusable for Focus {
    fn is_focused(&self) -> bool {
        self.focused
    }

    fn focus(&mut self) {
        self.focused = true;
        self.shown = true;
    }

    fn unfocus(&mut self) {
        Focus::unfocus(self);
    }
}

/// Moves keyboard focus among the widgets in its content. Put one around a window's content.
pub fn scope<'a, Message: 'a>(content: impl Into<Element<'a, Message>>) -> Scope<'a, Message> {
    Scope {
        content: content.into(),
    }
}

pub struct Scope<'a, Message> {
    content: Element<'a, Message>,
}

/// What a scope keeps: the keys whose presses a widget took, and what it knows of touches. Both
/// are shared with what floats over the content, which sees every event first.
#[derive(Debug, Default)]
struct ScopeState {
    held: RefCell<Held>,
    touches: Cell<Touches>,
}

/// What a scope knows of touches: the finger that's down, if any, and whether a finger has
/// lifted since a mouse last moved the pointer. A touch moves the pointer to the finger, and
/// nothing takes it away when the finger lifts (winit's), so what was tapped would stay hovered,
/// a sheet's tapped row lit: once a finger lifts, the content sees no pointer until a mouse moves
/// it. Every change is the same however often it's made, as an event can reach both what floats
/// over the content and the content.
#[derive(Debug, Clone, Copy, Default)]
struct Touches {
    finger: Option<touch::Finger>,
    lifted: bool,
}

impl Touches {
    /// Before an event is seen: a mouse moving, or a finger going down, brings the pointer back.
    fn before(&mut self, event: &Event) {
        if let Event::Mouse(mouse::Event::CursorMoved { .. })
        | Event::Touch(touch::Event::FingerPressed { .. }) = event
        {
            self.lifted = false;
        }
    }

    /// Whether an event goes on to the content: while a finger is down, another finger's don't,
    /// so a second finger's lift doesn't finish the first's press.
    fn admits(self, event: &Event) -> bool {
        match (event, self.finger) {
            (
                Event::Touch(
                    touch::Event::FingerPressed { id, .. }
                    | touch::Event::FingerMoved { id, .. }
                    | touch::Event::FingerLifted { id, .. }
                    | touch::Event::FingerLost { id, .. },
                ),
                Some(finger),
            ) => *id == finger,
            _ => true,
        }
    }

    /// The pointer the content sees, with an event or to draw: none once a finger has lifted;
    /// a touch's own events are where the finger is.
    fn cursor(self, event: Option<&Event>, cursor: mouse::Cursor) -> mouse::Cursor {
        if self.lifted && !matches!(event, Some(Event::Touch(_))) {
            mouse::Cursor::Unavailable
        } else {
            cursor
        }
    }

    /// After an event is seen: the finger that went down is the one, until it lifts.
    fn after(&mut self, event: &Event) {
        match event {
            Event::Touch(touch::Event::FingerPressed { id, .. }) => {
                self.finger.get_or_insert(*id);
            }
            Event::Touch(
                touch::Event::FingerLifted { id, .. } | touch::Event::FingerLost { id, .. },
            ) if self.finger.is_none_or(|finger| finger == *id) => {
                self.finger = None;
                self.lifted = true;
            }
            _ => {}
        }
    }
}

/// The keys whose presses a widget took, still held, and who took each.
#[derive(Debug, Default)]
struct Held(Vec<(Physical, Taker)>);

/// Who took a key's press: the widget focused then, by its focus's place in memory, which stays
/// as long as the widget does wherever focus order puts it (none if nothing was focused), or
/// what floats over the content, such as a menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Taker {
    Focused(Option<usize>),
    Overlay,
}

impl Held {
    /// A key pressed, not repeated: who took it now is who its repeats belong to, whatever an
    /// earlier press whose release went astray left.
    fn press(&mut self, key: &Physical, taker: Taker) {
        self.0.retain(|(held, _)| held != key);
        self.0.push((*key, taker));
    }

    fn taker(&self, key: &Physical) -> Option<Taker> {
        self.0
            .iter()
            .find_map(|(held, taker)| (held == key).then_some(*taker))
    }

    fn holds(&self, key: &Physical) -> bool {
        self.taker(key).is_some()
    }

    /// Whether the key's press was taken; it's let go of either way.
    fn release(&mut self, key: &Physical) -> bool {
        let before = self.0.len();
        self.0.retain(|(held, _)| held != key);
        self.0.len() != before
    }
}

/// Counts the focusable widgets, and finds the focused one.
#[derive(Default)]
struct Census {
    total: usize,
    focused: Option<usize>,
    bounds: Option<Rectangle>,
    /// The focused widget's own focus, by where it is in memory: the widget for as long as it
    /// lives, wherever focus order puts it.
    whose: Option<usize>,
}

impl Operation for Census {
    fn traverse(&mut self, operate: &mut dyn FnMut(&mut dyn Operation)) {
        operate(self);
    }

    fn focusable(&mut self, _id: Option<&Id>, bounds: Rectangle, state: &mut dyn Focusable) {
        if state.is_focused() {
            self.focused = Some(self.total);
            self.bounds = Some(bounds);
            self.whose = Some(std::ptr::from_mut(state).cast::<()>() as usize);
        }
        self.total += 1;
    }
}

/// Said of the focusable widget that comes next, through [`Operation::custom`]: what it does
/// can't be undone, such as a danger button's Erase.
pub(crate) struct Danger;

/// Focuses the widget at an index, or none, and unfocuses the others.
struct FocusOnly {
    target: Option<usize>,
    at: usize,
    /// Where the widget it focused is.
    bounds: Option<Rectangle>,
}

impl Operation for FocusOnly {
    fn traverse(&mut self, operate: &mut dyn FnMut(&mut dyn Operation)) {
        operate(self);
    }

    fn focusable(&mut self, _id: Option<&Id>, bounds: Rectangle, state: &mut dyn Focusable) {
        if self.target == Some(self.at) {
            state.focus();
            self.bounds = Some(bounds);
        } else {
            state.unfocus();
        }
        self.at += 1;
    }
}

/// Scrolls the scroll areas that hold these bounds until they're in sight.
struct Reveal {
    target: Rectangle,
}

/// Room left around a widget scrolled into view, for its focus outline.
const MARGIN: f32 = crate::ui::tokens::space::M;

impl Operation for Reveal {
    fn traverse(&mut self, operate: &mut dyn FnMut(&mut dyn Operation)) {
        operate(self);
    }

    fn scrollable(
        &mut self,
        _id: Option<&Id>,
        bounds: Rectangle,
        content_bounds: Rectangle,
        translation: Vector,
        state: &mut dyn Scrollable,
    ) {
        // Not another scroll area's, nor the area itself, focused as a whole: it isn't moved to
        // show itself.
        if !content_bounds.contains(self.target.center()) || bounds == self.target {
            return;
        }
        // Just enough to bring it in sight with the margin round it; one too long to fit shows
        // its start, where its outline begins, not its end.
        let offset = |start: f32, length: f32, target: f32, target_length: f32, scrolled: f32| {
            let shown = start + scrolled;
            let fits = target_length + 2.0 * MARGIN <= length;
            if target - MARGIN < shown || (!fits && target - MARGIN > shown) {
                Some(scrolled + target - MARGIN - shown)
            } else if fits && target + target_length + MARGIN > shown + length {
                Some(scrolled + target + target_length + MARGIN - shown - length)
            } else {
                None
            }
        };
        let t = self.target;
        let x = offset(bounds.x, bounds.width, t.x, t.width, translation.x);
        let y = offset(bounds.y, bounds.height, t.y, t.height, translation.y);
        if x.is_some() || y.is_some() {
            state.scroll_to(AbsoluteOffset {
                x: x.map(|x| x.max(0.0)),
                y: y.map(|y| y.max(0.0)),
            });
        }
    }
}

/// Where Tab moves focus from `focused` among `total`: forwards round the end, or backwards.
pub(crate) fn next(focused: Option<usize>, total: usize, forwards: bool) -> Option<usize> {
    if total == 0 {
        return None;
    }
    Some(match (focused, forwards) {
        (None, true) => 0,
        (None, false) => total - 1,
        (Some(at), true) => (at + 1) % total,
        (Some(at), false) => (at + total - 1) % total,
    })
}

impl<Message> Scope<'_, Message> {
    fn census(&mut self, tree: &mut Tree, layout: Layout<'_>, renderer: &Renderer) -> Census {
        let mut census = Census::default();
        self.content
            .as_widget_mut()
            .operate(tree, layout, renderer, &mut census);
        census
    }

    /// Focuses the widget at `target`, and only it: where it is, if there's one.
    fn focus_only(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        target: Option<usize>,
    ) -> Option<Rectangle> {
        let mut operation = FocusOnly {
            target,
            at: 0,
            bounds: None,
        };
        self.content
            .as_widget_mut()
            .operate(tree, layout, renderer, &mut operation);
        operation.bounds
    }
}

impl<Message> Widget<Message, Theme, Renderer> for Scope<'_, Message> {
    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }

    fn size_hint(&self) -> Size<Length> {
        self.content.as_widget().size_hint()
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.content));
    }

    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<ScopeState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(ScopeState::default())
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.content
            .as_widget_mut()
            .layout(&mut tree.children[0], renderer, limits)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        self.content
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, operation);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        let Tree {
            state, children, ..
        } = tree;
        let ScopeState { held, touches } = state.downcast_mut::<ScopeState>();
        let held = held.get_mut();
        let content = &mut children[0];
        let mut touched = touches.get();
        touched.before(event);
        if !touched.admits(event) {
            touches.set(touched);
            shell.capture_event();
            return;
        }
        let cursor = touched.cursor(Some(event), cursor);
        touched.after(event);
        if touched.lifted && !touches.get().lifted {
            // What was under the finger is no longer hovered.
            shell.request_redraw();
        }
        touches.set(touched);
        // A release whose press a widget took is taken before anything else sees it.
        if let Event::Keyboard(keyboard::Event::KeyReleased { physical_key, .. }) = event
            && held.release(physical_key)
        {
            shell.capture_event();
        }
        match event {
            Event::Keyboard(keyboard::Event::KeyPressed {
                key: Key::Named(Named::Tab),
                modifiers,
                physical_key,
                ..
            }) if !(modifiers.control() || modifiers.alt() || modifiers.logo()) => {
                held.press(physical_key, Taker::Focused(None));
                // Three walks of the tree: to count, to focus, and to scroll into view.
                let census = self.census(content, layout, renderer);
                let target = next(census.focused, census.total, !modifiers.shift());
                if let Some(bounds) = self.focus_only(content, layout, renderer, target) {
                    let mut reveal = Reveal { target: bounds };
                    self.content
                        .as_widget_mut()
                        .operate(content, layout, renderer, &mut reveal);
                }
                shell.capture_event();
                shell.request_redraw();
                return;
            }
            // What the click hits focuses itself; everything else lets go.
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
            | Event::Touch(touch::Event::FingerPressed { .. })
                if self.census(content, layout, renderer).focused.is_some() =>
            {
                let _ = self.focus_only(content, layout, renderer, None);
                shell.request_redraw();
            }
            _ => {}
        }
        // A held key's repeats go where its press went, or nowhere: once focus has moved, as
        // Tab or a page change moves it, a secret's last letter held down isn't typed into the
        // next field; nor are the repeats of a press a menu took, after it closed.
        if let Event::Keyboard(keyboard::Event::KeyPressed {
            physical_key,
            repeat: true,
            ..
        }) = event
            && let Some(taker) = held.taker(physical_key)
        {
            // Taken with nothing focused, by something that isn't there now, such as a dialog
            // gone: nobody's.
            let moved = match taker {
                Taker::Overlay | Taker::Focused(None) => true,
                Taker::Focused(whose) => self.census(content, layout, renderer).whose != whose,
            };
            if moved {
                shell.capture_event();
                return;
            }
        }
        self.content.as_widget_mut().update(
            content, event, layout, cursor, renderer, clipboard, shell, viewport,
        );
        if let Event::Keyboard(keyboard::Event::KeyPressed {
            physical_key,
            repeat,
            ..
        }) = event
        {
            if shell.is_event_captured() {
                if !repeat {
                    let whose = self.census(content, layout, renderer).whose;
                    held.press(physical_key, Taker::Focused(whose));
                }
            } else if !repeat {
                // Its last release went astray, and the app has its press now, so its release.
                held.release(physical_key);
            } else if held.holds(physical_key) {
                shell.capture_event();
            }
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let touches = tree.state.downcast_ref::<ScopeState>().touches.get();
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            layout,
            touches.cursor(None, cursor),
            viewport,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        let touches = tree.state.downcast_ref::<ScopeState>().touches.get();
        self.content.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            touches.cursor(None, cursor),
            viewport,
            renderer,
        )
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'b, Message, Theme, Renderer>> {
        let Tree {
            state, children, ..
        } = tree;
        let ScopeState { held, touches } = state.downcast_ref::<ScopeState>();
        self.content
            .as_widget_mut()
            .overlay(&mut children[0], layout, renderer, viewport, translation)
            .map(|overlay| {
                overlay::Element::new(Box::new(Touched {
                    overlay,
                    held,
                    touches,
                }))
            })
    }
}

/// What floats over a scope's content, such as an open menu, seeing touches as the content does:
/// one finger at a time, and no pointer once a finger lifts. It sees every event first, so it
/// keeps the scope's record of touches too.
struct Touched<'b, Message> {
    overlay: overlay::Element<'b, Message, Theme, Renderer>,
    held: &'b RefCell<Held>,
    touches: &'b Cell<Touches>,
}

impl<Message> overlay::Overlay<Message, Theme, Renderer> for Touched<'_, Message> {
    fn layout(&mut self, renderer: &Renderer, bounds: Size) -> layout::Node {
        self.overlay.as_overlay_mut().layout(renderer, bounds)
    }

    fn draw(
        &self,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
    ) {
        let cursor = self.touches.get().cursor(None, cursor);
        self.overlay
            .as_overlay()
            .draw(renderer, theme, style, layout, cursor);
    }

    fn operate(&mut self, layout: Layout<'_>, renderer: &Renderer, operation: &mut dyn Operation) {
        self.overlay
            .as_overlay_mut()
            .operate(layout, renderer, operation);
    }

    fn update(
        &mut self,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
    ) {
        let mut touched = self.touches.get();
        touched.before(event);
        if !touched.admits(event) {
            self.touches.set(touched);
            shell.capture_event();
            return;
        }
        let cursor = touched.cursor(Some(event), cursor);
        touched.after(event);
        if touched.lifted && !self.touches.get().lifted {
            shell.request_redraw();
        }
        self.touches.set(touched);
        self.overlay
            .as_overlay_mut()
            .update(event, layout, cursor, renderer, clipboard, shell);
        // A press it took is held as one the content took, its release and repeats kept from
        // the app: Enter held to choose in a menu doesn't go on to press Next.
        if let Event::Keyboard(keyboard::Event::KeyPressed {
            physical_key,
            repeat: false,
            ..
        }) = event
            && shell.is_event_captured()
        {
            self.held.borrow_mut().press(physical_key, Taker::Overlay);
        }
    }

    fn mouse_interaction(
        &self,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        let cursor = self.touches.get().cursor(None, cursor);
        self.overlay
            .as_overlay()
            .mouse_interaction(layout, cursor, renderer)
    }

    fn overlay<'a>(
        &'a mut self,
        layout: Layout<'a>,
        renderer: &Renderer,
    ) -> Option<overlay::Element<'a, Message, Theme, Renderer>> {
        let (held, touches) = (self.held, self.touches);
        self.overlay
            .as_overlay_mut()
            .overlay(layout, renderer)
            .map(|overlay| {
                overlay::Element::new(Box::new(Touched {
                    overlay,
                    held,
                    touches,
                }))
            })
    }

    fn index(&self) -> f32 {
        self.overlay.as_overlay().index()
    }
}

impl<'a, Message: 'a> From<Scope<'a, Message>> for Element<'a, Message> {
    fn from(scope: Scope<'a, Message>) -> Self {
        Element::new(scope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_keys_are_let_go_once() {
        let a = Physical::Code(keyboard::key::Code::KeyA);
        let seven = Physical::Code(keyboard::key::Code::Digit7);
        let mut held = Held::default();
        held.press(&a, Taker::Focused(Some(2)));
        held.press(&a, Taker::Overlay);
        held.press(&seven, Taker::Overlay);
        // The last press counts.
        assert_eq!(held.taker(&a), Some(Taker::Overlay));
        assert!(held.release(&a));
        assert!(!held.release(&a));
        assert!(held.release(&seven));
        assert!(held.0.is_empty());
    }

    #[test]
    fn tab_goes_round() {
        assert_eq!(next(None, 0, true), None);
        assert_eq!(next(None, 3, true), Some(0));
        assert_eq!(next(None, 3, false), Some(2));
        assert_eq!(next(Some(0), 3, true), Some(1));
        assert_eq!(next(Some(2), 3, true), Some(0));
        assert_eq!(next(Some(0), 3, false), Some(2));
        assert_eq!(next(Some(1), 3, false), Some(0));
    }

    #[test]
    fn keyboard_focus_shows_and_a_clicks_once_a_key_is_used() {
        let mut focus = Focus::default();
        assert!(!focus.is_focused() && !focus.is_shown());
        Focusable::focus(&mut focus);
        assert!(Focusable::is_focused(&focus) && focus.is_shown());
        focus.unfocus();
        assert!(!focus.is_focused() && !focus.is_shown());
        focus.click();
        assert!(focus.is_focused() && !focus.is_shown());
        focus.key();
        assert!(focus.is_shown());
        focus.unfocus();
        focus.key();
        assert!(!focus.is_shown());
    }

    #[test]
    fn reveal_scrolls_just_enough() {
        struct Spy(Option<AbsoluteOffset<Option<f32>>>);
        impl Scrollable for Spy {
            fn snap_to(&mut self, _: iced::widget::scrollable::RelativeOffset<Option<f32>>) {}
            fn scroll_to(&mut self, offset: AbsoluteOffset<Option<f32>>) {
                self.0 = Some(offset);
            }
            fn scroll_by(&mut self, _: AbsoluteOffset, _: Rectangle, _: Rectangle) {}
        }
        let viewport = Rectangle::new(iced::Point::ORIGIN, Size::new(100.0, 100.0));
        let content = Rectangle::new(iced::Point::ORIGIN, Size::new(100.0, 400.0));
        let scroll = |target: Rectangle, scrolled: f32| {
            let mut spy = Spy(None);
            Reveal { target }.scrollable(
                None,
                viewport,
                content,
                Vector::new(0.0, scrolled),
                &mut spy,
            );
            spy.0.and_then(|offset| offset.y)
        };
        let at = |y: f32| Rectangle::new(iced::Point::new(10.0, y), Size::new(50.0, 20.0));
        // In sight: nothing moves.
        assert_eq!(scroll(at(40.0), 0.0), None);
        // Below: scrolled until its bottom and the margin show.
        assert_eq!(scroll(at(150.0), 0.0), Some(78.0));
        // Above: scrolled back to its top, less the margin.
        assert_eq!(scroll(at(20.0), 100.0), Some(12.0));
        // Near the top: not past the start.
        assert_eq!(scroll(at(2.0), 50.0), Some(0.0));
        // Outside the content: another scroll area's.
        assert_eq!(scroll(at(500.0), 0.0), None);
        // Too tall to fit: its start shows, from below or above, and stays once it does.
        let tall = |y: f32| Rectangle::new(iced::Point::new(10.0, y), Size::new(50.0, 150.0));
        assert_eq!(scroll(tall(40.0), 0.0), Some(32.0));
        assert_eq!(scroll(tall(40.0), 100.0), Some(32.0));
        assert_eq!(scroll(tall(40.0), 32.0), None);
    }
}
