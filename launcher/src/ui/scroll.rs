// SPDX-License-Identifier: GPL-3.0-or-later

//! A scroll area: content taller than its window, such as a long form at a large scale. The wheel
//! scrolls it, a mouse wheel's steps gliding rather than jumping, and focus moved by the keyboard
//! scrolls the focused widget into view ([`crate::ui::focus`]). Scrolled on past an end, the content
//! gives a little and springs back.
//!
//! An area with more to show than fits and nothing in it that takes focus, such as a long text,
//! takes focus itself, as a browser's does, so nothing is left that only a pointer can reach:
//! focused, the arrow keys scroll it by a wheel's step, Page Up, Page Down and Space by what
//! shows less a step, and Home and End to its ends, gliding as a wheel's steps do.

use std::time::{Duration, Instant};

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::operation::{Focusable, Operation, Scrollable as Scrolled};
use iced::advanced::widget::{Id, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, overlay, renderer};
use iced::keyboard::{self, Key, key::Named};
use iced::widget::container;
use iced::widget::scrollable::{self, AbsoluteOffset, Direction, Scrollable, Scrollbar};
use iced::{
    Background, Border, Event, Length, Point, Rectangle, Shadow, Size, Vector, touch, window,
};

use crate::ui::tokens::motion;
use crate::ui::tokens::{radius, size, space};

use crate::ui::focus::Focus;
use crate::ui::theme::{Interaction, color as ink};
use crate::ui::{Element, Renderer, Theme, input, paint};

/// Scrolls its content up and down, a mouse wheel's steps gliding to where they lead.
pub fn scroll_area<'a, Message: 'a>(
    content: impl Into<Element<'a, Message>>,
) -> Smooth<'a, Message> {
    Smooth {
        // As wide as the room it's given, so its bar is at that room's edge, not its content's.
        inner: Scrollable::with_direction(content, Direction::Vertical(bar()))
            .width(Length::Fill)
            .class(Bar::Hidden)
            .into(),
        moves: !crate::ui::system::reduced_motion(),
    }
}

/// A scroll area whose mouse-wheel steps glide: iced jumps a step at once, and this puts it back
/// and moves there over [`motion::NORMAL`] on the design language's curve. A wheel a child takes, such as a
/// list's, a touchpad's smooth deltas and the scrollbar work as they always did. iced's own
/// scrollable takes no keys: an area with nothing in it that takes focus takes it itself, and
/// the keys scroll it (see the module).
///
/// A finger scrolls it from anywhere, a button or a checkbox too: what it pressed takes the
/// touch until it has moved further up or down than a slip ([`size::SLIP`]), and then lets go
/// (a lost touch), and the area follows the finger. A drag a child keeps, such as a switch's
/// knob, or one that goes across, stays the child's.
///
/// Scrolled on past an end, by a mouse wheel, a touchpad or a finger, the content gives (a
/// spring): a wheel's step bumps it, a touchpad or a finger pulls it, harder the further it is,
/// and it springs back.
pub struct Smooth<'a, Message> {
    inner: Element<'a, Message>,
    /// Whether it moves: glides and gives. Not with motion reduced.
    moves: bool,
}

/// A glide from one offset to another.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Glide {
    from: Vector,
    to: Vector,
    /// When it started: the first frame drawn after the step.
    start: Option<Instant>,
    /// Where it last put the area: moved from there by anything else, such as Tab's reveal, it
    /// stops, and doesn't put it back.
    put: Vector,
}

impl Glide {
    fn new(from: Vector, to: Vector) -> Self {
        Self {
            from,
            to,
            start: None,
            put: from,
        }
    }

    /// Where it is at `now`, and whether it's over.
    fn at(&mut self, now: Instant) -> (Vector, bool) {
        let start = *self.start.get_or_insert(now);
        let t =
            (now - start).as_secs_f32() / crate::ui::system::duration(motion::NORMAL).as_secs_f32();
        if t >= 1.0 {
            (self.to, true)
        } else {
            let k = motion::EASE.at(t);
            (self.from + (self.to - self.from) * k, false)
        }
    }
}

/// How far the content is past an end, a critically damped spring pulling it back: positive past
/// the bottom, the content moved up; negative past the top.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Stretch {
    /// Where it was and how fast it moved, in pixels a second, `since` then.
    at: f32,
    speed: f32,
    since: Option<Instant>,
}

/// The spring's rate, per second: back to rest in about [`motion::SLOW`].
const SPRING: f32 = 6.0 / motion::SLOW.as_secs_f32();
/// The furthest the content goes past an end, but never more than a third of the area.
const REACH: f32 = 80.0;
/// A wheel's step past an end bumps the content this share of the step at the most.
const BUMP: f32 = 1.0 / 3.0;
/// Below this far and this fast, the content is at rest.
const AT_REST: (f32, f32) = (0.25, 5.0);
/// iced's pixels for a wheel's line.
const LINE: f32 = 60.0;
/// How near an end counts as at it.
const EDGE: f32 = 0.5;

impl Stretch {
    /// Where it is and how fast it moves at `now`.
    fn at(self, now: Instant) -> (f32, f32) {
        let Some(since) = self.since else {
            return (0.0, 0.0);
        };
        let t = now.saturating_duration_since(since).as_secs_f32();
        let b = self.speed + SPRING * self.at;
        let decay = (-SPRING * t).exp();
        (
            (self.at + b * t) * decay,
            (self.speed - SPRING * b * t) * decay,
        )
    }

    fn set(&mut self, now: Instant, at: f32, speed: f32) {
        *self = if at.abs() < AT_REST.0 && speed.abs() < AT_REST.1 {
            Self::default()
        } else {
            Self {
                at,
                speed,
                since: Some(now),
            }
        };
    }

    /// A wheel's step `by` pixels past an end: a bump, smaller the further it already is, never
    /// past `reach`.
    fn kick(&mut self, now: Instant, by: f32, reach: f32) {
        let (at, speed) = self.at(now);
        let push = by * BUMP * SPRING * std::f32::consts::E;
        self.set(now, at, speed + push * give(at, by, reach));
    }

    /// A touchpad's pull `by` pixels past an end: the content follows, less the further it
    /// already is, never past `reach`, and comes to rest there before it springs back.
    fn pull(&mut self, now: Instant, by: f32, reach: f32) {
        let (at, _) = self.at(now);
        self.set(now, at + by * give(at, by, reach), 0.0);
    }

    /// A pull `by` pixels back toward the end: the content comes back first, as far as it went;
    /// what's left of `by` scrolls.
    fn give_back(&mut self, now: Instant, by: f32) -> f32 {
        let (at, _) = self.at(now);
        let to = at + by;
        if to * at > 0.0 {
            self.set(now, to, 0.0);
            0.0
        } else {
            *self = Self::default();
            to
        }
    }

    /// Whether it's still moving at `now`.
    fn moving(self, now: Instant) -> bool {
        let (at, speed) = self.at(now);
        at.abs() >= AT_REST.0 || speed.abs() >= AT_REST.1
    }
}

/// How much of a push `by` past an end the content takes, already `at` of `reach`: all of it back
/// toward the end, and less and less of it further out.
fn give(at: f32, by: f32, reach: f32) -> f32 {
    if at * by < 0.0 {
        return 1.0;
    }
    let left = (1.0 - at.abs() / reach).max(0.0);
    left * left
}

/// A finger on the area: where it went down, where it was last, and whose drag it is.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Touch {
    finger: touch::Finger,
    from: Point,
    last: Point,
    drag: Drag,
}

/// Whose a finger's drag is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drag {
    /// Not yet anyone's: it hasn't moved further than a slip.
    Undecided,
    /// The area's: it scrolls with the finger.
    Area,
    /// Someone else's: a child that drags, such as a switch's knob, a drag across, or iced's own
    /// scrolling of a press on nothing.
    Other,
}

impl Drag {
    /// Who a finger moved `by` since it went down drags, if it's been decided: up or down further
    /// than a slip, the area; across further, someone else.
    fn decide(by: Vector) -> Option<Self> {
        if by.y.abs() > size::SLIP && by.y.abs() >= by.x.abs() {
            Some(Self::Area)
        } else if by.x.abs() > size::SLIP {
            Some(Self::Other)
        } else {
            None
        }
    }
}

/// Whether an event ends a scroll transaction begun `since` ago, as iced's scrollable reckons it:
/// in one, the wheel is the area's own, and a child never sees it. Its times are iced_widget
/// 0.14.2's, private there; Cargo.lock holds iced_widget there, so they're checked again when
/// it moves.
fn ends_transaction(event: &Event, since: Duration) -> bool {
    match event {
        Event::Mouse(
            mouse::Event::ButtonPressed(_)
            | mouse::Event::ButtonReleased(_)
            | mouse::Event::CursorLeft,
        ) => true,
        Event::Mouse(mouse::Event::CursorMoved { .. }) => since > Duration::from_millis(100),
        _ => since > Duration::from_millis(1500),
    }
}

#[derive(Default)]
struct State {
    glide: Option<Glide>,
    /// How far the content is past an end, and where it was drawn.
    stretch: Stretch,
    stretched: f32,
    /// When the wheel last scrolled it, while iced's transaction lasts.
    transaction: Option<Instant>,
    /// Where it was scrolled to at the last frame, and its bounds and its content's.
    shown: Option<(Vector, Rectangle, Rectangle)>,
    /// When it last moved, and how opaque its bar is now.
    moved: Option<Instant>,
    bar: f32,
    /// Whether the pointer is over where the bar is: it shows while it is.
    over_bar: bool,
    /// The finger on it, if one is.
    touch: Option<Touch>,
    /// The thumb dragged by the pointer: where the press was, and how far the area was
    /// scrolled then.
    dragging: Option<(f32, f32)>,
    /// Its own focus, when it takes focus: more to show than fits, and nothing in it that does.
    focus: Focus,
    takes: bool,
}

/// How long the bar stays after the area stops moving, and how long it then takes to fade.
const BAR_STAYS: Duration = Duration::from_millis(700);
const BAR_FADES: Duration = motion::SLOW;

/// How opaque the bar is `since` the area last moved: whole while it stays, then fading out on
/// the design language's curve, both half as long on battery saver; and when to look again, if it's still
/// showing.
fn bar_opacity(since: Duration) -> (f32, Option<Duration>) {
    bar_opacity_over(
        since,
        crate::ui::system::duration(BAR_STAYS),
        crate::ui::system::duration(BAR_FADES),
    )
}

/// The same, the bar staying `stays`, then fading over `fades`.
fn bar_opacity_over(since: Duration, stays: Duration, fades: Duration) -> (f32, Option<Duration>) {
    if since < stays {
        (1.0, Some(stays - since))
    } else if since < stays + fades {
        let t = (since - stays).as_secs_f32() / fades.as_secs_f32();
        (1.0 - motion::EASE.at(t), Some(Duration::ZERO))
    } else {
        (0.0, None)
    }
}

/// Where the bar's thumb is: at the area's right edge, as iced's bar is, as far from the top and
/// the bottom as from the side, its length the share of the content shown. None when everything
/// shows.
fn thumb(translation: Vector, bounds: Rectangle, content: Rectangle) -> Option<Rectangle> {
    let hidden = content.height - bounds.height;
    if hidden <= 0.5 {
        return None;
    }
    let (length, travel) = thumb_length(bounds, content);
    let at = (translation.y / hidden).clamp(0.0, 1.0) * travel;
    Some(Rectangle::new(
        iced::Point::new(
            bounds.x + bounds.width - space::XXS - space::S,
            bounds.y + space::XXS + at,
        ),
        Size::new(space::S, length),
    ))
}

/// How long the bar's thumb is, its share of the track as the area's of the content but never
/// shorter than a finger's grip, and how far it travels from the top to the bottom.
fn thumb_length(bounds: Rectangle, content: Rectangle) -> (f32, f32) {
    let track = bounds.height - 2.0 * space::XXS;
    let length = (track * bounds.height / content.height).max(space::XXL);
    (length, (track - length).max(0.0))
}

/// Where the pointer shows the bar: a strip at the area's right edge, as wide as the bar and its
/// margins.
fn bar_strip(bounds: Rectangle) -> Rectangle {
    let width = space::S + 2.0 * space::XXS;
    Rectangle::new(
        iced::Point::new(bounds.x + bounds.width - width, bounds.y),
        Size::new(width, bounds.height),
    )
}

/// The scroll area's own offset: the first scrollable an operation meets, not a nested one's.
struct Offset {
    read: Option<Vector>,
    bounds: Option<(Rectangle, Rectangle)>,
    write: Option<Vector>,
    done: bool,
}

impl Operation for Offset {
    fn traverse(&mut self, operate: &mut dyn FnMut(&mut dyn Operation)) {
        // Once it's met its scrollable, nothing further in: the rest of the page isn't walked.
        if !self.done {
            operate(self);
        }
    }

    fn scrollable(
        &mut self,
        _id: Option<&Id>,
        bounds: Rectangle,
        content_bounds: Rectangle,
        translation: Vector,
        state: &mut dyn Scrolled,
    ) {
        if self.done {
            return;
        }
        self.done = true;
        self.read = Some(translation);
        self.bounds = Some((bounds, content_bounds));
        if let Some(to) = self.write {
            state.scroll_to(AbsoluteOffset {
                x: Some(to.x),
                y: Some(to.y),
            });
        }
    }
}

/// Counts the focusable widgets an operation meets.
struct Count(usize);

impl Operation for Count {
    fn traverse(&mut self, operate: &mut dyn FnMut(&mut dyn Operation)) {
        operate(self);
    }

    fn focusable(&mut self, _id: Option<&Id>, _bounds: Rectangle, _state: &mut dyn Focusable) {
        self.0 += 1;
    }
}

/// Where a key on the focused area scrolls it to, down from the top, `at` now, `shown` high with
/// `hidden` more below: the arrows by a wheel's step, Page Up, Page Down and Space (Shift+Space
/// up) by what shows less a step, so a line carries over, and Home and End to the ends. None for
/// a key that doesn't scroll.
fn scroll_key(key: &Key, shift: bool, at: f32, shown: f32, hidden: f32) -> Option<f32> {
    let page = (shown - LINE).max(LINE);
    let to = match key {
        Key::Named(Named::ArrowDown) => at + LINE,
        Key::Named(Named::ArrowUp) => at - LINE,
        Key::Named(Named::PageDown) => at + page,
        Key::Named(Named::PageUp) => at - page,
        Key::Named(Named::Space) if shift => at - page,
        Key::Named(Named::Space) => at + page,
        Key::Named(Named::Home) => 0.0,
        Key::Named(Named::End) => hidden,
        _ => return None,
    };
    Some(to.clamp(0.0, hidden.max(0.0)))
}

impl<Message> Smooth<'_, Message> {
    /// Whether it takes focus itself: there's more to show than fits, and nothing in it takes
    /// focus, which would scroll it into view.
    fn takes_focus(&mut self, tree: &mut Tree, layout: Layout<'_>, renderer: &Renderer) -> bool {
        let Some((_, bounds, content)) = self.geometry(tree, layout, renderer, None) else {
            return false;
        };
        if content.height - bounds.height <= EDGE {
            return false;
        }
        let mut count = Count(0);
        self.inner
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, &mut count);
        count.0 == 0
    }

    /// A key on it while it has focus: scrolled, gliding as a wheel's step does, and the key
    /// taken if it moved. Whether the key was one that scrolls.
    fn key(
        &mut self,
        key: &Key,
        modifiers: keyboard::Modifiers,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        shell: &mut Shell<'_, Message>,
    ) -> bool {
        let Some((translation, bounds, content)) = self.geometry(tree, layout, renderer, None)
        else {
            return false;
        };
        let state = tree.state.downcast_mut::<State>();
        // Pressed again while it glides, from where the glide was headed.
        let at = state.glide.map_or(translation.y, |glide| glide.to.y);
        let hidden = content.height - bounds.height;
        let Some(to) = scroll_key(key, modifiers.shift(), at, bounds.height, hidden) else {
            return false;
        };
        state.focus.key();
        shell.request_redraw();
        if (to - translation.y).abs() <= EDGE && state.glide.is_none() {
            return true;
        }
        let to = Vector::new(translation.x, to);
        if self.moves {
            state.glide = Some(Glide::new(translation, to));
        } else {
            let _ = self.offset(tree, layout, renderer, Some(to));
        }
        shell.capture_event();
        true
    }
}

impl<Message> Smooth<'_, Message> {
    /// Its offset now, or, given one, set to it.
    fn offset(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        write: Option<Vector>,
    ) -> Option<Vector> {
        self.geometry(tree, layout, renderer, write)
            .map(|(translation, _, _)| translation)
    }

    /// Its offset, its bounds and its content's, or, given an offset, set to it first.
    fn geometry(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        write: Option<Vector>,
    ) -> Option<(Vector, Rectangle, Rectangle)> {
        let mut offset = Offset {
            read: None,
            bounds: None,
            write,
            done: false,
        };
        self.inner
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, &mut offset);
        let (bounds, content) = offset.bounds?;
        Some((write.unwrap_or(offset.read?), bounds, content))
    }

    /// A wheel over the area, a mouse's or a touchpad's: iced scrolls it, then a mouse's step
    /// glides there; and scrolled on past an end, if the wheel is the area's own, the content
    /// gives (a spring).
    #[allow(clippy::too_many_arguments)]
    fn wheel(
        &mut self,
        delta: mouse::ScrollDelta,
        now: Instant,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        // How far down it's asked to scroll, as iced counts it.
        let (by, step) = match delta {
            mouse::ScrollDelta::Lines { y, .. } => (-y * LINE, true),
            mouse::ScrollDelta::Pixels { y, .. } => (-y, false),
        };
        let glides = self.moves && step;
        if !glides {
            // A touchpad takes over from the glide where it is.
            tree.state.downcast_mut::<State>().glide = None;
        }
        let before = self.geometry(tree, layout, renderer, None);
        self.inner.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
        let after = self.offset(tree, layout, renderer, None);
        let (Some((before, bounds, content)), Some(after)) = (before, after) else {
            return;
        };
        if !self.moves {
            return;
        }
        let state = tree.state.downcast_mut::<State>();
        if after != before {
            state.transaction = Some(now);
        }
        // A touchpad pulling back from past an end brings the content back before it scrolls.
        if !step && state.stretch.at(now).0 * by < 0.0 {
            let left = state.stretch.give_back(now, by);
            let _ = self.offset(
                tree,
                layout,
                renderer,
                Some(before + Vector::new(0.0, left)),
            );
            shell.capture_event();
            shell.request_redraw();
            return;
        }
        if after != before {
            if glides {
                // The area itself moved, not a child: back to where it was, and glide on from
                // where the glide was headed.
                let to = state.glide.map_or(before, |glide| glide.to) + (after - before);
                let _ = self.offset(tree, layout, renderer, Some(before));
                let state = tree.state.downcast_mut::<State>();
                state.glide = Some(Glide::new(before, to));
                shell.request_redraw();
            }
            return;
        }
        // It didn't move. Past an end, the wheel is the area's own if no child took it, or if
        // it's in a transaction, where no child sees the wheel.
        let ours = !shell.is_event_captured() || state.transaction.is_some();
        let hidden = content.height - bounds.height;
        let past = hidden > EDGE
            && ((by < 0.0 && before.y <= EDGE) || (by > 0.0 && before.y >= hidden - EDGE));
        if ours && past {
            let reach = REACH.min(bounds.height / 3.0);
            if step {
                state.stretch.kick(now, by, reach);
            } else {
                state.stretch.pull(now, by, reach);
            }
            shell.capture_event();
            shell.request_redraw();
        }
    }
}

impl<Message> Smooth<'_, Message> {
    /// The thumb the area draws, dragged by the pointer: the area follows it, the thumb's travel
    /// across the content. The area drags it itself, as iced's own draggable thumb is elsewhere
    /// on a long page, being shorter. Whether the event was the drag's.
    fn drag_thumb(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        shell: &mut Shell<'_, Message>,
    ) -> bool {
        let state = tree.state.downcast_mut::<State>();
        let Some((translation, bounds, content)) = state.shown else {
            return false;
        };
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                if state.bar > 0.0
                    && thumb(translation, bounds, content)
                        .is_some_and(|thumb| cursor.is_over(thumb.expand(space::XXS))) =>
            {
                let Some(at) = cursor.position() else {
                    return false;
                };
                state.dragging = Some((at.y, translation.y));
                state.glide = None;
                shell.capture_event();
                true
            }
            Event::Mouse(mouse::Event::CursorMoved { position }) => {
                let Some((from, scrolled)) = state.dragging else {
                    return false;
                };
                let (_, travel) = thumb_length(bounds, content);
                let hidden = (content.height - bounds.height).max(0.0);
                let by = if travel > 0.0 {
                    (position.y - from) * hidden / travel
                } else {
                    0.0
                };
                let to = (scrolled + by).clamp(0.0, hidden);
                let _ = self.offset(tree, layout, renderer, Some(Vector::new(translation.x, to)));
                shell.capture_event();
                shell.request_redraw();
                true
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
                if state.dragging.is_some() =>
            {
                state.dragging = None;
                state.moved = Some(Instant::now());
                shell.capture_event();
                shell.request_redraw();
                true
            }
            _ => false,
        }
    }

    /// The area scrolled `by` pixels down with a finger, as a touchpad's delta scrolls it: past an
    /// end the content gives, and pulled back it comes back first.
    fn follow(
        &mut self,
        by: f32,
        now: Instant,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
    ) {
        let Some((before, bounds, content)) = self.geometry(tree, layout, renderer, None) else {
            return;
        };
        let hidden = (content.height - bounds.height).max(0.0);
        let state = tree.state.downcast_mut::<State>();
        let by = if self.moves && state.stretch.at(now).0 * by < 0.0 {
            state.stretch.give_back(now, by)
        } else {
            by
        };
        let to = (before.y + by).clamp(0.0, hidden);
        let past = before.y + by - to;
        if self.moves && past != 0.0 && hidden > EDGE {
            state
                .stretch
                .pull(now, past, REACH.min(bounds.height / 3.0));
        }
        if to != before.y {
            let _ = self.offset(tree, layout, renderer, Some(Vector::new(before.x, to)));
        }
    }

    /// A finger's event, if it's one the area takes for itself rather than passing on: a drag
    /// that's become the area's moves it, and lifting it ends the drag. A drag the area takes
    /// over first tells what the finger pressed that it's lost the touch.
    #[allow(clippy::too_many_arguments)]
    fn touch(
        &mut self,
        event: &touch::Event,
        tree: &mut Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) -> bool {
        let state = tree.state.downcast_mut::<State>();
        let ours = |id: &touch::Finger| state.touch.is_some_and(|touch| touch.finger == *id);
        match *event {
            touch::Event::FingerPressed { id, position } => {
                if state.touch.is_none() && cursor.is_over(layout.bounds()) {
                    state.touch = Some(Touch {
                        finger: id,
                        from: position,
                        last: position,
                        drag: Drag::Undecided,
                    });
                }
                false
            }
            touch::Event::FingerMoved { id, position } if ours(&id) => {
                let Some(mut touch) = state.touch else {
                    return false;
                };
                match touch.drag {
                    Drag::Area => {
                        let by = touch.last.y - position.y;
                        touch.last = position;
                        tree.state.downcast_mut::<State>().touch = Some(touch);
                        self.follow(by, Instant::now(), tree, layout, renderer);
                        shell.capture_event();
                        shell.request_redraw();
                        true
                    }
                    Drag::Other => false,
                    Drag::Undecided => {
                        // What the finger pressed sees it move first: if it keeps the drag, or
                        // iced scrolls a press on nothing, the drag is theirs.
                        let before = self.offset(tree, layout, renderer, None);
                        self.inner.as_widget_mut().update(
                            &mut tree.children[0],
                            &Event::Touch(*event),
                            layout,
                            cursor,
                            renderer,
                            clipboard,
                            shell,
                            viewport,
                        );
                        let after = self.offset(tree, layout, renderer, None);
                        touch.drag = if shell.is_event_captured() || after != before {
                            Drag::Other
                        } else {
                            Drag::decide(position - touch.from).unwrap_or(Drag::Undecided)
                        };
                        if touch.drag == Drag::Area {
                            // The area's now: what the finger pressed lets go, as if the touch
                            // were taken away, and the area follows from here, without a jump.
                            touch.last = position;
                            self.inner.as_widget_mut().update(
                                &mut tree.children[0],
                                &Event::Touch(touch::Event::FingerLost { id, position }),
                                layout,
                                cursor,
                                renderer,
                                clipboard,
                                shell,
                                viewport,
                            );
                            shell.capture_event();
                            shell.request_redraw();
                        }
                        tree.state.downcast_mut::<State>().touch = Some(touch);
                        true
                    }
                }
            }
            touch::Event::FingerLifted { id, position }
            | touch::Event::FingerLost { id, position }
                if ours(&id) =>
            {
                let drag = state.touch.take().map(|touch| touch.drag);
                if drag != Some(Drag::Area) {
                    return false;
                }
                // What it pressed let go when the area took the drag: iced's own drag ends.
                self.inner.as_widget_mut().update(
                    &mut tree.children[0],
                    &Event::Touch(touch::Event::FingerLost { id, position }),
                    layout,
                    cursor,
                    renderer,
                    clipboard,
                    shell,
                    viewport,
                );
                shell.capture_event();
                shell.request_redraw();
                true
            }
            _ => false,
        }
    }
}

impl<Message> Widget<Message, Theme, Renderer> for Smooth<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.inner)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.inner));
    }

    fn size(&self) -> Size<Length> {
        self.inner.as_widget().size()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.inner
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
        let takes = self.takes_focus(tree, layout, renderer);
        let state = tree.state.downcast_mut::<State>();
        state.takes = takes;
        state.focus.keep_if(takes);
        if takes {
            operation.focusable(None, layout.bounds(), &mut state.focus);
        }
        self.inner
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
        let now = Instant::now();
        let state = tree.state.downcast_mut::<State>();
        state.transaction = state
            .transaction
            .filter(|&since| !ends_transaction(event, now.saturating_duration_since(since)));
        // Over where the bar is, the bar shows, while there's more to scroll to; let go of, it
        // stays a while and fades, as after scrolling.
        if let Event::Mouse(mouse::Event::CursorMoved { .. } | mouse::Event::CursorLeft) = event {
            let over = state.shown.is_some_and(|(translation, bounds, content)| {
                thumb(translation, bounds, content).is_some() && cursor.is_over(bar_strip(bounds))
            });
            if over != state.over_bar {
                state.over_bar = over;
                if !over {
                    state.moved = Some(now);
                }
                shell.request_redraw();
            }
        }
        // Taking focus itself, a click on it focuses it; the keys then scroll it.
        if let Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
        | Event::Touch(touch::Event::FingerPressed { .. }) = event
            && cursor.is_over(layout.bounds())
        {
            let state = tree.state.downcast_mut::<State>();
            if state.takes {
                state.focus.click();
            }
        }
        if let Event::Keyboard(keyboard::Event::KeyPressed { key, modifiers, .. }) = event
            && tree.state.downcast_ref::<State>().focus.is_focused()
            && input::plain(*modifiers)
            && self.key(key, *modifiers, tree, layout, renderer, shell)
        {
            return;
        }
        if self.drag_thumb(tree, event, layout, cursor, renderer, shell) {
            return;
        }
        if let Event::Mouse(mouse::Event::WheelScrolled { delta }) = event
            && cursor.is_over(layout.bounds())
        {
            self.wheel(
                *delta, now, tree, event, layout, cursor, renderer, clipboard, shell, viewport,
            );
            return;
        }
        if let Event::Touch(touch) = event {
            if cursor.is_over(layout.bounds()) {
                // A finger takes over from the glide where it is.
                tree.state.downcast_mut::<State>().glide = None;
            }
            if self.touch(
                touch, tree, layout, cursor, renderer, clipboard, shell, viewport,
            ) {
                return;
            }
        }
        if let Event::Window(window::Event::RedrawRequested(now)) = event {
            let mut write = None;
            if let Some(mut glide) = tree.state.downcast_mut::<State>().glide.take() {
                let moved_since = self
                    .offset(tree, layout, renderer, None)
                    .is_some_and(|offset| (offset - glide.put).y.abs() > EDGE);
                if !moved_since {
                    let (at, over) = glide.at(*now);
                    glide.put = at;
                    if !over {
                        tree.state.downcast_mut::<State>().glide = Some(glide);
                        shell.request_redraw();
                    }
                    write = Some(at);
                }
            }
            let state = tree.state.downcast_mut::<State>();
            // Past an end, the content springs back.
            state.stretched = state.stretch.at(*now).0;
            if state.stretch.moving(*now) {
                shell.request_redraw();
            } else {
                state.stretch = Stretch::default();
                state.stretched = 0.0;
            }
            // The bar shows while the area moves, by whatever means, then fades.
            let shown = self.geometry(tree, layout, renderer, write);
            let state = tree.state.downcast_mut::<State>();
            if let (Some((was, ..)), Some((now_at, ..))) = (state.shown, shown)
                && was != now_at
            {
                state.moved = Some(*now);
            }
            state.shown = shown;
            state.bar = 0.0;
            if state.over_bar || state.dragging.is_some() {
                state.bar = 1.0;
            } else if let Some(moved) = state.moved {
                let (opacity, again) = bar_opacity(now.saturating_duration_since(moved));
                state.bar = opacity;
                match again {
                    Some(wait) if wait.is_zero() => shell.request_redraw(),
                    Some(wait) => shell.request_redraw_at(*now + wait),
                    None => state.moved = None,
                }
            }
        } else if matches!(event, Event::Mouse(mouse::Event::ButtonPressed(_)))
            && cursor.is_over(layout.bounds())
        {
            // A press, as on the scrollbar, takes over from the glide where it is.
            tree.state.downcast_mut::<State>().glide = None;
        }
        // Past an end, the content is drawn moved by the stretch: the pointer's over what's drawn
        // under it.
        let stretched = Vector::new(0.0, tree.state.downcast_ref::<State>().stretched);
        self.inner.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor + stretched,
            renderer,
            clipboard,
            shell,
            viewport,
        );
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
        let state = tree.state.downcast_ref::<State>();
        if state.stretched == 0.0 {
            self.inner.as_widget().draw(
                &tree.children[0],
                renderer,
                theme,
                style,
                layout,
                cursor,
                viewport,
            );
        } else if let Some(shown) = layout.bounds().intersection(viewport) {
            // Past an end: the content moved by the stretch, and the pointer over what's drawn
            // under it. iced doesn't clip a layer to the one round it, so the area's own layer is
            // kept inside it by the viewport it's given: where it shows, moved back.
            let shift = Vector::new(0.0, -state.stretched);
            iced::advanced::Renderer::with_translation(renderer, shift, |renderer| {
                self.inner.as_widget().draw(
                    &tree.children[0],
                    renderer,
                    theme,
                    style,
                    layout,
                    cursor + Vector::new(0.0, state.stretched),
                    &(shown - shift),
                );
            });
        }
        if state.bar > 0.0
            && let Some((translation, bounds, content)) = state.shown
            && let Some(thumb) = thumb(translation, bounds, content)
        {
            let mut color = ink(theme.scroller(Interaction::Hovered));
            color.a *= state.bar;
            iced::advanced::Renderer::with_layer(renderer, bounds, |renderer| {
                iced::advanced::Renderer::fill_quad(
                    renderer,
                    renderer::Quad {
                        bounds: thumb,
                        border: Border {
                            radius: radius::ROUND.into(),
                            ..Border::default()
                        },
                        shadow: Shadow::default(),
                        snap: true,
                    },
                    Background::Color(color),
                );
            });
        }
        if state.takes && state.focus.is_shown() {
            paint::outline(renderer, theme, layout.bounds(), radius::CONTROL);
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        let stretched = Vector::new(0.0, tree.state.downcast_ref::<State>().stretched);
        self.inner.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            cursor + stretched,
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
        self.inner.as_widget_mut().overlay(
            &mut tree.children[0],
            layout,
            renderer,
            viewport,
            translation,
        )
    }
}

impl<'a, Message: 'a> From<Smooth<'a, Message>> for Element<'a, Message> {
    fn from(smooth: Smooth<'a, Message>) -> Self {
        Element::new(smooth)
    }
}

/// Scrolls its content sideways, such as a table wider than its window; the wheel still scrolls
/// what's round it up and down.
pub fn scroll_across<'a, Message: 'a>(
    content: impl Into<Element<'a, Message>>,
) -> Scrollable<'a, Message, Theme, Renderer> {
    // The bar takes room of its own below the content, rather than lying over its last row.
    Scrollable::with_direction(content, Direction::Horizontal(bar().spacing(space::S)))
}

/// Scrolls its content both ways, for a page wider than the window.
pub fn scroll_both<'a, Message: 'a>(
    content: impl Into<Element<'a, Message>>,
) -> Scrollable<'a, Message, Theme, Renderer> {
    Scrollable::with_direction(
        content,
        Direction::Both {
            vertical: bar(),
            horizontal: bar(),
        },
    )
}

fn bar() -> Scrollbar {
    Scrollbar::new()
        .width(space::S)
        .scroller_width(space::S)
        .margin(space::XXS)
}

/// Whether a scroll area's bar is drawn: always, or by the area itself, while it scrolls
/// ([`Smooth`]); hidden, iced's bar is still there to drag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bar {
    #[default]
    Shown,
    Hidden,
}

impl scrollable::Catalog for Theme {
    type Class<'a> = Bar;

    fn default<'a>() -> Self::Class<'a> {
        Bar::Shown
    }

    fn style(&self, class: &Bar, status: scrollable::Status) -> scrollable::Style {
        let (vertical, horizontal) = match status {
            scrollable::Status::Active { .. } => (Interaction::Idle, Interaction::Idle),
            scrollable::Status::Hovered {
                is_vertical_scrollbar_hovered,
                is_horizontal_scrollbar_hovered,
                ..
            } => (
                hovered(is_vertical_scrollbar_hovered),
                hovered(is_horizontal_scrollbar_hovered),
            ),
            scrollable::Status::Dragged {
                is_vertical_scrollbar_dragged,
                is_horizontal_scrollbar_dragged,
                ..
            } => (
                pressed(is_vertical_scrollbar_dragged),
                pressed(is_horizontal_scrollbar_dragged),
            ),
        };
        let rail = |interaction| scrollable::Rail {
            background: None,
            border: Border::default(),
            scroller: scrollable::Scroller {
                background: Background::Color(match class {
                    Bar::Shown => ink(self.scroller(interaction)),
                    Bar::Hidden => iced::Color::TRANSPARENT,
                }),
                border: Border {
                    radius: radius::ROUND.into(),
                    ..Border::default()
                },
            },
        };
        let badge = self.badge();
        scrollable::Style {
            container: container::Style::default(),
            vertical_rail: rail(vertical),
            horizontal_rail: rail(horizontal),
            gap: None,
            auto_scroll: scrollable::AutoScroll {
                background: Background::Color(ink(badge.fill.unwrap_or(badge.ink))),
                border: Border {
                    color: ink(badge.line.unwrap_or(badge.ink)),
                    width: crate::ui::tokens::size::BORDER,
                    radius: radius::ROUND.into(),
                },
                shadow: Shadow::default(),
                icon: ink(badge.ink),
            },
        }
    }
}

fn hovered(yes: bool) -> Interaction {
    if yes {
        Interaction::Hovered
    } else {
        Interaction::Idle
    }
}

fn pressed(yes: bool) -> Interaction {
    if yes {
        Interaction::Pressed
    } else {
        Interaction::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::widget::scrollable::Catalog as _;

    #[test]
    fn the_scroller_shows_what_the_pointer_does() {
        let theme = <Theme as Default>::default();
        let idle = theme.style(
            &Bar::Shown,
            scrollable::Status::Active {
                is_horizontal_scrollbar_disabled: false,
                is_vertical_scrollbar_disabled: false,
            },
        );
        let line = Background::Color(ink(theme.palette().line));
        assert_eq!(idle.vertical_rail.scroller.background, line);
        let dragged = theme.style(
            &Bar::Shown,
            scrollable::Status::Dragged {
                is_horizontal_scrollbar_dragged: false,
                is_vertical_scrollbar_dragged: true,
                is_horizontal_scrollbar_disabled: false,
                is_vertical_scrollbar_disabled: false,
            },
        );
        let dim = Background::Color(ink(theme.palette().dim));
        assert_eq!(dragged.vertical_rail.scroller.background, dim);
        assert_eq!(dragged.horizontal_rail.scroller.background, line);
        let hovered = theme.style(
            &Bar::Shown,
            scrollable::Status::Hovered {
                is_horizontal_scrollbar_hovered: true,
                is_vertical_scrollbar_hovered: false,
                is_horizontal_scrollbar_disabled: false,
                is_vertical_scrollbar_disabled: false,
            },
        );
        assert_eq!(hovered.horizontal_rail.scroller.background, dim);
        <Theme as scrollable::Catalog>::default();
        let _: Element<'_, ()> = scroll_area(crate::ui::text::interface("x")).into();
        let _: Scrollable<'_, (), Theme, Renderer> = scroll_both(crate::ui::text::interface("x"));
    }

    #[test]
    fn the_keys_scroll_by_a_step_a_page_less_a_step_or_to_an_end() {
        let named = |key| Key::Named(key);
        let to = |key, shift, at, shown| scroll_key(&named(key), shift, at, shown, 1000.0);
        assert_eq!(to(Named::ArrowDown, false, 0.0, 240.0), Some(60.0));
        assert_eq!(to(Named::ArrowUp, false, 30.0, 240.0), Some(0.0));
        assert_eq!(to(Named::PageDown, false, 100.0, 240.0), Some(280.0));
        assert_eq!(to(Named::PageUp, false, 100.0, 240.0), Some(0.0));
        assert_eq!(to(Named::Space, false, 100.0, 240.0), Some(280.0));
        assert_eq!(to(Named::Space, true, 300.0, 240.0), Some(120.0));
        assert_eq!(to(Named::PageDown, false, 900.0, 240.0), Some(1000.0));
        assert_eq!(to(Named::End, false, 0.0, 240.0), Some(1000.0));
        assert_eq!(to(Named::Home, false, 500.0, 240.0), Some(0.0));
        // An area shorter than a step still pages by one.
        assert_eq!(to(Named::PageDown, false, 0.0, 50.0), Some(60.0));
        assert_eq!(to(Named::Tab, false, 0.0, 240.0), None);
        assert_eq!(
            scroll_key(&Key::Character("j".into()), false, 0.0, 240.0, 1000.0),
            None
        );
    }

    #[test]
    fn a_glide_eases_to_its_end_and_stops() {
        let mut glide = Glide::new(Vector::new(0.0, 0.0), Vector::new(0.0, 120.0));
        let start = Instant::now();
        assert_eq!(glide.at(start), (Vector::new(0.0, 0.0), false));
        let (quarter, over) = glide.at(start + motion::NORMAL / 4);
        assert!(
            !over && quarter.y > 60.0 && quarter.y < 120.0,
            "{quarter:?}"
        );
        let (end, over) = glide.at(start + motion::NORMAL + Duration::from_millis(1));
        assert!(over && end == Vector::new(0.0, 120.0));
    }

    #[test]
    fn the_bar_stays_then_fades_then_goes() {
        assert_eq!(bar_opacity(Duration::ZERO), (1.0, Some(BAR_STAYS)));
        let (mid, again) = bar_opacity(BAR_STAYS + BAR_FADES / 2);
        assert!(
            mid > 0.0 && mid < 1.0 && again == Some(Duration::ZERO),
            "{mid}"
        );
        assert_eq!(bar_opacity(BAR_STAYS + BAR_FADES), (0.0, None));
    }

    #[test]
    fn the_thumb_is_the_share_shown_where_it_is() {
        let bounds = Rectangle::new(iced::Point::new(0.0, 0.0), Size::new(400.0, 300.0));
        let content = Rectangle::new(iced::Point::new(0.0, 0.0), Size::new(400.0, 900.0));
        let top = thumb(Vector::new(0.0, 0.0), bounds, content).unwrap();
        // As far in from the top and the bottom as from the side; a third of the track long.
        let track = 300.0 - 2.0 * space::XXS;
        assert_eq!((top.y, top.height), (space::XXS, track / 3.0));
        assert_eq!(top.x, 400.0 - space::XXS - space::S);
        let bottom = thumb(Vector::new(0.0, 600.0), bounds, content).unwrap();
        assert_eq!(bottom.y + bottom.height, 300.0 - space::XXS);
        // Everything shows: no bar.
        assert!(thumb(Vector::new(0.0, 0.0), bounds, bounds).is_none());
    }

    /// Where the stretch is at each frame for `ms`, at 60 frames a second.
    fn frames(stretch: Stretch, start: Instant, ms: u64) -> Vec<f32> {
        (0..=ms / 16)
            .map(|frame| stretch.at(start + Duration::from_millis(frame * 16)).0)
            .collect()
    }

    #[test]
    fn a_wheel_step_past_the_end_bumps_and_springs_back() {
        let start = Instant::now();
        let mut stretch = Stretch::default();
        stretch.kick(start, LINE, REACH);
        let path = frames(stretch, start, 600);
        let peak = path.iter().copied().fold(0.0, f32::max);
        // About a third of the step at the most, and never past the end it went from.
        assert!(
            peak > LINE * BUMP * 0.9 && peak <= LINE * BUMP + 0.01,
            "{peak}"
        );
        assert!(path.iter().all(|at| *at >= 0.0), "{path:?}");
        // Back at rest in about motion::SLOW.
        assert!(
            !stretch.moving(start + motion::SLOW + motion::FAST),
            "{path:?}"
        );
        // Past the top, the same the other way.
        let mut up = Stretch::default();
        up.kick(start, -LINE, REACH);
        let low = frames(up, start, 600).into_iter().fold(0.0, f32::min);
        assert!((low + peak).abs() < 0.01, "{low} {peak}");
    }

    #[test]
    fn a_pull_gives_less_the_further_it_goes_and_never_past_its_reach() {
        let start = Instant::now();
        let mut stretch = Stretch::default();
        let mut last = 0.0;
        let mut steps = Vec::new();
        for _ in 0..200 {
            stretch.pull(start, 10.0, REACH);
            let (at, speed) = stretch.at(start);
            assert_eq!(speed, 0.0);
            steps.push(at - last);
            last = at;
        }
        assert!(last < REACH, "{last}");
        assert!(steps.windows(2).all(|pair| pair[1] <= pair[0]), "{steps:?}");
        assert_eq!(steps[0], 10.0);
    }

    #[test]
    fn pulling_back_takes_the_stretch_in_first() {
        let start = Instant::now();
        let mut stretch = Stretch::default();
        stretch.pull(start, 30.0, REACH);
        let (at, _) = stretch.at(start);
        // Back by less than it went: nothing left to scroll.
        assert_eq!(stretch.give_back(start, -10.0), 0.0);
        assert!((stretch.at(start).0 - (at - 10.0)).abs() < 1e-4);
        // Back by more: at rest, and the rest scrolls.
        let left = stretch.give_back(start, -50.0);
        assert!((left - (at - 60.0)).abs() < 1e-4, "{left}");
        assert_eq!(stretch, Stretch::default());
        assert!(!stretch.moving(start));
    }

    #[test]
    fn give_is_whole_back_toward_the_end_and_less_further_out() {
        assert_eq!(give(0.0, 5.0, REACH), 1.0);
        assert_eq!(give(40.0, -5.0, REACH), 1.0);
        assert_eq!(give(-40.0, 5.0, REACH), 1.0);
        assert_eq!(give(40.0, 5.0, 80.0), 0.25);
        assert_eq!(give(-40.0, -5.0, 80.0), 0.25);
        assert_eq!(give(80.0, 5.0, 80.0), 0.0);
        assert_eq!(give(120.0, 5.0, 80.0), 0.0);
    }

    #[test]
    fn a_transaction_ends_as_iceds_does() {
        let click = Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left));
        let left = Event::Mouse(mouse::Event::CursorLeft);
        let moved = Event::Mouse(mouse::Event::CursorMoved {
            position: iced::Point::ORIGIN,
        });
        let wheel = Event::Mouse(mouse::Event::WheelScrolled {
            delta: mouse::ScrollDelta::Lines { x: 0.0, y: -1.0 },
        });
        let ms = Duration::from_millis;
        assert!(ends_transaction(&click, ms(0)));
        assert!(ends_transaction(&left, ms(0)));
        assert!(!ends_transaction(&moved, ms(100)) && ends_transaction(&moved, ms(101)));
        assert!(!ends_transaction(&wheel, ms(1500)) && ends_transaction(&wheel, ms(1501)));
    }
}
