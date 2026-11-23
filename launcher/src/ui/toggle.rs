// SPDX-License-Identifier: GPL-3.0-or-later

//! A toggle group: a few choices side by side in a rounded box, the chosen one raised, such as
//! dark or light, or a view.

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, renderer};
use iced::keyboard::{self, Key, key::Named};
use iced::{Event, Length, Point, Rectangle, Size, touch, window};

use crate::ui::tokens::{motion, size, space, typography};
use crate::ui::transition::Transition;

use crate::ui::focus::Focus;
use crate::ui::input;
use crate::ui::paint::{self, Plain};
use crate::ui::theme::{Interaction, Look, color};
use crate::ui::{Element, Renderer, Theme};

/// Room round the segments inside the group.
const INSET: f32 = crate::ui::tokens::size::SEGMENT_INSET;
/// Room either side of a segment's words.
const PAD: f32 = space::L;

/// The choices, with the chosen one's index. Without a message to send it is disabled.
pub fn toggle_group<'a, Message>(
    choices: impl IntoIterator<Item = impl Into<String>>,
    chosen: usize,
) -> ToggleGroup<'a, Message> {
    ToggleGroup {
        choices: choices.into_iter().map(Into::into).collect(),
        chosen,
        homogeneous: false,
        pill: false,
        height: size::TOGGLE,
        on_choose: None,
        preview: None,
    }
}

/// A toggle group: one Tab stop; Left and Right, Home and End move the choice, as a click does.
pub struct ToggleGroup<'a, Message> {
    choices: Vec<String>,
    chosen: usize,
    homogeneous: bool,
    pill: bool,
    height: f32,
    on_choose: Option<Box<dyn Fn(usize) -> Message + 'a>>,
    preview: Option<(usize, Interaction, bool)>,
}

impl<'a, Message> ToggleGroup<'a, Message> {
    /// The message for a choice, with its index: without one, it's disabled.
    pub fn on_choose(self, on_choose: impl Fn(usize) -> Message + 'a) -> Self {
        Self {
            on_choose: Some(Box::new(on_choose)),
            ..self
        }
    }

    /// Every segment as wide as the widest.
    pub fn homogeneous(self, homogeneous: bool) -> Self {
        Self {
            homogeneous,
            ..self
        }
    }

    /// Fully rounded, a pill, its segments too; otherwise its corners are a button's.
    pub fn pill(self, pill: bool) -> Self {
        Self { pill, ..self }
    }

    /// Its corners and its segments': a pill's, half its height and half theirs, or a button's
    /// and, inside them, running alongside.
    fn radii(&self) -> (f32, f32) {
        if self.pill {
            (self.height / 2.0, (self.height - 2.0 * INSET) / 2.0)
        } else {
            (
                crate::ui::tokens::radius::BUTTON,
                crate::ui::tokens::radius::SEGMENT,
            )
        }
    }

    /// Draws one segment as if in this state, and the focus outline, for the gallery.
    pub fn preview(self, segment: usize, interaction: Interaction, focused: bool) -> Self {
        Self {
            preview: Some((segment, interaction, focused)),
            ..self
        }
    }
}

/// Where each segment is, left to right, from the words' widths, in a group `height` tall: `widths`
/// of the words, all as wide as the widest if `homogeneous`.
pub(crate) fn segments(
    widths: &[f32],
    homogeneous: bool,
    at: Point,
    height: f32,
) -> Vec<Rectangle> {
    let widest = widths.iter().copied().fold(0.0, f32::max);
    let mut x = at.x + INSET;
    widths
        .iter()
        .map(|width| {
            // Whole pixels, so the group's edges and the segments' fall on them.
            let width = if homogeneous { widest } else { *width }.ceil() + 2.0 * PAD;
            let segment = Rectangle::new(
                Point::new(x, at.y + INSET),
                Size::new(width, height - 2.0 * INSET),
            );
            x += width;
            segment
        })
        .collect()
}

/// The choice a key moves to among `count`, from `chosen`.
pub(crate) fn step(chosen: usize, count: usize, key: &Key) -> Option<usize> {
    let last = count.checked_sub(1)?;
    let to = match key {
        Key::Named(Named::ArrowLeft) => chosen.saturating_sub(1),
        Key::Named(Named::ArrowRight) => (chosen + 1).min(last),
        Key::Named(Named::Home) => 0,
        Key::Named(Named::End) => last,
        _ => return None,
    };
    Some(to)
}

#[derive(Default)]
struct State {
    focus: Focus,
    words: Vec<Plain>,
    pressed: Option<usize>,
    drawn: Option<(Option<usize>, Option<usize>, bool, usize)>,
    /// Where the chosen segment's highlight is drawn, by segment, between them as it slides or
    /// is dragged; and the choice it was last drawn for.
    at: f32,
    seen: Option<usize>,
    slide: Option<Transition>,
    /// A press: where it began across, where the highlight was, and whether it has moved far
    /// enough to be a drag. As a switch's knob: dragged, the highlight follows the pointer, and
    /// the choice changes when it's let go.
    drag: Option<(f32, f32, bool)>,
    /// Its focus ring, fading in and out with keyboard focus on the glows' own timing.
    focus_lit: crate::ui::transition::Lit,
}

impl State {
    /// The highlight slides to `segment` from where it is, or with motion reduced, jumps there.
    fn slide_to(&mut self, segment: usize) {
        let to = segment as f32;
        self.slide = Transition::start(self.at, to, motion::FAST);
        if self.slide.is_none() {
            self.at = to;
        }
    }
}

/// The highlight's rectangle `at` the given place among `segments`: between two, as far between
/// them as `at` is.
fn highlight(segments: &[Rectangle], at: f32) -> Option<Rectangle> {
    let last = segments.len().checked_sub(1)?;
    let at = at.clamp(0.0, last as f32);
    let (from, to) = (at.floor() as usize, (at.ceil() as usize).min(last));
    let share = at - from as f32;
    let (a, b) = (segments[from], segments[to]);
    let mix = |x: f32, y: f32| x + (y - x) * share;
    Some(Rectangle::new(
        Point::new(mix(a.x, b.x), mix(a.y, b.y)),
        Size::new(mix(a.width, b.width), mix(a.height, b.height)),
    ))
}

impl<Message> ToggleGroup<'_, Message> {
    /// The segment under the pointer: its layout's children are the segments, laid out once.
    fn segment_at(layout: Layout<'_>, cursor: mouse::Cursor) -> Option<usize> {
        let point = cursor.position_over(layout.bounds())?;
        layout
            .children()
            .position(|segment| segment.bounds().contains(point))
    }

    fn choose(&mut self, segment: usize, shell: &mut Shell<'_, Message>) {
        if let Some(on_choose) = &self.on_choose
            && segment != self.chosen
            && segment < self.choices.len()
        {
            shell.publish(on_choose(segment));
            self.chosen = segment;
        }
    }
}

impl<Message> Widget<Message, Theme, Renderer> for ToggleGroup<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Shrink, Length::Fixed(self.height))
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        _renderer: &Renderer,
        _limits: &layout::Limits,
    ) -> layout::Node {
        let state = tree.state.downcast_mut::<State>();
        state.words.resize_with(self.choices.len(), Plain::default);
        for (plain, choice) in state.words.iter_mut().zip(&self.choices) {
            paint::shape(plain, &typography::INTERFACE, choice);
        }
        // As wide as its segments, whatever room it's given: a toggle group doesn't shrink.
        let widths: Vec<f32> = state
            .words
            .iter()
            .map(|plain| plain.min_bounds().width)
            .collect();
        let segments = segments(&widths, self.homogeneous, Point::ORIGIN, self.height);
        let width = segments
            .last()
            .map_or(0.0, |last| last.x + last.width + INSET);
        let nodes = segments
            .iter()
            .map(|segment| layout::Node::new(segment.size()).move_to(segment.position()))
            .collect();
        layout::Node::with_children(Size::new(width, self.height), nodes)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        _renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        if self.on_choose.is_some() && self.preview.is_none() && !self.choices.is_empty() {
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
        let state = tree.state.downcast_mut::<State>();
        state
            .focus
            .keep_if(self.on_choose.is_some() && !self.choices.is_empty());
        // The highlight slides, a frame at a time.
        if let Event::Window(window::Event::RedrawRequested(now)) = event
            && let Some(slide) = &mut state.slide
        {
            let going = slide.frame(*now);
            state.at = slide.at();
            if going {
                shell.request_redraw();
            } else {
                state.slide = None;
            }
        }
        // The choice changed from outside, as Ctrl+L or a key does: the highlight slides there.
        match state.seen {
            None => state.at = self.chosen as f32,
            Some(seen) if seen != self.chosen && state.drag.is_none() => {
                state.slide_to(self.chosen);
                shell.request_redraw();
            }
            Some(_) => {}
        }
        state.seen = Some(self.chosen);
        let over = Self::segment_at(layout, cursor);
        let last = self.choices.len().saturating_sub(1) as f32;
        if self.on_choose.is_some() {
            match event {
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerPressed { .. })
                    if over.is_some() =>
                {
                    state.pressed = over;
                    if let Some(at) = cursor.position() {
                        state.drag = Some((at.x, state.at, false));
                    }
                    state.focus.click();
                    shell.capture_event();
                }
                Event::Mouse(mouse::Event::CursorMoved { position })
                | Event::Touch(touch::Event::FingerMoved { position, .. }) => {
                    if let Some((from_x, from_at, dragged)) = &mut state.drag {
                        // Further than a slip, it drags the highlight rather than clicks.
                        let moved = position.x - *from_x;
                        *dragged |= moved.abs() > size::SLIP;
                        if *dragged {
                            let step =
                                (layout.bounds().width / self.choices.len().max(1) as f32).max(1.0);
                            state.slide = None;
                            state.at = (*from_at + moved / step).clamp(0.0, last);
                            shell.request_redraw();
                            shell.capture_event();
                        }
                    }
                }
                Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerLifted { .. }) => {
                    if state.pressed.take().is_some() {
                        // Dragged, the nearer segment; clicked, the one let go over. The choice
                        // changes now, as it's let go.
                        let dragged = state.drag.take().is_some_and(|(_, _, dragged)| dragged);
                        let segment = if dragged {
                            state.at.round() as usize
                        } else {
                            over.unwrap_or(self.chosen)
                        };
                        state.slide_to(segment);
                        state.seen = Some(segment);
                        self.choose(segment, shell);
                        shell.request_redraw();
                        shell.capture_event();
                    }
                }
                Event::Touch(touch::Event::FingerLost { .. }) => {
                    state.pressed = None;
                    if state.drag.take().is_some() {
                        state.slide_to(self.chosen);
                        shell.request_redraw();
                    }
                }
                Event::Keyboard(keyboard::Event::KeyPressed { key, modifiers, .. })
                    if state.focus.is_focused() && input::plain(*modifiers) =>
                {
                    if let Some(segment) = step(self.chosen, self.choices.len(), key) {
                        state.focus.key();
                        if segment != self.chosen {
                            state.slide_to(segment);
                            state.seen = Some(segment);
                            shell.request_redraw();
                        }
                        self.choose(segment, shell);
                        shell.capture_event();
                    }
                }
                _ => {}
            }
        } else {
            state.pressed = None;
            state.drag = None;
        }
        state.focus_lit.frame(event, shell);
        state.focus_lit.aim(state.focus.is_shown(), shell);
        let now = (over, state.pressed, state.focus.is_shown(), self.chosen);
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
        let bounds = layout.bounds();
        let disabled = self.on_choose.is_none();
        let (radius, segment_radius) = self.radii();
        paint::boxed(
            renderer,
            bounds,
            &Look {
                radius,
                ..theme.toggle_group()
            },
        );
        let over = Self::segment_at(layout, cursor);
        let height = typography::INTERFACE.size * typography::INTERFACE.line_height;
        let segments: Vec<Rectangle> = layout.children().map(|segment| segment.bounds()).collect();
        let interaction = |index: usize| match self.preview {
            Some((shown, interaction, _)) if shown == index => interaction,
            _ if disabled => Interaction::Disabled,
            _ if state.pressed == Some(index) => Interaction::Pressed,
            _ if over == Some(index) => Interaction::Hovered,
            _ => Interaction::Idle,
        };
        // Where the highlight is: the chosen segment's, sliding or dragged between them.
        let at = match (self.preview, state.seen) {
            (Some(_), _) | (None, None) => self.chosen as f32,
            (None, Some(_)) => state.at,
        };
        let nearest = (at.round().max(0.0) as usize).min(segments.len().saturating_sub(1));
        // The others first, as they look unchosen; then the highlight over them.
        for (index, segment) in segments.iter().enumerate() {
            let look = Look {
                radius: segment_radius,
                ..theme.toggle_segment(false, interaction(index))
            };
            paint::boxed(renderer, *segment, &look);
        }
        let shown = highlight(&segments, at);
        if let Some(shown) = shown {
            let look = Look {
                radius: segment_radius,
                ..theme.toggle_segment(true, interaction(nearest))
            };
            paint::boxed(renderer, shown, &look);
        }
        for (index, (segment, words)) in segments.iter().zip(&state.words).enumerate() {
            let look = theme.toggle_segment(index == nearest, interaction(index));
            let width = words.min_bounds().width;
            let place = Point::new(
                segment.center_x() - width / 2.0,
                segment.center_y() - height / 2.0,
            );
            paint::fill(renderer, words, place, color(look.ink), bounds);
        }
        let focused = match self.preview {
            Some((_, _, focused)) => focused,
            None => state.focus.is_shown(),
        };
        // The focus ring fades in and out with keyboard focus, round the highlight wherever it is.
        let ring = if self.preview.is_some() {
            f32::from(u8::from(focused))
        } else {
            state.focus_lit.share()
        };
        if let Some(shown) = shown {
            paint::outline_lit(renderer, theme, shown, segment_radius, ring);
        }
    }

    /// A hand that grabs: it's dragged across, as well as clicked. A header bar leaves the
    /// pointer's drag to it instead of moving the window.
    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        let pressed = tree.state.downcast_ref::<State>().pressed.is_some();
        if self.on_choose.is_some() && self.preview.is_none() && pressed {
            mouse::Interaction::Grabbing
        } else if self.on_choose.is_some()
            && self.preview.is_none()
            && Self::segment_at(layout, cursor).is_some()
        {
            mouse::Interaction::Grab
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, Message: 'a> From<ToggleGroup<'a, Message>> for Element<'a, Message> {
    fn from(group: ToggleGroup<'a, Message>) -> Self {
        Element::new(group)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_sit_side_by_side_inside_the_group() {
        let at = Point::new(10.0, 20.0);
        let s = segments(&[30.0, 50.0], false, at, size::TOGGLE);
        assert_eq!(
            s[0],
            Rectangle::new(Point::new(13.0, 23.0), Size::new(54.0, 26.0))
        );
        assert_eq!(s[1].x, 13.0 + 54.0);
        assert_eq!(s[1].width, 74.0);
        // Homogeneous: as wide as the widest.
        let s = segments(&[30.0, 50.0], true, at, size::TOGGLE);
        assert_eq!((s[0].width, s[1].width), (74.0, 74.0));
        assert!(segments(&[], false, at, size::TOGGLE).is_empty());
    }

    #[test]
    fn keys_move_the_choice_and_stop_at_the_ends() {
        let left = Key::Named(Named::ArrowLeft);
        let right = Key::Named(Named::ArrowRight);
        assert_eq!(step(0, 3, &left), Some(0));
        assert_eq!(step(0, 3, &right), Some(1));
        assert_eq!(step(2, 3, &right), Some(2));
        assert_eq!(step(1, 3, &Key::Named(Named::Home)), Some(0));
        assert_eq!(step(0, 3, &Key::Named(Named::End)), Some(2));
        assert_eq!(step(0, 3, &Key::Character("a".into())), None);
        assert_eq!(step(0, 0, &right), None);
    }
}
