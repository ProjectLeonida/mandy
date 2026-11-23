// SPDX-License-Identifier: GPL-3.0-or-later

//! A single choice among a few options, each a card with a title and a description: the Secure
//! Boot mode, the disk to install to.

use iced::advanced::layout::{self, Layout};
use iced::advanced::text::Renderer as _;
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, renderer};
use iced::keyboard::{self, Key, key::Named};
use iced::{Event, Length, Point, Rectangle, Size, touch};

use crate::ui::tokens::{size, space, typography};

use crate::ui::focus::Focus;
use crate::ui::input;
use crate::ui::paint::{self, Plain};
use crate::ui::theme::{Interaction, Look, color};
use crate::ui::transition::Lit;
use crate::ui::{Element, Renderer, Theme};

/// A single choice: what's chosen, if anything yet, and the message a new choice sends. Without
/// one it is disabled.
pub fn choice<'a, T: Copy + PartialEq, Message>(chosen: Option<T>) -> Choice<'a, T, Message> {
    Choice {
        options: Vec::new(),
        chosen,
        on_choose: None,
        preview: None,
    }
}

struct Choosable<T> {
    value: T,
    title: String,
    description: String,
    enabled: bool,
}

/// A single choice. It is one stop for Tab; the arrow keys, Home and End move the choice among
/// the options it can take, and Space takes the one the keyboard is on.
pub struct Choice<'a, T, Message> {
    options: Vec<Choosable<T>>,
    chosen: Option<T>,
    on_choose: Option<Box<dyn Fn(T) -> Message + 'a>>,
    preview: Option<(usize, Interaction, bool)>,
}

impl<'a, T: Copy + PartialEq, Message> Choice<'a, T, Message> {
    /// An option, with a title and a line or two saying what it means.
    pub fn option(
        mut self,
        value: T,
        title: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.options.push(Choosable {
            value,
            title: title.into(),
            description: description.into(),
            enabled: true,
        });
        self
    }

    /// An option that can't be taken here, saying why in its description.
    pub fn unavailable(
        mut self,
        value: T,
        title: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self = self.option(value, title, description);
        if let Some(option) = self.options.last_mut() {
            option.enabled = false;
        }
        self
    }

    /// The message for a choice, with the option's value: without one, it's disabled.
    pub fn on_choose(self, on_choose: impl Fn(T) -> Message + 'a) -> Self {
        Self {
            on_choose: Some(Box::new(on_choose)),
            ..self
        }
    }

    /// Draws one option as if in this state, and the focus outline, for the gallery.
    pub fn preview(self, option: usize, interaction: Interaction, focused: bool) -> Self {
        Self {
            preview: Some((option, interaction, focused)),
            ..self
        }
    }

    fn chosen_index(&self) -> Option<usize> {
        self.options
            .iter()
            .position(|option| Some(option.value) == self.chosen)
    }

    /// The option the keyboard is on: the chosen one, or the first that can be taken.
    fn current(&self) -> Option<usize> {
        self.chosen_index()
            .or_else(|| self.options.iter().position(|option| option.enabled))
    }

    fn interaction(&self, state: &State, index: usize, over: bool) -> Interaction {
        if self.on_choose.is_none() || !self.options[index].enabled {
            Interaction::Disabled
        } else if state.pressed == Some(index) {
            Interaction::Pressed
        } else if over {
            Interaction::Hovered
        } else {
            Interaction::Idle
        }
    }

    fn choose(&mut self, index: usize, shell: &mut Shell<'_, Message>) {
        let option = &self.options[index];
        if let Some(on_choose) = &self.on_choose
            && option.enabled
            && Some(option.value) != self.chosen
        {
            shell.publish(on_choose(option.value));
            // Taken at once, so a second key in the same batch of events starts from it, as the
            // app will.
            self.chosen = Some(option.value);
        }
    }
}

/// The option an arrow key moves to: the next that can be taken in that direction, or none.
pub(crate) fn step(enabled: &[bool], from: Option<usize>, key: &Named) -> Option<usize> {
    let mut indices: Box<dyn Iterator<Item = usize>> = match (key, from) {
        (Named::ArrowDown | Named::ArrowRight, Some(at)) => Box::new(at + 1..enabled.len()),
        (Named::ArrowUp | Named::ArrowLeft, Some(at)) => Box::new((0..at).rev()),
        (Named::Home, _) | (Named::ArrowDown | Named::ArrowRight, None) => {
            Box::new(0..enabled.len())
        }
        (Named::End, _) | (Named::ArrowUp | Named::ArrowLeft, None) => {
            Box::new((0..enabled.len()).rev())
        }
        _ => return None,
    };
    indices.find(|at| enabled[*at])
}

/// What it last drew: the option under the pointer, the pressed one, the outline, the current.
type Drawn = (Option<usize>, Option<usize>, bool, Option<usize>);

#[derive(Default)]
struct State {
    focus: Focus,
    pressed: Option<usize>,
    titles: Vec<Plain>,
    descriptions: Vec<Plain>,
    drawn: Option<Drawn>,
    /// Each card's glows and focus ring, fading in and out on the glows' own timing: under the
    /// pointer, chosen, and the focus ring round the current one.
    hover_lit: Vec<Lit>,
    chosen_lit: Vec<Lit>,
    ring_lit: Vec<Lit>,
}

/// The card's padding, across and down; the radio's gap to the text is `space::L` too.
const PAD_X: f32 = space::L;
const PAD_Y: f32 = space::M;

impl<T: Copy + PartialEq, Message> Widget<Message, Theme, Renderer> for Choice<'_, T, Message> {
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
        let width = limits.max().width;
        let text_width = (width - 2.0 * PAD_X - size::CHECK - space::L).max(0.0);
        state.titles.resize_with(self.options.len(), Plain::default);
        state
            .descriptions
            .resize_with(self.options.len(), Plain::default);
        let mut cards = Vec::with_capacity(self.options.len());
        let mut y = 0.0;
        for (at, option) in self.options.iter().enumerate() {
            paint::shape(&mut state.titles[at], &typography::INTERFACE, &option.title);
            let mut text = paint::unbounded(&typography::INTERFACE, &option.description);
            text.bounds = Size::new(text_width, f32::INFINITY);
            text.wrapping = iced::advanced::text::Wrapping::Word;
            state.descriptions[at].update(text);
            let height = 2.0 * PAD_Y
                + state.titles[at].min_height()
                + if option.description.is_empty() {
                    0.0
                } else {
                    space::XXS + state.descriptions[at].min_height()
                };
            cards.push(
                layout::Node::new(Size::new(width, height.ceil())).move_to(Point::new(0.0, y)),
            );
            y += height.ceil() + space::S;
        }
        let height = (y - space::S).max(0.0);
        layout::Node::with_children(Size::new(width, height), cards)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        _renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        if self.on_choose.is_some()
            && self.preview.is_none()
            && self.options.iter().any(|option| option.enabled)
        {
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
        state
            .focus
            .keep_if(self.on_choose.is_some() && self.options.iter().any(|option| option.enabled));
        let over = layout
            .children()
            .position(|card| cursor.is_over(card.bounds()));
        if self.on_choose.is_some() {
            match event {
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerPressed { .. }) => {
                    if let Some(at) = over.filter(|at| self.options[*at].enabled) {
                        state.pressed = Some(at);
                        state.focus.click();
                        shell.capture_event();
                    }
                }
                Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
                | Event::Touch(touch::Event::FingerLifted { .. }) => {
                    if let Some(at) = state.pressed.take() {
                        if over == Some(at) {
                            self.choose(at, shell);
                        }
                        shell.capture_event();
                    }
                }
                Event::Touch(touch::Event::FingerLost { .. }) => state.pressed = None,
                Event::Keyboard(keyboard::Event::KeyPressed {
                    key: Key::Named(key),
                    modifiers,
                    ..
                }) if state.focus.is_focused() && input::plain(*modifiers) => {
                    let enabled: Vec<_> = self.options.iter().map(|o| o.enabled).collect();
                    let target = match key {
                        Named::Space => self.current(),
                        _ => step(&enabled, self.chosen_index(), key),
                    };
                    if let Some(at) = target {
                        state.focus.key();
                        self.choose(at, shell);
                        shell.capture_event();
                    } else if matches!(
                        key,
                        Named::ArrowUp | Named::ArrowDown | Named::ArrowLeft | Named::ArrowRight
                    ) {
                        // At the end: the key stays here rather than scrolling the page.
                        state.focus.key();
                        shell.capture_event();
                    }
                }
                _ => {}
            }
        } else {
            state.pressed = None;
        }
        // Each card's glows and focus ring fade in and out on the glows' own timing.
        let count = self.options.len();
        state.hover_lit.resize_with(count, Lit::default);
        state.chosen_lit.resize_with(count, Lit::default);
        state.ring_lit.resize_with(count, Lit::default);
        let current = self.current();
        let shown = state.focus.is_shown();
        for at in 0..count {
            let interaction = self.interaction(state, at, over == Some(at));
            let enabled = interaction != Interaction::Disabled;
            let hovered = matches!(interaction, Interaction::Hovered | Interaction::Pressed);
            let chosen = Some(self.options[at].value) == self.chosen;
            state.hover_lit[at].frame(event, shell);
            state.hover_lit[at].aim(hovered, shell);
            state.chosen_lit[at].frame(event, shell);
            state.chosen_lit[at].aim(chosen && enabled, shell);
            state.ring_lit[at].frame(event, shell);
            state.ring_lit[at].aim(shown && current == Some(at), shell);
        }
        let now = (over, state.pressed, state.focus.is_shown(), self.current());
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
        let focused = match self.preview {
            Some((_, _, focused)) => focused,
            None => state.focus.is_shown(),
        };
        let current = self.current();
        for (at, card) in layout.children().enumerate() {
            let (Some(option), Some(title), Some(description)) = (
                self.options.get(at),
                state.titles.get(at),
                state.descriptions.get(at),
            ) else {
                continue;
            };
            let bounds = card.bounds();
            let interaction = match self.preview {
                Some((shown, interaction, _)) if shown == at => interaction,
                _ => self.interaction(state, at, cursor.is_over(bounds)),
            };
            let chosen = Some(option.value) == self.chosen;
            let look = theme.option(chosen, interaction);
            let enabled = interaction != Interaction::Disabled;
            let hovered = matches!(interaction, Interaction::Hovered | Interaction::Pressed);
            // How lit each glow is: on their fades, or whole in a preview, a picture of a state.
            let lit = |lits: &Vec<Lit>, whole: bool| match self.preview {
                Some(_) => f32::from(u8::from(whole)),
                None => lits.get(at).map_or(0.0, Lit::share),
            };
            let chosen_share = lit(&state.chosen_lit, chosen && enabled);
            let hover_share = lit(&state.hover_lit, hovered);
            let ring_share = lit(&state.ring_lit, focused && current == Some(at));
            // The card: the chosen glow at its share, and under it the pointer's glow, which
            // gives way to the chosen one as a card is chosen.
            let chosen_glow = theme.option(true, Interaction::Idle).glow;
            let hover_glow = theme.option(false, Interaction::Hovered).glow;
            if let Some(glow) = hover_glow {
                let card_hover = hover_share * (1.0 - chosen_share);
                paint::boxed(
                    renderer,
                    bounds,
                    &Look {
                        fill: None,
                        line: None,
                        ..look
                    }
                    .lit(Some(glow), card_hover),
                );
            }
            let look = Look { glow: None, ..look }.lit(chosen_glow, chosen_share);
            paint::boxed(renderer, bounds, &look);
            // The radio circle glows under the pointer, on the same fade.
            let radio = theme.radio(chosen, interaction);
            let radio_glow = theme.radio(chosen, Interaction::Hovered).glow;
            let radio = Look {
                glow: None,
                ..radio
            }
            .lit(radio_glow, hover_share);
            let first_line = title.min_height();
            let circle = Rectangle::new(
                Point::new(
                    bounds.x + PAD_X,
                    bounds.y + PAD_Y + (first_line - size::CHECK) / 2.0,
                ),
                Size::new(size::CHECK, size::CHECK),
            );
            paint::boxed(renderer, circle, &radio);
            if chosen {
                paint::dot(renderer, circle.center(), size::DOT, color(radio.ink), None);
            }
            let x = circle.x + size::CHECK + space::L;
            let ink = theme.label(interaction);
            paint::fill(
                renderer,
                title,
                Point::new(x, bounds.y + PAD_Y),
                color(ink),
                *viewport,
            );
            renderer.fill_paragraph(
                description.raw(),
                Point::new(x, bounds.y + PAD_Y + first_line + space::XXS),
                color(theme.ink(crate::ui::tokens::Ink::Muted)),
                *viewport,
            );
            // The focus ring round the current card fades in and out, and moves by fading.
            paint::outline_lit(renderer, theme, bounds, look.radius, ring_share);
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
        let over = layout
            .children()
            .zip(&self.options)
            .any(|(card, option)| option.enabled && cursor.is_over(card.bounds()));
        if self.on_choose.is_some() && over {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, T: Copy + PartialEq + 'a, Message: 'a> From<Choice<'a, T, Message>>
    for Element<'a, Message>
{
    fn from(choice: Choice<'a, T, Message>) -> Self {
        Element::new(choice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrows_skip_what_cant_be_taken() {
        let enabled = [true, false, true, true];
        assert_eq!(step(&enabled, Some(0), &Named::ArrowDown), Some(2));
        assert_eq!(step(&enabled, Some(2), &Named::ArrowUp), Some(0));
        assert_eq!(step(&enabled, Some(3), &Named::ArrowRight), None);
        assert_eq!(step(&enabled, Some(0), &Named::ArrowLeft), None);
        assert_eq!(step(&enabled, None, &Named::ArrowDown), Some(0));
        assert_eq!(step(&enabled, None, &Named::ArrowUp), Some(3));
        assert_eq!(step(&enabled, Some(2), &Named::Home), Some(0));
        assert_eq!(step(&enabled, Some(0), &Named::End), Some(3));
        assert_eq!(step(&enabled, Some(0), &Named::Enter), None);
        assert_eq!(step(&[false, false], None, &Named::Home), None);
    }
}
