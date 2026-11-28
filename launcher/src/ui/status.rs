// SPDX-License-Identifier: GPL-3.0-or-later

//! Status chips: a check's or a step's state, a dot and a word, never color alone.

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Tree, Widget, tree};
use iced::advanced::{mouse, renderer};
use iced::{Length, Point, Rectangle, Size};

use crate::ui::tokens::{size, space, typography};

use crate::ui::paint::{self, Plain};
use crate::ui::theme::{Status, color};
use crate::ui::{Element, Renderer, Theme};

/// A chip saying a state in a word: "Passed", "Needs setup mode".
pub fn status(status: Status, word: impl Into<String>) -> Chip {
    Chip {
        status,
        word: word.into(),
    }
}

/// A status chip ([`status`]): a dot and a few words in a pill, in its status's accent.
pub struct Chip {
    status: Status,
    word: String,
}

const PAD_X: f32 = space::M;
const PAD_Y: f32 = space::XXS;

impl<Message> Widget<Message, Theme, Renderer> for Chip {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<Plain>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(Plain::default())
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
        let word = tree.state.downcast_mut::<Plain>();
        paint::shape(word, &typography::META, &self.word);
        let text = word.min_bounds();
        let size = Size::new(
            (2.0 * PAD_X + size::DOT + space::S + text.width).ceil(),
            (2.0 * PAD_Y + text.height).ceil(),
        );
        layout::Node::new(limits.resolve(Length::Shrink, Length::Shrink, size))
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let word = tree.state.downcast_ref::<Plain>();
        let bounds = layout.bounds();
        let look = theme.status(self.status);
        paint::boxed(renderer, bounds, &look);
        let ink = color(look.ink);
        let dot = Point::new(bounds.x + PAD_X + size::DOT / 2.0, bounds.center_y());
        paint::dot(renderer, dot, size::DOT, ink, Some(theme.spark(look.ink)));
        let text = word.min_bounds();
        let at = Point::new(
            bounds.x + PAD_X + size::DOT + space::S,
            bounds.center_y() - text.height / 2.0,
        );
        paint::fill(renderer, word, at, ink, *viewport);
    }
}

impl<'a, Message: 'a> From<Chip> for Element<'a, Message> {
    fn from(chip: Chip) -> Self {
        Element::new(chip)
    }
}
