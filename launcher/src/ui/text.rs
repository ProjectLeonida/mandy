// SPDX-License-Identifier: GPL-3.0-or-later

//! Text in the launcher's type roles. The role sets the face, size, line height and case; the ink, set
//! with `.class(…)`, the color, `text` by default:
//!
//! ```
//! use leonida_launcher::ui::{Ink, text};
//!
//! let hint = text::interface("Choose a disk").class(Ink::Muted);
//! ```
//!
//! Labels are spaced out letter by letter, which iced's text can't do.

use std::borrow::Cow;

use crate::ui::tokens::Ink;
use crate::ui::tokens::typography::{self, Role};
use iced::advanced::layout::{self, Layout};
use iced::advanced::text::{self, Renderer as _};
use iced::advanced::widget::{Tree, Widget, tree};
use iced::advanced::{mouse, renderer};
use iced::widget::Text;
use iced::widget::text::{IntoFragment, LineHeight};
use iced::{Length, Pixels, Point, Rectangle, Size};
use unicode_segmentation::UnicodeSegmentation;

use crate::ui::paint::Plain;
use crate::ui::{Element, Renderer, Theme, font};

/// Where a text may break onto a new line, as `.wrapping(…)` takes it: at words by default, or
/// never, for a line that should be cut rather than broken.
pub use iced::widget::text::Wrapping;

/// Text set in a role. Labels are better made with [`label`], which spaces their letters.
pub fn role<'a>(role: &Role, content: impl IntoFragment<'a>) -> Text<'a, Theme, Renderer> {
    let content = content.into_fragment();
    let content = if role.uppercase {
        Cow::Owned(content.to_uppercase())
    } else {
        content
    };
    iced::widget::text(content)
        .font(font::role(role))
        .size(role.size)
        .line_height(LineHeight::Relative(role.line_height))
}

/// A page or window title.
pub fn title<'a>(content: impl IntoFragment<'a>) -> Text<'a, Theme, Renderer> {
    role(&typography::TITLE, content)
}

/// A section heading.
pub fn heading<'a>(content: impl IntoFragment<'a>) -> Text<'a, Theme, Renderer> {
    role(&typography::HEADING, content)
}

/// Interface text: controls, lists, messages.
pub fn interface<'a>(content: impl IntoFragment<'a>) -> Text<'a, Theme, Renderer> {
    role(&typography::INTERFACE, content)
}

/// Metadata, counts, paths and status, in the `muted` ink.
pub fn meta<'a>(content: impl IntoFragment<'a>) -> Text<'a, Theme, Renderer> {
    role(&typography::META, content).class(Ink::Muted)
}

/// A small uppercase label, spaced out, in the `muted` ink.
pub fn label(content: impl AsRef<str>) -> Label {
    Label {
        content: content.as_ref().to_uppercase(),
        ink: Ink::Muted,
        role: &typography::LABEL,
    }
}

/// An app's name, as its header bar shows it: a label a step larger and bolder.
pub fn brand(content: impl AsRef<str>) -> Label {
    Label {
        role: &typography::BRAND,
        ..label(content)
    }
}

/// A label: its letters set one by one, the role's letter spacing after each, one widget, each
/// letter shaped once.
#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    content: String,
    ink: Ink,
    role: &'static crate::ui::tokens::Role,
}

impl Label {
    pub fn ink(self, ink: impl Into<Ink>) -> Self {
        Self {
            ink: ink.into(),
            ..self
        }
    }

    /// The letters, as they're set: grapheme clusters, so an accent stays on its letter.
    #[cfg(test)]
    fn letters(&self) -> impl Iterator<Item = &str> {
        self.content.graphemes(true)
    }
}

impl<'a, Message: 'a> From<Label> for Element<'a, Message> {
    fn from(label: Label) -> Self {
        Element::new(label)
    }
}

/// A label's letters, each shaped once and again only when it changes, and where each goes.
#[derive(Default)]
struct Letters {
    shaped: Vec<Plain>,
    at: Vec<Point>,
}

impl Label {
    /// Its words, each its letters and, but for the last, a no-break space after it: a space
    /// alone would measure nothing.
    fn words(&self) -> impl Iterator<Item = impl Iterator<Item = &str>> {
        let last = self.content.split(' ').count().saturating_sub(1);
        self.content
            .split(' ')
            .enumerate()
            .map(move |(at, word)| word.graphemes(true).chain((at < last).then_some("\u{a0}")))
    }
}

impl<Message> Widget<Message, Theme, Renderer> for Label {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<Letters>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(Letters::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Shrink, Length::Shrink)
    }

    /// Letter by letter, the role's letter spacing after each, and word by word, so a label too
    /// long for its room wraps between words, as words do: as a row of words, each a row of its
    /// letters, would lay them out.
    fn layout(
        &mut self,
        tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let Letters { shaped, at } = tree.state.downcast_mut::<Letters>();
        let role = self.role;
        let spacing = role.letter_spacing_px();
        let room = limits.max();
        let mut count = 0;
        at.clear();
        let (mut x, mut y, mut line, mut widest) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        for word in self.words() {
            // The word's letters, from its start, and how wide and tall it is.
            let first = at.len();
            let (mut along, mut tall) = (0.0_f32, 0.0_f32);
            for letter in word {
                if count == shaped.len() {
                    shaped.push(Plain::default());
                }
                let plain = &mut shaped[count];
                let _ = plain.update(text::Text {
                    content: letter,
                    bounds: room,
                    size: Pixels(role.size),
                    line_height: LineHeight::Relative(role.line_height),
                    font: font::role(role),
                    align_x: text::Alignment::default(),
                    align_y: iced::alignment::Vertical::Top,
                    shaping: text::Shaping::default(),
                    wrapping: text::Wrapping::default(),
                });
                let size = limits.resolve(Length::Shrink, Length::Shrink, plain.min_bounds());
                at.push(Point::new(along, 0.0));
                along += size.width + spacing;
                tall = tall.max(size.height);
                count += 1;
            }
            let wide = (along - spacing).max(0.0);
            // Too wide for what's left of the line: on the next.
            if x != 0.0 && x + wide > room.width {
                widest = widest.max(x - spacing);
                (x, y, line) = (0.0, y + line, 0.0);
            }
            line = line.max(tall);
            for point in &mut at[first..] {
                *point = Point::new(point.x + x, y);
            }
            x += wide + spacing;
        }
        shaped.truncate(count);
        if x != 0.0 {
            widest = widest.max(x - spacing);
        }
        let size = limits.resolve(Length::Shrink, Length::Shrink, Size::new(widest, y + line));
        layout::Node::new(size)
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
        let letters = tree.state.downcast_ref::<Letters>();
        let ink = crate::ui::theme::color(theme.ink(self.ink));
        let origin = layout.position();
        for (plain, at) in letters.shaped.iter().zip(&letters.at) {
            let at = Point::new(origin.x + at.x, origin.y + at.y);
            renderer.fill_paragraph(plain.raw(), at, ink, *viewport);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_capitals_in_muted() {
        let label = label("Disks · éa");
        assert_eq!(label.content, "DISKS · ÉA");
        assert_eq!(label.ink, Ink::Muted);
        assert_eq!(
            label.clone().ink(crate::ui::tokens::Accent::Cyan).ink,
            Ink::Accent(crate::ui::tokens::Accent::Cyan)
        );
    }

    #[test]
    fn a_letter_keeps_its_accent() {
        // E and a combining acute accent stay one letter.
        let label = label("Cafe\u{301} x");
        let letters: Vec<_> = label.letters().collect();
        assert_eq!(letters, ["C", "A", "F", "E\u{301}", " ", "X"]);
        let _: Element<'_, ()> = label.into();
    }

    #[test]
    fn roles_build_text() {
        // Building them is what can be checked without a renderer; the gallery's screenshots
        // show how they look.
        let _: [Text<'_, Theme, Renderer>; 4] = [
            title("Title"),
            heading("Heading"),
            interface("Interface"),
            meta("meta"),
        ];
        let _ = role(&typography::LABEL, "upper");
    }
}
