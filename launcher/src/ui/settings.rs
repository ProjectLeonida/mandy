// SPDX-License-Identifier: GPL-3.0-or-later

//! A list of settings: a group's title, then a card of rows,
//! each a setting's name, what it does, and its control at the end, whatever the control is: a
//! switch, a dropdown, a field, a stepper. On or off is a switch, never a checkbox.
//!
//! ```ignore
//! group("Scrolling")
//!     .row(
//!         row("Scroll on output")
//!             .description("Scroll when applications in the terminal print")
//!             .control(switch(on).on_toggle(Message::ScrollOnOutput))
//!             .on_press(Message::ScrollOnOutput(!on)),
//!     )
//! ```

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, overlay, renderer};
use iced::widget::{Column, Space, column, container};
use iced::{Event, Length, Rectangle, Size, Vector, touch};

use crate::ui::tokens::{Ink, Surface, space};

use crate::ui::theme::Frame;
use crate::ui::{Element, Fill, Renderer, Theme, text};

/// The room inside a row, above and below its words and round its sides.
const ROW_PAD: [f32; 2] = [space::L, space::XL];

/// A setting: its name, and what follows.
pub fn row<'a, Message>(title: impl Into<String>) -> Row<'a, Message> {
    Row {
        title: title.into(),
        description: None,
        control: None,
        on_press: None,
    }
}

/// A setting's row: its name, what it does below it, and its control at the end.
pub struct Row<'a, Message> {
    title: String,
    description: Option<String>,
    control: Option<Element<'a, Message>>,
    on_press: Option<Message>,
}

impl<'a, Message: Clone + 'a> Row<'a, Message> {
    /// What the setting does, in a line or two below its name.
    pub fn description(self, description: impl Into<String>) -> Self {
        Self {
            description: Some(description.into()),
            ..self
        }
    }

    /// Its control, at the row's end: any widget.
    pub fn control(self, control: impl Into<Element<'a, Message>>) -> Self {
        Self {
            control: Some(control.into()),
            ..self
        }
    }

    /// What a click anywhere on the row sends, as a switch's row turns it over: pressed and let
    /// go on the row, as a button is, so a finger that lands on it to scroll the page sends
    /// nothing. A click on the control goes to the control alone.
    pub fn on_press(self, message: Message) -> Self {
        Self {
            on_press: Some(message),
            ..self
        }
    }
}

impl<'a, Message: Clone + 'a> From<Row<'a, Message>> for Element<'a, Message> {
    fn from(row: Row<'a, Message>) -> Self {
        let words = Column::with_children(
            std::iter::once(text::interface(row.title).into()).chain(
                row.description
                    .map(|description| text::interface(description).class(Ink::Muted).into()),
            ),
        )
        .spacing(space::XXS)
        .width(Fill);
        // The control beside the words while they have room, below them when they don't, as on
        // a phone.
        let content = container(
            crate::ui::flow::flow(std::iter::once(words.into()).chain(row.control))
                .spacing(space::XL),
        )
        .padding(ROW_PAD);
        match row.on_press {
            Some(on_press) => Element::new(Pressable {
                content: content.into(),
                on_press,
            }),
            None => content.into(),
        }
    }
}

/// A row that sends its message on a click: pressed on it and let go on it. What's in it, its
/// control, takes its own presses first.
struct Pressable<'a, Message> {
    content: Element<'a, Message>,
    on_press: Message,
}

/// Whether a press began on the row.
#[derive(Default)]
struct Pressed(bool);

impl<Message: Clone> Widget<Message, Theme, Renderer> for Pressable<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<Pressed>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(Pressed::default())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.content));
    }

    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
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
        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
        if shell.is_event_captured() {
            return;
        }
        let pressed = &mut tree.state.downcast_mut::<Pressed>().0;
        let over = cursor.is_over(layout.bounds());
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
            | Event::Touch(touch::Event::FingerPressed { .. })
                if over =>
            {
                *pressed = true;
                shell.capture_event();
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
            | Event::Touch(touch::Event::FingerLifted { .. })
                if *pressed =>
            {
                *pressed = false;
                if over {
                    shell.publish(self.on_press.clone());
                }
                shell.capture_event();
            }
            Event::Touch(touch::Event::FingerLost { .. }) => *pressed = false,
            _ => {}
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
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            layout,
            cursor,
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
        let inside = self.content.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            cursor,
            viewport,
            renderer,
        );
        if inside == mouse::Interaction::None && cursor.is_over(layout.bounds()) {
            mouse::Interaction::Pointer
        } else {
            inside
        }
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'b, Message, Theme, Renderer>> {
        self.content.as_widget_mut().overlay(
            &mut tree.children[0],
            layout,
            renderer,
            viewport,
            translation,
        )
    }
}

/// A group of settings under a title.
pub fn group<'a, Message>(title: impl Into<String>) -> Group<'a, Message> {
    Group {
        title: title.into(),
        description: None,
        rows: Vec::new(),
    }
}

/// A group of settings: its title, then its rows in a card, a line between each.
pub struct Group<'a, Message> {
    title: String,
    description: Option<String>,
    rows: Vec<Element<'a, Message>>,
}

impl<'a, Message: Clone + 'a> Group<'a, Message> {
    /// A line or two below the title, about the group.
    pub fn description(self, description: impl Into<String>) -> Self {
        Self {
            description: Some(description.into()),
            ..self
        }
    }

    /// Adds a row.
    pub fn row(mut self, row: Row<'a, Message>) -> Self {
        self.rows.push(row.into());
        self
    }

    /// Adds anything else to the card, with a row's room round it, such as a picture choice.
    pub fn item(mut self, item: impl Into<Element<'a, Message>>) -> Self {
        self.rows
            .push(container(item).padding(space::XL).width(Fill).into());
        self
    }
}

impl<'a, Message: Clone + 'a> From<Group<'a, Message>> for Element<'a, Message> {
    fn from(group: Group<'a, Message>) -> Self {
        let mut rows = Vec::with_capacity(group.rows.len() * 2);
        for (at, row) in group.rows.into_iter().enumerate() {
            if at > 0 {
                rows.push(
                    container(Space::new())
                        .width(Fill)
                        .height(crate::ui::tokens::size::BORDER)
                        .class(Frame::Divider)
                        .into(),
                );
            }
            rows.push(row);
        }
        let card = container(Column::with_children(rows))
            .width(Fill)
            .class(Frame::Card(Surface::Panel));
        let heading = Column::with_children(
            std::iter::once(text::label(group.title).into()).chain(
                group
                    .description
                    .map(|description| text::interface(description).class(Ink::Muted).into()),
            ),
        )
        .spacing(space::XS);
        column![heading, card].spacing(space::M).into()
    }
}
