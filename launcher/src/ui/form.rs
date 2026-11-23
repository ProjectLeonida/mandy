// SPDX-License-Identifier: GPL-3.0-or-later

//! A form: fields one under another, each its name above its control, then a line that helps or,
//! in its place, what's wrong. Names go above, not beside: they
//! fit a phone's width and a translation's longer words.
//!
//! An error says what's wrong and what to do, in words, under the control, which the app also
//! marks (`invalid(true)`), so it's never color alone. What was typed stays.
//!
//! ```ignore
//! form([
//!     field("Host name", text_field("leonida", &state.host).on_input(Message::Host))
//!         .help("How other machines on the network see this one")
//!         .error(state.host_error.clone())
//!         .into(),
//! ])
//! ```

use iced::widget::Column;

use crate::ui::tokens::{Accent, Ink, space};

use crate::ui::{Element, text};

/// Fields one under another.
pub fn form<'a, Message: 'a>(
    fields: impl IntoIterator<Item = Element<'a, Message>>,
) -> Column<'a, Message, crate::ui::Theme, crate::ui::Renderer> {
    Column::with_children(fields).spacing(space::XL)
}

/// A field: its name, then its control.
pub fn field<'a, Message: 'a>(
    name: impl Into<String>,
    control: impl Into<Element<'a, Message>>,
) -> Field<'a, Message> {
    Field {
        name: name.into(),
        control: control.into(),
        help: None,
        error: None,
    }
}

/// A form's field ([`field`]).
pub struct Field<'a, Message> {
    name: String,
    control: Element<'a, Message>,
    help: Option<String>,
    error: Option<String>,
}

impl<'a, Message: 'a> Field<'a, Message> {
    /// A line under the control that helps: what the field is for, or what it takes.
    pub fn help(self, help: impl Into<String>) -> Self {
        Self {
            help: Some(help.into()),
            ..self
        }
    }

    /// What's wrong, in the help's place, in red: what happened and what to do. `None` when
    /// nothing is.
    pub fn error(self, error: Option<String>) -> Self {
        Self { error, ..self }
    }
}

impl<'a, Message: 'a> From<Field<'a, Message>> for Element<'a, Message> {
    fn from(field: Field<'a, Message>) -> Self {
        let mut parts = vec![
            text::interface(field.name).class(Ink::Text).into(),
            field.control,
        ];
        if let Some(error) = field.error {
            parts.push(
                text::interface(error)
                    .class(Ink::Accent(Accent::Red))
                    .into(),
            );
        } else if let Some(help) = field.help {
            parts.push(text::interface(help).class(Ink::Muted).into());
        }
        Column::with_children(parts).spacing(space::S).into()
    }
}
