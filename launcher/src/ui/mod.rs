// SPDX-License-Identifier: GPL-3.0-or-later

//! The launcher's design system on stock iced 0.14: its UI kit.
//!
//! - [`tokens`]: colors (Sunset dark and light, from Leonida's logo), type, spacing, radii,
//!   motion and glow.
//! - [`theme`], [`text`], [`font`]: the theme every widget draws from, text roles, the fonts.
//! - [`focus`]: the focus model; Tab and Shift+Tab move focus, Enter and Space press.
//! - The widgets: [`button`], [`field`] (a text field), [`loader`], [`progress`], [`scroll`],
//!   [`status`], [`toggle`] (the light and dark pill), and [`chrome`], the window's own
//!   decorations, which [`window`] runs.

pub mod button;
pub mod checkbox;
pub mod choice;
pub mod chrome;
pub mod edit;
pub mod field;
pub mod flow;
pub mod focus;
pub mod font;
pub mod form;
mod icon;
mod input;
pub mod list;
pub mod loader;
mod paint;
pub mod progress;
pub mod scroll;
pub mod settings;
mod shape;
pub mod status;
pub mod switch;
pub mod system;
pub mod text;
pub mod theme;
pub mod toggle;
pub mod tokens;
pub(crate) mod transition;
pub mod window;

pub use button::button;
pub use checkbox::checkbox;
pub use choice::choice;
pub use field::text_field;
pub use list::navigation;
pub use loader::loader;
pub use progress::progress;
pub use scroll::scroll_area;
pub use status::status;
pub use switch::switch;
pub use theme::{Emphasis, Frame, Interaction, Status, Theme};
pub use toggle::toggle_group;
pub use tokens::{Accent, Ink, Surface};

pub use iced::{Alignment, Center, Fill, Length, Padding, Pixels, Point, Shrink, Size, keyboard};

/// The layout widgets, which draw nothing themselves. A `container` draws a [`Frame`].
pub mod widget {
    pub use crate::ui::flow::{Flow, flow};
    pub use iced::widget::{
        Column, Container, MouseArea, Row, Space, column, container, mouse_area, row, space,
    };
}

/// iced's renderer: wgpu, or tiny-skia where wgpu has no adapter (and offscreen, in the tests).
pub type Renderer = iced::Renderer;

/// A piece of interface, in the launcher's theme.
pub type Element<'a, Message> = iced::Element<'a, Message, Theme, Renderer>;
