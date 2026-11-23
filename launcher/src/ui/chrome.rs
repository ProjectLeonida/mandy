// SPDX-License-Identifier: GPL-3.0-or-later

//! A window's own decorations: the launcher draws them itself (client-side), the same on every
//! desktop. [`crate::ui::window`] puts them round the window, and the app gives what its
//! header bar holds:
//!
//! - [`header_bar`]: the bar at the top. What shows goes in its three slots ([`header`]): the
//!   app's name at the start, something in the middle, and the light and dark pill at the end.
//!   Dragging the bar moves the window, from its controls too once the pointer goes far enough,
//!   and double-clicking its empty area maximizes the window or restores it.
//! - [`close_button`]: the one window button.
//! - [`window_frame`]: the window's fill, and while it floats its rounded corners and line, and
//!   the zones inside its edges that resize it ([`resize_edge`]).
//!
//! The widgets only say what was asked, by message; the window module moves, maximizes, resizes and
//! closes the window. Their looks are the theme's ([`Theme::header_bar`],
//! [`Theme::close_button`], [`Theme::window`]), their sizes `crate::ui::tokens::window`'s.

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Renderer as _};
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, overlay};
use iced::gradient::Linear;
use iced::{
    Background, Border, Degrees, Event, Gradient, Length, Point, Rectangle, Shadow, Size, Vector,
};

use crate::ui::tokens::{size, window};

mod fold;
pub use fold::{Folded, folded};

use crate::ui::icon::Icon;
use crate::ui::shape;
use crate::ui::theme::color;
use crate::ui::{Element, Renderer, Theme};

/// An edge or a corner of a window, to resize it from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Edge {
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
}

impl Edge {
    /// The direction iced resizes a window in from it.
    pub fn direction(self) -> iced::window::Direction {
        use iced::window::Direction;
        match self {
            Self::North => Direction::North,
            Self::South => Direction::South,
            Self::East => Direction::East,
            Self::West => Direction::West,
            Self::NorthEast => Direction::NorthEast,
            Self::NorthWest => Direction::NorthWest,
            Self::SouthEast => Direction::SouthEast,
            Self::SouthWest => Direction::SouthWest,
        }
    }

    /// The pointer's shape over it.
    pub fn cursor(self) -> mouse::Interaction {
        match self {
            Self::North | Self::South => mouse::Interaction::ResizingVertically,
            Self::East | Self::West => mouse::Interaction::ResizingHorizontally,
            Self::NorthWest | Self::SouthEast => mouse::Interaction::ResizingDiagonallyDown,
            Self::NorthEast | Self::SouthWest => mouse::Interaction::ResizingDiagonallyUp,
        }
    }
}

/// What a header bar holds: three slots, each optional, each the app's, which always show: the
/// app's name at the start, something in the middle, and what comes last before the close
/// button, such as the light and dark pill.
pub struct Header<'a, Message> {
    start: Option<Element<'a, Message>>,
    center: Option<Element<'a, Message>>,
    end: Option<Element<'a, Message>>,
}

/// An empty header: fill its slots.
pub fn header<'a, Message>() -> Header<'a, Message> {
    Header {
        start: None,
        center: None,
        end: None,
    }
}

impl<'a, Message: 'a> Header<'a, Message> {
    /// What comes first: the app's name or brand, the window's title.
    pub fn start(self, content: impl Into<Element<'a, Message>>) -> Self {
        Self {
            start: Some(content.into()),
            ..self
        }
    }

    /// What sits in the middle of the room the others leave.
    pub fn center(self, content: impl Into<Element<'a, Message>>) -> Self {
        Self {
            center: Some(content.into()),
            ..self
        }
    }

    /// What comes last, before the close button.
    pub fn end(self, content: impl Into<Element<'a, Message>>) -> Self {
        Self {
            end: Some(content.into()),
            ..self
        }
    }

    /// The same, sending other messages.
    pub fn map<B: 'a>(self, f: impl Fn(Message) -> B + Clone + 'a) -> Header<'a, B> {
        let map = |slot: Option<Element<'a, Message>>| slot.map(|slot| slot.map(f.clone()));
        Header {
            start: map(self.start),
            center: map(self.center),
            end: map(self.end),
        }
    }

    /// Whether every slot is empty.
    pub fn is_empty(&self) -> bool {
        self.start.is_none() && self.center.is_none() && self.end.is_none()
    }
}

/// A header bar holding a header's slots. Without [`on_close`](HeaderBar::on_close) it has no
/// close button.
pub fn header_bar<'a, Message>(header: Header<'a, Message>) -> HeaderBar<'a, Message> {
    HeaderBar {
        header,
        on_drag: None,
        on_maximize: None,
        on_close: None,
        on_back: None,
        rounded: false,
    }
}

/// A header bar, made into an [`Element`] once it's set up.
pub struct HeaderBar<'a, Message> {
    header: Header<'a, Message>,
    on_drag: Option<Message>,
    on_maximize: Option<Message>,
    on_close: Option<Message>,
    on_back: Option<Message>,
    rounded: bool,
}

impl<Message> HeaderBar<'_, Message> {
    /// Adds the back button before everything else, which sends this: a folded window showing
    /// its page goes back to its sidebar.
    pub fn on_back(self, message: Message) -> Self {
        Self {
            on_back: Some(message),
            ..self
        }
    }

    /// Its top corners rounded as the window's, inside its line: where the frame's own rounding
    /// can't be drawn (`shape::round` on a renderer without custom primitives), the bar's fill
    /// keeps out of the window's corners itself.
    pub fn rounded(self, rounded: bool) -> Self {
        Self { rounded, ..self }
    }

    /// What pressing its empty area and dragging sends: the window should move with the pointer.
    pub fn on_drag(self, message: Message) -> Self {
        Self {
            on_drag: Some(message),
            ..self
        }
    }

    /// What double-clicking its empty area sends: the window should maximize, or restore.
    pub fn on_maximize(self, message: Message) -> Self {
        Self {
            on_maximize: Some(message),
            ..self
        }
    }

    /// Adds the close button, which sends this.
    pub fn on_close(self, message: Message) -> Self {
        Self {
            on_close: Some(message),
            ..self
        }
    }
}

impl<'a, Message: Clone + 'a> From<HeaderBar<'a, Message>> for Element<'a, Message> {
    fn from(bar: HeaderBar<'a, Message>) -> Self {
        let Header { start, center, end } = bar.header;
        let close = bar
            .on_close
            .map(|message| Element::from(close_button(message)));
        // In place of an overflow menu (items that move into a menu when they don't fit): the
        // three slots in a row, the room between them empty, so it drags the window.
        let gap = |width: f32| Element::from(iced::widget::space().width(width));
        let fill = || Element::from(iced::widget::space().width(Length::Fill));
        let mut row = Vec::new();
        match bar.on_back {
            Some(message) => {
                row.push(gap(window::BAR_PAD_END));
                row.push(Element::from(back_button(message)));
                row.push(gap(window::BAR_GAP));
            }
            None => row.push(gap(window::BAR_PAD_START)),
        }
        row.extend(start);
        row.push(fill());
        if let Some(center) = center {
            row.push(center);
            row.push(fill());
        }
        if let Some(end) = end {
            row.push(end);
            row.push(gap(window::BAR_GAP));
        }
        match close {
            Some(close) => {
                row.push(close);
                row.push(gap(window::BAR_PAD_END));
            }
            None => row.push(gap(window::BAR_PAD_START)),
        }
        let content = iced::widget::Row::with_children(row)
            .align_y(iced::Alignment::Center)
            .height(Length::Fill)
            .into();
        Element::new(Bar {
            content,
            on_drag: bar.on_drag,
            on_maximize: bar.on_maximize,
            rounded: bar.rounded,
        })
    }
}

/// The bar itself: its fill, tints and line, its items, and drags and double-clicks on what of it
/// no item takes.
struct Bar<'a, Message> {
    content: Element<'a, Message>,
    on_drag: Option<Message>,
    on_maximize: Option<Message>,
    rounded: bool,
}

#[derive(Debug, Default)]
struct Drag {
    /// Where the button went down on the bar, until the pointer moves away or it comes up, and
    /// whether on an item, such as a button.
    pressed: Option<Point>,
    on_item: bool,
    last: Option<mouse::Click>,
}

impl<Message: Clone> Widget<Message, Theme, Renderer> for Bar<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<Drag>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(Drag::default())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.content));
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fixed(window::BAR_HEIGHT))
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.resolve(Length::Fill, Length::Fixed(window::BAR_HEIGHT), Size::ZERO);
        // The items sit above the bottom line.
        let inside = layout::Limits::new(
            Size::ZERO,
            Size::new(size.width, size.height - size::BORDER),
        );
        let content = self
            .content
            .as_widget_mut()
            .layout(&mut tree.children[0], renderer, &inside);
        layout::Node::with_children(size, vec![content])
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        let content = inside(layout);
        self.content
            .as_widget_mut()
            .operate(&mut tree.children[0], content, renderer, operation);
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
        let content = inside(layout);
        // A press anywhere on the bar, on an item too, moves the window once the pointer goes
        // far enough with it held, as a header bar does in GTK. The items then hear the button
        // come up away from them, so the one pressed lets go without acting.
        let drag = tree.state.downcast_mut::<Drag>();
        if let Event::Mouse(mouse::Event::CursorMoved { .. }) = event
            && let (Some(from), Some(at)) = (drag.pressed, cursor.position())
            && from.distance(at) >= window::DRAG_THRESHOLD
        {
            drag.pressed = None;
            if drag.on_item {
                self.content.as_widget_mut().update(
                    &mut tree.children[0],
                    &Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
                    content,
                    mouse::Cursor::Unavailable,
                    renderer,
                    clipboard,
                    shell,
                    viewport,
                );
            }
            if let Some(message) = &self.on_drag {
                shell.publish(message.clone());
            }
            return;
        }
        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            content,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
        // The button up, whoever took it, ends the press.
        if let Event::Mouse(
            mouse::Event::ButtonReleased(mouse::Button::Left) | mouse::Event::CursorLeft,
        ) = event
        {
            tree.state.downcast_mut::<Drag>().pressed = None;
        }
        if shell.is_event_captured() {
            // An item took it. A press on one may still move the window, but not on text, where
            // a drag selects, nor on what's dragged itself, such as the light and dark pill.
            if let Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) = event {
                let on_text = matches!(
                    self.content.as_widget().mouse_interaction(
                        &tree.children[0],
                        content,
                        cursor,
                        viewport,
                        renderer,
                    ),
                    mouse::Interaction::Text
                        | mouse::Interaction::Grab
                        | mouse::Interaction::Grabbing
                );
                let drag = tree.state.downcast_mut::<Drag>();
                drag.pressed = cursor
                    .position_over(layout.bounds())
                    .filter(|_| !on_text && self.on_drag.is_some());
                drag.on_item = true;
                drag.last = None;
            }
            return;
        }
        let drag = tree.state.downcast_mut::<Drag>();
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let Some(at) = cursor.position_over(layout.bounds()) else {
                    return;
                };
                let click = mouse::Click::new(at, mouse::Button::Left, drag.last);
                drag.last = Some(click);
                if click.kind() == mouse::click::Kind::Double {
                    drag.pressed = None;
                    if let Some(message) = &self.on_maximize {
                        shell.publish(message.clone());
                    }
                } else {
                    drag.pressed = Some(at);
                    drag.on_item = false;
                }
                shell.capture_event();
            }
            Event::Mouse(
                mouse::Event::ButtonReleased(mouse::Button::Left) | mouse::Event::CursorLeft,
            ) => drag.pressed = None,
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
        let bounds = layout.bounds();
        let look = theme.header_bar();
        // Rounded, its top corners keep inside the window's, a line in from its edge.
        let radius = if self.rounded {
            let r = window::CORNER_RADIUS - window::BORDER;
            iced::border::Radius {
                top_left: r,
                top_right: r,
                bottom_right: 0.0,
                bottom_left: 0.0,
            }
        } else {
            iced::border::Radius::default()
        };
        fill_rounded(
            renderer,
            bounds,
            Background::Color(color(look.fill)),
            radius,
        );
        fill_rounded(
            renderer,
            bounds,
            Background::Gradient(edge_tints(look.start, look.end, look.reach)),
            radius,
        );
        let line = Rectangle {
            y: bounds.y + bounds.height - size::BORDER,
            height: size::BORDER,
            ..bounds
        };
        fill(renderer, line, Background::Color(color(look.line)));
        let content = inside(layout);
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            content,
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
        let content = inside(layout);
        self.content.as_widget().mouse_interaction(
            &tree.children[0],
            content,
            cursor,
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
        let content = inside(layout);
        self.content.as_widget_mut().overlay(
            &mut tree.children[0],
            content,
            renderer,
            viewport,
            translation,
        )
    }
}

/// The edge tints as a gradient, left to right: the start's tint fading to nothing across
/// `reach` of the width, nothing, then the end's tint fading in across the last `reach`, as the
/// mockup's CSS. iced eases between a gradient's stops where CSS goes straight, so each fade
/// gets two more stops, on the straight line, which keeps it within a fraction of a level of it.
fn edge_tints(
    start: crate::ui::tokens::Color,
    end: crate::ui::tokens::Color,
    reach: f32,
) -> Gradient {
    let start = color(start);
    let end = color(end);
    let fade = |tint: iced::Color, amount: f32| iced::Color {
        a: tint.a * amount,
        ..tint
    };
    let third = reach / 3.0;
    Gradient::Linear(
        Linear::new(Degrees(90.0))
            .add_stop(0.0, start)
            .add_stop(third, fade(start, 2.0 / 3.0))
            .add_stop(2.0 * third, fade(start, 1.0 / 3.0))
            .add_stop(reach, fade(start, 0.0))
            .add_stop(1.0 - reach, fade(end, 0.0))
            .add_stop(1.0 - 2.0 * third, fade(end, 1.0 / 3.0))
            .add_stop(1.0 - third, fade(end, 2.0 / 3.0))
            .add_stop(1.0, end),
    )
}

/// A rectangle filled edge to edge.
fn fill(renderer: &mut Renderer, bounds: Rectangle, background: Background) {
    fill_rounded(
        renderer,
        bounds,
        background,
        iced::border::Radius::default(),
    );
}

/// A rectangle filled, its corners rounded each by its own radius.
fn fill_rounded(
    renderer: &mut Renderer,
    bounds: Rectangle,
    background: Background,
    radius: iced::border::Radius,
) {
    renderer.fill_quad(
        renderer::Quad {
            bounds,
            border: Border {
                radius,
                ..Border::default()
            },
            shadow: Shadow::default(),
            snap: true,
        },
        background,
    );
}

/// The close button, a window button whose hover is red: Enter or Space closes the window when it
/// has focus, as a click does.
pub fn close_button<Message>(on_press: Message) -> WindowButton<Message> {
    crate::ui::icon::window_button(Icon::CLOSE, on_press).danger()
}

/// The back button, at a folded window's header bar's start: back from the page to the sidebar.
pub fn back_button<Message>(on_press: Message) -> WindowButton<Message> {
    crate::ui::icon::window_button(Icon::BACK, on_press)
}

pub use crate::ui::icon::WindowButton;

/// How a floating window's corners are made.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Corners {
    /// Square: maximized, or the frame drawn by no one (`LEONIDA_WINDOW_FRAME=0`).
    #[default]
    Square,
    /// Rounded by the app, on a transparent window: everything drawn is cut to the rounded
    /// rectangle and the line drawn round it, as the design language has it.
    Drawn,
    /// Rounded by the system, on an opaque window (Windows 11's own corners): the app draws only
    /// its line, at the system's radius.
    System,
}

/// A window's frame round its content, filling the window: bg behind everything. Square and
/// without resize zones until set up.
pub fn window_frame<'a, Message>(
    content: impl Into<Element<'a, Message>>,
) -> WindowFrame<'a, Message> {
    WindowFrame {
        content: content.into(),
        corners: Corners::Square,
        on_resize: None,
    }
}

/// What a header bar or a window frame holds: its layout's one child, which its layout always
/// has.
fn inside(layout: Layout<'_>) -> Layout<'_> {
    layout.children().next().unwrap_or(layout)
}

/// A window's frame round its content ([`window_frame`]): made into an [`Element`] once it's set
/// up.
pub struct WindowFrame<'a, Message> {
    content: Element<'a, Message>,
    corners: Corners,
    on_resize: Option<Box<dyn Fn(Edge) -> Message + 'a>>,
}

impl<'a, Message> WindowFrame<'a, Message> {
    /// Its corners while it floats, and a line round the edge unless they're square. Drawn
    /// corners need a transparent window.
    pub fn corners(self, corners: Corners) -> Self {
        Self { corners, ..self }
    }

    /// Zones inside its edges that resize the window ([`resize_edge`]): pressing one sends the
    /// direction to resize in, and the pointer over one shows it.
    pub fn on_resize(self, on_resize: impl Fn(Edge) -> Message + 'a) -> Self {
        Self {
            on_resize: Some(Box::new(on_resize)),
            ..self
        }
    }

    fn rounded(&self) -> bool {
        self.corners != Corners::Square
    }

    fn inset(&self) -> f32 {
        if self.rounded() { window::BORDER } else { 0.0 }
    }

    /// The resize zone under the pointer, if it has any.
    fn edge(&self, layout: Layout<'_>, cursor: mouse::Cursor) -> Option<Edge> {
        let bounds = layout.bounds();
        self.on_resize.as_ref()?;
        let at = cursor.position_over(bounds)?;
        resize_edge(bounds.size(), Point::new(at.x - bounds.x, at.y - bounds.y))
    }

    /// The pointer as the content sees it: over a resize zone, it's the zone's.
    fn content_cursor(&self, layout: Layout<'_>, cursor: mouse::Cursor) -> mouse::Cursor {
        if self.edge(layout, cursor).is_some() {
            cursor.levitate()
        } else {
            cursor
        }
    }
}

impl<Message> Widget<Message, Theme, Renderer> for WindowFrame<'_, Message> {
    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.content));
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.resolve(Length::Fill, Length::Fill, Size::ZERO);
        let inset = self.inset();
        let inside = layout::Limits::new(
            Size::ZERO,
            Size::new(
                (size.width - 2.0 * inset).max(0.0),
                (size.height - 2.0 * inset).max(0.0),
            ),
        );
        let content = self
            .content
            .as_widget_mut()
            .layout(&mut tree.children[0], renderer, &inside)
            .move_to(Point::new(inset, inset));
        layout::Node::with_children(size, vec![content])
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        let content = inside(layout);
        self.content
            .as_widget_mut()
            .operate(&mut tree.children[0], content, renderer, operation);
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
        if let Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) = event
            && let Some(direction) = self.edge(layout, cursor)
            && let Some(on_resize) = &self.on_resize
        {
            shell.publish(on_resize(direction));
            shell.capture_event();
            return;
        }
        let content = inside(layout);
        let cursor = self.content_cursor(layout, cursor);
        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            content,
            cursor,
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
        let bounds = layout.bounds();
        let look = theme.window();
        // Drawn corners: the fill keeps inside them itself too, for a renderer that can't round
        // what's drawn (`shape::round`).
        let radius = if self.corners == Corners::Drawn {
            look.radius
        } else {
            0.0
        };
        fill_rounded(
            renderer,
            bounds,
            Background::Color(color(look.fill)),
            radius.into(),
        );
        let content = inside(layout);
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            content,
            self.content_cursor(layout, cursor),
            viewport,
        );
        draw_corners(renderer, theme, bounds, self.corners);
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        if let Some(direction) = self.edge(layout, cursor) {
            return direction.cursor();
        }
        let content = inside(layout);
        self.content.as_widget().mouse_interaction(
            &tree.children[0],
            content,
            cursor,
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
        let content = inside(layout);
        let (rounded, corners) = (self.rounded(), self.corners);
        let overlay = self.content.as_widget_mut().overlay(
            &mut tree.children[0],
            content,
            renderer,
            viewport,
            translation,
        )?;
        // An overlay, such as an open menu, draws after everything, the rounding too: rounded
        // again over it, it keeps the window's corners, as a dialog in the window does.
        Some(if rounded {
            overlay::Element::new(Box::new(Rounded {
                overlay,
                bounds: layout.bounds() + translation,
                corners,
            }))
        } else {
            overlay
        })
    }
}

/// A rounded window's overlay, rounded at the window's corners and lined at its edge again once
/// it's drawn ([`shape::round`]).
struct Rounded<'b, Message> {
    overlay: overlay::Element<'b, Message, Theme, Renderer>,
    /// The window's.
    bounds: Rectangle,
    corners: Corners,
}

/// The frame's corners and line, last, in a layer of their own: drawn corners round everything
/// drawn before them ([`shape::round`]); the system's get only the line, at Windows 11's radius.
fn draw_corners(renderer: &mut Renderer, theme: &Theme, bounds: Rectangle, corners: Corners) {
    let look = theme.window();
    let (radius, clear) = match corners {
        Corners::Square => return,
        Corners::Drawn => (look.radius, true),
        Corners::System => (SYSTEM_CORNER_RADIUS, false),
    };
    renderer.with_layer(bounds, |renderer| {
        shape::round(
            renderer,
            bounds,
            radius,
            window::BORDER,
            color(look.line),
            clear,
        );
    });
}

/// Windows 11's corners on a window that asks for them rounded (`DWMWCP_ROUND`): 8 pixels.
pub const SYSTEM_CORNER_RADIUS: f32 = 8.0;

impl<Message> overlay::Overlay<Message, Theme, Renderer> for Rounded<'_, Message> {
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
        self.overlay
            .as_overlay()
            .draw(renderer, theme, style, layout, cursor);
        draw_corners(renderer, theme, self.bounds, self.corners);
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
        self.overlay
            .as_overlay_mut()
            .update(event, layout, cursor, renderer, clipboard, shell);
    }

    fn mouse_interaction(
        &self,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.overlay
            .as_overlay()
            .mouse_interaction(layout, cursor, renderer)
    }

    fn overlay<'a>(
        &'a mut self,
        layout: Layout<'a>,
        renderer: &Renderer,
    ) -> Option<overlay::Element<'a, Message, Theme, Renderer>> {
        let (bounds, corners) = (self.bounds, self.corners);
        self.overlay
            .as_overlay_mut()
            .overlay(layout, renderer)
            .map(|overlay| {
                overlay::Element::new(Box::new(Rounded {
                    overlay,
                    bounds,
                    corners,
                }))
            })
    }

    fn index(&self) -> f32 {
        self.overlay.as_overlay().index()
    }
}

impl<'a, Message: 'a> From<WindowFrame<'a, Message>> for Element<'a, Message> {
    fn from(frame: WindowFrame<'a, Message>) -> Self {
        Element::new(frame)
    }
}

/// Which edge or corner of a window `size` big a point in it resizes, if any: a band
/// `RESIZE_BAND` wide inside each edge, and at each corner the band's parts reaching
/// `RESIZE_CORNER` along both edges (`crate::ui::tokens::window`). A window too small for both of
/// opposite bands resizes from the top and the left.
pub fn resize_edge(size: Size, at: Point) -> Option<Edge> {
    let (band, corner) = (window::RESIZE_BAND, window::RESIZE_CORNER);
    if !(0.0..size.width).contains(&at.x) || !(0.0..size.height).contains(&at.y) {
        return None;
    }
    let (left, right) = (at.x < band, at.x >= size.width - band);
    let (top, bottom) = (at.y < band, at.y >= size.height - band);
    let sides = left || right;
    let ends = top || bottom;
    let north = top || (sides && at.y < corner);
    let south = !north && (bottom || (sides && at.y >= size.height - corner));
    let west = left || (ends && at.x < corner);
    let east = !west && (right || (ends && at.x >= size.width - corner));
    Some(match (north, south, west, east) {
        (true, _, true, _) => Edge::NorthWest,
        (true, _, _, true) => Edge::NorthEast,
        (_, true, true, _) => Edge::SouthWest,
        (_, true, _, true) => Edge::SouthEast,
        (true, ..) => Edge::North,
        (_, true, ..) => Edge::South,
        (_, _, true, _) => Edge::West,
        (_, _, _, true) => Edge::East,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Size = Size::new(800.0, 600.0);

    fn at(x: f32, y: f32) -> Option<Edge> {
        resize_edge(WINDOW, Point::new(x, y))
    }

    #[test]
    fn the_middle_resizes_nothing() {
        assert_eq!(at(400.0, 300.0), None);
        assert_eq!(at(6.0, 300.0), None);
        assert_eq!(at(400.0, 6.0), None);
        assert_eq!(at(793.0, 300.0), None);
        assert_eq!(at(400.0, 593.0), None);
        // Outside the window: not its.
        assert_eq!(at(-1.0, 300.0), None);
        assert_eq!(at(800.0, 300.0), None);
        assert_eq!(at(400.0, 600.0), None);
    }

    #[test]
    fn edges_resize_in_their_direction() {
        assert_eq!(at(0.0, 300.0), Some(Edge::West));
        assert_eq!(at(5.9, 300.0), Some(Edge::West));
        assert_eq!(at(799.0, 300.0), Some(Edge::East));
        assert_eq!(at(794.0, 300.0), Some(Edge::East));
        assert_eq!(at(400.0, 0.0), Some(Edge::North));
        assert_eq!(at(400.0, 599.5), Some(Edge::South));
        // Just past a corner's reach, an edge again.
        assert_eq!(at(14.0, 0.0), Some(Edge::North));
        assert_eq!(at(0.0, 14.0), Some(Edge::West));
        assert_eq!(at(785.9, 599.0), Some(Edge::South));
    }

    #[test]
    fn corners_reach_along_both_edges() {
        for (x, y) in [(0.0, 0.0), (13.9, 2.0), (2.0, 13.9), (5.0, 5.0)] {
            assert_eq!(at(x, y), Some(Edge::NorthWest), "{x}, {y}");
        }
        assert_eq!(at(799.0, 0.0), Some(Edge::NorthEast));
        assert_eq!(at(787.0, 3.0), Some(Edge::NorthEast));
        assert_eq!(at(797.0, 13.0), Some(Edge::NorthEast));
        assert_eq!(at(0.0, 599.0), Some(Edge::SouthWest));
        assert_eq!(at(10.0, 597.0), Some(Edge::SouthWest));
        assert_eq!(at(799.0, 599.0), Some(Edge::SouthEast));
        assert_eq!(at(796.0, 587.0), Some(Edge::SouthEast));
        // Inside the corner's square but off the band: nothing.
        assert_eq!(at(10.0, 10.0), None);
    }

    #[test]
    fn a_tiny_window_resizes_from_the_top_left() {
        let tiny = Size::new(10.0, 10.0);
        assert_eq!(
            resize_edge(tiny, Point::new(5.0, 5.0)),
            Some(Edge::NorthWest)
        );
        assert_eq!(resize_edge(Size::ZERO, Point::ORIGIN), None);
    }

    #[test]
    fn the_pointer_shows_the_direction() {
        use mouse::Interaction::*;
        let shapes = [
            (Edge::North, ResizingVertically),
            (Edge::South, ResizingVertically),
            (Edge::East, ResizingHorizontally),
            (Edge::West, ResizingHorizontally),
            (Edge::NorthWest, ResizingDiagonallyDown),
            (Edge::SouthEast, ResizingDiagonallyDown),
            (Edge::NorthEast, ResizingDiagonallyUp),
            (Edge::SouthWest, ResizingDiagonallyUp),
        ];
        for (direction, shape) in shapes {
            assert_eq!(direction.cursor(), shape);
        }
    }

    #[test]
    fn edge_tints_fade_straight_to_nothing() {
        let theme = <Theme as Default>::default();
        let look = theme.header_bar();
        let Gradient::Linear(linear) = edge_tints(look.start, look.end, look.reach);
        let stops: Vec<_> = linear.stops.iter().flatten().collect();
        assert_eq!(stops.len(), 8);
        assert_eq!(stops[0].color, color(look.start));
        assert_eq!(stops[7].color, color(look.end));
        assert_eq!(stops[3].offset, 0.28);
        assert_eq!(stops[3].color.a, 0.0);
        assert_eq!(stops[4].color.a, 0.0);
        // On the straight line from the tint to nothing.
        assert!((stops[1].color.a - color(look.start).a * 2.0 / 3.0).abs() < 1e-6);
        assert!((stops[6].color.a - color(look.end).a * 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn headers_map_their_slots() {
        let empty = header::<u8>();
        assert!(empty.is_empty());
        let full = header::<u8>()
            .start(crate::ui::text::interface("Gallery"))
            .center(crate::ui::text::interface("Search"))
            .end(crate::ui::text::interface("Synced"));
        assert!(!full.is_empty());
        let mapped = full.map(u32::from);
        assert!(mapped.start.is_some() && mapped.center.is_some() && mapped.end.is_some());
        let only_end = header::<u8>().end(crate::ui::text::interface("Synced"));
        assert!(!only_end.is_empty());
    }
}
