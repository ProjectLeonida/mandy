// SPDX-License-Identifier: GPL-3.0-or-later

//! Drawing what the theme gives: boxes, dots and the focus outline. Shared by the widgets, which
//! draw only [`Look`]s.

use iced::advanced::renderer::{self, Renderer as _};
use iced::advanced::text::{self, Renderer as _};
use iced::{Background, Border, Color, Pixels, Point, Rectangle, Shadow, Size};

use crate::ui::theme::{Halo, Look, color};
use crate::ui::{Renderer, Theme};

/// A look's box, filling the bounds.
pub(crate) fn boxed(renderer: &mut Renderer, bounds: Rectangle, look: &Look) {
    framed(renderer, bounds, look, crate::ui::tokens::size::BORDER);
}

/// A look's box with its line this wide, such as a chosen picture's. Nothing's drawn of one that
/// can't be seen, such as a fade's far end.
pub(crate) fn framed(renderer: &mut Renderer, bounds: Rectangle, look: &Look, line: f32) {
    let seen = |color: Option<crate::ui::tokens::Color>| color.is_some_and(|c| c.alpha() > 0.0);
    if !seen(look.fill) && !seen(look.line) && !seen(look.glow.map(|glow| glow.color)) {
        return;
    }
    let radius = look.radius.min(bounds.width / 2.0).min(bounds.height / 2.0);
    renderer.fill_quad(
        renderer::Quad {
            bounds,
            border: Border {
                color: look.line.map(color).unwrap_or(Color::TRANSPARENT),
                width: if look.line.is_some() { line } else { 0.0 },
                radius: radius.into(),
            },
            shadow: look.glow.map(halo).unwrap_or_default(),
            snap: true,
        },
        look.fill
            .map(|fill| Background::Color(color(fill)))
            .unwrap_or(Background::Color(Color::TRANSPARENT)),
    );
}

/// A glow as iced draws it: a quad's shadow, round the quad and never inside it.
fn halo(glow: Halo) -> Shadow {
    Shadow {
        color: color(glow.color),
        offset: iced::Vector::new(0.0, glow.offset),
        blur_radius: glow.blur,
    }
}

/// The focus outline around bounds whose corners have this radius, with focus's glow round it:
/// as a shadow never shows inside its quad, the control inside the outline stays clear.
pub(crate) fn outline(renderer: &mut Renderer, theme: &Theme, bounds: Rectangle, radius: f32) {
    outline_glowing(renderer, theme, bounds, radius, theme.focus_glow());
}

/// The focus outline, with this glow round it: a window button's is small.
pub(crate) fn outline_glowing(
    renderer: &mut Renderer,
    theme: &Theme,
    bounds: Rectangle,
    radius: f32,
    glow: Halo,
) {
    outline_faded(renderer, theme, bounds, radius, glow, 1.0);
}

/// The focus outline with focus's glow, at `share` of its strength: a focus ring fades in and
/// out on the glows' own timing, kept by the widget in a [`crate::ui::transition::Lit`].
pub(crate) fn outline_lit(
    renderer: &mut Renderer,
    theme: &Theme,
    bounds: Rectangle,
    radius: f32,
    share: f32,
) {
    outline_faded(renderer, theme, bounds, radius, theme.focus_glow(), share);
}

/// The focus outline with this glow, its line and glow both at `share` of their strength;
/// nothing at all at none.
pub(crate) fn outline_faded(
    renderer: &mut Renderer,
    theme: &Theme,
    bounds: Rectangle,
    radius: f32,
    glow: Halo,
    share: f32,
) {
    let share = share.clamp(0.0, 1.0);
    if share <= 0.0 {
        return;
    }
    let outline = theme.outline();
    let grow = outline.offset + outline.width;
    let radius = radius.min(bounds.width / 2.0).min(bounds.height / 2.0);
    let bounds = bounds.expand(grow);
    let glow = Halo {
        color: glow.color.opacity(glow.color.alpha() * share),
        ..glow
    };
    renderer.fill_quad(
        renderer::Quad {
            bounds,
            border: Border {
                color: color(outline.color.opacity(outline.color.alpha() * share)),
                width: outline.width,
                radius: (radius + grow).into(),
            },
            shadow: halo(glow),
            snap: true,
        },
        Background::Color(Color::TRANSPARENT),
    );
}

/// A round dot, glowing if it's a light, such as a status chip's.
pub(crate) fn dot(
    renderer: &mut Renderer,
    center: Point,
    diameter: f32,
    ink: Color,
    glow: Option<Halo>,
) {
    let bounds = Rectangle::new(
        Point::new(center.x - diameter / 2.0, center.y - diameter / 2.0),
        Size::new(diameter, diameter),
    );
    renderer.fill_quad(
        renderer::Quad {
            bounds,
            border: Border {
                radius: (diameter / 2.0).into(),
                ..Border::default()
            },
            shadow: glow.map(halo).unwrap_or_default(),
            snap: false,
        },
        Background::Color(ink),
    );
}

/// A line of text kept shaped in a widget's state, reshaped only when it changes.
pub(crate) type Plain =
    iced::advanced::text::paragraph::Plain<<Renderer as text::Renderer>::Paragraph>;

/// A single line in a role, unbounded, to shape into a [`Plain`].
pub(crate) fn unbounded<'a>(
    role: &crate::ui::tokens::Role,
    content: &'a str,
) -> text::Text<&'a str, iced::Font> {
    text::Text {
        content,
        bounds: Size::INFINITE,
        size: Pixels(role.size),
        line_height: text::LineHeight::Relative(role.line_height),
        font: crate::ui::font::role(role),
        align_x: text::Alignment::Left,
        align_y: iced::alignment::Vertical::Top,
        shaping: text::Shaping::Auto,
        wrapping: text::Wrapping::None,
    }
}

/// Shapes `content` into `plain` if it changed: whether it did.
pub(crate) fn shape(plain: &mut Plain, role: &crate::ui::tokens::Role, content: &str) -> bool {
    plain.update(unbounded(role, content))
}

/// Draws a shaped line, its top left at a point.
pub(crate) fn fill(renderer: &mut Renderer, plain: &Plain, at: Point, ink: Color, clip: Rectangle) {
    renderer.fill_paragraph(plain.raw(), at, ink, clip);
}
