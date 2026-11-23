// SPDX-License-Identifier: GPL-3.0-or-later

//! The design tokens: colors, type, spacing, shape, motion and effects, as plain data with no
//! dependencies. They're the only source of colors, sizes and durations for the launcher's
//! widgets.
//!
//! Colors are roles, and a [`Palette`] gives each a value: [`Palette::SUNSET_DARK`] and
//! [`Palette::SUNSET_LIGHT`], from Leonida's logo.
//!
//! Text takes an [`Ink`], and `dim` isn't one: it's for placeholders, icons and decoration only,
//! as it falls below WCAG's 4.5:1 for normal-size text. The tests check the contrast of every ink
//! on every surface.

pub mod color;
pub mod effect;
pub mod motion;
pub mod typography;

pub use color::{Accent, Color, Ink, Mode, Palette, Surface, ThemeName};
pub use typography::{Face, Family, Role, Slant};

/// The spacing scale for padding and gaps, in logical pixels.
pub mod space {
    pub const XXS: f32 = 2.0;
    pub const XS: f32 = 4.0;
    pub const S: f32 = 6.0;
    pub const M: f32 = 8.0;
    pub const L: f32 = 12.0;
    pub const XL: f32 = 16.0;
    pub const XXL: f32 = 24.0;
    pub const XXXL: f32 = 32.0;

    /// The whole scale, smallest first.
    pub const SCALE: [f32; 8] = [XXS, XS, S, M, L, XL, XXL, XXXL];
}

/// Standard sizes, in logical pixels.
pub mod size {
    /// A loader's ring, across: small, the default and large; and its line.
    pub const LOADER_SMALL: f32 = 24.0;
    pub const LOADER: f32 = 36.0;
    pub const LOADER_LARGE: f32 = 48.0;
    pub const LOADER_RING: f32 = 2.0;
    /// A window's title bar.
    pub const TITLE_BAR: f32 = 47.0;
    /// Buttons and text fields.
    pub const CONTROL: f32 = 36.0;
    /// A toggle group's pill: a little lower than a control, so it fits a header bar.
    pub const TOGGLE: f32 = 32.0;
    /// A status chip's dot.
    pub const DOT: f32 = 6.0;
    /// A progress bar's thickness.
    pub const PROGRESS: f32 = 6.0;
    /// Every border: 1px everywhere.
    pub const BORDER: f32 = 1.0;
    /// The keyboard focus outline, always shown with focus (glow is never the only sign), and its
    /// gap from what it outlines.
    pub const FOCUS_OUTLINE: f32 = 2.0;
    pub const FOCUS_OFFSET: f32 = 2.0;
    /// A line icon.
    pub const ICON: f32 = 16.0;
    /// An icon's stroke, with round caps and joins.
    pub const ICON_STROKE: f32 = 1.4;
    /// Room round a toggle group's segments inside it.
    pub const SEGMENT_INSET: f32 = 3.0;
    /// The text caret's width.
    pub const CARET: f32 = 2.0;
    /// A list or table row.
    pub const ROW: f32 = 28.0;
    /// A sidebar's row: taller than a list's.
    pub const SIDEBAR_ROW: f32 = 36.0;
    /// The mark at the chosen sidebar row's start: a pill this wide, set in from the row's start
    /// as far as from its top and bottom.
    pub const SIDEBAR_MARK: f32 = 4.0;
    pub const SIDEBAR_MARK_INSET: f32 = 8.0;
    /// A switch's track: a pill, its knob inside it.
    pub const SWITCH_WIDTH: f32 = 40.0;
    pub const SWITCH_HEIGHT: f32 = 22.0;
    /// A checkbox's box and a choice's radio circle.
    pub const CHECK: f32 = 16.0;
    /// The least a flow's filling child, such as a setting's words, is beside the others:
    /// narrower, they go one under another.
    pub const FLOW_LEAST: f32 = 160.0;
    /// How far a press may move and still be a press: further, it's a drag, such as a finger
    /// scrolling the page from a button.
    pub const SLIP: f32 = super::space::S;
}

/// Corner radii, in logical pixels.
pub mod radius {
    /// A checkbox's box.
    pub const CHECK: f32 = 4.0;
    /// Chips and key caps (4–6px).
    pub const CHIP: f32 = 6.0;
    /// Controls and cards (8–10px).
    pub const CONTROL: f32 = 10.0;
    /// Buttons, toggle groups and window buttons.
    pub const BUTTON: f32 = 8.0;
    /// A toggle group's segment: a button's, its group's, less the room round it.
    pub const SEGMENT: f32 = BUTTON - super::size::SEGMENT_INSET;
    /// Windows (12–14px).
    pub const WINDOW: f32 = 14.0;
    /// Round ends: status chips, progress bars, dots. Larger than anything it rounds.
    pub const ROUND: f32 = 999.0;
}

/// A window's own decorations: the launcher draws them itself (client-side), the same on every
/// desktop. The header bar, its close button, the frame and the zones that resize the window.
pub mod window {
    /// The header bar's height, its bottom line included.
    pub const BAR_HEIGHT: f32 = super::size::TITLE_BAR;
    /// Room round a sidebar's places, at its sides.
    pub const SIDEBAR_PAD: f32 = super::space::S;
    /// Room before the bar's first item: as far in as a sidebar's words start, so the app's name
    /// lines up with them.
    pub const BAR_PAD_START: f32 =
        SIDEBAR_PAD + 2.0 * super::size::SIDEBAR_MARK_INSET + super::size::SIDEBAR_MARK;
    /// A sidebar beside the page: always this wide.
    pub const SIDEBAR_WIDTH: f32 = 240.0;
    /// Narrower than this, in logical pixels, a window is as a phone's: its sidebar folds,
    /// showing the sidebar or the page, and the page's header bar a way back.
    pub const FOLD_BELOW: f32 = 640.0;
    /// Between the bar's items.
    pub const BAR_GAP: f32 = 14.0;
    /// A window button: close, the only one. Square, and as tall as everything on the bar.
    pub const BUTTON_SIZE: f32 = 32.0;
    /// The close button's room on each side: above and below it in the bar, and from the
    /// window's edge.
    pub const BUTTON_MARGIN: f32 = (BAR_HEIGHT - super::size::BORDER - BUTTON_SIZE) / 2.0;
    /// Room after the bar's last item: the close button's margin.
    pub const BAR_PAD_END: f32 = BUTTON_MARGIN;
    /// A window button's corners: a button's.
    pub const BUTTON_RADIUS: f32 = super::radius::BUTTON;
    /// The close button's cross, in its 16px icon box, in the icon stroke. Each stroke
    /// `[x0, y0, x1, y1]`.
    pub const CLOSE_CROSS: [[f32; 4]; 2] = [[3.7, 3.7, 12.3, 12.3], [12.3, 3.7, 3.7, 12.3]];
    /// The back button's chevron, pointing left, in the same box and stroke.
    pub const BACK_CHEVRON: [[f32; 4]; 2] = [[10.0, 3.5, 5.5, 8.0], [5.5, 8.0, 10.0, 12.5]];
    /// The frame's corners while the window floats; maximized it's square.
    pub const CORNER_RADIUS: f32 = super::radius::WINDOW;
    /// The frame's line, while it floats.
    pub const BORDER: f32 = super::size::BORDER;
    /// The zones that resize a floating window: a band this wide inside each edge, and at each
    /// corner a zone reaching this far along both edges.
    pub const RESIZE_BAND: f32 = 6.0;
    pub const RESIZE_CORNER: f32 = 14.0;
    /// How far the pointer moves with the button held on the header bar before the window
    /// moves with it, so a double-click doesn't start a move.
    pub const DRAG_THRESHOLD: f32 = 4.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spacing_scale_grows() {
        assert!(space::SCALE.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn radii_are_in_the_design_languages_ranges() {
        assert!((4.0..=6.0).contains(&radius::CHIP));
        assert!((8.0..=10.0).contains(&radius::CONTROL));
        assert!((12.0..=14.0).contains(&radius::WINDOW));
    }

    #[test]
    fn the_window_decorations() {
        assert_eq!(window::BAR_HEIGHT, 47.0);
        assert_eq!((window::BUTTON_MARGIN, window::BAR_PAD_END), (7.0, 7.0));
        assert_eq!((window::BUTTON_SIZE, window::BUTTON_RADIUS), (32.0, 8.0));
        assert_eq!((window::CORNER_RADIUS, window::BORDER), (14.0, 1.0));
        const { assert!(window::RESIZE_CORNER > window::RESIZE_BAND) };
        const { assert!(window::RESIZE_CORNER >= window::CORNER_RADIUS) };
        for stroke in window::CLOSE_CROSS {
            for c in stroke {
                assert!((0.0..=size::ICON).contains(&c));
            }
        }
        // The close button sits inside the rounded corner's arc, so the corner needs no clipping.
        let corner = window::CORNER_RADIUS;
        let from_center = ((corner - window::BUTTON_MARGIN).powi(2) * 2.0).sqrt();
        assert!(from_center < corner);
    }
}
