// SPDX-License-Identifier: GPL-3.0-or-later

//! Effects: glow, and the faint tints on title bars. Light comes from interaction: glow shows
//! focus, the selected item and a primary button on hover, never resting UI.
//!
//! A glow's shape is a palette's: in dark it is a halo all round, in light a soft colored shadow
//! below ([`super::Palette::glows`]).

use super::color::{Accent, Color, Palette};

/// A glow: a ring of the accent's line tint, and a blurred shadow of the accent around it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glow {
    pub accent: Accent,
    /// The ring's width, in logical pixels.
    pub ring: f32,
    /// How far down the shadow sits: 0 for a halo all round.
    pub offset: f32,
    /// The shadow's blur radius, in logical pixels.
    pub blur: f32,
    /// The shadow's opacity.
    pub opacity: f32,
}

impl Glow {
    /// The ring's color in a palette.
    pub const fn ring_color(self, palette: &Palette) -> Color {
        palette.accent_line(self.accent)
    }

    /// The halo's color in a palette.
    pub const fn halo_color(self, palette: &Palette) -> Color {
        palette.accent(self.accent).opacity(self.opacity)
    }
}

/// A palette's glows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glows {
    /// Keyboard focus and hover, in cyan, the secondary color. Never the only sign of focus: a 2px
    /// outline always shows it.
    pub focus: Glow,
    /// The selected item, and the main button on hover: in the secondary color too, as every glow.
    pub selected: Glow,
    /// The small lights, such as status dots: their accent, blurred this much.
    pub spark: Spark,
}

/// A small light's glow: its own accent, blurred.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spark {
    pub blur: f32,
    pub opacity: f32,
}

/// Sunset dark's: halos all round, Neon dark's shapes.
pub(crate) const SUNSET_DARK: Glows = Glows {
    focus: Glow {
        accent: Accent::Cyan,
        ring: 1.0,
        offset: 0.0,
        blur: 22.0,
        opacity: 0.16,
    },
    selected: Glow {
        accent: Accent::Cyan,
        ring: 1.0,
        offset: 0.0,
        blur: 26.0,
        opacity: 0.20,
    },
    spark: Spark {
        blur: 8.0,
        opacity: 1.0,
    },
};

/// Sunset light's: soft colored shadows, 4px down.
pub(crate) const SUNSET_LIGHT: Glows = Glows {
    focus: Glow {
        accent: Accent::Cyan,
        ring: 1.0,
        offset: 4.0,
        blur: 14.0,
        opacity: 0.16,
    },
    selected: Glow {
        accent: Accent::Cyan,
        ring: 1.0,
        offset: 4.0,
        blur: 14.0,
        opacity: 0.16,
    },
    spark: Spark {
        blur: 4.0,
        opacity: 0.45,
    },
};

/// An accent at an opacity, fading in from one edge of a bar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeTint {
    pub accent: Accent,
    /// The opacity at the edge.
    pub opacity: f32,
}

impl EdgeTint {
    /// The color at the edge, in a palette.
    pub const fn color(self, palette: &Palette) -> Color {
        palette.accent(self.accent).opacity(self.opacity)
    }
}

/// A title bar's edge tints, over its `panel`: the first accent fading from the left edge to
/// nothing, and the second from the right edge, each across `reach` of the bar's width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeTints {
    pub start: EdgeTint,
    pub end: EdgeTint,
    /// How far in from its edge each tint fades to nothing, as a fraction of the width.
    pub reach: f32,
}

/// Sunset dark's title bar: the main color at 12% from the left, the secondary at 8% from the
/// right.
pub(crate) const SUNSET_DARK_TITLE_BAR: EdgeTints = EdgeTints {
    start: EdgeTint {
        accent: Accent::Magenta,
        opacity: 0.12,
    },
    end: EdgeTint {
        accent: Accent::Cyan,
        opacity: 0.08,
    },
    reach: 0.28,
};

/// Sunset light's: the same, fainter (6% and 5%), in light's deeper accents.
pub(crate) const SUNSET_LIGHT_TITLE_BAR: EdgeTints = EdgeTints {
    start: EdgeTint {
        accent: Accent::Magenta,
        opacity: 0.06,
    },
    end: EdgeTint {
        accent: Accent::Cyan,
        opacity: 0.05,
    },
    reach: 0.28,
};

/// The brand gradient, for the brand word: the logo's blue to its magenta.
pub const BRAND: (Accent, Accent) = (Accent::Cyan, Accent::Magenta);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glows_follow_the_design_language() {
        // Dark: a 1px ring and a halo of 22–26px at 16–20%, all round.
        let dark = Palette::SUNSET_DARK;
        for glow in [dark.glows.focus, dark.glows.selected] {
            assert_eq!((glow.ring, glow.offset), (1.0, 0.0));
            assert!((22.0..=26.0).contains(&glow.blur));
            assert!((0.16..=0.20).contains(&glow.opacity));
            assert_eq!(glow.ring_color(&dark), dark.accent_line(glow.accent));
            let halo = glow.halo_color(&dark);
            assert_eq!(halo.rgb8(), dark.accent(glow.accent).rgb8());
            assert_eq!(halo.alpha(), glow.opacity);
        }
        // Light: the same ring, and a shadow 4px down, 14px soft, at 16%.
        let light = Palette::SUNSET_LIGHT;
        for glow in [light.glows.focus, light.glows.selected] {
            let shape = (glow.ring, glow.offset, glow.blur, glow.opacity);
            assert_eq!(shape, (1.0, 4.0, 14.0, 0.16));
        }
        for palette in Palette::ALL {
            assert_eq!(palette.glows.focus.accent, Accent::Cyan);
            assert_eq!(palette.glows.selected.accent, Accent::Cyan);
        }
    }

    #[test]
    fn title_bar_tints_are_faint() {
        for palette in Palette::ALL {
            let bar = palette.title_bar;
            assert!(bar.start.opacity <= 0.12 && bar.end.opacity <= 0.10);
            assert!(bar.start.opacity > 0.0 && bar.end.opacity > 0.0);
            assert_eq!(bar.reach, 0.28);
            let start = bar.start.color(&palette);
            assert_eq!(start.rgb8(), palette.accent(bar.start.accent).rgb8());
            assert_eq!(start.alpha(), bar.start.opacity);
        }
    }
}
