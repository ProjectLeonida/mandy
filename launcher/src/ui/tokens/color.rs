// SPDX-License-Identifier: GPL-3.0-or-later

//! Colors: the roles colors play, apart from their values. The roles are the surfaces, the lines,
//! the inks text takes, `dim`, and the accents with their meanings and tints. A [`Palette`] gives
//! every role a value.
//!
//! A theme is a named look ([`ThemeName`]), and the mode, light or dark ([`Mode`]), is a switch
//! apart that applies to every theme, so each theme has a palette for each mode. Sunset is the one
//! theme here: [`Palette::SUNSET_DARK`], the default, and [`Palette::SUNSET_LIGHT`]. Its values come
//! from the logo's pixels: the gradient's stops (blue, violet, magenta, pink, coral, orange) give
//! the accents, its deep purple outline the lines, and its near-black purple shadow the dark
//! surfaces and the light palette's text. An accent keeps its stop's hue, lightened (dark) or
//! deepened (light) only as far as it must to read as text on every surface.
//!
//! Also WCAG 2.1 contrast, which the tests check for every palette.

use std::fmt;

/// An sRGB color with straight alpha. Only this module makes them, so every color in the launcher
/// is a token, a tint of one, or tokens painted over each other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: f32,
}

impl Color {
    pub(crate) const fn hex(rgb: u32) -> Self {
        Self {
            r: (rgb >> 16) as u8,
            g: (rgb >> 8) as u8,
            b: rgb as u8,
            a: 1.0,
        }
    }

    /// The same color at this opacity.
    pub const fn opacity(self, a: f32) -> Self {
        Self { a, ..self }
    }

    /// Red, green and blue.
    pub const fn rgb8(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    /// The opacity, from 0 to 1.
    pub const fn alpha(self) -> f32 {
        self.a
    }

    /// This color painted over an opaque one, blended in sRGB (as iced blends with `web-colors`).
    pub fn over(self, below: Color) -> Color {
        let mix = |top: u8, bottom: u8| {
            (f32::from(top) * self.a + f32::from(bottom) * (1.0 - self.a)).round() as u8
        };
        Color {
            r: mix(self.r, below.r),
            g: mix(self.g, below.g),
            b: mix(self.b, below.b),
            a: 1.0,
        }
    }

    /// On the way from this color to `to`, `share` of the way, its opacity too: blended in sRGB,
    /// as [`Color::over`] blends.
    pub fn mix(self, to: Color, share: f32) -> Color {
        let share = share.clamp(0.0, 1.0);
        let mix = |from: u8, to: u8| {
            (f32::from(from) + (f32::from(to) - f32::from(from)) * share).round() as u8
        };
        Color {
            r: mix(self.r, to.r),
            g: mix(self.g, to.g),
            b: mix(self.b, to.b),
            a: self.a + (to.a - self.a) * share,
        }
    }

    /// The relative luminance of WCAG 2.1, from 0 for black to 1 for white, ignoring opacity.
    pub fn luminance(self) -> f64 {
        let linear = |channel: u8| {
            let c = f64::from(channel) / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * linear(self.r) + 0.7152 * linear(self.g) + 0.0722 * linear(self.b)
    }
}

/// `#rrggbb`, and the opacity in percent after a tint's: `#3a8ede 14%`.
impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02x}{:02x}{:02x}", self.r, self.g, self.b)?;
        if self.a < 1.0 {
            write!(f, " {}%", (self.a * 100.0).round())?;
        }
        Ok(())
    }
}

/// The contrast ratio of WCAG 2.1 between two opaque colors, from 1 to 21. Paint a tint over its
/// surface first ([`Color::over`]).
pub fn contrast(a: Color, b: Color) -> f64 {
    let (la, lb) = (a.luminance(), b.luminance());
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

/// The logo's own colors, as sampled from `assets/logo.png` (5×5 averages, fully opaque pixels).
pub mod logo {
    use super::Color;

    /// The gradient's top: blue.
    pub const BLUE: Color = Color::hex(0x036ed5);
    /// Below it: violet.
    pub const VIOLET: Color = Color::hex(0x984ecb);
    /// The middle: magenta.
    pub const MAGENTA: Color = Color::hex(0xe733ae);
    /// Pink, where magenta turns to coral.
    pub const PINK: Color = Color::hex(0xfa4093);
    /// Coral.
    pub const CORAL: Color = Color::hex(0xff5d72);
    /// The bottom: orange.
    pub const ORANGE: Color = Color::hex(0xfd7744);
    /// The outline round the letters: deep purple.
    pub const OUTLINE: Color = Color::hex(0x5c2694);
    /// The shadow and the outline's inner edge: near-black purple.
    pub const SHADOW: Color = Color::hex(0x1a012b);

    /// The gradient, top to bottom.
    pub const GRADIENT: [Color; 6] = [BLUE, VIOLET, MAGENTA, PINK, CORAL, ORANGE];
}

/// The surfaces text and controls sit on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Surface {
    /// Behind windows.
    Void,
    /// The app background.
    Bg,
    /// Bars and title bars.
    Panel,
    /// Hovered rows and raised parts.
    Raise,
    /// Fields and embedded content, such as a log.
    Sunk,
}

impl Surface {
    pub const ALL: [Self; 5] = [Self::Void, Self::Bg, Self::Panel, Self::Raise, Self::Sunk];

    /// The token's name in the design language.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Void => "void",
            Self::Bg => "bg",
            Self::Panel => "panel",
            Self::Raise => "raise",
            Self::Sunk => "sunk",
        }
    }
}

/// The accents. Each has one meaning, and color is never the only signal. The names are the
/// design language's roles; in Sunset, cyan is the logo's blue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Accent {
    Cyan,
    Magenta,
    Violet,
    Mint,
    Amber,
    Red,
    CodePink,
}

impl Accent {
    pub const ALL: [Self; 7] = [
        Self::Cyan,
        Self::Magenta,
        Self::Violet,
        Self::Mint,
        Self::Amber,
        Self::Red,
        Self::CodePink,
    ];

    /// The token's name in the design language.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cyan => "cyan",
            Self::Magenta => "magenta",
            Self::Violet => "violet",
            Self::Mint => "mint",
            Self::Amber => "amber",
            Self::Red => "red",
            Self::CodePink => "code-pink",
        }
    }

    /// What the accent means, whatever its value in a palette.
    pub const fn meaning(self) -> &'static str {
        match self {
            Self::Cyan => "Focus, links, interactive elements, the text caret",
            Self::Magenta => "Selection, current item, brand",
            Self::Violet => "Information, callouts, planned states",
            Self::Mint => "Success, synced, done",
            Self::Amber => "Warnings, conflicts",
            Self::Red => "Errors, destructive actions",
            Self::CodePink => "Inline code",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// The colors text takes. In every palette each reaches WCAG's 4.5:1 for normal-size text on
/// every surface; `dim` doesn't, so it isn't one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ink {
    /// Primary text and titles.
    Text,
    /// Long-form reading text.
    Body,
    /// Secondary text, small metadata such as counts and the status line included.
    Muted,
    Accent(Accent),
}

impl Ink {
    pub const ALL: [Self; 10] = [
        Self::Text,
        Self::Body,
        Self::Muted,
        Self::Accent(Accent::Cyan),
        Self::Accent(Accent::Magenta),
        Self::Accent(Accent::Violet),
        Self::Accent(Accent::Mint),
        Self::Accent(Accent::Amber),
        Self::Accent(Accent::Red),
        Self::Accent(Accent::CodePink),
    ];

    /// The token's name in the design language.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Body => "body",
            Self::Muted => "muted",
            Self::Accent(accent) => accent.name(),
        }
    }
}

impl From<Accent> for Ink {
    fn from(accent: Accent) -> Self {
        Self::Accent(accent)
    }
}

/// An accent's value: its color, and the opacities of its two tints, a soft fill for backgrounds
/// (7–15%) and a line for borders (38–45%).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tinted {
    pub color: Color,
    pub soft: f32,
    pub line: f32,
}

/// A named look. Sunset, from Leonida's logo, is the one there is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ThemeName {
    #[default]
    Sunset,
}

/// Light or dark, for every theme. Dark is the default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    #[default]
    Dark,
    Light,
}

impl Mode {
    /// The other one.
    pub const fn toggled(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::Dark,
        }
    }
}

/// A value for every color role: one theme in one mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Palette {
    pub theme: ThemeName,
    pub mode: Mode,
    /// What it's called: "sunset-dark".
    pub name: &'static str,
    void: Color,
    bg: Color,
    panel: Color,
    raise: Color,
    sunk: Color,
    /// Borders.
    pub line: Color,
    /// Inner dividers, and a progress bar's track.
    pub line_soft: Color,
    text: Color,
    body: Color,
    muted: Color,
    /// Placeholders, icons and decoration, never other text.
    pub dim: Color,
    /// The main action's fill: the main color as the logo has it, too bright or too dark to be
    /// text on the page itself; and the ink on it.
    pub main_fill: Color,
    pub on_main: Color,
    accents: [Tinted; 7],
    /// A function's name in code.
    pub code_function: Color,
    pub glows: super::effect::Glows,
    /// A title bar's edge tints.
    pub title_bar: super::effect::EdgeTints,
}

impl Palette {
    /// Sunset in dark: the logo's near-black purple shadow as the app's background, surfaces a
    /// step either side of it, its deep purple outline as the line. The logo's dark purple is the
    /// main color (the main action, selection, the brand) and its light orange the secondary
    /// (focus, what can be acted on, every glow).
    pub const SUNSET_DARK: Palette = Palette {
        theme: ThemeName::Sunset,
        mode: Mode::Dark,
        name: "sunset-dark",
        void: Color::hex(0x0e0118),
        bg: logo::SHADOW,
        panel: Color::hex(0x210436),
        raise: Color::hex(0x2c0a47),
        sunk: Color::hex(0x130120),
        line: logo::OUTLINE,
        line_soft: Color::hex(0x3b1561),
        text: Color::hex(0xf7efff),
        body: Color::hex(0xe4d5f3),
        muted: Color::hex(0xb9a2d2),
        dim: Color::hex(0x7c5e9c),
        // The main action's fill is the logo's dark purple itself, with light text on it.
        main_fill: logo::OUTLINE,
        on_main: Color::hex(0xf7efff),
        // Each accent's soft and line opacities: Neon dark's, but amber's soft (8%).
        accents: [
            // Cyan, the secondary: the logo's light orange, as it is.
            Tinted {
                color: logo::ORANGE,
                soft: 0.14,
                line: 0.38,
            },
            // Magenta, the main: the logo's dark purple, 41% of the way to white to read as
            // text and lines on the dark surfaces.
            Tinted {
                color: Color::hex(0x9f7fc0),
                soft: 0.15,
                line: 0.45,
            },
            // Violet, information: the logo's blue, 22% of the way to white.
            Tinted {
                color: Color::hex(0x3a8ede),
                soft: 0.10,
                line: 0.45,
            },
            // The logo has no green: success keeps a mint.
            Tinted {
                color: Color::hex(0x3ee0a6),
                soft: 0.12,
                line: 0.38,
            },
            // Amber, warnings: a gold, apart from the orange.
            Tinted {
                color: Color::hex(0xffc643),
                soft: 0.08,
                line: 0.45,
            },
            // Red, errors: the logo's coral, as it is.
            Tinted {
                color: logo::CORAL,
                soft: 0.15,
                line: 0.45,
            },
            // Its pink, as it is.
            Tinted {
                color: logo::PINK,
                soft: 0.10,
                line: 0.40,
            },
        ],
        code_function: Color::hex(0xffc08a),
        glows: super::effect::SUNSET_DARK,
        title_bar: super::effect::SUNSET_DARK_TITLE_BAR,
    };

    /// Sunset in light: pale lavender surfaces and the logo's shadow as text. The logo's light
    /// orange is the main color and its dark purple the secondary, every glow included.
    pub const SUNSET_LIGHT: Palette = Palette {
        theme: ThemeName::Sunset,
        mode: Mode::Light,
        name: "sunset-light",
        void: Color::hex(0xefe5f8),
        bg: Color::hex(0xfdf9ff),
        panel: Color::hex(0xf6eefc),
        raise: Color::hex(0xecdff7),
        sunk: Color::hex(0xfaf4fe),
        line: Color::hex(0xc9afe2),
        line_soft: Color::hex(0xe3d3f1),
        text: logo::SHADOW,
        body: Color::hex(0x33104f),
        muted: Color::hex(0x5e4379),
        dim: Color::hex(0x9077aa),
        // The main action's fill is the logo's light orange itself, with dark text on it.
        main_fill: logo::ORANGE,
        on_main: logo::SHADOW,
        // Soft tints at 9% (violet and amber 8%, red 10%), lines at 40% (mint 35%): Neon light's.
        accents: [
            // Cyan, the secondary: the logo's dark purple, as it is.
            Tinted {
                color: logo::OUTLINE,
                soft: 0.09,
                line: 0.40,
            },
            // Magenta, the main: the logo's light orange, 37% of the way to black to read as
            // text and lines on the light surfaces.
            Tinted {
                color: Color::hex(0x9f4b2b),
                soft: 0.09,
                line: 0.40,
            },
            // Violet, information: the logo's blue, 11% of the way to black.
            Tinted {
                color: Color::hex(0x0362be),
                soft: 0.08,
                line: 0.40,
            },
            Tinted {
                color: Color::hex(0x1f7053),
                soft: 0.09,
                line: 0.35,
            },
            // Amber, warnings: a deep gold, apart from the orange.
            Tinted {
                color: Color::hex(0x784c00),
                soft: 0.08,
                line: 0.40,
            },
            // Red, errors: the logo's coral, 34% of the way to black.
            Tinted {
                color: Color::hex(0xa83d4b),
                soft: 0.10,
                line: 0.40,
            },
            // Its pink, 28% of the way to black.
            Tinted {
                color: Color::hex(0xb42e6a),
                soft: 0.09,
                line: 0.40,
            },
        ],
        code_function: Color::hex(0x7f4a00),
        glows: super::effect::SUNSET_LIGHT,
        title_bar: super::effect::SUNSET_LIGHT_TITLE_BAR,
    };

    /// Every palette there is: the tests check each.
    pub const ALL: [Palette; 2] = [Self::SUNSET_DARK, Self::SUNSET_LIGHT];

    /// A theme's palette in a mode.
    pub const fn of(theme: ThemeName, mode: Mode) -> Palette {
        match (theme, mode) {
            (ThemeName::Sunset, Mode::Dark) => Self::SUNSET_DARK,
            (ThemeName::Sunset, Mode::Light) => Self::SUNSET_LIGHT,
        }
    }

    /// On the way from this palette to `to`, `share` of the way, as one mode fades into the
    /// other: every color, tint and glow between the two. Which theme and mode it is, and its
    /// name, are the nearer one's, and so are the accents its glows and tints take.
    pub fn mixed(&self, to: &Palette, share: f32) -> Palette {
        use super::effect::{EdgeTint, EdgeTints, Glow, Glows, Spark};
        let share = share.clamp(0.0, 1.0);
        let nearer = if share < 0.5 { self } else { to };
        let color = |from: Color, to: Color| from.mix(to, share);
        let number = |from: f32, to: f32| from + (to - from) * share;
        let mut accents = self.accents;
        for (tinted, to) in accents.iter_mut().zip(&to.accents) {
            *tinted = Tinted {
                color: color(tinted.color, to.color),
                soft: number(tinted.soft, to.soft),
                line: number(tinted.line, to.line),
            };
        }
        let glow = |from: Glow, to: Glow| Glow {
            accent: if share < 0.5 { from.accent } else { to.accent },
            ring: number(from.ring, to.ring),
            offset: number(from.offset, to.offset),
            blur: number(from.blur, to.blur),
            opacity: number(from.opacity, to.opacity),
        };
        let tint = |from: EdgeTint, to: EdgeTint| EdgeTint {
            accent: if share < 0.5 { from.accent } else { to.accent },
            opacity: number(from.opacity, to.opacity),
        };
        Palette {
            theme: nearer.theme,
            mode: nearer.mode,
            name: nearer.name,
            void: color(self.void, to.void),
            bg: color(self.bg, to.bg),
            panel: color(self.panel, to.panel),
            raise: color(self.raise, to.raise),
            sunk: color(self.sunk, to.sunk),
            line: color(self.line, to.line),
            line_soft: color(self.line_soft, to.line_soft),
            text: color(self.text, to.text),
            body: color(self.body, to.body),
            muted: color(self.muted, to.muted),
            dim: color(self.dim, to.dim),
            main_fill: color(self.main_fill, to.main_fill),
            on_main: color(self.on_main, to.on_main),
            accents,
            code_function: color(self.code_function, to.code_function),
            glows: Glows {
                focus: glow(self.glows.focus, to.glows.focus),
                selected: glow(self.glows.selected, to.glows.selected),
                spark: Spark {
                    blur: number(self.glows.spark.blur, to.glows.spark.blur),
                    opacity: number(self.glows.spark.opacity, to.glows.spark.opacity),
                },
            },
            title_bar: EdgeTints {
                start: tint(self.title_bar.start, to.title_bar.start),
                end: tint(self.title_bar.end, to.title_bar.end),
                reach: number(self.title_bar.reach, to.title_bar.reach),
            },
        }
    }

    pub const fn surface(&self, surface: Surface) -> Color {
        match surface {
            Surface::Void => self.void,
            Surface::Bg => self.bg,
            Surface::Panel => self.panel,
            Surface::Raise => self.raise,
            Surface::Sunk => self.sunk,
        }
    }

    pub const fn ink(&self, ink: Ink) -> Color {
        match ink {
            Ink::Text => self.text,
            Ink::Body => self.body,
            Ink::Muted => self.muted,
            Ink::Accent(accent) => self.accent(accent),
        }
    }

    pub const fn accent(&self, accent: Accent) -> Color {
        self.accents[accent.index()].color
    }

    /// An accent's opacities.
    pub const fn tinted(&self, accent: Accent) -> Tinted {
        self.accents[accent.index()]
    }

    /// An accent's soft tint, a fill for backgrounds.
    pub const fn soft(&self, accent: Accent) -> Color {
        let tinted = self.tinted(accent);
        tinted.color.opacity(tinted.soft)
    }

    /// An accent's line tint, for borders.
    pub const fn accent_line(&self, accent: Accent) -> Color {
        let tinted = self.tinted(accent);
        tinted.color.opacity(tinted.line)
    }

    /// The theme's own gradient, as drawn pictures take it (the Play page's scene): for Sunset,
    /// the logo's stops, top to bottom.
    pub const fn gradient(&self) -> [Color; 6] {
        match self.theme {
            ThemeName::Sunset => logo::GRADIENT,
        }
    }

    /// The theme's outline color, as drawn pictures take it: for Sunset, the logo's deep purple.
    pub const fn outline(&self) -> Color {
        match self.theme {
            ThemeName::Sunset => logo::OUTLINE,
        }
    }

    /// The theme's deepest color, as drawn pictures take it: for Sunset, the logo's shadow.
    pub const fn shadow(&self) -> Color {
        match self.theme {
            ThemeName::Sunset => logo::SHADOW,
        }
    }

    /// The logo's gradient in this palette's accents, for what shows how far something is: cyan
    /// (the blue), violet, magenta, then amber (the orange), each one that reads on the surfaces.
    pub const fn sunset(&self) -> [Color; 2] {
        match self.mode {
            Mode::Dark => [self.accent(Accent::Magenta), self.accent(Accent::Cyan)],
            Mode::Light => [self.accent(Accent::Cyan), self.accent(Accent::Magenta)],
        }
    }
}

impl Default for Palette {
    /// The default theme in the default mode: Sunset, dark.
    fn default() -> Self {
        Self::of(ThemeName::default(), Mode::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAIN: [Surface; 3] = [Surface::Bg, Surface::Panel, Surface::Raise];

    fn on(palette: &Palette, color: Color, surface: Surface) -> f64 {
        contrast(color, palette.surface(surface))
    }

    #[test]
    fn contrast_runs_from_1_to_21() {
        let black = Color::hex(0x000000);
        let white = Color::hex(0xffffff);
        assert_eq!(black.luminance(), 0.0);
        assert!((white.luminance() - 1.0).abs() < 1e-12);
        assert!((contrast(black, white) - 21.0).abs() < 1e-9);
        assert_eq!(contrast(white, black), contrast(black, white));
        assert_eq!(contrast(logo::BLUE, logo::BLUE), 1.0);
    }

    #[test]
    fn every_ink_reads_on_every_surface() {
        for palette in Palette::ALL {
            for ink in Ink::ALL {
                for surface in Surface::ALL {
                    let ratio = on(&palette, palette.ink(ink), surface);
                    assert!(
                        ratio >= 4.5,
                        "{}: {} on {}: {ratio:.2}",
                        palette.name,
                        ink.name(),
                        surface.name()
                    );
                }
            }
        }
    }

    #[test]
    fn dim_is_no_ink() {
        for palette in Palette::ALL {
            assert!(Ink::ALL.iter().all(|ink| palette.ink(*ink) != palette.dim));
            for surface in MAIN {
                let ratio = on(&palette, palette.dim, surface);
                assert!(
                    ratio < 4.5,
                    "{}: dim would pass as text on {surface:?}",
                    palette.name
                );
            }
        }
    }

    #[test]
    fn dim_meets_3_to_1_for_icons() {
        // WCAG's 3:1 for icons and other non-text, which dim is for.
        for palette in Palette::ALL {
            for surface in Surface::ALL {
                let ratio = on(&palette, palette.dim, surface);
                assert!(
                    ratio >= 3.0,
                    "{}: dim on {}: {ratio:.2}",
                    palette.name,
                    surface.name()
                );
            }
        }
    }

    #[test]
    fn text_reads_on_every_soft_tint() {
        // A selected or tinted part: text, body and muted on any accent's soft fill.
        for palette in Palette::ALL {
            for accent in Accent::ALL {
                for surface in MAIN {
                    let fill = palette.soft(accent).over(palette.surface(surface));
                    for ink in [Ink::Text, Ink::Body, Ink::Muted] {
                        let ratio = contrast(palette.ink(ink), fill);
                        assert!(
                            ratio >= 4.5,
                            "{}: {} on {}-soft over {}: {ratio:.2}",
                            palette.name,
                            ink.name(),
                            accent.name(),
                            surface.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn accents_read_on_their_own_tint_over_bg_and_panel() {
        // A status chip: the accent on its own soft fill.
        for palette in Palette::ALL {
            for accent in Accent::ALL {
                for surface in [Surface::Bg, Surface::Panel] {
                    let fill = palette.soft(accent).over(palette.surface(surface));
                    let ratio = contrast(palette.accent(accent), fill);
                    assert!(
                        ratio >= 4.5,
                        "{}: {} on its tint over {}: {ratio:.2}",
                        palette.name,
                        accent.name(),
                        surface.name()
                    );
                }
            }
        }
    }

    #[test]
    fn tints_are_in_their_ranges() {
        // The design language's: dark soft 7–15%, line 38–45%; light 8–10% and 35–40%.
        let ranges = |mode| match mode {
            Mode::Dark => (0.07..=0.15, 0.38..=0.45),
            Mode::Light => (0.08..=0.10, 0.35..=0.40),
        };
        for palette in Palette::ALL {
            let (soft, line) = ranges(palette.mode);
            for accent in Accent::ALL {
                assert_eq!(palette.soft(accent).rgb8(), palette.accent(accent).rgb8());
                let (s, l) = (
                    palette.soft(accent).alpha(),
                    palette.accent_line(accent).alpha(),
                );
                assert!(soft.contains(&s), "{}: {accent:?} soft {s}", palette.name);
                assert!(line.contains(&l), "{}: {accent:?} line {l}", palette.name);
            }
        }
        let dark = Palette::SUNSET_DARK;
        assert_eq!(dark.soft(Accent::Cyan).to_string(), "#fd7744 14%");
    }

    #[test]
    fn code_reads_on_every_surface() {
        for palette in Palette::ALL {
            for surface in Surface::ALL {
                let ratio = on(&palette, palette.code_function, surface);
                assert!(
                    ratio >= 4.5,
                    "{}: on {}: {ratio:.2}",
                    palette.name,
                    surface.name()
                );
            }
        }
    }

    #[test]
    fn the_palettes_come_from_the_logo() {
        let (dark, light) = (Palette::SUNSET_DARK, Palette::SUNSET_LIGHT);
        // The shadow is dark's background and light's text; the outline is dark's line.
        assert_eq!(dark.surface(Surface::Bg), logo::SHADOW);
        assert_eq!(light.ink(Ink::Text), logo::SHADOW);
        assert_eq!(dark.line, logo::OUTLINE);
        // The two colors: in dark, purple the main and orange the secondary; in light, the other
        // way round. The main's fill is the logo's color itself.
        assert_eq!(
            (dark.main_fill, light.main_fill),
            (logo::OUTLINE, logo::ORANGE)
        );
        assert_eq!(dark.accent(Accent::Cyan), logo::ORANGE);
        assert_eq!(light.accent(Accent::Cyan), logo::OUTLINE);
        // As text, the main keeps its hue: purple's blue over red, orange's red over the rest.
        let [r, g, b] = dark.accent(Accent::Magenta).rgb8();
        assert!(b > r && r > g, "dark main {r} {g} {b}");
        let [r, g, b] = light.accent(Accent::Magenta).rgb8();
        assert!(r > g && g > b, "light main {r} {g} {b}");
        // The main's fill carries its ink at 4.5:1 and more.
        for palette in Palette::ALL {
            let ratio = contrast(palette.on_main, palette.main_fill);
            assert!(ratio >= 4.5, "{}: {ratio:.2}", palette.name);
        }
        assert_eq!(dark.accent(Accent::Red), logo::CORAL);
        assert_eq!(dark.accent(Accent::CodePink), logo::PINK);
        assert_eq!(logo::GRADIENT[0], logo::BLUE);
        // A running bar goes from purple to orange in both.
        assert_eq!(dark.sunset(), [dark.accent(Accent::Magenta), logo::ORANGE]);
        assert_eq!(
            light.sunset(),
            [logo::OUTLINE, light.accent(Accent::Magenta)]
        );
    }

    #[test]
    fn modes_choose_palettes() {
        assert_eq!(Mode::Dark.toggled(), Mode::Light);
        assert_eq!(Mode::Light.toggled(), Mode::Dark);
        assert_eq!(
            Palette::of(ThemeName::Sunset, Mode::Light),
            Palette::SUNSET_LIGHT
        );
        let light = Palette::SUNSET_LIGHT;
        assert_eq!((light.mode, light.name), (Mode::Light, "sunset-light"));
        assert_eq!(Palette::default(), Palette::SUNSET_DARK);
    }

    #[test]
    fn painting_blends_in_srgb() {
        let bg = Palette::SUNSET_DARK.surface(Surface::Bg);
        assert_eq!(logo::BLUE.over(bg), logo::BLUE);
        assert_eq!(logo::BLUE.opacity(0.0).over(bg), bg);
    }

    #[test]
    fn roles_have_the_design_languages_names() {
        let names: Vec<_> = Ink::ALL.iter().map(|ink| ink.name()).collect();
        assert_eq!(
            names,
            [
                "text",
                "body",
                "muted",
                "cyan",
                "magenta",
                "violet",
                "mint",
                "amber",
                "red",
                "code-pink"
            ]
        );
        let surfaces: Vec<_> = Surface::ALL.iter().map(|s| s.name()).collect();
        assert_eq!(surfaces, ["void", "bg", "panel", "raise", "sunk"]);
        assert!(Accent::ALL.iter().all(|a| !a.meaning().is_empty()));
        assert_eq!(Ink::from(Accent::Mint), Ink::Accent(Accent::Mint));
    }
}
