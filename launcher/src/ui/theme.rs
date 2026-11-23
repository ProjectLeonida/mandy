// SPDX-License-Identifier: GPL-3.0-or-later

//! The theme: the tokens mapped to iced's styles and to each widget's [`Look`].
//!
//! A theme carries a [`Palette`], one named look in one mode, and every color any widget draws
//! comes from it: the surfaces, lines and inks, the accents' tints, the focus outline. Sunset in
//! dark is the default, and Sunset in light the other mode.
//!
//! It is its own type, not iced's `Theme`, so the tokens stay the only source of colors: a text's
//! class is an [`Ink`] and a container's a [`Frame`], and neither takes a color of the app's
//! choosing.

use crate::ui::tokens::{Accent, Ink, Palette, Surface, ThemeName, radius, size, window};
use iced::theme::{self, Base, Mode};
use iced::widget::{container, text};
use iced::{Background, Border, Shadow};

/// The theme: a palette, Sunset dark by default, and whether it saves battery.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Theme {
    palette: Palette,
    /// On battery saver, glow is half as soft ([`crate::ui::system`]).
    saving: bool,
}

/// A token's color in iced.
pub fn color(token: crate::ui::tokens::Color) -> iced::Color {
    let [r, g, b] = token.rgb8();
    iced::Color::from_rgba8(r, g, b, token.alpha())
}

impl Theme {
    /// A theme in a palette.
    pub const fn new(palette: Palette) -> Self {
        Self {
            palette,
            saving: false,
        }
    }

    /// The same theme, saving battery or not: on battery saver, glow blurs half as far.
    pub const fn saving_battery(self, saving: bool) -> Self {
        Self { saving, ..self }
    }

    /// Whether it saves battery.
    pub const fn saves_battery(&self) -> bool {
        self.saving
    }

    /// A named theme in a mode.
    pub const fn of(theme: ThemeName, mode: crate::ui::tokens::Mode) -> Self {
        Self::new(Palette::of(theme, mode))
    }

    /// What it's called, as screenshots are filed: "sunset-dark".
    pub const fn name(&self) -> &'static str {
        self.palette.name
    }

    pub const fn palette(&self) -> &Palette {
        &self.palette
    }

    /// An ink's color.
    pub const fn ink(&self, ink: Ink) -> crate::ui::tokens::Color {
        self.palette.ink(ink)
    }

    /// A surface's color.
    pub const fn surface(&self, surface: Surface) -> crate::ui::tokens::Color {
        self.palette.surface(surface)
    }

    /// On the way from this theme to `to`, `share` of the way, as one fades into the other
    /// ([`Palette::mixed`]).
    pub fn mixed(&self, to: &Theme, share: f32) -> Self {
        Self::new(self.palette.mixed(&to.palette, share)).saving_battery(self.saving || to.saving)
    }
}

impl Base for Theme {
    fn default(_preference: Mode) -> Self {
        <Self as Default>::default()
    }

    fn mode(&self) -> Mode {
        match self.palette.mode {
            crate::ui::tokens::Mode::Dark => Mode::Dark,
            crate::ui::tokens::Mode::Light => Mode::Light,
        }
    }

    fn base(&self) -> theme::Style {
        theme::Style {
            background_color: color(self.surface(Surface::Bg)),
            text_color: color(self.ink(Ink::Text)),
        }
    }

    fn palette(&self) -> Option<theme::Palette> {
        let p = &self.palette;
        Some(theme::Palette {
            background: color(p.surface(Surface::Bg)),
            text: color(p.ink(Ink::Text)),
            primary: color(p.accent(Accent::Cyan)),
            success: color(p.accent(Accent::Mint)),
            warning: color(p.accent(Accent::Amber)),
            danger: color(p.accent(Accent::Red)),
        })
    }

    fn name(&self) -> &str {
        self.palette.name
    }
}

impl text::Catalog for Theme {
    type Class<'a> = Ink;

    fn default<'a>() -> Self::Class<'a> {
        Ink::Text
    }

    fn style(&self, ink: &Ink) -> text::Style {
        text::Style {
            color: Some(color(self.ink(*ink))),
        }
    }
}

/// What a container draws behind its content.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Frame {
    /// Nothing.
    #[default]
    None,
    /// A surface's fill, edge to edge: a page.
    Fill(Surface),
    /// A surface's fill with a border, at a control's radius.
    Card(Surface),
    /// The selected item: magenta's soft fill and line.
    Selected,
    /// An accent's soft fill and line, at a chip's radius.
    Chip(Accent),
    /// A line between two parts: the line's color, edge to edge.
    Divider,
}

impl container::Catalog for Theme {
    type Class<'a> = Frame;

    fn default<'a>() -> Self::Class<'a> {
        Frame::None
    }

    fn style(&self, frame: &Frame) -> container::Style {
        let p = &self.palette;
        let border = |line: crate::ui::tokens::Color, radius: f32| Border {
            color: color(line),
            width: size::BORDER,
            radius: radius.into(),
        };
        let (background, border) = match *frame {
            Frame::None => (None, Border::default()),
            Frame::Fill(surface) => (Some(p.surface(surface)), Border::default()),
            Frame::Divider => (Some(p.line), Border::default()),
            Frame::Card(surface) => (Some(p.surface(surface)), border(p.line, radius::CONTROL)),
            Frame::Selected => (
                Some(p.soft(Accent::Magenta)),
                border(p.accent_line(Accent::Magenta), radius::CONTROL),
            ),
            Frame::Chip(accent) => (
                Some(p.soft(accent)),
                border(p.accent_line(accent), radius::CHIP),
            ),
        };
        container::Style {
            text_color: None,
            background: background.map(|token| Background::Color(color(token))),
            border,
            shadow: Shadow::default(),
            snap: true,
        }
    }
}

/// How a widget looks in one state: its fill, its line and corner radius, and its ink (text, icon,
/// dot or bar). Only the theme makes them, from its palette; a widget draws what it's given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    pub fill: Option<crate::ui::tokens::Color>,
    pub line: Option<crate::ui::tokens::Color>,
    pub radius: f32,
    pub ink: crate::ui::tokens::Color,
    /// The glow round it: on hover, the selected item, a primary button on hover. Never at rest.
    pub glow: Option<Halo>,
}

/// How much of `text` tints a window button at rest: always faintly there.
const WINDOW_BUTTON_TINT: f32 = 0.05;

/// A glow, drawn round a box and never inside it: the accent's color, how far down it sits (0
/// for a halo all round), and how far it blurs, from a palette's glows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Halo {
    pub color: crate::ui::tokens::Color,
    pub offset: f32,
    pub blur: f32,
}

impl Halo {
    fn of(glow: crate::ui::tokens::effect::Glow, palette: &Palette) -> Self {
        Self {
            color: glow.halo_color(palette),
            offset: glow.offset,
            blur: glow.blur,
        }
    }

    /// A small glow, for a window button: half as far and half as bright.
    pub fn small(self) -> Self {
        Self {
            color: self.color.opacity(self.color.alpha() / 2.0),
            offset: self.offset / 2.0,
            blur: self.blur / 2.0,
        }
    }
}

impl Look {
    fn boxed(
        fill: crate::ui::tokens::Color,
        line: crate::ui::tokens::Color,
        radius: f32,
        ink: crate::ui::tokens::Color,
    ) -> Self {
        Self {
            fill: Some(fill),
            line: Some(line),
            radius,
            ink,
            glow: None,
        }
    }

    /// Just a fill, in its ink's color: a bar, a caret, a scroller.
    pub fn solid(ink: crate::ui::tokens::Color, radius: f32) -> Self {
        Self {
            fill: Some(ink),
            line: None,
            radius,
            ink,
            glow: None,
        }
    }

    fn bare(radius: f32, ink: crate::ui::tokens::Color) -> Self {
        Self {
            fill: None,
            line: None,
            radius,
            ink,
            glow: None,
        }
    }

    /// The same, glowing.
    pub fn glowing(self, glow: Halo) -> Self {
        Self {
            glow: Some(glow),
            ..self
        }
    }

    /// The same with `glow` at a share of its strength, in place of its own: a glow fading in or
    /// out as the pointer comes and goes ([`crate::ui::transition::Lit`]).
    pub fn lit(self, glow: Option<Halo>, share: f32) -> Self {
        let share = share.clamp(0.0, 1.0);
        Self {
            glow: glow.filter(|_| share > 0.0).map(|glow| Halo {
                color: glow.color.opacity(glow.color.alpha() * share),
                ..glow
            }),
            ..self
        }
    }

    /// The same at a share of its opacity, fill, line and glow: two looks crossfade as one goes
    /// and the other comes.
    pub fn faded(self, share: f32) -> Self {
        let share = share.clamp(0.0, 1.0);
        let fade = |color: crate::ui::tokens::Color| color.opacity(color.alpha() * share);
        Self {
            fill: self.fill.map(fade),
            line: self.line.map(fade),
            glow: self.glow.map(|glow| Halo {
                color: fade(glow.color),
                ..glow
            }),
            ..self
        }
    }
}

/// What the pointer is doing to a widget, or that it's off. Keyboard focus is apart: it adds the
/// focus outline, whatever the interaction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Interaction {
    #[default]
    Idle,
    Hovered,
    Pressed,
    Disabled,
}

/// A button's weight.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Emphasis {
    /// The step forward: Check, Build, Play. One a screen.
    Primary,
    /// Everything else.
    #[default]
    Secondary,
    /// What can't be undone, or stops what's under way: Cancel.
    Danger,
}

/// A long step's state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Progress {
    #[default]
    Running,
    Done,
    Failed,
}

/// A check's or a step's state, as a status chip shows it: a dot and a word, never color alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Status {
    Passed,
    Warning,
    Failed,
    /// Under way.
    Running,
    /// For information, or planned.
    Info,
    /// Not started.
    #[default]
    Waiting,
}

/// The keyboard focus outline: 2px of cyan (Sunset's blue), 2px out from what it outlines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outline {
    pub color: crate::ui::tokens::Color,
    pub width: f32,
    pub offset: f32,
}

impl Theme {
    /// Focus's glow, cyan: keyboard focus, round the outline, and a control under the pointer.
    pub fn focus_glow(&self) -> Halo {
        self.saved(Halo::of(self.palette.glows.focus, &self.palette))
    }

    /// The selected item's glow, magenta, and a primary button's under the pointer.
    pub fn selected_glow(&self) -> Halo {
        self.saved(Halo::of(self.palette.glows.selected, &self.palette))
    }

    /// A glow in an accent, focus's shape: a status chip's, a progress bar's.
    pub fn accent_glow(&self, accent: Accent) -> Halo {
        let glow = crate::ui::tokens::effect::Glow {
            accent,
            ..self.palette.glows.focus
        };
        self.saved(Halo::of(glow, &self.palette))
    }

    /// A small light's glow, such as a status dot's: its own color, blurred (the palette's
    /// spark).
    pub fn spark(&self, ink: crate::ui::tokens::Color) -> Halo {
        let spark = self.palette.glows.spark;
        self.saved(Halo {
            color: ink.opacity(ink.alpha() * spark.opacity),
            offset: 0.0,
            blur: spark.blur,
        })
    }

    /// What can't be undone glows red under the pointer: the selected glow's shape, in red.
    pub fn danger_glow(&self) -> Halo {
        let glow = crate::ui::tokens::effect::Glow {
            accent: Accent::Red,
            ..self.palette.glows.selected
        };
        self.saved(Halo::of(glow, &self.palette))
    }

    /// A glow as this theme draws it: on battery saver, half as soft.
    fn saved(&self, glow: Halo) -> Halo {
        if self.saving {
            Halo {
                blur: glow.blur / 2.0,
                ..glow
            }
        } else {
            glow
        }
    }

    /// The focus outline, shown whenever a widget has keyboard focus.
    pub fn outline(&self) -> Outline {
        Outline {
            color: self.palette.accent(Accent::Cyan),
            width: size::FOCUS_OUTLINE,
            offset: size::FOCUS_OFFSET,
        }
    }

    /// Anything that's off: panel, a soft line, muted text.
    fn disabled(&self, radius: f32) -> Look {
        let p = &self.palette;
        Look::boxed(
            p.surface(Surface::Panel),
            p.line_soft,
            radius,
            p.ink(Ink::Muted),
        )
    }

    /// A control's label: text, or muted when it's off.
    pub fn label(&self, interaction: Interaction) -> crate::ui::tokens::Color {
        match interaction {
            Interaction::Disabled => self.ink(Ink::Muted),
            _ => self.ink(Ink::Text),
        }
    }

    pub fn button(&self, emphasis: Emphasis, interaction: Interaction) -> Look {
        let p = &self.palette;
        let text = p.ink(Ink::Text);
        let tinted = |accent: Accent| {
            let (fill, line) = match interaction {
                Interaction::Idle => (p.soft(accent), p.accent_line(accent)),
                Interaction::Hovered => (p.soft(accent), p.accent(accent)),
                _ => (p.accent_line(accent), p.accent(accent)),
            };
            let look = Look::boxed(fill, line, radius::BUTTON, text);
            // Under the pointer a primary button glows as the selected item does; a danger
            // button, red.
            match (accent, interaction) {
                (_, Interaction::Idle) => look,
                (Accent::Red, _) => look.glowing(self.danger_glow()),
                _ => look.glowing(self.selected_glow()),
            }
        };
        if interaction == Interaction::Disabled {
            return self.disabled(radius::BUTTON);
        }
        match emphasis {
            // The main action: the main color itself as its fill, its own ink on it; under the
            // pointer it glows in the secondary color, and pressed, the secondary lines it.
            Emphasis::Primary => {
                let look = Look::boxed(p.main_fill, p.main_fill, radius::BUTTON, p.on_main);
                match interaction {
                    Interaction::Idle => look,
                    Interaction::Hovered => look.glowing(self.selected_glow()),
                    _ => Look {
                        line: Some(p.accent(Accent::Cyan)),
                        ..look
                    }
                    .glowing(self.selected_glow()),
                }
            }
            Emphasis::Danger => tinted(Accent::Red),
            Emphasis::Secondary => {
                // As a field: sunk, its line round it, cyan's tint under the pointer, with its
                // glow; pressed, raised.
                match interaction {
                    Interaction::Idle => {
                        Look::boxed(p.surface(Surface::Sunk), p.line, radius::BUTTON, text)
                    }
                    Interaction::Hovered => Look::boxed(
                        p.surface(Surface::Sunk),
                        p.accent_line(Accent::Cyan),
                        radius::BUTTON,
                        text,
                    )
                    .glowing(self.focus_glow()),
                    _ => Look::boxed(
                        p.surface(Surface::Raise),
                        p.accent(Accent::Cyan),
                        radius::BUTTON,
                        text,
                    )
                    .glowing(self.focus_glow()),
                }
            }
        }
    }

    /// A text field. An invalid one's line is red.
    pub fn field(&self, interaction: Interaction, focused: bool, invalid: bool) -> Look {
        let p = &self.palette;
        let line = match (interaction, focused, invalid) {
            (Interaction::Disabled, _, _) => return self.disabled(radius::CONTROL),
            (_, true, true) => p.accent(Accent::Red),
            (_, false, true) => p.accent_line(Accent::Red),
            (_, true, false) => p.accent(Accent::Cyan),
            (Interaction::Hovered | Interaction::Pressed, false, false) => {
                p.accent_line(Accent::Cyan)
            }
            (Interaction::Idle, false, false) => p.line,
        };
        let look = Look::boxed(
            p.surface(Surface::Sunk),
            line,
            radius::CONTROL,
            p.ink(Ink::Text),
        );
        if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            look.glowing(self.focus_glow())
        } else {
            look
        }
    }

    /// A field's placeholder: `dim`, the one text-like use it has.
    pub fn placeholder(&self) -> crate::ui::tokens::Color {
        self.palette.dim
    }

    /// Selected text's highlight: magenta, which means selection.
    pub fn selection(&self) -> crate::ui::tokens::Color {
        self.palette.accent_line(Accent::Magenta)
    }

    /// The text caret.
    pub fn caret(&self) -> crate::ui::tokens::Color {
        self.palette.accent(Accent::Cyan)
    }

    /// A toggle group's pill ([`crate::ui::toggle`]): sunk, lined, its corners a button's.
    pub fn toggle_group(&self) -> Look {
        let p = &self.palette;
        Look::boxed(
            p.surface(Surface::Sunk),
            p.line,
            radius::BUTTON,
            p.ink(Ink::Muted),
        )
    }

    /// A segment of a toggle group: the chosen one raised, on cyan's line tint; the others muted,
    /// and `text` under the pointer.
    pub fn toggle_segment(&self, chosen: bool, interaction: Interaction) -> Look {
        // Only the chosen segment glows, in its line's cyan; the others don't, even hovered.
        let look = self.toggle_segment_look(chosen, interaction);
        if chosen && interaction != Interaction::Disabled {
            look.glowing(self.focus_glow())
        } else {
            look
        }
    }

    fn toggle_segment_look(&self, chosen: bool, interaction: Interaction) -> Look {
        let p = &self.palette;
        let radius = radius::SEGMENT;
        match (chosen, interaction) {
            (true, Interaction::Disabled) => Look {
                fill: Some(p.surface(Surface::Raise)),
                ..Look::bare(radius, p.ink(Ink::Muted))
            },
            (false, Interaction::Disabled) => Look::bare(radius, p.dim),
            (true, _) => Look::boxed(
                p.surface(Surface::Raise),
                p.accent_line(Accent::Cyan),
                radius,
                p.ink(Ink::Text),
            ),
            (false, Interaction::Hovered | Interaction::Pressed) => {
                Look::bare(radius, p.ink(Ink::Text))
            }
            (false, Interaction::Idle) => Look::bare(radius, p.ink(Ink::Muted)),
        }
    }

    /// A scroll indicator: a scroll area's. Decoration, so dim is allowed.
    pub fn scroller(&self, interaction: Interaction) -> crate::ui::tokens::Color {
        match interaction {
            Interaction::Hovered | Interaction::Pressed => self.palette.dim,
            _ => self.palette.line,
        }
    }

    /// A progress bar: its track, and the done part in the ink.
    pub fn progress(&self, progress: Progress) -> Look {
        let p = &self.palette;
        Look::boxed(
            p.line_soft,
            p.line_soft,
            radius::ROUND,
            p.accent(progress_accent(progress)),
        )
    }

    /// A progress bar's done part: its accent, glowing.
    pub fn progress_bar(&self, progress: Progress) -> Look {
        let accent = progress_accent(progress);
        Look::solid(self.palette.accent(accent), radius::ROUND).glowing(self.accent_glow(accent))
    }

    /// A running bar's done part in Sunset: the logo's gradient, in this palette's accents, from
    /// the bar's start to its end ([`crate::ui::tokens::Palette::sunset`]).
    pub fn progress_gradient(&self) -> [crate::ui::tokens::Color; 2] {
        self.palette.sunset()
    }

    /// A loader's ring ([`crate::ui::loader`]): a cyan line, glowing, nothing inside it.
    pub fn loader(&self) -> Look {
        let cyan = self.palette.accent(Accent::Cyan);
        Look {
            line: Some(cyan),
            ..Look::bare(radius::ROUND, cyan)
        }
        .glowing(self.accent_glow(Accent::Cyan))
    }

    /// A status chip; the dot and the word take its ink.
    pub fn status(&self, status: Status) -> Look {
        let p = &self.palette;
        // Each lit, glowing in its accent; waiting isn't.
        let tinted = |accent: Accent| {
            Look::boxed(
                p.soft(accent),
                p.accent_line(accent),
                radius::ROUND,
                p.accent(accent),
            )
            .glowing(self.accent_glow(accent))
        };
        match status {
            Status::Passed => tinted(Accent::Mint),
            Status::Warning => tinted(Accent::Amber),
            Status::Failed => tinted(Accent::Red),
            Status::Running => tinted(Accent::Cyan),
            Status::Info => tinted(Accent::Violet),
            // Waiting isn't lit. In dark, though, its line and its small light are the secondary
            // color, the logo's orange: muted purple on the dark purple page hardly shows.
            Status::Waiting => match p.mode {
                crate::ui::tokens::Mode::Dark => Look {
                    line: Some(p.accent_line(Accent::Cyan)),
                    ..Look::bare(radius::ROUND, p.accent(Accent::Cyan))
                },
                crate::ui::tokens::Mode::Light => Look {
                    line: Some(p.line),
                    ..Look::bare(radius::ROUND, p.ink(Ink::Muted))
                },
            },
        }
    }

    /// A scroll area's auto-scroll badge: panel, line and muted.
    pub fn badge(&self) -> Look {
        let p = &self.palette;
        Look::boxed(
            p.surface(Surface::Panel),
            p.line,
            radius::ROUND,
            p.ink(Ink::Muted),
        )
    }

    /// A checkbox's box, glowing under the pointer; its ink is the check mark's.
    pub fn check(&self, checked: bool, interaction: Interaction) -> Look {
        let look = self.check_box(checked, interaction);
        self.glow_hovered(look, interaction)
    }

    /// A look with focus's glow under the pointer, as every control has.
    fn glow_hovered(&self, look: Look, interaction: Interaction) -> Look {
        if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            look.glowing(self.focus_glow())
        } else {
            look
        }
    }

    fn check_box(&self, checked: bool, interaction: Interaction) -> Look {
        let p = &self.palette;
        let cyan = p.accent(Accent::Cyan);
        // Sunk as every control; its line `dim`, though, not `line`: a box this small is known by
        // its edge alone, which needs 3:1.
        let bg = p.surface(Surface::Sunk);
        match (checked, interaction) {
            (_, Interaction::Disabled) => Look::boxed(
                p.surface(Surface::Panel),
                p.line,
                radius::CHECK,
                p.ink(Ink::Muted),
            ),
            (true, _) => Look::boxed(p.soft(Accent::Cyan), cyan, radius::CHECK, cyan),
            (false, Interaction::Hovered | Interaction::Pressed) => {
                Look::boxed(bg, p.accent_line(Accent::Cyan), radius::CHECK, cyan)
            }
            (false, Interaction::Idle) => Look::boxed(bg, p.dim, radius::CHECK, cyan),
        }
    }

    /// One option of a single choice: a card, magenta when it's the chosen one.
    pub fn option(&self, chosen: bool, interaction: Interaction) -> Look {
        let p = &self.palette;
        let text = p.ink(Ink::Text);
        match (chosen, interaction) {
            (_, Interaction::Disabled) => self.disabled(radius::CONTROL),
            (true, _) => Look::boxed(
                p.soft(Accent::Magenta),
                p.accent_line(Accent::Magenta),
                radius::CONTROL,
                text,
            )
            .glowing(self.selected_glow()),
            (false, Interaction::Hovered | Interaction::Pressed) => Look::boxed(
                p.surface(Surface::Sunk),
                p.accent_line(Accent::Cyan),
                radius::CONTROL,
                text,
            )
            .glowing(self.focus_glow()),
            (false, Interaction::Idle) => {
                Look::boxed(p.surface(Surface::Sunk), p.line, radius::CONTROL, text)
            }
        }
    }

    /// An option's radio circle, glowing under the pointer; its ink is the chosen one's dot.
    pub fn radio(&self, chosen: bool, interaction: Interaction) -> Look {
        let look = self.radio_circle(chosen, interaction);
        self.glow_hovered(look, interaction)
    }

    fn radio_circle(&self, chosen: bool, interaction: Interaction) -> Look {
        let p = &self.palette;
        let cyan = p.accent(Accent::Cyan);
        let (line, ink) = match (chosen, interaction) {
            (_, Interaction::Disabled) => (p.line, p.ink(Ink::Muted)),
            (true, _) => (cyan, cyan),
            (false, Interaction::Hovered | Interaction::Pressed) => {
                (p.accent_line(Accent::Cyan), cyan)
            }
            (false, Interaction::Idle) => (p.dim, cyan),
        };
        Look::boxed(p.surface(Surface::Sunk), line, radius::ROUND, ink)
    }

    /// A selection list's frame: it glows under the pointer, as every control does; its rows
    /// don't.
    pub fn list(&self, interaction: Interaction) -> Look {
        let p = &self.palette;
        if interaction == Interaction::Disabled {
            self.disabled(radius::CONTROL)
        } else {
            let look = Look::boxed(
                p.surface(Surface::Sunk),
                p.line,
                radius::CONTROL,
                p.ink(Ink::Text),
            );
            self.glow_hovered(look, interaction)
        }
    }

    /// A list's row: the chosen one magenta, a hovered one raised.
    pub fn row(&self, chosen: bool, interaction: Interaction) -> Look {
        let p = &self.palette;
        let text = p.ink(Ink::Text);
        match (chosen, interaction) {
            (_, Interaction::Disabled) => Look::bare(radius::CHIP, p.ink(Ink::Muted)),
            (true, _) => Look::boxed(
                p.soft(Accent::Magenta),
                p.accent_line(Accent::Magenta),
                radius::CHIP,
                text,
            ),
            (false, Interaction::Hovered | Interaction::Pressed) => Look {
                fill: Some(p.surface(Surface::Raise)),
                ..Look::bare(radius::CHIP, text)
            },
            (false, Interaction::Idle) => Look::bare(radius::CHIP, text),
        }
    }

    /// A switch ([`mod@crate::ui::switch`]): off, sunk and lined with a muted knob; on,
    /// cyan's soft fill and line with a cyan knob, light from interaction. Under the pointer its
    /// line brightens.
    pub fn switch(&self, on: bool, interaction: Interaction) -> SwitchLook {
        // On, it's lit: it glows. Off, it doesn't, even under the pointer.
        let look = self.switch_parts(on, interaction);
        let track = if on && interaction != Interaction::Disabled {
            look.track.glowing(self.focus_glow())
        } else {
            look.track
        };
        SwitchLook { track, ..look }
    }

    fn switch_parts(&self, on: bool, interaction: Interaction) -> SwitchLook {
        let p = &self.palette;
        let radius = crate::ui::tokens::size::SWITCH_HEIGHT / 2.0;
        let hovered = matches!(interaction, Interaction::Hovered | Interaction::Pressed);
        match (on, interaction) {
            (_, Interaction::Disabled) => SwitchLook {
                track: Look::boxed(p.surface(Surface::Sunk), p.line, radius, p.dim),
                knob: if on { p.ink(Ink::Muted) } else { p.dim },
            },
            (true, _) => SwitchLook {
                track: Look::boxed(
                    p.soft(Accent::Cyan),
                    if hovered {
                        p.accent(Accent::Cyan)
                    } else {
                        p.accent_line(Accent::Cyan)
                    },
                    radius,
                    p.ink(Ink::Text),
                ),
                knob: p.accent(Accent::Cyan),
            },
            (false, _) => SwitchLook {
                track: Look::boxed(
                    p.surface(Surface::Sunk),
                    if hovered { p.dim } else { p.line },
                    radius,
                    p.ink(Ink::Text),
                ),
                knob: if hovered {
                    p.ink(Ink::Text)
                } else {
                    p.ink(Ink::Muted)
                },
            },
        }
    }

    /// A row of a sidebar ([`crate::ui::list::navigation`]): muted at rest, raised under
    /// the pointer; the chosen one in magenta's soft fill, which [`Theme::sidebar_mark`] edges.
    pub fn sidebar_row(&self, chosen: bool, interaction: Interaction) -> Look {
        let p = &self.palette;
        match (chosen, interaction) {
            (_, Interaction::Disabled) => Look::bare(radius::CHIP, p.ink(Ink::Muted)),
            (true, _) => Look {
                fill: Some(p.soft(Accent::Magenta)),
                ..Look::bare(radius::CHIP, p.ink(Ink::Text))
            },
            (false, Interaction::Hovered | Interaction::Pressed) => Look {
                fill: Some(p.surface(Surface::Raise)),
                ..Look::bare(radius::CHIP, p.ink(Ink::Text))
            },
            (false, Interaction::Idle) => Look::bare(radius::CHIP, p.ink(Ink::Muted)),
        }
    }

    /// The mark at the chosen sidebar row's start: a short bar of magenta, the selection's accent.
    pub fn sidebar_mark(&self) -> crate::ui::tokens::Color {
        // The logo's orange in both modes, against the row's purple-tinted fill: the main color
        // in light, the secondary in dark.
        match self.palette.mode {
            crate::ui::tokens::Mode::Light => self.palette.accent(Accent::Magenta),
            crate::ui::tokens::Mode::Dark => self.palette.accent(Accent::Cyan),
        }
    }
}

/// A switch's look: its track, and its knob's color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwitchLook {
    pub track: Look,
    pub knob: crate::ui::tokens::Color,
}

/// A header bar's look: its fill, the tints fading in from its edges, and its bottom line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarLook {
    pub fill: crate::ui::tokens::Color,
    /// The tint at the left edge, and at the right, each fading to nothing across `reach` of the
    /// bar's width.
    pub start: crate::ui::tokens::Color,
    pub end: crate::ui::tokens::Color,
    pub reach: f32,
    pub line: crate::ui::tokens::Color,
}

/// A window frame's look: the fill behind everything, and while the window floats, its line and
/// corner radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowLook {
    pub fill: crate::ui::tokens::Color,
    pub line: crate::ui::tokens::Color,
    pub radius: f32,
}

/// A window's own decorations (`crate::ui::chrome`). Their sizes are
/// `crate::ui::tokens::window`'s.
impl Theme {
    /// The header bar: panel, the palette's edge tints and a line below.
    pub fn header_bar(&self) -> BarLook {
        let p = &self.palette;
        BarLook {
            fill: p.surface(Surface::Panel),
            start: p.title_bar.start.color(p),
            end: p.title_bar.end.color(p),
            reach: p.title_bar.reach,
            line: p.line,
        }
    }

    /// The close button: its cross `muted`, and red on red's soft tint under the pointer.
    pub fn close_button(&self, interaction: Interaction) -> Look {
        let p = &self.palette;
        match interaction {
            Interaction::Hovered | Interaction::Pressed => Look {
                fill: Some(p.soft(Accent::Red)),
                ..Look::bare(window::BUTTON_RADIUS, p.accent(Accent::Red))
            }
            .glowing(self.danger_glow().small()),
            Interaction::Idle | Interaction::Disabled => self.window_button_at_rest(),
        }
    }

    /// A window button at rest: its glyph `muted` on a faint tint of `text`, always there.
    fn window_button_at_rest(&self) -> Look {
        let p = &self.palette;
        Look {
            fill: Some(p.ink(Ink::Text).opacity(WINDOW_BUTTON_TINT)),
            ..Look::bare(window::BUTTON_RADIUS, p.ink(Ink::Muted))
        }
    }

    /// Any other window button: its glyph `muted`, as the close button's, and `text` under the
    /// pointer on a tint of `text`.
    pub fn window_button(&self, interaction: Interaction) -> Look {
        let p = &self.palette;
        let tint = |opacity| {
            Look {
                fill: Some(p.ink(Ink::Text).opacity(opacity)),
                ..Look::bare(window::BUTTON_RADIUS, p.ink(Ink::Text))
            }
            .glowing(self.focus_glow().small())
        };
        match interaction {
            Interaction::Hovered => tint(0.14),
            Interaction::Pressed => tint(0.2),
            Interaction::Idle | Interaction::Disabled => self.window_button_at_rest(),
        }
    }

    /// A window's frame: bg behind everything, and while it floats, a line round its rounded
    /// edge: the logo's deep purple outline in dark.
    pub fn window(&self) -> WindowLook {
        let p = &self.palette;
        WindowLook {
            fill: p.surface(Surface::Bg),
            line: p.line,
            radius: window::CORNER_RADIUS,
        }
    }
}

/// A progress bar's accent: cyan running, mint done, red failed.
fn progress_accent(progress: Progress) -> Accent {
    match progress {
        Progress::Running => Accent::Cyan,
        Progress::Done => Accent::Mint,
        Progress::Failed => Accent::Red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::widget::container::Catalog as _;

    const DARK: Theme = Theme::new(Palette::SUNSET_DARK);

    fn frame(frame: Frame) -> container::Style {
        DARK.style(&frame)
    }

    #[test]
    fn tokens_map_to_iced_colors() {
        let p = Palette::SUNSET_DARK;
        assert_eq!(
            color(p.accent(Accent::Cyan)),
            iced::Color::from_rgb8(0xfd, 0x77, 0x44)
        );
        let soft = color(p.soft(Accent::Cyan));
        assert_eq!(soft.a, 0.14);
        assert_eq!(
            iced::Color { a: 1.0, ..soft },
            color(p.accent(Accent::Cyan))
        );
    }

    #[test]
    fn the_base_is_text_on_bg() {
        let theme = <Theme as Base>::default(Mode::Light);
        assert_eq!(theme, DARK);
        assert_eq!(theme.palette(), &Palette::SUNSET_DARK);
        assert_eq!(theme.mode(), Mode::Dark);
        assert_eq!(Base::name(&theme), "sunset-dark");
        let light = Theme::of(ThemeName::Sunset, crate::ui::tokens::Mode::Light);
        assert_eq!((light.mode(), light.name()), (Mode::Light, "sunset-light"));
        let base = theme.base();
        assert_eq!(base.background_color, color(theme.surface(Surface::Bg)));
        assert_eq!(base.text_color, color(theme.ink(Ink::Text)));
    }

    #[test]
    fn text_takes_its_ink() {
        let default = <Theme as text::Catalog>::default();
        assert_eq!(default, Ink::Text);
        for ink in Ink::ALL {
            let style = text::Catalog::style(&DARK, &ink);
            assert_eq!(style.color, Some(color(DARK.ink(ink))));
        }
    }

    #[test]
    fn frames_draw_the_palette() {
        let p = Palette::SUNSET_DARK;
        let none = frame(<Theme as container::Catalog>::default());
        assert_eq!(none.background, None);
        let card = frame(Frame::Card(Surface::Raise));
        assert_eq!(card.border.color, color(p.line));
        assert_eq!(card.border.width, 1.0);
        let chip = frame(Frame::Chip(Accent::Mint));
        assert_eq!(chip.border.radius, radius::CHIP.into());
    }

    #[test]
    fn no_look_draws_text_in_dim_in_any_palette() {
        for palette in Palette::ALL {
            let theme = Theme::new(palette);
            for interaction in [
                Interaction::Idle,
                Interaction::Hovered,
                Interaction::Pressed,
                Interaction::Disabled,
            ] {
                let mut looks = Vec::new();
                for emphasis in [Emphasis::Primary, Emphasis::Secondary, Emphasis::Danger] {
                    looks.push(theme.button(emphasis, interaction));
                }
                for (focused, invalid) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    looks.push(theme.field(interaction, focused, invalid));
                }
                looks.push(theme.toggle_segment(true, interaction));
                for look in looks {
                    assert_ne!(look.ink, palette.dim, "{}: {look:?}", palette.name);
                }
                assert_ne!(theme.label(interaction), palette.dim);
            }
        }
    }

    #[test]
    fn every_look_comes_from_the_palette() {
        let p = Palette::SUNSET_DARK;
        // The main action is filled with the main color; focus and the glows take the secondary.
        let primary = DARK.button(Emphasis::Primary, Interaction::Idle);
        assert_eq!((primary.fill, primary.ink), (Some(p.main_fill), p.on_main));
        assert_eq!(DARK.outline().color, p.accent(Accent::Cyan));
        assert_eq!(DARK.progress(Progress::Running).ink, p.accent(Accent::Cyan));
        assert_eq!(DARK.progress_gradient()[1], p.accent(Accent::Cyan));
        let disabled = DARK.button(Emphasis::Primary, Interaction::Disabled);
        assert_eq!(disabled.ink, p.ink(Ink::Muted));
        assert_eq!(
            DARK.field(Interaction::Idle, true, true).line,
            Some(p.accent(Accent::Red))
        );
        assert!(DARK.progress_bar(Progress::Running).glow.is_some());
        assert!(DARK.status(Status::Passed).glow.is_some());
        assert!(DARK.status(Status::Waiting).glow.is_none());
        assert!(DARK.toggle_segment(true, Interaction::Idle).glow.is_some());
        assert!(
            DARK.toggle_segment(false, Interaction::Hovered)
                .glow
                .is_none()
        );
        assert_eq!(DARK.scroller(Interaction::Hovered), p.dim);
        assert_eq!(DARK.progress(Progress::Failed).ink, p.accent(Accent::Red));
        assert_eq!(DARK.status(Status::Info).fill, Some(p.soft(Accent::Violet)));
        assert_eq!(DARK.placeholder(), p.dim);
        assert_eq!(DARK.caret(), p.accent(Accent::Cyan));
    }

    #[test]
    fn window_decorations_come_from_the_palette() {
        for palette in Palette::ALL {
            let theme = Theme::new(palette);
            let bar = theme.header_bar();
            assert_eq!(bar.fill, palette.surface(Surface::Panel));
            assert_eq!(bar.line, palette.line);
            let idle = theme.close_button(Interaction::Idle);
            assert_eq!(idle.ink, palette.ink(Ink::Muted));
            for interaction in [Interaction::Hovered, Interaction::Pressed] {
                let hovered = theme.close_button(interaction);
                assert_eq!(hovered.fill, Some(palette.soft(Accent::Red)));
                assert_eq!(hovered.ink, palette.accent(Accent::Red));
            }
            let window = theme.window();
            assert_eq!(window.fill, palette.surface(Surface::Bg));
            assert_eq!((window.line, window.radius), (palette.line, 14.0));
        }
    }
}
