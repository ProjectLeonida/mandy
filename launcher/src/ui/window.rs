// SPDX-License-Identifier: GPL-3.0-or-later

//! The launcher's window: its own decorations ([`crate::ui::chrome`]), as the window runs
//! them: the window's state, what its header bar, sidebar, frame and events say, and what's done
//! about it.

use std::ffi::OsStr;

use iced::time::Instant;
use iced::{Color, Event, Size, Subscription, Task, window};

use crate::ui::chrome::{Corners, Edge, Header, header_bar, window_frame};
use crate::ui::theme::Frame;
use crate::ui::tokens::{Surface, space, window as decorations};
use crate::ui::widget::{Row, Space, container};
use crate::ui::{Element, Fill, Theme, scroll_area};
use iced::theme::{self, Base};

/// The environment variable that says who draws the window's frame.
pub const FRAME_VARIABLE: &str = "LEONIDA_WINDOW_FRAME";

/// Who draws the window's frame, as [`FRAME_VARIABLE`] says.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FrameSetting {
    /// The launcher draws it: rounded corners and a line while the window floats.
    #[default]
    Drawn,
    /// Nobody: an opaque, square window, for a desktop that draws its own corners and border, or
    /// a driver that misbehaves with a transparent window.
    Off,
}

impl FrameSetting {
    /// The setting a value of [`FRAME_VARIABLE`] gives: unset or `1` draws the frame, `0` doesn't,
    /// and anything else is as unset, with the line to say once on stderr.
    pub fn parse(value: Option<&OsStr>) -> (Self, Option<String>) {
        match value.map(|value| value.to_str()) {
            None => (Self::Drawn, None),
            Some(Some("1")) => (Self::Drawn, None),
            Some(Some("0")) => (Self::Off, None),
            Some(other) => (
                Self::Drawn,
                Some(format!(
                    "{FRAME_VARIABLE}={} isn't 0 or 1: the launcher draws its frame",
                    other.unwrap_or("(not text)")
                )),
            ),
        }
    }

    /// The setting from the environment, saying on stderr what it doesn't understand.
    pub fn from_env() -> Self {
        let (setting, warning) = Self::parse(std::env::var_os(FRAME_VARIABLE).as_deref());
        if let Some(warning) = warning {
            eprintln!("leonida-launcher: {warning}");
        }
        setting
    }
}

/// The system the window opens on, as far as its frame goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Windows: no transparent window with stock iced, but Windows 11 rounds a window that asks.
    Windows,
    /// Linux (Wayland and X11) and the rest: a transparent window the launcher rounds itself.
    Other,
}

impl Platform {
    /// The one it's built for.
    pub const fn current() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else {
            Self::Other
        }
    }
}

/// Whether the window is transparent: when the launcher draws its frame's corners itself.
pub fn transparent(platform: Platform, frame: FrameSetting) -> bool {
    frame == FrameSetting::Drawn && platform == Platform::Other
}

/// The window's corners: drawn by the launcher on a transparent window, the system's on Windows,
/// and square while maximized, full-screen, or with the frame off.
pub fn corners(platform: Platform, frame: FrameSetting, floating: bool) -> Corners {
    match (frame, floating, platform) {
        (FrameSetting::Off, ..) | (_, false, _) => Corners::Square,
        (FrameSetting::Drawn, true, Platform::Other) => Corners::Drawn,
        (FrameSetting::Drawn, true, Platform::Windows) => Corners::System,
    }
}

/// The window's settings when it opens: without the system's decorations, as the launcher draws
/// its own; transparent where it rounds its corners itself; on Windows, asking for Windows 11's
/// rounded corners unless the frame is off.
pub fn settings(size: Size, min: Size, frame: FrameSetting) -> window::Settings {
    let platform = Platform::current();
    window::Settings {
        size,
        min_size: Some(min),
        decorations: false,
        transparent: transparent(platform, frame),
        platform_specific: platform_specific(frame),
        ..window::Settings::default()
    }
}

#[cfg(windows)]
fn platform_specific(frame: FrameSetting) -> window::settings::PlatformSpecific {
    use window::settings::platform::CornerPreference;
    window::settings::PlatformSpecific {
        corner_preference: match frame {
            FrameSetting::Drawn => CornerPreference::Round,
            FrameSetting::Off => CornerPreference::DoNotRound,
        },
        undecorated_shadow: true,
        ..window::settings::PlatformSpecific::default()
    }
}

#[cfg(not(windows))]
fn platform_specific(_frame: FrameSetting) -> window::settings::PlatformSpecific {
    window::settings::PlatformSpecific::default()
}

/// A window's state, as its decorations need it.
#[derive(Debug, Clone, PartialEq)]
pub struct Chrome {
    frame: FrameSetting,
    platform: Platform,
    id: Option<window::Id>,
    maximized: bool,
    fullscreen: bool,
    focused: bool,
    /// Its size, in logical pixels.
    size: Size,
    /// What a folded window shows: its sidebar or its page.
    pane: Pane,
    /// The theme it's fading from, after the app's changed.
    fade: Option<Fade>,
}

/// A fade from one theme into the app's: from what showed when it changed, over
/// [`THEME_FADE`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct Fade {
    from: Theme,
    start: Instant,
    /// The frame it's at.
    at: Instant,
}

/// How long one theme takes to fade into another.
pub const THEME_FADE: std::time::Duration = crate::ui::tokens::motion::SLOW;

impl Fade {
    /// How far into the new theme it is, eased.
    fn share(&self) -> f32 {
        let fade = crate::ui::system::duration(THEME_FADE);
        let t = self.at.saturating_duration_since(self.start).as_secs_f32() / fade.as_secs_f32();
        crate::ui::tokens::motion::EASE.at(t.clamp(0.0, 1.0))
    }
}

/// What a folded window shows ([`Chrome::folded`]). It starts on the sidebar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pane {
    #[default]
    Sidebar,
    Page,
}

/// What a window's decorations and its events say.
#[derive(Debug, Clone, PartialEq)]
pub enum ChromeMessage {
    /// The header bar's empty area was dragged: the window moves with the pointer.
    Drag,
    /// The header bar's empty area was double-clicked: the window maximizes, or restores.
    ToggleMaximize,
    /// The close button was pressed.
    Close,
    /// A resize zone was pressed.
    Resize(Edge),
    /// The window opened, or was found.
    Opened(window::Id),
    /// The window gained focus, or lost it.
    Focused(window::Id, bool),
    /// The window opened at this size, or changed to it.
    Resized(window::Id, Size),
    /// A folded window's back button was pressed: from the page back to the sidebar.
    Back,
    /// Whether the window is maximized now.
    Maximized(bool),
    /// The window's mode now: full-screen or not.
    Mode(window::Mode),
    /// A frame, while a theme fades in.
    Frame(Instant),
}

impl Chrome {
    /// A window's, before it opens: focused and floating.
    pub fn new(size: Size, frame: FrameSetting) -> Self {
        Self::on(Platform::current(), size, frame)
    }

    /// The same, on a platform: for the tests.
    pub fn on(platform: Platform, size: Size, frame: FrameSetting) -> Self {
        Self {
            frame,
            platform,
            id: None,
            maximized: false,
            fullscreen: false,
            focused: true,
            size,
            pane: Pane::default(),
            fade: None,
        }
    }

    /// While the theme fades: the one it fades from and how far into the new one it is.
    pub fn theme_fade(&self) -> Option<(Theme, f32)> {
        self.fade.map(|fade| (fade.from, fade.share()))
    }

    /// The theme to draw in: the app's, `theme`, or while it fades in, on the way to it.
    pub fn theme(&self, theme: Theme) -> Theme {
        match &self.fade {
            Some(fade) => fade.from.mixed(&theme, fade.share()),
            None => theme,
        }
    }

    /// The app's theme changed from `before`, `now`: it fades in from what shows, unless motion
    /// is reduced (`LEONIDA_REDUCE_MOTION`).
    pub fn theme_changed(&mut self, before: Theme, now: Instant) {
        self.fade = (!crate::ui::system::reduced_motion()).then(|| Fade {
            from: self.theme(before),
            start: now,
            at: now,
        });
    }

    /// Whether a theme is fading in: it wants every frame till it's done.
    pub fn is_fading(&self) -> bool {
        self.fade.is_some()
    }

    /// Its size, in logical pixels.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Whether a window with a sidebar folds: narrower than [`decorations::FOLD_BELOW`].
    pub fn folded(&self) -> bool {
        self.size.width < decorations::FOLD_BELOW
    }

    /// What a folded window shows.
    pub fn pane(&self) -> Pane {
        self.pane
    }

    /// Its size changed to this, as its events say; or, drawing it without a window, as in a
    /// screenshot test, the size it's drawn at.
    pub fn resize(&mut self, size: Size) {
        self.size = size;
    }

    /// A place was chosen: a folded window shows its page.
    pub fn show_page(&mut self) {
        self.pane = Pane::Page;
    }

    /// Whether the window is transparent, for the launcher to draw its rounded corners.
    pub fn transparent(&self) -> bool {
        transparent(self.platform, self.frame)
    }

    /// Whether it floats: not maximized or full-screen.
    pub fn floating(&self) -> bool {
        !(self.maximized || self.fullscreen)
    }

    /// Its corners now.
    pub fn corners(&self) -> Corners {
        corners(self.platform, self.frame, self.floating())
    }

    /// Whether its edges resize it: while it floats.
    pub fn resizable(&self) -> bool {
        self.floating()
    }

    pub fn is_maximized(&self) -> bool {
        self.maximized
    }

    pub fn is_focused(&self) -> bool {
        self.focused
    }

    /// What the window learns at start: its ID, once it has opened.
    pub fn boot() -> Task<ChromeMessage> {
        window::oldest().and_then(|id| Task::done(ChromeMessage::Opened(id)))
    }

    /// The window events the decorations follow.
    pub fn subscription() -> Subscription<ChromeMessage> {
        iced::event::listen_with(|event, _status, id| match event {
            Event::Window(window::Event::Opened { size, .. }) => {
                Some(ChromeMessage::Resized(id, size))
            }
            Event::Window(window::Event::Focused) => Some(ChromeMessage::Focused(id, true)),
            Event::Window(window::Event::Unfocused) => Some(ChromeMessage::Focused(id, false)),
            Event::Window(window::Event::Resized(size)) => Some(ChromeMessage::Resized(id, size)),
            _ => None,
        })
    }

    /// Acts on a message: moves, maximizes, resizes or closes the window, or notes its state.
    pub fn update(&mut self, message: ChromeMessage) -> Task<ChromeMessage> {
        match message {
            ChromeMessage::Drag => self.act(window::drag),
            ChromeMessage::ToggleMaximize => self.act(window::toggle_maximize),
            ChromeMessage::Close => self.act(window::close),
            ChromeMessage::Resize(edge) if self.resizable() => {
                self.act(move |id| window::drag_resize(id, edge.direction()))
            }
            ChromeMessage::Resize(_) => Task::none(),
            ChromeMessage::Back => {
                self.pane = Pane::Sidebar;
                Task::none()
            }
            ChromeMessage::Resized(id, size) => {
                self.resize(size);
                self.update(ChromeMessage::Opened(id))
            }
            ChromeMessage::Opened(id) => {
                self.id = Some(id);
                // Whether it's maximized or full-screen now: winit says nothing of it otherwise.
                Task::batch([
                    window::is_maximized(id).map(ChromeMessage::Maximized),
                    window::mode(id).map(ChromeMessage::Mode),
                ])
            }
            ChromeMessage::Focused(id, focused) => {
                self.id = Some(id);
                self.focused = focused;
                Task::none()
            }
            ChromeMessage::Maximized(maximized) => {
                self.maximized = maximized;
                Task::none()
            }
            ChromeMessage::Frame(at) => {
                if let Some(fade) = &mut self.fade {
                    fade.at = at;
                    if fade.share() >= 1.0 {
                        self.fade = None;
                    }
                }
                Task::none()
            }
            ChromeMessage::Mode(mode) => {
                self.fullscreen = mode == window::Mode::Fullscreen;
                Task::none()
            }
        }
    }

    /// A task on the window: on its ID, or the first window's until it's known.
    fn act(
        &self,
        task: impl Fn(window::Id) -> Task<ChromeMessage> + Send + 'static,
    ) -> Task<ChromeMessage> {
        match self.id {
            Some(id) => task(id),
            None => window::oldest().and_then(task),
        }
    }

    /// The window: its frame round the header bar and the content, and with a sidebar, the
    /// sidebar beside the content, or, folded, one of them, the header bar's back button
    /// returning from the page.
    pub fn view<'a, M: Clone + 'a>(
        &self,
        header: Header<'a, M>,
        sidebar: Option<Element<'a, M>>,
        content: Element<'a, M>,
        wrap: fn(ChromeMessage) -> M,
    ) -> Element<'a, M> {
        let folded = sidebar.is_some() && self.folded();
        let back = folded && self.pane == Pane::Page;
        let corners = self.corners();
        let mut bar = header_bar(header)
            .rounded(corners == Corners::Drawn)
            .on_drag(wrap(ChromeMessage::Drag))
            .on_maximize(wrap(ChromeMessage::ToggleMaximize))
            .on_close(wrap(ChromeMessage::Close));
        if back {
            bar = bar.on_back(wrap(ChromeMessage::Back));
        }
        let bar = Element::from(bar);
        let pane = |sidebar: Element<'a, M>, width: f32| -> Element<'a, M> {
            let width = if folded { Fill } else { width.into() };
            container(scroll_area(
                container(sidebar)
                    .padding([space::M, decorations::SIDEBAR_PAD])
                    .width(Fill),
            ))
            .width(width)
            .height(Fill)
            .class(Frame::Fill(Surface::Panel))
            .into()
        };
        let body = match sidebar {
            None => content,
            // Folded, one pane at a time, sliding between them; the page passes over the sidebar,
            // so it fills itself.
            Some(sidebar) if folded => {
                let page: Element<'a, M> = container(content)
                    .width(Fill)
                    .height(Fill)
                    .class(Frame::Fill(Surface::Bg))
                    .into();
                crate::ui::chrome::folded(pane(sidebar, 0.0), page, back).into()
            }
            Some(sidebar) => Row::with_children([
                pane(sidebar, decorations::SIDEBAR_WIDTH),
                container(Space::new())
                    .width(decorations::BORDER)
                    .height(Fill)
                    .class(Frame::Divider)
                    .into(),
                content,
            ])
            .height(Fill)
            .into(),
        };
        let window: Element<'a, M> = iced::widget::Column::with_children([bar, body])
            .height(Fill)
            .into();
        let frame = window_frame(window).corners(corners);
        if self.resizable() {
            frame
                .on_resize(move |edge| wrap(ChromeMessage::Resize(edge)))
                .into()
        } else {
            frame.into()
        }
    }

    /// The window's background: clear while the launcher draws its frame, for the rounded
    /// corners, and the theme's otherwise.
    pub fn style(&self, theme: &Theme) -> theme::Style {
        let base = theme.base();
        if self.transparent() {
            theme::Style {
                background_color: Color::TRANSPARENT,
                ..base
            }
        } else {
            base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: Size = Size::new(960.0, 640.0);

    #[test]
    fn the_switch_reads_as_documented() {
        assert_eq!(FrameSetting::parse(None), (FrameSetting::Drawn, None));
        assert_eq!(
            FrameSetting::parse(Some(OsStr::new("1"))),
            (FrameSetting::Drawn, None)
        );
        assert_eq!(
            FrameSetting::parse(Some(OsStr::new("0"))),
            (FrameSetting::Off, None)
        );
        let (setting, warning) = FrameSetting::parse(Some(OsStr::new("yes")));
        assert_eq!(setting, FrameSetting::Drawn);
        assert!(warning.unwrap().contains("LEONIDA_WINDOW_FRAME=yes"));
    }

    #[test]
    fn the_frame_follows_the_platform_the_switch_and_the_window() {
        use FrameSetting::{Drawn, Off};
        use Platform::{Other, Windows};
        // Linux: transparent and rounded by the launcher while floating.
        assert!(transparent(Other, Drawn));
        assert_eq!(corners(Other, Drawn, true), Corners::Drawn);
        assert_eq!(corners(Other, Drawn, false), Corners::Square);
        // Windows: opaque, Windows 11's own corners and the launcher's line while floating.
        assert!(!transparent(Windows, Drawn));
        assert_eq!(corners(Windows, Drawn, true), Corners::System);
        assert_eq!(corners(Windows, Drawn, false), Corners::Square);
        // Off: opaque and square everywhere.
        for platform in [Other, Windows] {
            assert!(!transparent(platform, Off));
            for floating in [true, false] {
                assert_eq!(corners(platform, Off, floating), Corners::Square);
            }
        }
    }

    #[test]
    fn maximized_its_square_and_doesnt_resize() {
        let mut chrome = Chrome::on(Platform::Other, SIZE, FrameSetting::Drawn);
        assert!(chrome.floating() && chrome.resizable() && chrome.transparent());
        assert_eq!(chrome.corners(), Corners::Drawn);
        let _ = chrome.update(ChromeMessage::Maximized(true));
        assert!(chrome.is_maximized() && !chrome.resizable());
        assert_eq!(chrome.corners(), Corners::Square);
        let _ = chrome.update(ChromeMessage::Maximized(false));
        let _ = chrome.update(ChromeMessage::Mode(window::Mode::Fullscreen));
        assert_eq!(chrome.corners(), Corners::Square);
    }

    #[test]
    fn a_narrow_window_folds_and_a_chosen_place_shows_its_page() {
        let mut chrome = Chrome::on(Platform::Other, SIZE, FrameSetting::Drawn);
        assert!(!chrome.folded());
        chrome.resize(Size::new(500.0, 640.0));
        assert!(chrome.folded());
        assert_eq!(chrome.pane(), Pane::Sidebar);
        chrome.show_page();
        assert_eq!(chrome.pane(), Pane::Page);
        let _ = chrome.update(ChromeMessage::Back);
        assert_eq!(chrome.pane(), Pane::Sidebar);
    }

    #[test]
    fn a_new_theme_fades_in_from_what_showed_and_then_is_itself() {
        use crate::ui::tokens::Palette;
        if crate::ui::system::reduced_motion() {
            return;
        }
        let (dark, light) = (
            Theme::new(Palette::SUNSET_DARK),
            Theme::new(Palette::SUNSET_LIGHT),
        );
        let mut chrome = Chrome::on(Platform::Other, SIZE, FrameSetting::Drawn);
        assert_eq!(chrome.theme(light), light);
        let start = Instant::now();
        chrome.theme_changed(dark, start);
        assert!(chrome.is_fading());
        assert_eq!(chrome.theme(light), dark, "it starts from what showed");
        let _ = chrome.update(ChromeMessage::Frame(start + THEME_FADE / 4));
        let between = chrome.theme(light);
        assert!(between != dark && between != light);
        let _ = chrome.update(ChromeMessage::Frame(start + THEME_FADE));
        assert!(!chrome.is_fading());
        assert_eq!(chrome.theme(light), light);
    }

    #[test]
    fn the_background_is_clear_only_when_the_corners_are_drawn() {
        let theme = <Theme as Default>::default();
        let drawn = Chrome::on(Platform::Other, SIZE, FrameSetting::Drawn);
        assert_eq!(drawn.style(&theme).background_color, Color::TRANSPARENT);
        let windows = Chrome::on(Platform::Windows, SIZE, FrameSetting::Drawn);
        assert_ne!(windows.style(&theme).background_color, Color::TRANSPARENT);
    }
}
