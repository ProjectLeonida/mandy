// SPDX-License-Identifier: GPL-3.0-or-later

//! Drawing and driving the launcher's view offscreen, with no window: iced's user interface on
//! its tiny-skia renderer, all on the CPU.
//! Nothing here touches a display or a GPU.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use iced::advanced::renderer::{Headless, Style};
use iced::advanced::{clipboard, mouse};
use iced::keyboard::{self, Key, Location, Modifiers, key::Physical};
use iced::time::Instant;
use iced::{Event, Pixels, Size, window};
use iced_runtime::user_interface::{Cache, UserInterface};

use leonida_launcher::app::{Launcher, Message};
use leonida_launcher::ui::font;
use leonida_launcher::ui::window::FrameSetting;

/// A tiny-skia renderer, the launcher's fonts loaded.
pub fn renderer() -> iced::Renderer {
    font::load();
    iced::futures::executor::block_on(<iced::Renderer as Headless>::new(
        font::DEFAULT,
        Pixels(14.0),
        Some(&std::env::var("LEONIDA_TEST_RENDERER").unwrap_or_else(|_| "tiny-skia".to_owned())),
    ))
    .expect("tiny-skia draws without a GPU")
}

/// A launcher with no settings file, for a made-up repository.
pub fn launcher() -> Launcher {
    let mut launcher = Launcher::new(
        Ok(PathBuf::from(if cfg!(windows) {
            "C:\\src\\Leonida"
        } else {
            "/src/Leonida"
        })),
        None,
        FrameSetting::Drawn,
    );
    launcher.note = None;
    launcher
}

/// A view driven by events, as a window drives it.
pub struct Session {
    pub launcher: Launcher,
    renderer: iced::Renderer,
    cache: Cache,
    size: Size,
    cursor: mouse::Cursor,
    pub messages: Vec<Message>,
}

impl Session {
    pub fn new(launcher: Launcher, size: Size) -> Self {
        let mut session = Self {
            launcher,
            renderer: renderer(),
            cache: Cache::default(),
            size,
            cursor: mouse::Cursor::Unavailable,
            messages: Vec::new(),
        };
        session.launcher.resize(size);
        session
    }

    /// The window resized, as its event says: drawn at the new size from now on.
    pub fn resize_to(&mut self, size: Size) {
        self.size = size;
        let id = window::Id::unique();
        let _ = self.launcher.update(Message::Chrome(
            leonida_launcher::ui::window::ChromeMessage::Resized(id, size),
        ));
    }

    /// Makes the Play page's scene for the current size, mode and scale, at once, as the window
    /// would once its size has held.
    pub fn settle_scene(&mut self, scale: f32) {
        self.launcher.synchronous = true;
        let _ = self.launcher.update(Message::Scale(scale));
        let later = Instant::now() + leonida_launcher::backdrop::DEBOUNCE * 2;
        let _ = self.launcher.update(Message::SceneTick(later));
        // Its fade, run to the end, as the window's frames would.
        let _ = self.launcher.update(Message::SceneFrame(later));
        let whole = later + leonida_launcher::ui::tokens::motion::SCENE_IN;
        let _ = self.launcher.update(Message::SceneFrame(whole));
    }

    /// Sends events to the view; the messages it publishes are kept, not applied.
    pub fn send(&mut self, events: &[Event]) {
        let cache = std::mem::take(&mut self.cache);
        let mut ui =
            UserInterface::build(self.launcher.view(), self.size, cache, &mut self.renderer);
        let mut clipboard = clipboard::Null;
        let _ = ui.update(
            events,
            self.cursor,
            &mut self.renderer,
            &mut clipboard,
            &mut self.messages,
        );
        self.cache = ui.into_cache();
    }

    /// Sends events, then applies what they published, as the app would.
    pub fn apply(&mut self, events: &[Event]) {
        self.send(events);
        for message in std::mem::take(&mut self.messages) {
            let _ = self.launcher.update(message);
        }
    }

    /// A key pressed and released.
    pub fn press(&mut self, key: Key, modifiers: Modifiers) {
        self.send(&[key_event(key.clone(), modifiers, true)]);
        self.send(&[key_event(key, modifiers, false)]);
    }

    /// Draws a frame (so animations settle on a time), then the picture, RGBA.
    pub fn render(&mut self, scale: f32) -> Vec<u8> {
        self.send(&[Event::Window(
            window::Event::RedrawRequested(Instant::now()),
        )]);
        let cache = std::mem::take(&mut self.cache);
        let mut ui =
            UserInterface::build(self.launcher.view(), self.size, cache, &mut self.renderer);
        let theme = self.launcher.theme();
        let style = Style {
            text_color: iced::theme::Base::base(&theme).text_color,
        };
        ui.draw(&mut self.renderer, &theme, &style, self.cursor);
        self.cache = ui.into_cache();
        let background = self.launcher.style(&theme).background_color;
        let physical = Size::new(
            (self.size.width * scale).round() as u32,
            (self.size.height * scale).round() as u32,
        );
        self.renderer.screenshot(physical, scale, background)
    }
}

pub fn key_event(key: Key, modifiers: Modifiers, pressed: bool) -> Event {
    let physical = Physical::Unidentified(keyboard::key::NativeCode::Unidentified);
    let text = match &key {
        Key::Character(c) if pressed => Some(c.clone()),
        Key::Named(keyboard::key::Named::Space) if pressed => Some(" ".into()),
        _ => None,
    };
    Event::Keyboard(if pressed {
        keyboard::Event::KeyPressed {
            key: key.clone(),
            modified_key: key,
            physical_key: physical,
            location: Location::Standard,
            modifiers,
            text,
            repeat: false,
        }
    } else {
        keyboard::Event::KeyReleased {
            key: key.clone(),
            modified_key: key,
            physical_key: physical,
            location: Location::Standard,
            modifiers,
        }
    })
}

/// Writes RGBA pixels as a PNG.
/// Draws any view of the launcher's kit, in a theme, at a size: for pictures of parts.
pub fn render_view(
    view: leonida_launcher::ui::Element<'_, Message>,
    theme: &leonida_launcher::ui::Theme,
    size: Size,
) -> Vec<u8> {
    let mut renderer = renderer();
    let mut ui = UserInterface::build(view, size, Cache::default(), &mut renderer);
    let style = Style {
        text_color: iced::theme::Base::base(theme).text_color,
    };
    ui.draw(&mut renderer, theme, &style, mouse::Cursor::Unavailable);
    let background = iced::theme::Base::base(theme).background_color;
    let physical = Size::new(size.width as u32, size.height as u32);
    renderer.screenshot(physical, 1.0, background)
}

pub fn write_png(path: &Path, size: (u32, u32), rgba: &[u8]) {
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder).unwrap();
    }
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), size.0, size.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(rgba).unwrap();
}
