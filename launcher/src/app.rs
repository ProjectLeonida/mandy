// SPDX-License-Identifier: GPL-3.0-or-later

//! The launcher's window: a sidebar with three pages (Play, Settings, Performance) in the
//! design language's browsing window, the light and dark pill in the header bar, and the commands run from it.

use std::path::{Path, PathBuf};

use std::time::{Duration, Instant};

use iced::keyboard::{self, key::Named};
use iced::widget::image;
use iced::{Size, Subscription, Task};

use crate::backdrop::{self, Backdrop, DEBOUNCE, Key};
use crate::scene::{self, Picture};
use crate::ui::chrome::Corners;
use crate::ui::tokens::window as decorations;

use crate::commands::{self, Command, Missing, Platform};
use crate::config::{self, Checked, STAGES, Settings, Shade, Unimplemented, Wave64};
use crate::output::Tracker;
use crate::process;
use crate::repo;
use crate::state::{self, Action, Facts, Kind, Outcome, Tone};
use crate::ui::chrome::header;
use crate::ui::settings::{group, row};
use crate::ui::text;
use crate::ui::theme::{Emphasis, Frame, Status};
use crate::ui::tokens::{Mode, Palette, Surface, size, space};
use crate::ui::widget::{Column, Row, column, container, row as across, space as gap};
use crate::ui::window::{Chrome, ChromeMessage, FrameSetting};
use crate::ui::{
    Center, Element, Fill, Ink, Theme, button, choice, form, loader, navigation, progress,
    scroll_area, status, switch, text_field, toggle_group,
};

/// The launcher's version, from its Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The window's size when it opens, and the least it takes.
pub const SIZE: Size = Size::new(980.0, 680.0);
pub const LEAST: Size = Size::new(420.0, 480.0);

/// How many of the output's last lines the Play page shows.
const SHOWN_LINES: usize = 200;

/// The pages, in the sidebar's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Page {
    #[default]
    Play,
    GameFiles,
    Build,
    Performance,
    Diagnostics,
}

impl Page {
    pub const ALL: [Self; 5] = [
        Self::Play,
        Self::GameFiles,
        Self::Build,
        Self::Performance,
        Self::Diagnostics,
    ];

    pub const fn title(self) -> &'static str {
        match self {
            Self::Play => "Play",
            Self::GameFiles => "Game files",
            Self::Build => "Build",
            Self::Performance => "Performance",
            Self::Diagnostics => "Diagnostics",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|page| *page == self).unwrap_or(0)
    }

    /// The page after this one, or before it, round the ends.
    pub fn step(self, forwards: bool) -> Self {
        let count = Self::ALL.len();
        let at = self.index();
        Self::ALL[if forwards {
            (at + 1) % count
        } else {
            (at + count - 1) % count
        }]
    }
}

/// One of the folders the settings name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Folder {
    Package,
    Output,
    Work,
    /// Where the game's writes go (`gta6rt --overlay`).
    Overlay,
}

#[derive(Debug, Clone)]
pub enum Message {
    Chrome(ChromeMessage),
    /// A page chosen in the sidebar.
    Show(Page),
    /// Ctrl+Page Down and Ctrl+Page Up.
    Step(bool),
    Shade(Shade),
    /// Ctrl+L.
    ToggleShade,
    Folder(Folder, String),
    Pick(Folder),
    Picked(Folder, Option<PathBuf>),
    Stage(&'static str, bool),
    Wav(bool),
    FullCheck(bool),
    Gpu(String),
    AutoWave64(bool),
    Wave64(Wave64),
    Jobs(String),
    Validation(bool),
    TraceD3d12(bool),
    Unimplemented(Unimplemented),
    Act(Action),
    Process(process::Event),
    /// The window's scale factor.
    Scale(f32),
    /// A wait for the scene's key ended: generate it if it held.
    SceneTick(Instant),
    /// A scene was generated for this key.
    SceneReady(Key, Option<Picture>),
}

/// A command under way.
#[derive(Debug, Clone)]
pub struct Running {
    pub kind: Kind,
    pub command: Command,
    pub pid: Option<u32>,
    pub tracker: Tracker,
    pub cancelled: bool,
}

/// What exists on disk, as last looked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OnDisk {
    pub package: bool,
    pub loader: bool,
    pub exe: bool,
}

/// The launcher's state.
pub struct Launcher {
    pub repo: Result<PathBuf, String>,
    pub platform: Platform,
    pub exe: String,
    pub settings_path: Option<PathBuf>,
    pub settings: Settings,
    /// Something to say about the settings file, once.
    pub note: Option<String>,
    pub page: Page,
    pub chrome: Chrome,
    pub disk: OnDisk,
    pub running: Option<Running>,
    /// The last command that finished: how it ended, its command line and its output.
    pub last: Option<(Outcome, Command, Tracker)>,
    /// The Play page's scene, cached.
    pub backdrop: Backdrop,
    /// Generate scenes on the interface's thread, at once: for the tests, which run no tasks.
    pub synchronous: bool,
    /// Physical pixels to a logical one.
    pub scale: f32,
    /// The logo, decoded once: a handle made anew in every view would be a new image every frame,
    /// uploaded again while anything animates.
    logo: image::Handle,
}

/// The Play page's text column: its padding either side of its content, and its content's width,
/// in logical pixels. Text sits on the page only inside it; the scene's fade keeps it readable.
pub const COLUMN_PAD: f32 = space::XXL;
pub const COLUMN_CONTENT: f32 = 420.0;
/// The column's whole width from the page's left edge, as the scene's fade takes it.
pub const COLUMN: f32 = COLUMN_PAD + COLUMN_CONTENT + space::XXL;

impl Launcher {
    /// The launcher for a repository, with its settings file (none: not saved).
    pub fn new(
        repo: Result<PathBuf, String>,
        settings_path: Option<PathBuf>,
        frame: FrameSetting,
    ) -> Self {
        let loaded = settings_path
            .as_deref()
            .map_or(config::Loaded::Missing, config::load);
        let exe = repo
            .as_deref()
            .map_or_else(|_| repo::DEFAULT_EXE.to_owned(), repo::expected_exe);
        let mut launcher = Self {
            repo,
            platform: Platform::current(),
            exe,
            settings_path,
            settings: loaded.settings(),
            note: loaded.note(),
            page: Page::Play,
            chrome: Chrome::new(SIZE, frame),
            disk: OnDisk::default(),
            running: None,
            last: None,
            backdrop: Backdrop::default(),
            synchronous: false,
            scale: 1.0,
            logo: image::Handle::from_bytes(include_bytes!("../../assets/logo.png").as_slice()),
        };
        launcher.refresh();
        launcher
            .backdrop
            .want(launcher.scene_input(), Instant::now());
        launcher
    }

    /// The Play page's size, in logical pixels: the window less its frame, header bar and,
    /// unless it's folded, sidebar.
    pub fn page_size(&self) -> Size {
        let window = self.chrome.size();
        let inset = if self.chrome.corners() == Corners::Square {
            0.0
        } else {
            decorations::BORDER
        };
        let sidebar = if self.chrome.folded() {
            0.0
        } else {
            decorations::SIDEBAR_WIDTH + decorations::BORDER
        };
        Size::new(
            (window.width - 2.0 * inset - sidebar).max(0.0),
            (window.height - 2.0 * inset - decorations::BAR_HEIGHT).max(0.0),
        )
    }

    /// What the scene would be generated from now, and its key.
    pub fn scene_input(&self) -> Option<(Key, scene::Input)> {
        let page = self.page_size();
        let theme = self.chosen_theme();
        let input = scene::Input {
            colors: scene::Colors::of(theme.palette()),
            width: (page.width * self.scale).round() as u32,
            height: (page.height * self.scale).round() as u32,
            scale: self.scale,
            column: COLUMN,
            corner: (self.chrome.corners() == Corners::Drawn)
                .then_some(decorations::CORNER_RADIUS - decorations::BORDER),
        };
        Some((Key::of(&input, theme.name()), input))
    }

    /// Handles a message, then notes the scene the page wants now; a new one starts its wait.
    pub fn update(&mut self, message: Message) -> Task<Message> {
        let task = self.handle(message);
        if self.backdrop.want(self.scene_input(), Instant::now()) {
            Task::batch([task, wait(DEBOUNCE)])
        } else {
            task
        }
    }

    /// Generates the scene that's due, if one is: off this thread, or at once when synchronous.
    fn generate_scene(&mut self, now: Instant) -> Task<Message> {
        let Some((key, input)) = self.backdrop.due(now) else {
            return Task::none();
        };
        let generator = self.backdrop.generator;
        if self.synchronous {
            self.backdrop.finished(key, generator(&input));
            return Task::none();
        }
        let (sender, receiver) = iced::futures::channel::oneshot::channel();
        std::thread::spawn(move || {
            let _ = sender.send(generator(&input));
        });
        Task::perform(receiver, move |picture| {
            Message::SceneReady(key, picture.ok().flatten())
        })
    }

    /// Looks again at what exists: the package, the built loader, the game's executable.
    pub fn refresh(&mut self) {
        let package = self.settings.package.trim();
        let output = self.settings.output.trim();
        let is_dir = |path: &str| !path.is_empty() && Path::new(path).is_dir();
        let is_file = |path: PathBuf| path.is_file();
        self.disk = OnDisk {
            package: is_dir(package),
            loader: !output.is_empty() && is_file(Path::new(output).join(self.platform.loader())),
            exe: !package.is_empty() && is_file(Path::new(package).join(&self.exe)),
        };
    }

    /// What the state machine reads.
    pub fn facts(&self) -> Facts {
        let package = self.settings.package.trim().to_owned();
        Facts {
            checked: self
                .settings
                .checked
                .as_ref()
                .is_some_and(|checked| checked.package == package),
            package,
            package_exists: self.disk.package,
            output: self.settings.output.trim().to_owned(),
            loader_exists: self.disk.loader,
            loader: self.platform.loader().to_owned(),
            exe_exists: self.disk.exe,
            exe: self.exe.clone(),
            running: self.running.as_ref().map(|running| running.kind),
            last: self.last.as_ref().map(|(outcome, ..)| outcome.clone()),
        }
    }

    /// The theme the window draws in now: the chosen one, or while it fades in, on the way.
    pub fn theme(&self) -> Theme {
        self.chrome.theme(self.chosen_theme())
    }

    /// The theme the settings choose.
    pub fn chosen_theme(&self) -> Theme {
        Theme::new(match self.settings.shade {
            Shade::Dark => Palette::SUNSET_DARK,
            Shade::Light => Palette::SUNSET_LIGHT,
        })
    }

    /// Light or dark, fading from what shows into it.
    fn set_shade(&mut self, shade: Shade) {
        if shade == self.settings.shade {
            return;
        }
        let before = self.chosen_theme();
        self.settings.shade = shade;
        self.chrome.theme_changed(before, Instant::now());
        self.save();
    }

    fn save(&mut self) {
        if let Some(path) = &self.settings_path
            && let Err(error) = config::save(path, &self.settings)
        {
            self.note = Some(format!("Settings couldn't be saved: {error}"));
        }
    }

    /// The command for an action, or why there's none.
    fn command(&self, kind: Kind) -> Result<Command, String> {
        let repo = self.repo.as_ref().map_err(Clone::clone)?;
        let repo = repo.display().to_string();
        let made = match kind {
            Kind::Check => commands::check(self.platform, &repo, &self.settings),
            Kind::Build => commands::build(self.platform, &repo, &self.settings),
            Kind::Play => commands::play(self.platform, &self.exe, &self.settings),
        };
        made.map_err(|missing| match missing {
            Missing::Package => "Choose the game package folder in Settings.".to_owned(),
            Missing::Output => "Choose an output folder in Settings.".to_owned(),
        })
    }

    fn handle(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Chrome(message) => {
                // Once the window's known, its scale factor too.
                let scale = match &message {
                    ChromeMessage::Opened(id) | ChromeMessage::Resized(id, _) => {
                        iced::window::scale_factor(*id).map(Message::Scale)
                    }
                    _ => Task::none(),
                };
                return Task::batch([self.chrome.update(message).map(Message::Chrome), scale]);
            }
            Message::Scale(scale) => {
                if scale > 0.0 {
                    self.scale = scale;
                }
            }
            Message::SceneTick(now) => return self.generate_scene(now),
            Message::SceneReady(key, picture) => {
                self.backdrop.finished(key, picture);
                // Another key may have come and held meanwhile.
                return self.generate_scene(Instant::now());
            }
            Message::Show(page) => {
                self.page = page;
                self.chrome.show_page();
            }
            Message::Step(forwards) => {
                self.page = self.page.step(forwards);
                self.chrome.show_page();
            }
            Message::Shade(shade) => self.set_shade(shade),
            Message::ToggleShade => self.set_shade(match self.settings.shade {
                Shade::Dark => Shade::Light,
                Shade::Light => Shade::Dark,
            }),
            Message::Folder(folder, value) => {
                *self.folder_mut(folder) = value;
                self.refresh();
                self.save();
            }
            Message::Pick(folder) => {
                let start = self.folder(folder).trim().to_owned();
                let title = match folder {
                    Folder::Package => "Choose your game package folder",
                    Folder::Output => "Choose the output folder",
                    Folder::Work => "Choose the work folder",
                    Folder::Overlay => "Choose the folder for the game's writes",
                };
                let mut dialog = rfd::AsyncFileDialog::new().set_title(title);
                if !start.is_empty() && Path::new(&start).is_dir() {
                    dialog = dialog.set_directory(&start);
                }
                return Task::perform(dialog.pick_folder(), move |picked| {
                    Message::Picked(folder, picked.map(|handle| handle.path().to_path_buf()))
                });
            }
            Message::Picked(folder, Some(path)) => {
                *self.folder_mut(folder) = path.display().to_string();
                self.refresh();
                self.save();
            }
            Message::Picked(_, None) => {}
            Message::Stage(stage, on) => {
                self.settings.stages.retain(|chosen| chosen != stage);
                if on {
                    self.settings.stages.push(stage.to_owned());
                }
                self.save();
            }
            Message::Wav(on) => {
                self.settings.wav = on;
                self.save();
            }
            Message::FullCheck(on) => {
                self.settings.full_check = on;
                self.save();
            }
            Message::Gpu(value) => {
                self.settings.gpu = value;
                self.save();
            }
            Message::AutoWave64(auto) => {
                self.settings.wave64 = if auto { Wave64::Auto } else { Wave64::On };
                self.save();
            }
            Message::Wave64(mode) => {
                self.settings.wave64 = mode;
                self.save();
            }
            Message::Jobs(value) => {
                self.settings.jobs = value.chars().filter(char::is_ascii_digit).take(4).collect();
                self.save();
            }
            Message::TraceD3d12(on) => {
                self.settings.trace_d3d12 = on;
                self.save();
            }
            Message::Unimplemented(policy) => {
                self.settings.unimplemented = policy;
                self.save();
            }
            Message::Validation(on) => {
                self.settings.validation = on;
                self.save();
            }
            Message::Act(Action::Cancel) => {
                if let Some(running) = &mut self.running {
                    running.cancelled = true;
                    if let Some(pid) = running.pid {
                        process::cancel(pid);
                    }
                }
            }
            Message::Act(Action::Run(kind)) => return self.start(kind),
            Message::Process(event) => self.follow(event),
        }
        Task::none()
    }

    fn folder(&self, folder: Folder) -> &str {
        match folder {
            Folder::Package => &self.settings.package,
            Folder::Output => &self.settings.output,
            Folder::Work => &self.settings.work,
            Folder::Overlay => &self.settings.overlay,
        }
    }

    fn folder_mut(&mut self, folder: Folder) -> &mut String {
        match folder {
            Folder::Package => &mut self.settings.package,
            Folder::Output => &mut self.settings.output,
            Folder::Work => &mut self.settings.work,
            Folder::Overlay => &mut self.settings.overlay,
        }
    }

    /// Starts a command, unless one runs.
    fn start(&mut self, kind: Kind) -> Task<Message> {
        if self.running.is_some() {
            return Task::none();
        }
        self.refresh();
        let command = match self.command(kind) {
            Ok(command) => command,
            Err(why) => {
                self.last = Some((
                    Outcome {
                        kind,
                        code: None,
                        cancelled: false,
                        words: Some(why),
                    },
                    Command {
                        program: String::new(),
                        args: Vec::new(),
                        env: Vec::new(),
                        cwd: String::new(),
                    },
                    Tracker::plain(),
                ));
                return Task::none();
            }
        };
        let tracker = match kind {
            Kind::Build => Tracker::build(commands::stages(&self.settings).as_deref()),
            Kind::Check | Kind::Play => Tracker::plain(),
        };
        let stream = process::run(&command);
        self.running = Some(Running {
            kind,
            command,
            pid: None,
            tracker,
            cancelled: false,
        });
        Task::run(stream, Message::Process)
    }

    /// What a running command said.
    pub fn follow(&mut self, event: process::Event) {
        let Some(running) = &mut self.running else {
            return;
        };
        let ended = match event {
            process::Event::Started(pid) => {
                running.pid = Some(pid);
                None
            }
            process::Event::Line(line) => {
                running.tracker.feed(&line);
                None
            }
            process::Event::Exited(code) => Some((code, None)),
            process::Event::Failed(why) => Some((None, Some(why))),
        };
        let Some((code, why)) = ended else {
            return;
        };
        let Some(running) = self.running.take() else {
            return;
        };
        let succeeded = code == Some(0) && !running.cancelled;
        let outcome = Outcome {
            kind: running.kind,
            code,
            cancelled: running.cancelled,
            words: why.or_else(|| (!succeeded).then(|| running.tracker.last_words()).flatten()),
        };
        if running.kind == Kind::Check {
            let package = self.settings.package.trim().to_owned();
            if succeeded {
                let game = running
                    .tracker
                    .game
                    .clone()
                    .unwrap_or_else(|| "a supported version".to_owned());
                self.settings.checked = Some(Checked { package, game });
            } else if !running.cancelled {
                self.settings.checked = None;
            }
            self.save();
        }
        self.last = Some((outcome, running.command, running.tracker));
        self.refresh();
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let keys = keyboard::listen().filter_map(|event| match event {
            keyboard::Event::KeyPressed { key, modifiers, .. }
                if modifiers.control() && !modifiers.alt() =>
            {
                match key.as_ref() {
                    keyboard::Key::Named(Named::PageDown) => Some(Message::Step(true)),
                    keyboard::Key::Named(Named::PageUp) => Some(Message::Step(false)),
                    keyboard::Key::Character("l" | "L") => Some(Message::ToggleShade),
                    _ => None,
                }
            }
            _ => None,
        });
        let fading = if self.chrome.is_fading() {
            iced::window::frames().map(|at| Message::Chrome(ChromeMessage::Frame(at)))
        } else {
            Subscription::none()
        };
        Subscription::batch([keys, Chrome::subscription().map(Message::Chrome), fading])
    }

    /// The window: header bar, sidebar and the page.
    pub fn view(&self) -> Element<'_, Message> {
        let shade = usize::from(self.settings.shade == Shade::Light);
        let pill = toggle_group(["Dark", "Light"], shade)
            .pill(true)
            .on_choose(|at| Message::Shade(if at == 1 { Shade::Light } else { Shade::Dark }));
        let bar = header().start(text::brand("Leonida")).end(pill);
        let sidebar = navigation(
            Page::ALL.iter().map(|page| page.title()),
            Some(self.page.index()),
        )
        .on_choose(|at| Message::Show(Page::ALL[at]));
        let page = match self.page {
            Page::Play => self.play_page(),
            Page::GameFiles => self.game_files_page(),
            Page::Build => self.build_page(),
            Page::Diagnostics => self.diagnostics_page(),
            Page::Performance => self.performance_page(),
        };
        let scrolled: Element<'_, Message> =
            scroll_area(container(page).padding(COLUMN_PAD).width(Fill)).into();
        // The Play page's scene, under it: the cached picture, never made here. While a resize
        // is under way, the last one, scaled to cover and kept to the bottom right.
        let content = match (self.page, self.backdrop.current()) {
            (
                Page::Play,
                Some(backdrop::Shown {
                    handle: Some(handle),
                    size,
                }),
            ) => {
                let shown = backdrop::cover(*size, self.page_size());
                let picture = image(handle.clone())
                    .width(shown.width)
                    .height(shown.height)
                    .content_fit(iced::ContentFit::Fill);
                let scene = container(picture)
                    .width(Fill)
                    .height(Fill)
                    .align_x(iced::alignment::Horizontal::Right)
                    .align_y(iced::alignment::Vertical::Bottom)
                    .clip(true);
                iced::widget::Stack::with_children([scene.into(), scrolled]).into()
            }
            _ => scrolled,
        };
        // One focus scope round the whole window, as every window has it: Tab goes through the
        // header bar, the sidebar and the page.
        crate::ui::focus::scope(self.chrome.view(
            bar,
            Some(sidebar.into()),
            content,
            Message::Chrome,
        ))
        .into()
    }

    fn page_title<'a>(&self, words: &'a str) -> Element<'a, Message> {
        text::heading(words).into()
    }

    fn play_page(&self) -> Element<'_, Message> {
        let facts = self.facts();
        let running = self.running.as_ref();
        let tracker = running
            .map(|running| &running.tracker)
            .or(self.last.as_ref().map(|(_, _, tracker)| tracker));
        let under_way = running.and_then(|running| running.tracker.stage());
        let line = state::status(&facts, under_way.as_deref());
        let game = self
            .settings
            .checked
            .as_ref()
            .filter(|checked| checked.package == self.settings.package.trim())
            .map_or_else(
                || "not checked yet".to_owned(),
                |checked| checked.game.clone(),
            );
        let logo = image(self.logo.clone()).width(96).height(96);
        let identity = across![
            logo,
            column![
                text::title("Leonida"),
                text::meta(format!("Launcher {VERSION}")),
                text::meta(format!("Game copy: {game}")),
            ]
            .spacing(space::XS),
        ]
        .spacing(space::XL)
        .align_y(Center);

        let tone = match line.tone {
            Tone::Waiting => Status::Waiting,
            Tone::Running => Status::Running,
            Tone::Passed => Status::Passed,
            Tone::Info => Status::Info,
            Tone::Failed => Status::Failed,
        };
        let status_line = across![
            status(tone, line.word),
            text::interface(line.text).class(match line.tone {
                Tone::Failed => Ink::Text,
                _ => Ink::Body,
            }),
        ]
        .spacing(space::L)
        .align_y(Center);

        let primary = state::primary(&facts);
        let mut actions = Row::new().spacing(space::L).align_y(Center);
        let main = button(primary.action.label())
            .large()
            .emphasis(match primary.action {
                Action::Cancel => Emphasis::Danger,
                Action::Run(_) => Emphasis::Primary,
            })
            .on_press_maybe(primary.enabled().then_some(Message::Act(primary.action)));
        actions = actions.push(main);
        for (action, enabled) in state::secondary(&facts) {
            actions = actions.push(
                button(action.label()).on_press_maybe(enabled.then_some(Message::Act(action))),
            );
        }
        if running.is_some() {
            actions = actions.push(loader().size(size::LOADER_SMALL));
        }

        // The text column: text sits on the page only here, where the scene's fade keeps it
        // readable.
        let mut column_content = Column::new()
            .spacing(space::XL)
            .width(Fill)
            .push(identity)
            .push(status_line)
            .push(actions);
        if let Some(note) = &self.note {
            column_content = column_content.push(text::meta(note.clone()));
        }
        if let Err(why) = &self.repo {
            column_content = column_content
                .push(text::interface(why.clone()).class(Ink::Accent(crate::ui::Accent::Red)));
        }
        let text_column = container(column_content).max_width(COLUMN_CONTENT);
        // Everything else lies on an opaque surface.
        let bar: Option<Element<'_, Message>> = match (running, &self.last) {
            (Some(running), _) => Some(match running.tracker.progress() {
                Some(done) if running.kind == Kind::Build => progress(done).into(),
                _ => crate::ui::progress::busy().into(),
            }),
            (None, Some((outcome, _, tracker)))
                if !outcome.succeeded() && outcome.kind == Kind::Build =>
            {
                Some(progress(tracker.progress().unwrap_or(1.0)).failed().into())
            }
            (None, Some((outcome, _, _))) if outcome.succeeded() && outcome.kind == Kind::Build => {
                Some(progress(1.0).done().into())
            }
            _ => None,
        };
        let command = running
            .map(|running| &running.command)
            .or(self.last.as_ref().map(|(_, command, _)| command))
            .filter(|command| !command.program.is_empty());
        let bar_or_output = bar.is_some() || tracker.is_some();
        let mut surface = Column::new().spacing(space::M).width(Fill);
        if let Some(bar) = bar {
            surface = surface.push(bar);
        }
        if let Some(tracker) = tracker {
            let lines = tracker.lines.iter().rev().take(SHOWN_LINES).rev();
            let log = Column::with_children(
                lines.map(|line| text::meta(line.clone()).class(Ink::Body).into()),
            )
            .spacing(space::XXS)
            .width(Fill);
            surface = surface.push(text::label("Output"));
            if let Some(command) = command {
                surface = surface.push(text::meta(command.display()));
            }
            surface = surface.push(
                container(scroll_area(container(log).padding(space::L).width(Fill)))
                    .height(220)
                    .width(Fill)
                    .class(Frame::Card(Surface::Sunk)),
            );
        }
        let mut page = Column::new().spacing(space::XL).push(text_column);
        if bar_or_output {
            page = page.push(
                container(surface)
                    .padding(space::L)
                    .width(Fill)
                    .class(Frame::Card(Surface::Panel)),
            );
        }
        page.max_width(760).into()
    }

    fn folder_field(&self, folder: Folder) -> Element<'_, Message> {
        let (name, help, placeholder) = match folder {
            Folder::Package => (
                "Game package",
                "Your own copy: the folder that holds MicrosoftGame.Config and the .rpf archives.",
                "The package folder",
            ),
            Folder::Output => (
                "Output folder",
                "Where the build puts the port and the loader (make_game.py --out).",
                "Documents/Leonida",
            ),
            Folder::Work => (
                "Work folder",
                "Hundreds of GB of intermediate files (--work); empty puts them in the output \
                 folder's work.",
                "<output>/work",
            ),
            Folder::Overlay => (
                "Writes folder",
                "Where the game's writes go instead of the package (gta6rt --overlay); empty is \
                 the package's overlay folder. Needed for a read-only package.",
                "<package>/overlay",
            ),
        };
        let missing = folder == Folder::Package
            && !self.settings.package.trim().is_empty()
            && !self.disk.package;
        let field = text_field(placeholder, self.folder(folder))
            .on_input(move |value| Message::Folder(folder, value))
            .invalid(missing);
        let control = across![field, button("Browse…").on_press(Message::Pick(folder))]
            .spacing(space::M)
            .align_y(Center);
        let error = missing.then(|| "There's no folder there.".to_owned());
        form::field(name, control).help(help).error(error).into()
    }

    fn game_files_page(&self) -> Element<'_, Message> {
        let folders = group("Folders")
            .description("The folders the build uses. Nothing is copied into the repository.")
            .item(
                form::form([
                    self.folder_field(Folder::Package),
                    self.folder_field(Folder::Output),
                    self.folder_field(Folder::Work),
                ])
                .padding([space::L, space::XL]),
            );
        let check = group("Checking your copy").row(
            row("Compare every file's SHA-256")
                .description("Slow: reads the whole package (make_game.py --full-check).")
                .control(switch(self.settings.full_check).on_toggle(Message::FullCheck)),
        );
        column![self.page_title("Game files"), folders, check]
            .spacing(space::XL)
            .max_width(760)
            .into()
    }

    fn build_page(&self) -> Element<'_, Message> {
        let stage_words = |stage: &'static str| match stage {
            "shaders" => (
                "Shaders",
                "Carve, disassemble and translate them to SPIR-V: the longest step.",
            ),
            "assets" => (
                "Assets",
                "Export the models, textures and audio: hundreds of GB.",
            ),
            "runtime" => ("Runtime", "Build the loader, gta6rt."),
            "pipelines" => (
                "Pipelines",
                "Create every translated shader on this GPU's driver and save its cache.",
            ),
            _ => (stage, ""),
        };
        let mut stages = group("Stages").description(
            "What Build runs after checking your copy (make_game.py --stages). A stopped build \
             continues where it left off.",
        );
        for stage in STAGES {
            let on = self.settings.stages.iter().any(|chosen| chosen == stage);
            let (title, words) = stage_words(stage);
            stages = stages.row(
                row(title)
                    .description(words)
                    .control(switch(on).on_toggle(move |on| Message::Stage(stage, on))),
            );
        }
        let audio = group("Audio").row(
            row("Also decode audio to WAV")
                .description(
                    "16-bit PCM WAV beside the Ogg Opus files: several times larger (--wav).",
                )
                .control(switch(self.settings.wav).on_toggle(Message::Wav)),
        );
        column![self.page_title("Build"), stages, audio]
            .spacing(space::XL)
            .max_width(760)
            .into()
    }

    fn diagnostics_page(&self) -> Element<'_, Message> {
        let loader = group("The loader")
            .description("For finding what stops the game: docs/boot.md has the triage.")
            .row(
                row("Trace D3D12 calls")
                    .description(
                        "Every D3D12 call and its outcome in the output (gta6rt --trace-d3d12).",
                    )
                    .control(switch(self.settings.trace_d3d12).on_toggle(Message::TraceD3d12)),
            )
            .item(
                container(
                    form::field(
                        "Unimplemented imports",
                        choice(Some(self.settings.unimplemented))
                            .option(
                                Unimplemented::Log,
                                "Log",
                                "Log each once and return 0: the loader's default.",
                            )
                            .option(
                                Unimplemented::Fail,
                                "Fail",
                                "Return E_NOTIMPL, to see how the game's own failure path reacts.",
                            )
                            .option(
                                Unimplemented::Stop,
                                "Stop",
                                "End at the first, with its caller (exit code 3).",
                            )
                            .on_choose(Message::Unimplemented),
                    )
                    .help("LEONIDA_UNIMPLEMENTED"),
                )
                .padding([space::L, space::XL]),
            )
            .item(container(self.folder_field(Folder::Overlay)).padding([space::L, space::XL]));
        column![self.page_title("Diagnostics"), loader]
            .spacing(space::XL)
            .max_width(760)
            .into()
    }

    fn performance_page(&self) -> Element<'_, Message> {
        let gpu = group("Graphics")
            .row(
                row("GPU")
                    .description(
                        "Part of the GPU's name, such as 5070 or radeon: the loader and the \
                         build's GPU checks use the first that matches (LEONIDA_VK_DEVICE). \
                         Empty lets the runtime choose.",
                    )
                    .control(
                        text_field("Any", &self.settings.gpu)
                            .on_input(Message::Gpu)
                            .width(180),
                    ),
            )
            .row(
                row("Vulkan validation")
                    .description(
                        "The Khronos validation layer at play, a diagnostic: slower to start and \
                         to run (LEONIDA_VK_VALIDATION=1).",
                    )
                    .control(switch(self.settings.validation).on_toggle(Message::Validation)),
            );
        let auto = self.settings.wave64 == Wave64::Auto;
        let mode = if auto {
            None
        } else {
            Some(self.settings.wave64)
        };
        let mut wave = choice(mode)
            .option(
                Wave64::On,
                "Emulate",
                "Exact on any GPU; needed where subgroups are 32 wide (NVIDIA, Intel).",
            )
            .option(
                Wave64::Off,
                "Native",
                "64-wide subgroups as the console runs them (AMD).",
            );
        if !auto {
            wave = wave.on_choose(Message::Wave64);
        }
        let build = group("Build")
            .row(
                row("Choose wave64 translation automatically")
                    .description(
                        "Asks the GPU whether it runs 64-wide subgroups (--wave64-emulate auto). \
                         Changing it translates the shaders again.",
                    )
                    .control(switch(auto).on_toggle(Message::AutoWave64)),
            )
            .item(container(wave).padding([space::L, space::XL]))
            .row(
                row("Jobs")
                    .description(
                        "Processes the shader translation runs on (--jobs). Empty is half the \
                         CPU threads.",
                    )
                    .control(
                        text_field("Auto", &self.settings.jobs)
                            .on_input(Message::Jobs)
                            .max_len(4)
                            .width(96),
                    ),
            );
        column![
            self.page_title("Performance"),
            gpu,
            build,
            gap().height(space::XL)
        ]
        .spacing(space::XL)
        .max_width(760)
        .into()
    }

    /// The window's background: clear where the launcher rounds its own corners.
    pub fn style(&self, theme: &Theme) -> iced::theme::Style {
        self.chrome.style(theme)
    }

    /// For the window's events: the scale-free size the page is drawn at, in a test.
    pub fn resize(&mut self, size: Size) {
        self.chrome.resize(size);
    }

    /// The mode, for the screenshots' names.
    pub fn mode(&self) -> Mode {
        match self.settings.shade {
            Shade::Dark => Mode::Dark,
            Shade::Light => Mode::Light,
        }
    }
}

/// The end of a scene key's wait, `delay` from now, from a thread that sleeps: no timer runs once
/// it has fired.
fn wait(delay: Duration) -> Task<Message> {
    let (sender, receiver) = iced::futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        let _ = sender.send(Instant::now());
    });
    Task::perform(receiver, |at| {
        Message::SceneTick(at.unwrap_or_else(|_| Instant::now()))
    })
}
