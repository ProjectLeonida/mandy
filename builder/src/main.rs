// Leonida: a launcher that builds the port with make_game.py, manages its mods and starts it.
#![cfg_attr(all(not(debug_assertions), windows), windows_subsystem = "windows")]
mod backend;
mod home;
mod play;

use backend::{Build, Msg, Progress, State, STAGES};
use eframe::egui::{
    self, pos2, vec2, Align2, Color32, FontFamily, FontId, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, Vec2,
};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};

const SIZE: Vec2 = vec2(960.0, 600.0);
const PAGE: Color32 = Color32::from_rgb(19, 18, 24);
const INK: Color32 = Color32::from_rgb(240, 238, 244);
const SOFT: Color32 = Color32::from_rgb(166, 161, 178);
const FAINT: Color32 = Color32::from_rgb(133, 127, 147);
const PINK: Color32 = Color32::from_rgb(238, 62, 140);
const PINK_HOVER: Color32 = Color32::from_rgb(250, 88, 160);
const ORANGE: Color32 = Color32::from_rgb(255, 152, 84);
const PURPLE: Color32 = Color32::from_rgb(150, 82, 210);
const GO: Color32 = Color32::from_rgb(95, 190, 145);
const BAD: Color32 = Color32::from_rgb(238, 116, 108);

fn glass(a: u8) -> Color32 {
    Color32::from_rgba_premultiplied(a, a, a, a)
}
fn reg(s: f32) -> FontId {
    FontId::new(s, FontFamily::Proportional)
}
fn semi(s: f32) -> FontId {
    FontId::new(s, FontFamily::Name("semi".into()))
}
fn bold(s: f32) -> FontId {
    FontId::new(s, FontFamily::Name("bold".into()))
}
fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t.clamp(0.0, 1.0)) as u8;
    Color32::from_rgb(m(a.r(), b.r()), m(a.g(), b.g()), m(a.b(), b.b()))
}
fn sunset(t: f32) -> Color32 {
    if t < 0.5 { mix(PURPLE, PINK, t * 2.0) } else { mix(PINK, ORANGE, t * 2.0 - 1.0) }
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Home,
    Mods,
    Build,
    Settings,
    Log,
}

#[derive(Clone, Copy, PartialEq)]
enum Page {
    Files,
    Building,
    Ready,
    Failed,
}

struct Found {
    python: Option<Vec<String>>,
    out: Option<String>,
    games: Vec<(String, String)>,
}

struct App {
    logo: egui::TextureHandle,
    root: Option<PathBuf>,
    python: Option<Vec<String>>,
    searching: Option<Receiver<Found>>,
    games: Vec<(String, String)>,
    chosen: Option<usize>,
    out: String,
    default_out: String,
    jobs: usize,
    stages: [bool; 5],
    page: Page,
    check_only: bool,
    show_log: bool,
    cancelled: bool,
    progress: Progress,
    build: Option<Build>,
    started: std::time::Instant,
    took: u64,
    note: String,
    tab: Tab,
    config: play::Config,
    log_path: PathBuf,
    log: Vec<String>,
    log_read: Option<std::time::Instant>,
    port: play::Port,
    port_checked: Option<std::time::Instant>,
    game: Option<play::Game>,
    stopping: bool,
    play_note: String,
    mods: play::Mods,
    mods_rx: Option<Receiver<play::Mods>>,
    #[cfg(feature = "shots")]
    shot: usize,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = &cc.egui_ctx;
        let mut f = egui::FontDefinitions::default();
        let fallback = f.families[&FontFamily::Proportional].clone();
        for (name, bytes) in [
            ("regular", include_bytes!("../fonts/Inter-Regular.ttf").as_slice()),
            ("semi", include_bytes!("../fonts/Inter-SemiBold.ttf")),
            ("bold", include_bytes!("../fonts/Inter-Bold.ttf")),
        ] {
            f.font_data.insert(name.into(), egui::FontData::from_static(bytes));
            let family = if name == "regular" { FontFamily::Proportional } else { FontFamily::Name(name.into()) };
            f.families.insert(family, std::iter::once(name.to_string()).chain(fallback.clone()).collect());
        }
        ctx.set_fonts(f);
        ctx.set_theme(egui::Theme::Dark);
        let mut v = egui::Visuals::dark();
        v.override_text_color = Some(INK);
        v.selection.bg_fill = PINK.gamma_multiply(0.45);
        ctx.set_visuals(v);

        // The mark is off-centre in a square of empty space: crop to it.
        let logo = image::load_from_memory(include_bytes!("../../assets/logo.png"))
            .expect("assets/logo.png")
            .crop_imm(0, 145, 1024, 740)
            .resize(512, 512, image::imageops::FilterType::Lanczos3)
            .to_rgba8();
        let size = [logo.width() as usize, logo.height() as usize];
        let logo = ctx.load_texture("logo", egui::ColorImage::from_rgba_unmultiplied(size, &logo), egui::TextureOptions::LINEAR);

        let dir = play::config_dir();
        let config = play::Config::load(dir.join("launcher.conf"));
        let root = backend::find_root();
        let (tx, rx) = channel();
        let (r, c) = (root.clone(), ctx.clone());
        std::thread::spawn(move || {
            let python = backend::find_python();
            let (out, games) = match (&python, &r) {
                (Some(p), Some(r)) => (backend::default_out(p, r), backend::find_games(p, r)),
                _ => (None, vec![]),
            };
            let _ = tx.send(Found { python, out, games });
            c.request_repaint();
        });
        Self {
            logo,
            root,
            python: None,
            searching: Some(rx),
            games: vec![],
            chosen: None,
            out: config.get("out").into(),
            default_out: String::new(),
            jobs: (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1),
            stages: [true; 5],
            page: Page::Files,
            check_only: false,
            show_log: false,
            cancelled: false,
            progress: Progress::new(&[true; 5]),
            build: None,
            started: std::time::Instant::now(),
            took: 0,
            note: String::new(),
            tab: Tab::Home,
            config,
            log_path: dir.join("game.log"),
            log: vec![],
            log_read: None,
            port: play::Port::NotBuilt,
            port_checked: None,
            game: None,
            stopping: false,
            play_note: String::new(),
            mods: play::Mods::default(),
            mods_rx: None,
            #[cfg(feature = "shots")]
            shot: 0,
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        self.poll_game();
        if self.port_checked.is_none_or(|t| t.elapsed().as_secs() >= 2) {
            self.port = play::port(&self.out);
            self.port_checked = Some(std::time::Instant::now());
        }
        if let Some(f) = self.searching.as_ref().and_then(|r| r.try_recv().ok()) {
            self.searching = None;
            self.python = f.python;
            self.default_out = f.out.unwrap_or_default();
            if self.out.is_empty() {
                self.out = self.default_out.clone();
            }
            for g in f.games {
                if !self.games.iter().any(|(d, _)| *d == g.0) {
                    self.games.push(g);
                }
            }
            if self.games.len() == 1 {
                self.chosen = Some(0);
            }
            self.port_checked = None;
            self.port = play::port(&self.out);
            if self.port != play::Port::NotBuilt {
                self.load_mods(vec![], ctx);
            }
            if self.root.is_none() {
                self.note = "make_game.py wasn't found. Keep this app inside the Leonida folder.".into();
            } else if self.python.is_none() {
                self.note = "Python 3.12 or newer is needed. Run Build.bat once to install it, or get it from python.org.".into();
            }
        }
        let Some(b) = &self.build else { return };
        let mut end = None;
        while let Ok(m) = b.rx.try_recv() {
            match m {
                Msg::Line(l) => self.progress.line(l),
                Msg::Done(c) => end = Some(c),
            }
        }
        if let Some(code) = end {
            let ok = code == Some(0) && !self.cancelled;
            self.build = None;
            self.took = self.started.elapsed().as_secs();
            self.progress.finish(ok);
            self.port_checked = None;
            self.load_mods(vec![], ctx);
            self.page = match (ok, self.cancelled) {
                (true, _) => Page::Ready,
                (_, true) => Page::Files,
                _ => Page::Failed,
            };
            if self.cancelled {
                self.note = "Stopped. Build again to pick up where it left off.".into();
            }
        }
    }

    fn add_game(&mut self, dir: String) {
        let i = self.games.iter().position(|(d, _)| *d == dir).unwrap_or_else(|| {
            self.games.push((dir, "Chosen by you".into()));
            self.games.len() - 1
        });
        self.chosen = Some(i);
    }

    fn start(&mut self, check_only: bool, ctx: &egui::Context) {
        let (Some(root), Some(py), Some(i)) = (&self.root, &self.python, self.chosen) else { return };
        let mut args: Vec<String> = ["--no-tui", "--game", &self.games[i].0, "--jobs"].map(String::from).into();
        args.push(self.jobs.to_string());
        let out = self.out.trim();
        if !out.is_empty() && out != self.default_out {
            args.extend(["--out".into(), out.into()]);
        }
        let mut run = self.stages;
        run[0] = true; // make_game.py always checks the files
        if check_only {
            args.push("--check".into());
            run = [true, false, false, false, false];
        } else if run != [true; 5] {
            // Left out when everything is on: make_game.py then skips what this system can't build.
            let list: Vec<_> = STAGES.iter().zip(run).filter(|(_, on)| *on).map(|(s, _)| s.0).collect();
            args.extend(["--stages".into(), list.join(",")]);
        }
        self.progress = Progress::new(&run);
        let c = ctx.clone();
        match backend::start(py, root, &args, move || c.request_repaint()) {
            Ok(b) => {
                self.build = Some(b);
                self.page = Page::Building;
                self.check_only = check_only;
                self.cancelled = false;
                self.show_log = false;
                self.note.clear();
                self.started = std::time::Instant::now();
            }
            Err(e) => self.note = format!("The build couldn't start: {e}"),
        }
    }
}

// ---- drawing ----

fn r(x: f32, y: f32, w: f32, h: f32) -> Rect {
    Rect::from_min_size(pos2(x, y), vec2(w, h))
}

fn text(ui: &Ui, p: Pos2, a: Align2, s: &str, f: FontId, c: Color32) -> Rect {
    ui.painter().text(p, a, s, f, c)
}

/// One line, cut with an ellipsis at `w`.
fn line(ui: &Ui, p: Pos2, w: f32, s: &str, f: FontId, c: Color32) {
    let mut job = egui::text::LayoutJob::simple_singleline(s.into(), f, c);
    job.wrap = egui::text::TextWrapping::truncate_at_width(w);
    let g = ui.fonts(|x| x.layout_job(job));
    ui.painter().galley(p - vec2(0.0, g.size().y / 2.0), g, c);
}

fn card(ui: &Ui, rect: Rect) {
    ui.painter().rect(rect, 12.0, glass(9), Stroke::new(1.0_f32, glass(20)));
}

fn glow(ui: &Ui, c: Pos2, radius: f32, color: Color32) {
    let mut m = egui::Mesh::default();
    m.colored_vertex(c, color);
    for i in 0..=64 {
        let a = i as f32 / 64.0 * std::f32::consts::TAU;
        m.colored_vertex(c + radius * vec2(a.cos(), a.sin()), Color32::TRANSPARENT);
        if i > 0 {
            m.add_triangle(0, i, i + 1);
        }
    }
    ui.painter().add(Shape::mesh(m));
}

fn arc(ui: &Ui, c: Pos2, rad: f32, width: f32, from: f32, len: f32, color: impl Fn(f32) -> Color32) {
    if len <= 0.0 {
        return;
    }
    let at = |t: f32| {
        let a = (from + t * len - 0.25) * std::f32::consts::TAU;
        c + rad * vec2(a.cos(), a.sin())
    };
    let n = (len * 120.0).ceil().max(2.0) as usize;
    for i in 0..n {
        let (a, b) = (i as f32 / n as f32, (i as f32 + 1.3) / n as f32);
        ui.painter().line_segment([at(a), at(b.min(1.0))], Stroke::new(width, color(a)));
    }
    for t in [0.0, 1.0] {
        ui.painter().circle_filled(at(t), width / 2.0, color(t));
    }
}

fn tick(ui: &Ui, c: Pos2, s: f32, stroke: Stroke) {
    let p = [c + vec2(-s, 0.0), c + vec2(-s * 0.3, s * 0.7), c + vec2(s, -s * 0.7)];
    ui.painter().add(Shape::line(p.to_vec(), stroke));
}

fn hit(ui: &mut Ui, rect: Rect, id: &str, label: &str, enabled: bool) -> Response {
    let mut resp = ui.interact(rect, ui.id().with(id), if enabled { Sense::click() } else { Sense::hover() });
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, enabled, label));
    // egui drops `hovered` for a pressed widget when another one under it could start a drag.
    resp.hovered |= enabled && resp.is_pointer_button_down_on() && resp.contains_pointer();
    if enabled && resp.has_focus() {
        ui.painter().rect_stroke(rect.expand(3.0), 12.0, Stroke::new(1.5_f32, PINK_HOVER));
    }
    if enabled { resp.on_hover_cursor(egui::CursorIcon::PointingHand) } else { resp }
}

fn button(ui: &mut Ui, rect: Rect, label: &str, primary: bool, enabled: bool) -> bool {
    let resp = hit(ui, rect, label, label, enabled);
    let t = ui.ctx().animate_bool(resp.id, enabled && resp.hovered());
    let (fill, stroke, ink) = match (primary, enabled) {
        (true, true) => (mix(PINK, PINK_HOVER, t), Color32::TRANSPARENT, Color32::WHITE),
        (_, true) => (glass((10.0 + 14.0 * t) as u8), glass(34), INK),
        _ => (glass(8), glass(14), FAINT),
    };
    ui.painter().rect(rect, 10.0, fill, Stroke::new(1.0_f32, stroke));
    text(ui, rect.center(), Align2::CENTER_CENTER, label, semi(14.0), ink);
    resp.clicked()
}

fn link(ui: &mut Ui, p: Pos2, label: &str, color: Color32) -> (bool, f32) {
    let g = ui.painter().layout_no_wrap(label.into(), semi(13.0), color);
    let rect = Rect::from_min_size(p - vec2(0.0, g.size().y / 2.0), g.size());
    let resp = hit(ui, rect.expand2(vec2(2.0, 5.0)), label, label, true);
    if resp.hovered() {
        ui.painter().hline(rect.x_range(), rect.bottom(), Stroke::new(1.0_f32, color));
    }
    ui.painter().galley(rect.min, g, color);
    (resp.clicked(), rect.width())
}

fn spinner(ui: &Ui, c: Pos2, rad: f32) {
    let t = (ui.input(|i| i.time) * 0.9).fract() as f32;
    ui.painter().circle_stroke(c, rad, Stroke::new(2.0_f32, glass(26)));
    arc(ui, c, rad, 2.0, t, 0.28, |_| PINK_HOVER);
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
}

fn clock(s: u64) -> String {
    match s {
        0..=59 => format!("{s} s"),
        60..=3599 => format!("{} min {:02} s", s / 60, s % 60),
        _ => format!("{} h {:02} min", s / 3600, s / 60 % 60),
    }
}

fn elide_home(p: &str) -> String {
    let home = std::env::var(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).unwrap_or_default();
    match p.strip_prefix(&home) {
        Some(rest) if !home.is_empty() && rest.starts_with(['/', '\\']) => format!("~{rest}"),
        _ => p.into(),
    }
}

impl App {
    fn chrome(&mut self, ui: &mut Ui) {
        let full = ui.max_rect();
        ui.painter().rect_filled(full, 0.0, PAGE);
        glow(ui, pos2(90.0, 20.0), 420.0, Color32::from_rgba_unmultiplied(150, 60, 200, 34));
        glow(ui, pos2(900.0, 620.0), 460.0, Color32::from_rgba_unmultiplied(255, 90, 120, 26));
        // macOS draws its window buttons over the content: the whole strip moves the window.
        // Either side of the tabs, so pressing a tab can't start moving the window.
        for (id, strip) in [("bar1", r(0.0, 0.0, 280.0, 64.0)), ("bar2", r(680.0, 0.0, SIZE.x - 680.0, 64.0))] {
            let bar = ui.interact(strip, ui.id().with(id), Sense::drag());
            if cfg!(target_os = "macos") && bar.drag_started() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
        }
        let x = if cfg!(target_os = "macos") { 92.0 } else { 28.0 };
        self.logo_at(ui, pos2(x + 18.0, 32.0), 36.0);
        text(ui, pos2(x + 46.0, 32.0), Align2::LEFT_CENTER, "Leonida", bold(15.0), INK);

        let tabs = [(Tab::Home, "Play"), (Tab::Mods, "Mods"), (Tab::Build, "Build"), (Tab::Settings, "Settings")];
        let now = tabs.iter().position(|t| t.0 == self.tab).unwrap_or(0);
        let w = 96.0;
        let pill = Rect::from_center_size(pos2(SIZE.x / 2.0, 32.0), vec2(w * 4.0 + 8.0, 40.0));
        ui.painter().rect(pill, 20.0, glass(10), Stroke::new(1.0_f32, glass(22)));
        let slide = ui.ctx().animate_value_with_time(ui.id().with("pill"), now as f32, 0.18);
        ui.painter().rect_filled(r(pill.left() + 4.0 + slide * w, pill.top() + 4.0, w, 32.0), 16.0, glass(26));
        for (i, (tab, name)) in tabs.iter().enumerate() {
            let rect = r(pill.left() + 4.0 + w * i as f32, pill.top() + 4.0, w, 32.0);
            let resp = hit(ui, rect, &format!("tab{name}"), name, true);
            let ink = if i == now || resp.hovered() { INK } else { SOFT };
            text(ui, rect.center(), Align2::CENTER_CENTER, name, semi(13.0), ink);
            if *tab == Tab::Build && self.build.is_some() && i != now {
                ui.painter().circle_filled(pos2(rect.right() - 16.0, rect.center().y), 3.0, PINK_HOVER);
            }
            if resp.clicked() {
                self.tab = *tab;
            }
        }
    }

    fn logo_at(&self, ui: &Ui, c: Pos2, w: f32) {
        let s = self.logo.size_vec2();
        let rect = Rect::from_center_size(c, vec2(w, w * s.y / s.x));
        ui.painter().image(self.logo.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    }

    fn label(&self, ui: &Ui, x: f32, y: f32, s: &str) {
        text(ui, pos2(x, y), Align2::LEFT_CENTER, s, semi(12.0), SOFT);
    }

    fn files_page(&mut self, ui: &mut Ui) {
        text(ui, pos2(64.0, 100.0), Align2::LEFT_CENTER, "Build your game", bold(24.0), INK);
        let sub = "Pick your game files and where the game goes. Leonida does the rest.";
        text(ui, pos2(64.0, 128.0), Align2::LEFT_CENTER, sub, reg(14.0), SOFT);

        self.label(ui, 64.0, 168.0, "Found on this computer");
        let list = r(64.0, 184.0, 832.0, 112.0);
        card(ui, list);
        if self.games.is_empty() {
            let busy = self.searching.is_some();
            let x = if busy { spinner(ui, pos2(96.0, 240.0), 9.0); 122.0 } else { 88.0 };
            let (a, b) = if busy {
                ("Looking for your game files", "Downloads, Desktop, Documents and the top of every drive")
            } else {
                ("No game files found", "Browse to the folder that holds MicrosoftGame.Config, or drop it on this window")
            };
            text(ui, pos2(x, 230.0), Align2::LEFT_CENTER, a, semi(14.0), INK);
            text(ui, pos2(x, 251.0), Align2::LEFT_CENTER, b, reg(13.0), SOFT);
        }
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(list.shrink(6.0)));
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(&mut child, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            for (i, (dir, what)) in self.games.iter().enumerate() {
                let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 50.0), Sense::hover());
                let on = self.chosen == Some(i);
                let resp = hit(ui, rect, &format!("game{i}"), what, true);
                if on || resp.hovered() {
                    ui.painter().rect_filled(rect, 8.0, if on { PINK.gamma_multiply(0.16) } else { glass(8) });
                }
                let c = pos2(rect.left() + 22.0, rect.center().y);
                ui.painter().circle_stroke(c, 8.0, Stroke::new(1.5_f32, if on { PINK_HOVER } else { FAINT }));
                if on {
                    ui.painter().circle_filled(c, 4.0, PINK_HOVER);
                }
                let w = rect.width() - 60.0;
                line(ui, pos2(rect.left() + 44.0, c.y - 9.0), w, what, semi(14.0), INK);
                line(ui, pos2(rect.left() + 44.0, c.y + 10.0), w, &elide_home(dir), reg(12.0), SOFT);
                if resp.clicked() {
                    self.chosen = Some(i);
                }
            }
        });
        text(ui, pos2(64.0, 316.0), Align2::LEFT_CENTER, "Not listed?", reg(13.0), SOFT);
        let (browse, w) = link(ui, pos2(138.0, 316.0), "Browse", PINK_HOVER);
        text(ui, pos2(146.0 + w, 316.0), Align2::LEFT_CENTER, "or drop the folder on this window", reg(13.0), FAINT);
        if browse {
            if let Some(p) = rfd::FileDialog::new().set_title("Choose your game folder").pick_folder() {
                self.add_game(p.display().to_string());
            }
        }

        self.label(ui, 64.0, 356.0, "Put the game in");
        self.folder_field(ui, r(64.0, 372.0, 538.0, 40.0));
        self.label(ui, 728.0, 356.0, "Parallel jobs");
        let box_ = r(728.0, 372.0, 168.0, 40.0);
        ui.painter().rect(box_, 10.0, glass(9), Stroke::new(1.0_f32, glass(26)));
        text(ui, box_.center(), Align2::CENTER_CENTER, &self.jobs.to_string(), semi(14.0), INK);
        for (x, s, name) in [(728.0, "-", "Fewer jobs"), (856.0, "+", "More jobs")] {
            let rect = r(x, 372.0, 40.0, 40.0);
            let resp = hit(ui, rect, name, name, true);
            let ink = if resp.hovered() { INK } else { SOFT };
            let c = rect.center();
            ui.painter().hline(c.x - 5.0..=c.x + 5.0, c.y, Stroke::new(1.6_f32, ink));
            if s == "+" {
                ui.painter().vline(c.x, c.y - 5.0..=c.y + 5.0, Stroke::new(1.6_f32, ink));
            }
            if resp.clicked() {
                self.jobs = if s == "+" { (self.jobs + 1).min(128) } else { (self.jobs - 1).max(1) };
            }
        }

        self.label(ui, 64.0, 444.0, "Steps to run");
        let mut x = 64.0;
        for i in 1..5 {
            let name = STAGES[i].1;
            let g = ui.painter().layout_no_wrap(name.into(), semi(13.0), INK);
            let rect = r(x, 460.0, g.size().x + 46.0, 36.0);
            let resp = hit(ui, rect, &format!("stage{i}"), name, true);
            let on = self.stages[i];
            let (fill, edge) = if on { (PINK.gamma_multiply(0.18), PINK.gamma_multiply(0.6)) } else { (glass(8), glass(24)) };
            ui.painter().rect(rect, 18.0, fill, Stroke::new(1.0_f32, if resp.hovered() { PINK_HOVER } else { edge }));
            let c = pos2(rect.left() + 20.0, rect.center().y);
            if on { tick(ui, c, 5.0, Stroke::new(2.0_f32, PINK_HOVER)) } else { ui.painter().circle_stroke(c, 4.5, Stroke::new(1.5_f32, FAINT)); }
            ui.painter().galley_with_override_text_color(pos2(rect.left() + 33.0, rect.center().y - g.size().y / 2.0), g, if on { INK } else { SOFT });
            if resp.clicked() {
                self.stages[i] = !on;
            }
            x = rect.right() + 10.0;
        }

        let ready = self.root.is_some() && self.python.is_some() && self.chosen.is_some() && self.searching.is_none() && self.game.is_none();
        let any = self.stages[1..].iter().any(|s| *s);
        let (msg, ink) = if !self.note.is_empty() {
            (self.note.as_str(), ORANGE)
        } else if !any {
            ("Choose at least one step, or just check the files.", SOFT)
        } else if self.chosen.is_none() && self.searching.is_none() {
            ("Choose your game files to continue.", SOFT)
        } else {
            ("The first build takes hours and hundreds of GB. You can stop and resume.", SOFT)
        };
        let g = ui.painter().layout(msg.into(), reg(13.0), ink, 480.0);
        ui.painter().galley(pos2(64.0, 546.0 - g.size().y / 2.0), g, ink);
        if button(ui, r(578.0, 524.0, 140.0, 44.0), "Check files", false, ready) {
            self.start(true, ui.ctx());
        }
        if button(ui, r(728.0, 524.0, 168.0, 44.0), "Build the game", true, ready && any) {
            self.start(false, ui.ctx());
        }
    }

    fn building_page(&mut self, ui: &mut Ui) {
        let failed = self.page == Page::Failed;
        let (done, total) = self.progress.count();
        let c = pos2(116.5, 152.0);
        let part: f32 = self.progress.state.iter().zip(self.progress.percent).map(|(s, p)| match s {
            State::Done => 1.0,
            State::Running => p.unwrap_or(0) as f32 / 100.0 * 0.9,
            _ => 0.0,
        }).sum::<f32>() / total.max(1) as f32;
        ui.painter().circle_stroke(c, 48.0, Stroke::new(9.0_f32, glass(18)));
        let p = ui.ctx().animate_value_with_time(ui.id().with("ring"), part, 0.4);
        arc(ui, c, 48.0, 9.0, 0.0, p, |t| if failed { BAD } else { sunset(t) });
        text(ui, c - vec2(0.0, 7.0), Align2::CENTER_CENTER, &format!("{done} of {total}"), bold(18.0), INK);
        text(ui, c + vec2(0.0, 14.0), Align2::CENTER_CENTER, "steps", reg(12.0), SOFT);

        let title = match (failed, self.cancelled, self.check_only) {
            (true, ..) => "The build stopped",
            (_, true, _) => "Stopping",
            (_, _, true) => "Checking your files",
            _ => "Building your game",
        };
        text(ui, pos2(194.0, 136.0), Align2::LEFT_CENTER, title, bold(24.0), INK);
        let sub = if failed {
            if self.progress.error.is_empty() { "Open the log below to see what happened.".into() } else { self.progress.error.clone() }
        } else {
            format!("{} so far. You can keep using your computer.", clock(self.started.elapsed().as_secs()))
        };
        let mut job = egui::text::LayoutJob::simple(sub, reg(14.0), if failed { BAD } else { SOFT }, 700.0);
        job.wrap.max_rows = 2;
        let g = ui.fonts(|f| f.layout_job(job));
        ui.painter().galley(pos2(194.0, 156.0), g, SOFT);
        if !failed {
            ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
        }

        let body = r(64.0, 224.0, 832.0, 282.0);
        if self.show_log {
            ui.painter().rect(body, 12.0, Color32::from_black_alpha(90), Stroke::new(1.0_f32, glass(20)));
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body.shrink(12.0)));
            let log = &self.progress.log;
            let area = egui::ScrollArea::both().stick_to_bottom(true).auto_shrink([false, false]);
            area.show_rows(&mut child, 14.0, log.len(), |ui, rows| {
                for l in &log[rows] {
                    let ink = if l.contains("ERROR") { BAD } else if l.contains("WARNING") { ORANGE } else { SOFT };
                    let t = egui::RichText::new(l).font(FontId::monospace(12.0)).color(ink);
                    ui.add(egui::Label::new(t).extend());
                }
            });
        } else {
            card(ui, body);
            for (i, (_, name, what)) in STAGES.iter().enumerate() {
                let y = body.top() + 1.0 + 56.0 * i as f32;
                let mid = y + 28.0;
                if i > 0 {
                    ui.painter().hline(82.0..=878.0, y, Stroke::new(1.0_f32, glass(12)));
                }
                let dot = pos2(94.0, mid);
                let s = self.progress.state[i];
                match s {
                    State::Done => {
                        ui.painter().circle_filled(dot, 10.0, GO.gamma_multiply(0.22));
                        tick(ui, dot, 4.5, Stroke::new(2.0_f32, GO));
                    }
                    State::Running => spinner(ui, dot, 9.0),
                    State::Failed => {
                        ui.painter().circle_filled(dot, 10.0, BAD.gamma_multiply(0.22));
                        for d in [vec2(3.5, 3.5), vec2(3.5, -3.5)] {
                            ui.painter().line_segment([dot - d, dot + d], Stroke::new(2.0_f32, BAD));
                        }
                    }
                    State::Waiting => { ui.painter().circle_stroke(dot, 9.0, Stroke::new(1.5_f32, glass(40))); }
                    State::Skipped => { ui.painter().hline(88.0..=100.0, mid, Stroke::new(1.5_f32, FAINT)); }
                }
                let quiet = matches!(s, State::Waiting | State::Skipped);
                text(ui, pos2(120.0, mid - 9.0), Align2::LEFT_CENTER, name, semi(14.0), if quiet { FAINT } else { INK });
                let d = match s {
                    State::Running | State::Failed if !self.progress.detail[i].is_empty() => &self.progress.detail[i],
                    _ => *what,
                };
                line(ui, pos2(120.0, mid + 10.0), 560.0, d, reg(12.0), if quiet { FAINT } else { SOFT });
                let right = match (s, self.progress.percent[i]) {
                    (State::Running, Some(p)) => format!("{p}%"),
                    (State::Running, _) => "Working".into(),
                    (State::Done, _) => "Done".into(),
                    (State::Failed, _) => "Stopped".into(),
                    (State::Skipped, _) => "Not run".into(),
                    _ => String::new(),
                };
                let ink = match s { State::Done => GO, State::Failed => BAD, State::Running => INK, _ => FAINT };
                text(ui, pos2(878.0, mid), Align2::RIGHT_CENTER, &right, semi(13.0), ink);
            }
        }
        if link(ui, pos2(64.0, 546.0), if self.show_log { "Hide log" } else { "Show log" }, PINK_HOVER).0 {
            self.show_log = !self.show_log;
        }
        if failed {
            if button(ui, r(618.0, 524.0, 110.0, 44.0), "Back", false, true) {
                self.page = Page::Files;
            }
            if button(ui, r(738.0, 524.0, 158.0, 44.0), "Try again", true, true) {
                self.start(self.check_only, ui.ctx());
            }
        } else if button(ui, r(786.0, 524.0, 110.0, 44.0), "Cancel", false, !self.cancelled) {
            self.cancelled = true;
            if let Some(b) = &self.build {
                b.cancel(false);
            }
        }
    }

    fn ready_page(&mut self, ui: &mut Ui) {
        let x = SIZE.x / 2.0;
        glow(ui, pos2(x, 240.0), 250.0, Color32::from_rgba_unmultiplied(240, 70, 150, 60));
        self.logo_at(ui, pos2(x, 230.0), 230.0);
        let (title, sub) = if self.check_only {
            ("Your files are good", "Everything the build needs is there.".to_string())
        } else {
            ("Your build is finished", format!("Built in {}. It's in {}", clock(self.took), elide_home(&self.out)))
        };
        text(ui, pos2(x, 362.0), Align2::CENTER_CENTER, title, bold(26.0), INK);
        let g = ui.painter().layout(sub, reg(14.0), SOFT, 700.0);
        ui.painter().galley(pos2(x - g.size().x / 2.0, 386.0), g, SOFT);
        if self.check_only {
            if button(ui, r(x - 90.0, 450.0, 180.0, 46.0), "Continue", true, true) {
                self.page = Page::Files;
            }
        } else {
            if button(ui, r(x - 175.0, 450.0, 180.0, 46.0), "Continue", true, true) {
                (self.tab, self.page) = (Tab::Home, Page::Files);
            }
            if button(ui, r(x + 15.0, 450.0, 160.0, 46.0), "Build again", false, true) {
                self.page = Page::Files;
            }
        }
    }

    #[cfg(feature = "shots")]
    fn shots(&mut self, ctx: &egui::Context) {
        let Some(dir) = std::env::var_os("LEONIDA_SHOTS").map(PathBuf::from) else { return };
        let n = self.shot / 30;
        if self.shot % 30 == 0 {
            (self.searching, self.python, self.note) = (None, Some(vec!["python3".into()]), String::new());
            self.out = "/Users/you/Leonida".into();
            self.games = vec![("/Users/you/Games/Leonida".into(), "Example game 1.0.2.1".into()), ("/Volumes/Games/Copy".into(), "Example game 1.0.2.1".into())];
            self.chosen = Some(0);
            self.progress = Progress::new(&[true; 5]);
            for l in ["12:00:00  check: all files present", "12:00:01  shaders translate: 620/1000 (62%), ~31 min left", "12:00:02  assets: texture pool 3"] {
                self.progress.line(l.into());
            }
            
            self.took = 5120;
            self.config = play::Config::default();
            for (k, v) in [("seconds", 8_040), ("last", play::now() - 7_300), ("keyboard", 1)] {
                let _ = self.config.set(k, v);
            }
            self.port = play::Port::Ready;
            let m = |folder: &str, name: &str, enabled| play::Mod { folder: folder.into(), name: name.into(), enabled, note: String::new() };
            self.mods.list = vec![m("example_one", "Example mod", true), m("example_two", "Another example", false)];
            self.page = Page::Files;
            self.play_note.clear();
            match n {
                0 => self.port = play::Port::NotBuilt,
                1 => self.tab = Tab::Home,
                2 => self.play_note = "Closed after 42 s with code 3. The log says more.".into(),
                3 => self.tab = Tab::Mods,
                4 => self.tab = Tab::Build,
                5 => self.page = Page::Building,
                6 => self.tab = Tab::Settings,
                7 => {
                    self.log = vec!["[rt] example line".into()];
                    self.log_read = None;
                    self.log_path = PathBuf::new();
                    self.tab = Tab::Log;
                }
                _ => return ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            }
        }
        self.port_checked = Some(std::time::Instant::now());
        if self.shot % 30 == 24 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
        }
        for e in ctx.input(|i| i.events.clone()) {
            if let egui::Event::Screenshot { image: im, .. } = e {
                let bytes: Vec<u8> = im.pixels.iter().flat_map(|p| p.to_array()).collect();
                let _ = image::save_buffer(dir.join(format!("{n}.png")), &bytes, im.size[0] as u32, im.size[1] as u32, image::ColorType::Rgba8);
            }
        }
        self.shot += 1;
        ctx.request_repaint_after(std::time::Duration::from_millis(30));
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll(ctx);
        #[cfg(feature = "shots")]
        self.shots(ctx);
        if self.tab == Tab::Build && self.page == Page::Files {
            let dropped = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone()));
            if let Some(p) = dropped {
                let dir = if p.is_dir() { Some(p) } else { p.parent().map(PathBuf::from) };
                if let Some(d) = dir {
                    self.add_game(d.display().to_string());
                }
            }
        }
        egui::CentralPanel::default().frame(egui::Frame::none()).show(ctx, |ui| {
            self.chrome(ui);
            match (self.tab, self.page) {
                (Tab::Home, _) => self.home_page(ui),
                (Tab::Log, _) => self.log_page(ui),
                (Tab::Mods, _) => self.mods_page(ui),
                (Tab::Settings, _) => self.settings_page(ui),
                (_, Page::Files) => self.files_page(ui),
                (_, Page::Building | Page::Failed) => self.building_page(ui),
                (_, Page::Ready) => self.ready_page(ui),
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(b) = &self.build {
            b.cancel(true);
        }
    }
}

fn main() -> eframe::Result<()> {
    let mut vp = egui::ViewportBuilder::default()
        .with_inner_size(SIZE)
        .with_resizable(false)
        .with_maximize_button(false);
    if cfg!(target_os = "macos") {
        vp = vp.with_fullsize_content_view(true).with_titlebar_shown(false).with_title_shown(false);
    }
    if let Ok(img) = image::load_from_memory(include_bytes!("../../assets/logo.png")) {
        let rgba = img.resize(256, 256, image::imageops::FilterType::Lanczos3).to_rgba8();
        let (width, height) = rgba.dimensions();
        vp = vp.with_icon(egui::IconData { rgba: rgba.into_raw(), width, height });
    }
    let options = eframe::NativeOptions { viewport: vp, centered: true, ..Default::default() };
    eframe::run_native("Leonida", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cursor on each frame while the pointer hovers, presses, holds and releases on a button.
    fn cursors(at: Pos2) -> Vec<egui::CursorIcon> {
        let ctx = egui::Context::default();
        let press = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        let move_ = || vec![egui::Event::PointerMoved(at)];
        let steps = [move_(), vec![], vec![], vec![press(true)], vec![], vec![], vec![press(false)], vec![], vec![]];
        let mut out = vec![];
        for (i, events) in steps.into_iter().enumerate() {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), SIZE)),
                time: Some(i as f64 * 0.05),
                events,
                ..Default::default()
            };
            let o = ctx.run(input, |ctx| {
                egui::CentralPanel::default().frame(egui::Frame::none()).show(ctx, |ui| {
                    ui.interact(r(0.0, 0.0, SIZE.x, 64.0), ui.id().with("bar"), Sense::drag());
                    hit(ui, r(400.0, 16.0, 96.0, 32.0), "tab", "tab", true);
                    hit(ui, r(64.0, 296.0, 190.0, 56.0), "button", "button", true);
                });
            });
            out.push(o.platform_output.cursor_icon);
        }
        out
    }

    #[test]
    fn the_hand_stays_while_clicking() {
        for at in [pos2(440.0, 32.0), pos2(150.0, 320.0)] {
            let c = cursors(at);
            assert!(c[2..].iter().all(|c| *c == egui::CursorIcon::PointingHand), "{at:?}: {c:?}");
        }
    }
}
