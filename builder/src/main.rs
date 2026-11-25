// Leonida Builder: a desktop window around `make_game.py --no-tui`.
#![cfg_attr(all(not(debug_assertions), windows), windows_subsystem = "windows")]

use eframe::egui::{self, Color32, RichText};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

const STAGES: [&str; 5] = ["check", "shaders", "assets", "runtime", "pipelines"];
const ACCENT: Color32 = Color32::from_rgb(255, 99, 132);

enum Msg {
    Line(String),
    Done(Option<i32>),
}

struct App {
    root: Option<PathBuf>,
    game: String,
    out: String,
    jobs: u32,
    stages: [bool; 5],
    log: Vec<String>,
    rx: Option<Receiver<Msg>>,
    child: Arc<Mutex<Option<Child>>>,
    status: String,
    current_stage: Option<usize>,
}

fn find_root() -> Option<PathBuf> {
    let mut starts = vec![std::env::current_dir().ok()];
    starts.push(std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)));
    for s in starts.into_iter().flatten() {
        let mut d: &Path = &s;
        loop {
            if d.join("make_game.py").is_file() {
                return Some(d.to_path_buf());
            }
            match d.parent() {
                Some(p) => d = p,
                None => break,
            }
        }
    }
    None
}

fn default_out() -> String {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from);
    match home {
        Some(h) => {
            let docs = h.join("Documents");
            (if docs.is_dir() { docs } else { h }).join("Leonida").display().to_string()
        }
        None => "Leonida".into(),
    }
}

fn python_candidates() -> Vec<Vec<&'static str>> {
    if cfg!(windows) {
        vec![vec!["py", "-3"], vec!["python"]]
    } else {
        vec![vec!["python3.14"], vec!["python3.13"], vec!["python3.12"], vec!["python3"], vec!["python"]]
    }
}

fn find_python() -> Option<Vec<&'static str>> {
    python_candidates().into_iter().find(|c| {
        let mut cmd = Command::new(c[0]);
        cmd.args(&c[1..]).args(["-c", "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"]);
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        hide_window(&mut cmd);
        cmd.status().map(|s| s.success()).unwrap_or(false)
    })
}

#[cfg(windows)]
fn hide_window(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x0800_0000);
}
#[cfg(not(windows))]
fn hide_window(_: &mut Command) {}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut v = egui::Visuals::dark();
        v.selection.bg_fill = ACCENT;
        v.widgets.hovered.bg_stroke.color = ACCENT;
        v.window_rounding = 10.0.into();
        cc.egui_ctx.set_visuals(v);
        cc.egui_ctx.style_mut(|s| {
            s.spacing.item_spacing = egui::vec2(10.0, 10.0);
            s.spacing.button_padding = egui::vec2(14.0, 7.0);
        });
        Self {
            root: find_root(),
            game: String::new(),
            out: default_out(),
            jobs: (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2) / 2).max(1) as u32,
            stages: [true; 5],
            log: vec![],
            rx: None,
            child: Arc::new(Mutex::new(None)),
            status: "Ready".into(),
            current_stage: None,
        }
    }

    fn running(&self) -> bool {
        self.rx.is_some()
    }

    fn start(&mut self, check_only: bool, ctx: &egui::Context) {
        let Some(root) = self.root.clone() else {
            self.status = "make_game.py not found: run this from the Leonida folder".into();
            return;
        };
        let Some(py) = find_python() else {
            self.status = "Python 3.12 or newer is needed (python.org/downloads)".into();
            return;
        };
        let mut cmd = Command::new(py[0]);
        cmd.args(&py[1..]).args(["-I", "-u"]).arg(root.join("make_game.py"));
        cmd.args(["--no-tui", "--jobs", &self.jobs.to_string()]);
        if !self.game.trim().is_empty() {
            cmd.args(["--game", self.game.trim()]);
        }
        if !self.out.trim().is_empty() {
            cmd.args(["--out", self.out.trim()]);
        }
        if check_only {
            cmd.arg("--check");
        } else if self.stages.iter().any(|s| !s) {
            let list: Vec<_> = STAGES.iter().zip(self.stages).filter(|(_, on)| *on).map(|(s, _)| *s).collect();
            if list.is_empty() {
                self.status = "Tick at least one stage".into();
                return;
            }
            cmd.args(["--stages", &list.join(",")]);
        }
        cmd.current_dir(&root).stdout(Stdio::piped()).stderr(Stdio::piped());
        hide_window(&mut cmd);
        self.log.clear();
        self.current_stage = None;
        match cmd.spawn() {
            Err(e) => self.status = format!("Could not start: {e}"),
            Ok(mut child) => {
                let out = child.stdout.take().unwrap();
                let err = child.stderr.take().unwrap();
                *self.child.lock().unwrap() = Some(child);
                let (tx, rx) = channel();
                self.rx = Some(rx);
                self.status = "Working...".into();
                let pump = |r: Box<dyn std::io::Read + Send>, tx: Sender<Msg>, ctx: egui::Context| {
                    std::thread::spawn(move || {
                        for l in BufReader::new(r).lines().map_while(Result::ok) {
                            let _ = tx.send(Msg::Line(l));
                            ctx.request_repaint();
                        }
                    })
                };
                let h1 = pump(Box::new(out), tx.clone(), ctx.clone());
                let h2 = pump(Box::new(err), tx.clone(), ctx.clone());
                let handle = self.child.clone();
                let ctx2 = ctx.clone();
                std::thread::spawn(move || {
                    let _ = h1.join();
                    let _ = h2.join();
                    let code = handle.lock().unwrap().take().and_then(|mut c| c.wait().ok()).and_then(|s| s.code());
                    let _ = tx.send(Msg::Done(code));
                    ctx2.request_repaint();
                });
            }
        }
    }

    fn stop(&mut self) {
        if let Some(c) = self.child.lock().unwrap().as_mut() {
            let _ = c.kill();
        }
    }

    fn drain(&mut self) {
        let Some(rx) = &self.rx else { return };
        let mut done = None;
        while let Ok(m) = rx.try_recv() {
            match m {
                Msg::Line(l) => {
                    let low = l.to_lowercase();
                    for (i, s) in STAGES.iter().enumerate() {
                        if low.contains(&format!("stage {s}")) || low.starts_with(&format!("[{s}]")) {
                            self.current_stage = Some(i);
                        }
                    }
                    self.log.push(l);
                }
                Msg::Done(c) => done = Some(c),
            }
        }
        if let Some(c) = done {
            self.rx = None;
            self.status = match c {
                Some(0) => "Done".into(),
                Some(n) => format!("Stopped (exit code {n})"),
                None => "Stopped".into(),
            };
        }
    }
}

fn folder_row(ui: &mut egui::Ui, label: &str, hint: &str, value: &mut String) {
    ui.label(RichText::new(label).strong());
    ui.horizontal(|ui| {
        let w = ui.available_width() - 100.0;
        ui.add(egui::TextEdit::singleline(value).hint_text(hint).desired_width(w));
        if ui.button("Browse").clicked() {
            if let Some(p) = rfd::FileDialog::new().pick_folder() {
                *value = p.display().to_string();
            }
        }
    });
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.drain();
        let running = self.running();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("LEONIDA").size(30.0).strong().color(ACCENT));
                ui.label(RichText::new("Game Builder").size(18.0).weak());
            });
            ui.separator();
            ui.add_enabled_ui(!running, |ui| {
                folder_row(ui, "Game folder", "Leave empty to find it automatically", &mut self.game);
                folder_row(ui, "Output folder", "", &mut self.out);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Stages").strong());
                    for (i, s) in STAGES.iter().enumerate() {
                        ui.checkbox(&mut self.stages[i], *s);
                    }
                    ui.separator();
                    ui.label("Jobs");
                    ui.add(egui::DragValue::new(&mut self.jobs).range(1..=128));
                });
            });
            if running {
                if let Some(i) = self.current_stage {
                    ui.add(egui::ProgressBar::new((i as f32 + 0.5) / STAGES.len() as f32)
                        .text(format!("{} ({}/{})", STAGES[i], i + 1, STAGES.len())).animate(true));
                } else {
                    ui.add(egui::ProgressBar::new(0.0).text("Starting...").animate(true));
                }
            }
            ui.horizontal(|ui| {
                if ui.add_enabled(!running, egui::Button::new(RichText::new("Build").strong()).fill(ACCENT)).clicked() {
                    self.start(false, ctx);
                }
                if ui.add_enabled(!running, egui::Button::new("Check game files")).clicked() {
                    self.start(true, ctx);
                }
                if ui.add_enabled(running, egui::Button::new("Stop")).clicked() {
                    self.stop();
                }
                ui.label(RichText::new(&self.status).weak());
            });
            egui::Frame::none().fill(Color32::from_rgb(18, 18, 22)).rounding(8.0).inner_margin(8.0).show(ui, |ui| {
                egui::ScrollArea::both().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                    for l in &self.log {
                        let low = l.to_lowercase();
                        let c = if low.contains("error") || low.contains("failed") {
                            Color32::from_rgb(255, 120, 120)
                        } else if low.contains("warn") {
                            Color32::from_rgb(240, 200, 100)
                        } else {
                            Color32::LIGHT_GRAY
                        };
                        ui.label(RichText::new(l).monospace().color(c));
                    }
                });
            });
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop();
    }
}

fn main() -> eframe::Result<()> {
    let mut vp = egui::ViewportBuilder::default().with_inner_size([860.0, 620.0]).with_min_inner_size([600.0, 420.0]);
    if let Ok(img) = image::load_from_memory(include_bytes!("../../assets/logo.png")) {
        let rgba = img.resize(256, 256, image::imageops::FilterType::Lanczos3).to_rgba8();
        let (w, h) = rgba.dimensions();
        vp = vp.with_icon(egui::IconData { rgba: rgba.into_raw(), width: w, height: h });
    }
    eframe::run_native(
        "Leonida Builder",
        eframe::NativeOptions { viewport: vp, ..Default::default() },
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
