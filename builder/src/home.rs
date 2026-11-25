// The Play, Mods and Settings pages.
use super::*;
use crate::play::{self, Port};

fn ago(then: u64) -> String {
    match play::now().saturating_sub(then) {
        _ if then == 0 => "Never".into(),
        0..=59 => "Just now".into(),
        s @ 60..=3599 => format!("{} min ago", s / 60),
        s @ 3600..=86399 => format!("{} h ago", s / 3600),
        s if s < 172_800 => "Yesterday".into(),
        s => format!("{} days ago", s / 86400),
    }
}

fn chip(ui: &Ui, p: Pos2, label: &str, color: Color32) {
    let g = ui.painter().layout_no_wrap(label.into(), semi(12.0), color);
    let rect = Rect::from_min_size(p, vec2(g.size().x + 38.0, 28.0));
    ui.painter().rect(rect, 14.0, color.gamma_multiply(0.13), Stroke::new(1.0_f32, color.gamma_multiply(0.35)));
    ui.painter().circle_filled(pos2(rect.left() + 15.0, rect.center().y), 3.5, color);
    ui.painter().galley(pos2(rect.left() + 26.0, rect.center().y - g.size().y / 2.0), g, color);
}

pub fn toggle(ui: &mut Ui, rect: Rect, id: &str, label: &str, on: bool, enabled: bool) -> bool {
    let resp = hit(ui, rect, id, label, enabled);
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, enabled, on, label));
    let t = ui.ctx().animate_bool(resp.id, on);
    let c = Rect::from_center_size(rect.center(), vec2(40.0, 24.0));
    ui.painter().rect_filled(c, 12.0, mix(Color32::from_rgb(58, 54, 68), PINK, t));
    ui.painter().circle_filled(pos2(c.left() + 12.0 + 16.0 * t, c.center().y), 9.0, Color32::WHITE);
    resp.clicked()
}

fn log_view(ui: &mut Ui, body: Rect, log: &[String], empty: &str) {
    ui.painter().rect(body, 12.0, Color32::from_black_alpha(120), Stroke::new(1.0_f32, glass(20)));
    if log.is_empty() {
        text(ui, body.center(), Align2::CENTER_CENTER, empty, reg(13.0), FAINT);
    }
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body.shrink(12.0)));
    let area = egui::ScrollArea::both().stick_to_bottom(true).auto_shrink([false, false]);
    area.show_rows(&mut child, 14.0, log.len(), |ui, rows| {
        for l in &log[rows] {
            let ink = if l.contains("ERROR") { BAD } else if l.contains("WARNING") { ORANGE } else { SOFT };
            ui.add(egui::Label::new(egui::RichText::new(l).font(FontId::monospace(12.0)).color(ink)).extend());
        }
    });
}

impl App {
    pub fn save(&mut self, key: &str, value: impl ToString) {
        if let Err(e) = self.config.set(key, value) {
            self.play_note = format!("Settings couldn't be saved: {e}");
        }
    }

    pub fn poll_game(&mut self) {
        if let Some((code, secs)) = self.game.as_ref().and_then(|g| g.ended().map(|c| (c, g.started.elapsed().as_secs()))) {
            self.game = None;
            let total = self.config.num("seconds") + secs;
            self.save("seconds", total);
            self.save("last", play::now());
            self.play_note = match code {
                _ if self.stopping => "Stopped.".into(),
                Some(0) => String::new(),
                Some(127) if !cfg!(windows) => "Wine wasn't found. Install it, or choose it in Settings.".into(),
                Some(c) => format!("Closed after {} with code {c}. The log says more.", clock(secs)),
                None => format!("Closed unexpectedly after {}. The log says more.", clock(secs)),
            };
            self.stopping = false;
        }
        if let Some(m) = self.mods_rx.as_ref().and_then(|r| r.try_recv().ok()) {
            self.mods_rx = None;
            self.mods = m;
        }
    }

    fn play(&mut self, ctx: &egui::Context) {
        let o = play::Options {
            keyboard: self.config.on("keyboard"),
            splash: !self.config.on("no_splash"),
            diagnostics: self.config.on("diagnostics"),
            wine: self.config.get("wine").into(),
        };
        let c = ctx.clone();
        let run = play::command(std::path::Path::new(self.out.trim()), &o)
            .ok_or_else(|| std::io::Error::other("game_path.txt is missing"))
            .and_then(|cmd| play::start(cmd, &self.log_path, move || c.request_repaint()));
        self.play_note.clear();
        match run {
            Ok(g) => self.game = Some(g),
            Err(e) => self.play_note = format!("It couldn't start: {e}"),
        }
    }

    pub fn load_mods(&mut self, args: Vec<String>, ctx: &egui::Context) {
        let (Some(py), Some(root)) = (self.python.clone(), self.root.clone()) else { return };
        if self.mods_rx.is_some() {
            return;
        }
        let (tx, rx) = channel();
        self.mods_rx = Some(rx);
        let (out, c) = (self.out.trim().to_string(), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(play::mods(&py, &root, &out, &args));
            c.request_repaint();
        });
    }

    pub fn home_page(&mut self, ui: &mut Ui) {
        let running = self.game.is_some();
        let port = self.port.clone();
        let c = pos2(690.0, 250.0);
        // A slow breath, only while the game runs, so an idle launcher draws nothing.
        let beat = if running {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
            (ui.input(|i| i.time) * 1.6).sin() as f32
        } else {
            0.0
        };
        glow(ui, c, 300.0 + 14.0 * beat, Color32::from_rgba_unmultiplied(240, 70, 150, 70));
        glow(ui, c + vec2(90.0, 60.0), 220.0, Color32::from_rgba_unmultiplied(255, 150, 80, 36));
        self.logo_at(ui, c + vec2(0.0, 5.0 * beat), 400.0);

        let (label, color) = match (&port, running) {
            (_, true) => ("Running", GO),
            (Port::Ready, _) => ("Ready to start", GO),
            (Port::Missing(_), _) => ("File missing", BAD),
            _ if self.build.is_some() => ("Building", PINK_HOVER),
            _ => ("Not built yet", ORANGE),
        };
        chip(ui, pos2(64.0, 116.0), label, color);
        text(ui, pos2(62.0, 188.0), Align2::LEFT_CENTER, "Leonida", bold(58.0), INK);
        let sub = match (&port, running) {
            (_, true) => format!("Running for {}.", clock(self.game.as_ref().map_or(0, |g| g.started.elapsed().as_secs()))),
            (Port::Missing(f), _) => format!("{f} is missing, so there is nothing to start. Check your game files, then build again."),
            (Port::NotBuilt, _) => "Build the port from your own copy of the game, once. Then start it here.".into(),
            _ => "An early port: it isn't playable yet. Start it to see how far it gets.".into(),
        };
        let g = ui.painter().layout(sub, reg(15.0), SOFT, 350.0);
        ui.painter().galley(pos2(64.0, 226.0), g, SOFT);

        let main = r(64.0, 296.0, 190.0, 56.0);
        if running {
            if button(ui, main, if self.stopping { "Stopping" } else { "Stop" }, false, !self.stopping) {
                self.stopping = true;
                if let Some(g) = &self.game {
                    g.stop();
                }
            }
        } else if port == Port::Ready {
            if button(ui, main, "Play", true, self.build.is_none()) {
                self.play(ui.ctx());
            }
        } else if button(ui, main, if self.build.is_some() { "See the build" } else { "Build the game" }, true, true) {
            self.tab = Tab::Build;
        }
        let has = std::path::Path::new(self.out.trim()).is_dir();
        if button(ui, r(266.0, 296.0, 130.0, 56.0), "Open folder", false, has) {
            backend::open_folder(self.out.trim());
        }
        let g = ui.painter().layout(self.play_note.clone(), reg(13.0), ORANGE, 350.0);
        ui.painter().galley(pos2(64.0, 366.0), g, ORANGE);

        let on = self.mods.list.iter().filter(|m| m.enabled).count();
        let time = match self.config.num("seconds") { 0 => "None yet".into(), s => clock(s) };
        let last = if running { "Now".into() } else { ago(self.config.num("last")) };
        let cards = [("Last started", last), ("Time running", time), ("Mods on", format!("{on} of {}", self.mods.list.len()))];
        for (i, (k, v)) in cards.iter().enumerate() {
            let x = 64.0 + 281.0 * i as f32;
            card(ui, r(x, 434.0, 270.0, 76.0));
            text(ui, pos2(x + 18.0, 458.0), Align2::LEFT_CENTER, k, semi(12.0), SOFT);
            line(ui, pos2(x + 18.0, 484.0), 234.0, v, bold(17.0), INK);
        }
        line(ui, pos2(64.0, 546.0), 640.0, &elide_home(&self.out), reg(13.0), FAINT);
        let (go, _) = link(ui, pos2(838.0, 546.0), "Show log", PINK_HOVER);
        if go {
            self.tab = Tab::Log;
        }
        if running {
            ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
        }
    }

    pub fn log_page(&mut self, ui: &mut Ui) {
        text(ui, pos2(64.0, 100.0), Align2::LEFT_CENTER, "Game log", bold(24.0), INK);
        text(ui, pos2(64.0, 128.0), Align2::LEFT_CENTER, "From the last start. It can quote the game, so don't post it publicly.", reg(14.0), SOFT);
        if self.log_read.is_none_or(|t| t.elapsed().as_millis() > 700) {
            self.log = play::tail(&self.log_path);
            self.log_read = Some(std::time::Instant::now());
        }
        log_view(ui, r(64.0, 156.0, 832.0, 350.0), &self.log, "Nothing has been started yet");
        if self.game.is_some() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(750));
        }
        if button(ui, r(786.0, 524.0, 110.0, 44.0), "Back", false, true) {
            self.tab = Tab::Home;
        }
    }

    pub fn mods_page(&mut self, ui: &mut Ui) {
        text(ui, pos2(64.0, 100.0), Align2::LEFT_CENTER, "Mods", bold(24.0), INK);
        text(ui, pos2(64.0, 128.0), Align2::LEFT_CENTER, "Changes apply the next time you start. Your game files are never touched.", reg(14.0), SOFT);
        let busy = self.mods_rx.is_some();
        let usable = self.port != Port::NotBuilt && self.python.is_some() && !busy;
        if busy {
            spinner(ui, pos2(748.0, 112.0), 9.0);
        }
        if button(ui, r(776.0, 92.0, 120.0, 40.0), "Add a mod", true, usable) {
            let pick = rfd::FileDialog::new().set_title("Choose a mod package").add_filter("Mod packages", &["zip", "leonidamod"]).pick_file();
            if let Some(p) = pick {
                self.load_mods(vec!["install".into(), "--disable".into(), p.display().to_string()], ui.ctx());
            }
        }
        let body = r(64.0, 156.0, 832.0, 350.0);
        card(ui, body);
        if self.mods.list.is_empty() {
            self.logo_at(ui, body.center() - vec2(0.0, 52.0), 90.0);
            let (a, b) = if self.port == Port::NotBuilt {
                ("Build the game first", "Mods are kept next to the finished build.")
            } else {
                ("No mods yet", "Add a .zip or .leonidamod package to get started.")
            };
            text(ui, body.center() + vec2(0.0, 14.0), Align2::CENTER_CENTER, a, semi(16.0), INK);
            text(ui, body.center() + vec2(0.0, 38.0), Align2::CENTER_CENTER, b, reg(13.0), SOFT);
        }
        let mut change = None;
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body.shrink(6.0)));
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(&mut child, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            for m in &self.mods.list {
                let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 60.0), Sense::hover());
                let y = rect.center().y;
                line(ui, pos2(rect.left() + 16.0, y - 9.0), 640.0, &m.name, semi(14.0), INK);
                let (note, ink) = if m.note.is_empty() { (&m.folder, SOFT) } else { (&m.note, ORANGE) };
                line(ui, pos2(rect.left() + 16.0, y + 11.0), 640.0, note, reg(12.0), ink);
                let sw = Rect::from_center_size(pos2(rect.right() - 36.0, y), vec2(48.0, 32.0));
                if toggle(ui, sw, &m.folder, &m.name, m.enabled, usable) {
                    change = Some(vec![if m.enabled { "disable" } else { "enable" }.to_string(), m.folder.clone()]);
                }
            }
        });
        if let Some(a) = change {
            self.load_mods(a, ui.ctx());
        }
        line(ui, pos2(64.0, 546.0), 830.0, &self.mods.error, reg(13.0), ORANGE);
    }

    pub fn settings_page(&mut self, ui: &mut Ui) {
        text(ui, pos2(64.0, 100.0), Align2::LEFT_CENTER, "Settings", bold(24.0), INK);
        text(ui, pos2(64.0, 128.0), Align2::LEFT_CENTER, "Saved as you change them.", reg(14.0), SOFT);
        self.label(ui, 64.0, 168.0, "Game folder");
        if self.folder_field(ui, r(64.0, 184.0, 730.0, 40.0)) {
            self.port_checked = None;
        }
        let rows = [
            ("keyboard", false, "Keyboard as a controller", "The game reads gamepads only; this maps your keys to one"),
            ("no_splash", true, "Splash screen", "Show the package's splash while the game starts"),
            ("diagnostics", false, "Graphics diagnostics", "Validation and device-loss reports in the log; slower, needs the Vulkan layers"),
        ];
        card(ui, r(64.0, 248.0, 832.0, 180.0));
        for (i, (key, inverted, name, what)) in rows.iter().enumerate() {
            let mid = 278.0 + 60.0 * i as f32;
            if i > 0 {
                ui.painter().hline(82.0..=878.0, mid - 30.0, Stroke::new(1.0_f32, glass(12)));
            }
            text(ui, pos2(84.0, mid - 9.0), Align2::LEFT_CENTER, name, semi(14.0), INK);
            text(ui, pos2(84.0, mid + 11.0), Align2::LEFT_CENTER, what, reg(12.0), SOFT);
            let on = self.config.on(key) != *inverted;
            if toggle(ui, Rect::from_center_size(pos2(850.0, mid), vec2(48.0, 32.0)), key, name, on, true) {
                let v = on == *inverted;
                self.save(key, v as u8);
            }
        }
        if !cfg!(windows) {
            self.label(ui, 64.0, 456.0, "Wine command");
            let field = r(64.0, 472.0, 730.0, 40.0);
            ui.painter().rect(field, 10.0, Color32::from_black_alpha(60), Stroke::new(1.0_f32, glass(26)));
            let mut wine = self.config.get("wine").to_string();
            let edit = egui::TextEdit::singleline(&mut wine).frame(false).font(reg(14.0))
                .vertical_align(egui::Align::Center).hint_text("wine");
            if ui.put(field.shrink2(vec2(10.0, 4.0)), edit).changed() {
                self.save("wine", wine);
            }
            if button(ui, r(804.0, 472.0, 92.0, 40.0), "Browse", false, true) {
                if let Some(p) = rfd::FileDialog::new().set_title("Choose Wine").pick_file() {
                    self.save("wine", p.display());
                }
            }
        }
        line(ui, pos2(64.0, 546.0), 830.0, &self.play_note, reg(13.0), ORANGE);
    }

    /// The output folder field with its Change button; true when it changed.
    pub fn folder_field(&mut self, ui: &mut Ui, field: Rect) -> bool {
        let before = self.out.clone();
        ui.painter().rect(field, 10.0, Color32::from_black_alpha(60), Stroke::new(1.0_f32, glass(26)));
        let locked = self.build.is_some() || self.game.is_some();
        let edit = egui::TextEdit::singleline(&mut self.out).frame(false).font(reg(14.0)).vertical_align(egui::Align::Center);
        ui.add_enabled_ui(!locked, |ui| ui.put(field.shrink2(vec2(10.0, 4.0)), edit));
        if button(ui, r(field.right() + 10.0, field.top(), 92.0, 40.0), "Change", false, !locked) {
            if let Some(p) = rfd::FileDialog::new().set_title("Choose where the game goes").pick_folder() {
                self.out = p.display().to_string();
            }
        }
        if self.out == before {
            return false;
        }
        let v = self.out.trim().to_string();
        self.save("out", v);
        true
    }
}
