// Starting the built port, the launcher's own settings, and the mod manager's command line. No UI in here.
#[cfg(windows)]
use crate::backend::quiet;
use crate::backend::{kill_tree, python};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

pub fn config_dir() -> PathBuf {
    let var = |n| std::env::var_os(n).filter(|v| !v.is_empty()).map(PathBuf::from);
    let home = || var("HOME").unwrap_or_default();
    if cfg!(windows) {
        var("APPDATA").unwrap_or_default().join("Leonida Launcher")
    } else if cfg!(target_os = "macos") {
        home().join("Library/Application Support/Leonida Launcher")
    } else {
        var("XDG_CONFIG_HOME").unwrap_or_else(|| home().join(".config")).join("leonida-launcher")
    }
}

/// `key=value` lines. Unknown keys are kept, so an older launcher doesn't drop a newer one's settings.
#[derive(Default)]
pub struct Config {
    path: PathBuf,
    map: BTreeMap<String, String>,
}

impl Config {
    pub fn load(path: PathBuf) -> Self {
        let text = fs::read_to_string(&path).unwrap_or_default();
        let map = text.lines().filter_map(|l| l.split_once('=')).map(|(k, v)| (k.into(), v.into())).collect();
        Self { path, map }
    }
    pub fn get(&self, k: &str) -> &str {
        self.map.get(k).map_or("", String::as_str)
    }
    pub fn num(&self, k: &str) -> u64 {
        self.get(k).parse().unwrap_or(0)
    }
    pub fn on(&self, k: &str) -> bool {
        self.get(k) == "1"
    }
    pub fn set(&mut self, k: &str, v: impl ToString) -> std::io::Result<()> {
        self.map.insert(k.into(), v.to_string().replace(['\n', '\r'], " "));
        if self.path.as_os_str().is_empty() {
            return Ok(()); // in memory only
        }
        if let Some(d) = self.path.parent() {
            fs::create_dir_all(d)?;
        }
        let text: String = self.map.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
        let part = self.path.with_extension("part");
        fs::write(&part, text)?;
        fs::rename(part, &self.path)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Port {
    NotBuilt,
    /// Built, but this file is missing.
    Missing(String),
    Ready,
}

const EXE: &str = "game_scarlett_master.exe"; // tools/leonida_mods.py, EXPECTED_EXE

fn game_dir(out: &Path) -> Option<PathBuf> {
    let s = fs::read_to_string(out.join("game_path.txt")).ok()?;
    let s = s.trim_end_matches(['\r', '\n']);
    (!s.is_empty()).then(|| s.into())
}

fn starter(out: &Path) -> PathBuf {
    out.join(if cfg!(windows) { "gta6rt.exe" } else { "gta6rt.sh" })
}

pub fn port(out: &str) -> Port {
    if out.trim().is_empty() {
        return Port::NotBuilt;
    }
    let out = Path::new(out.trim());
    if !starter(out).is_file() || !out.join("gta6rt.exe").is_file() {
        return Port::NotBuilt;
    }
    match game_dir(out) {
        None => Port::Missing("game_path.txt".into()),
        Some(d) if !d.join(EXE).is_file() => Port::Missing(EXE.into()),
        Some(_) => Port::Ready,
    }
}

pub struct Options {
    pub keyboard: bool,
    pub splash: bool,
    pub diagnostics: bool,
    pub wine: String,
}

/// The documented start (docs/boot.md, section 2). On Linux and macOS gta6rt.sh finds Wine and adds the overlay itself.
pub fn command(out: &Path, o: &Options) -> Option<Command> {
    let game = game_dir(out)?;
    let mut c = Command::new(starter(out));
    c.arg("--run");
    if cfg!(windows) {
        c.arg(game.join(EXE)).arg("--game-dir").arg(&game);
        c.arg("--overlay").arg(out.join("overlay")).arg("--mods").arg(out.join("mods"));
    } else if !o.wine.trim().is_empty() {
        c.env("WINE", o.wine.trim());
    }
    if std::env::var_os("LEONIDA_D3D12").is_none() {
        c.env("LEONIDA_D3D12", "vulkan");
    }
    if o.keyboard {
        c.env("LEONIDA_KEYBOARD_PAD", "1");
    }
    if !o.splash {
        c.env("LEONIDA_PACKAGE_SPLASH", "0");
    }
    if o.diagnostics {
        c.env("LEONIDA_DEVICE_FAULT", "1").env("LEONIDA_VK_VALIDATION", "1");
    }
    c.current_dir(out);
    Some(c)
}

pub struct Game {
    pub started: Instant,
    done: Receiver<Option<i32>>,
    pid: u32,
    alive: Arc<Mutex<bool>>,
}

/// Its output goes to a file, so the game carries on when the launcher is closed.
pub fn start(mut c: Command, log: &Path, wake: impl Fn() + Send + 'static) -> std::io::Result<Game> {
    if let Some(d) = log.parent() {
        fs::create_dir_all(d)?;
    }
    let f = fs::File::create(log)?;
    c.stdin(Stdio::null()).stdout(f.try_clone()?).stderr(f);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        c.process_group(0);
    }
    #[cfg(windows)]
    quiet(&mut c);
    let mut child = c.spawn()?;
    let (pid, alive) = (child.id(), Arc::new(Mutex::new(true)));
    let (tx, done) = channel();
    let flag = alive.clone();
    std::thread::spawn(move || {
        let code = child.wait().ok().and_then(|s| s.code());
        *flag.lock().unwrap() = false;
        let _ = tx.send(code);
        wake();
    });
    Ok(Game { started: Instant::now(), done, pid, alive })
}

impl Game {
    pub fn ended(&self) -> Option<Option<i32>> {
        self.done.try_recv().ok()
    }
    pub fn stop(&self) {
        kill_tree(self.pid, self.alive.clone(), false);
    }
}

/// The last part of the log, as lines.
pub fn tail(log: &Path) -> Vec<String> {
    const MAX: u64 = 128 * 1024;
    let read = || -> std::io::Result<Vec<u8>> {
        let mut f = fs::File::open(log)?;
        let n = f.metadata()?.len();
        f.seek(SeekFrom::Start(n.saturating_sub(MAX)))?;
        let mut b = vec![];
        f.take(MAX).read_to_end(&mut b)?;
        Ok(b)
    };
    String::from_utf8_lossy(&read().unwrap_or_default()).lines().map(String::from).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mod {
    pub folder: String,
    pub name: String,
    pub enabled: bool,
    pub note: String,
}

#[derive(Default)]
pub struct Mods {
    pub list: Vec<Mod>,
    pub error: String,
}

/// Runs `leonida_mods.py <args>` (none: nothing), then reads the list again.
pub fn mods(py: &[String], root: &Path, out: &str, args: &[String]) -> Mods {
    let tool = |args: &[String]| {
        let mut c = python(py, root);
        c.arg(root.join("tools").join("leonida_mods.py")).args(["--port", out]).args(args).output()
    };
    let fault = |o: std::io::Result<std::process::Output>| match o {
        Ok(o) if o.status.success() => Ok(o),
        Ok(o) => {
            let e = String::from_utf8_lossy(&o.stderr);
            let e = e.trim().lines().last().unwrap_or("The mod manager stopped.");
            Err(e.trim_start_matches("ERROR: ").to_string())
        }
        Err(e) => Err(e.to_string()),
    };
    let mut error = String::new();
    if !args.is_empty() {
        error = fault(tool(args)).err().unwrap_or_default();
    }
    match fault(tool(&["list".into(), "--json".into()])) {
        Ok(o) => {
            let mut m = parse_mods(&String::from_utf8_lossy(&o.stdout));
            if !error.is_empty() {
                m.error = error;
            }
            m
        }
        Err(e) => Mods { list: vec![], error: if error.is_empty() { e } else { error } },
    }
}

fn parse_mods(text: &str) -> Mods {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else {
        return Mods { list: vec![], error: "The mod list couldn't be read.".into() };
    };
    let s = |v: &serde_json::Value| v.as_str().unwrap_or_default().to_string();
    let list = v["mods"].as_array().into_iter().flatten().map(|m| Mod {
        folder: s(&m["folder"]),
        name: s(&m["name"]),
        enabled: m["enabled"].as_bool().unwrap_or(false),
        note: s(&m["warnings"][0]),
    });
    Mods { list: list.collect(), error: s(&v["errors"][0]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("leonida-launcher-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn build(d: &Path) {
        fs::write(d.join("gta6rt.exe"), "").unwrap();
        fs::write(d.join("gta6rt.sh"), "#!/bin/sh\necho \"$1 $WINE $LEONIDA_D3D12 $LEONIDA_KEYBOARD_PAD\"\nsleep \"${NAP:-0}\"\nexit 7\n").unwrap();
        fs::write(d.join("game_path.txt"), d.join("game").to_str().unwrap()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(d.join("gta6rt.sh"), fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    fn options() -> Options {
        Options { keyboard: true, splash: true, diagnostics: false, wine: "my-wine".into() }
    }

    #[test]
    fn port_states() {
        let d = folder("port");
        let s = d.to_str().unwrap();
        assert_eq!(port(s), Port::NotBuilt);
        build(&d);
        assert_eq!(port(s), Port::Missing(EXE.into()));
        fs::create_dir(d.join("game")).unwrap();
        fs::write(d.join("game").join(EXE), "").unwrap();
        assert_eq!(port(s), Port::Ready);
        fs::remove_dir_all(d).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn runs_the_port_and_reports_its_exit() {
        let d = folder("run");
        build(&d);
        let log = d.join("logs").join("game.log");
        let g = start(command(&d, &options()).unwrap(), &log, || {}).unwrap();
        assert_eq!(g.done.recv_timeout(std::time::Duration::from_secs(10)).unwrap(), Some(7));
        assert!(tail(&log)[0].starts_with("--run my-wine"));
        assert!(tail(&log)[0].ends_with(" 1"));
        fs::remove_dir_all(d).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn stop_ends_the_whole_tree() {
        let d = folder("stop");
        build(&d);
        let mut c = command(&d, &options()).unwrap();
        c.env("NAP", "60");
        let g = start(c, &d.join("game.log"), || {}).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        g.stop();
        assert!(g.done.recv_timeout(std::time::Duration::from_secs(12)).is_ok());
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn settings_survive_a_reload() {
        let d = folder("conf");
        let p = d.join("a").join("launcher.conf");
        let mut c = Config::load(p.clone());
        c.set("out", "/a=b/c").unwrap();
        c.set("out", "/a=b/d").unwrap();
        assert_eq!(Config::load(p).get("out"), "/a=b/d");
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn reads_the_mod_managers_own_output() {
        let root = crate::backend::find_root().unwrap();
        let py = crate::backend::find_python().unwrap();
        let d = folder("mods");
        build(&d);
        let out = d.to_str().unwrap();
        let src = root.join("mods").join("first_person").display().to_string();
        let m = mods(&py, &root, out, &["install".into(), "--enable".into(), src]);
        assert_eq!(m.error, "");
        assert_eq!(m.list.len(), 1);
        assert!(m.list[0].enabled);
        let m = mods(&py, &root, out, &["disable".into(), m.list[0].folder.clone()]);
        assert!(!m.list[0].enabled, "{}", m.error);
        assert_ne!(mods(&py, &root, out, &["enable".into(), "nope".into()]).error, "");
        fs::remove_dir_all(d).unwrap();
    }
}
