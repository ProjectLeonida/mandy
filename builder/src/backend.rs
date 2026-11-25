// Runs make_game.py and turns its log into stage states. No UI in here.
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const STAGES: [(&str, &str, &str); 5] = [
    ("check", "Check game files", "Every file is there and the right size"),
    ("shaders", "Translate shaders", "RDNA2 to SPIR-V, the longest step"),
    ("assets", "Export assets", "Models, textures and audio"),
    ("runtime", "Build the runtime", "The loader and its libraries"),
    ("pipelines", "Test pipelines", "Every shader, on this GPU"),
];
const MAX_LOG: usize = 20_000;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum State {
    Waiting,
    Running,
    Done,
    Skipped,
    Failed,
}

pub enum Msg {
    Line(String),
    Done(Option<i32>),
}

pub struct Build {
    pub rx: Receiver<Msg>,
    pid: u32,
    alive: Arc<Mutex<bool>>,
}

/// What the log has said so far.
pub struct Progress {
    pub state: [State; 5],
    pub percent: [Option<u8>; 5],
    pub detail: [String; 5],
    pub log: Vec<String>,
    pub error: String,
}

impl Progress {
    pub fn new(selected: &[bool; 5]) -> Self {
        Self {
            state: selected.map(|on| if on { State::Waiting } else { State::Skipped }),
            percent: [None; 5],
            detail: Default::default(),
            log: vec![],
            error: String::new(),
        }
    }

    pub fn line(&mut self, raw: String) {
        // "12:34:56  shaders: 10/40 (25%), ~3 min left"
        let text = match raw.split_once("  ") {
            Some((t, rest)) if t.len() == 8 && t.as_bytes()[2] == b':' => rest,
            _ => raw.as_str(),
        };
        if let Some(e) = text.strip_prefix("ERROR: ") {
            self.error = e.to_string();
        }
        for (i, (key, ..)) in STAGES.iter().enumerate() {
            let Some(rest) = text.strip_prefix(key) else { continue };
            let Some(colon) = rest.find(':') else { continue };
            // "shaders:" and a progress label such as "shaders translate:", but not "runtime, pipelines: left out"
            let label = &rest[..colon];
            if label.starts_with(',') || rest.contains("left out") {
                if rest.contains("left out") {
                    self.state[i] = State::Skipped;
                }
                continue;
            }
            if !(label.is_empty() || label.starts_with(' ')) {
                continue;
            }
            let body = rest[colon + 1..].trim();
            if self.state[i] != State::Failed {
                self.state[i] = State::Running;
            }
            self.percent[i] = body.find('(').and_then(|p| body[p + 1..].split_once("%)")).and_then(|(n, _)| n.parse().ok());
            self.detail[i] = body.to_string();
            let done = match *key {
                "check" => body.starts_with("all files present"),
                "runtime" => body.starts_with("built") || body.starts_with("using the build"),
                _ => body == "done",
            };
            if done {
                self.state[i] = State::Done;
                self.percent[i] = None;
            }
        }
        if self.log.len() >= MAX_LOG {
            self.log.drain(..MAX_LOG / 4);
        }
        self.log.push(raw);
    }

    pub fn finish(&mut self, ok: bool) {
        for s in &mut self.state {
            *s = match (*s, ok) {
                (State::Running, true) => State::Done,
                (State::Waiting, true) => State::Skipped,
                (State::Running, false) => State::Failed,
                (s, _) => s,
            };
        }
    }

    /// (finished, total) over the stages this build runs.
    pub fn count(&self) -> (usize, usize) {
        let n = |f: fn(&State) -> bool| self.state.iter().filter(|s| f(s)).count();
        (n(|s| *s == State::Done), n(|s| *s != State::Skipped))
    }
}

pub fn find_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    [exe, std::env::current_dir().ok()]
        .into_iter()
        .flatten()
        .find_map(|p| p.ancestors().find(|d| d.join("make_game.py").is_file()).map(Path::to_path_buf))
}

pub fn quiet(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd.stdin(Stdio::null())
}

pub fn find_python() -> Option<Vec<String>> {
    let mut list: Vec<Vec<String>> = vec![];
    let names: &[&[&str]] = if cfg!(windows) {
        &[&["py", "-3.14"], &["py", "-3.13"], &["py", "-3.12"], &["py", "-3"], &["python"]]
    } else {
        &[&["python3.14"], &["python3.13"], &["python3.12"], &["python3"], &["python"]]
    };
    // An app opened from Finder has a bare PATH, so look where Homebrew and python.org install too.
    let dirs: &[&str] = if cfg!(target_os = "macos") { &["", "/opt/homebrew/bin/", "/usr/local/bin/"] } else { &[""] };
    for dir in dirs {
        for n in names {
            let mut c: Vec<String> = n.iter().map(|s| s.to_string()).collect();
            c[0] = format!("{dir}{}", c[0]);
            list.push(c);
        }
    }
    list.into_iter().find(|c| {
        quiet(Command::new(&c[0]).args(&c[1..]))
            .args(["-c", "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

pub fn python(py: &[String], root: &Path) -> Command {
    let mut c = Command::new(&py[0]);
    quiet(&mut c).args(&py[1..]).args(["-I", "-u", "-X", "utf8"]).current_dir(root);
    c
}

/// make_game.py's own default output folder (iCloud and redirected Documents are its business).
pub fn default_out(py: &[String], root: &Path) -> Option<String> {
    let code = "import sys; sys.path.insert(0, sys.argv[1]); import make_game; print(make_game.default_out())";
    let o = python(py, root).args(["-c", code]).arg(root).output().ok()?;
    let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
    (o.status.success() && !s.is_empty()).then_some(s)
}

/// `make_game.py --find`: (folder, title and version) of each copy on this computer.
pub fn find_games(py: &[String], root: &Path) -> Vec<(String, String)> {
    let Ok(o) = python(py, root).arg("make_game.py").arg("--find").output() else { return vec![] };
    parse_find(&String::from_utf8_lossy(&o.stdout))
}

fn parse_find(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter(|l| !l.starts_with("not usable: "))
        .filter_map(|l| l.trim_end().strip_suffix(')')?.rsplit_once("  ("))
        .map(|(d, v)| (d.to_string(), v.to_string()))
        .collect()
}

pub fn start(py: &[String], root: &Path, args: &[String], wake: impl Fn() + Send + Clone + 'static) -> std::io::Result<Build> {
    let mut cmd = python(py, root);
    cmd.arg("make_game.py").args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0); // so Cancel reaches the compilers it starts
    }
    let mut child = cmd.spawn()?;
    let pid = child.id();
    let alive = Arc::new(Mutex::new(true));
    let (tx, rx) = channel();
    let pump = |r: Box<dyn Read + Send>| {
        let (tx, wake) = (tx.clone(), wake.clone());
        std::thread::spawn(move || {
            let mut r = BufReader::new(r);
            let mut buf = vec![];
            while r.read_until(b'\n', &mut buf).is_ok_and(|n| n > 0) {
                let line = String::from_utf8_lossy(&buf).trim_end().to_string();
                buf.clear();
                if !line.is_empty() {
                    let _ = tx.send(Msg::Line(line));
                    wake();
                }
            }
        })
    };
    let out = pump(Box::new(child.stdout.take().unwrap()));
    let err = pump(Box::new(child.stderr.take().unwrap()));
    let flag = alive.clone();
    std::thread::spawn(move || {
        let _ = out.join();
        let _ = err.join();
        let code = child.wait().ok().and_then(|s| s.code());
        *flag.lock().unwrap() = false;
        let _ = tx.send(Msg::Done(code));
        wake();
    });
    Ok(Build { rx, pid, alive })
}

impl Build {
    /// make_game.py treats the interrupt as Ctrl+C, so the next run resumes.
    pub fn cancel(&self, now: bool) {
        kill_tree(self.pid, self.alive.clone(), now);
    }
}

/// Stops a process group (a tree on Windows): an interrupt first, then by force, or a terminate at once.
pub fn kill_tree(pid: u32, alive: Arc<Mutex<bool>>, now: bool) {
    let stop = move |sig: &str| {
        if !*alive.lock().unwrap() {
            return;
        }
        let mut c = if cfg!(windows) {
            let mut c = Command::new("taskkill");
            c.args(["/F", "/T", "/PID", &pid.to_string()]);
            c
        } else {
            let mut c = Command::new("/bin/kill");
            c.args([sig, "--", &format!("-{pid}")]);
            c
        };
        let _ = quiet(&mut c).stdout(Stdio::null()).stderr(Stdio::null()).status();
    };
    if now {
        return stop("-TERM");
    }
    std::thread::spawn(move || {
        stop("-INT");
        std::thread::sleep(Duration::from_secs(8));
        stop("-KILL");
    });
}

pub fn open_folder(path: &str) {
    let tool = if cfg!(windows) { "explorer" } else if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    let _ = Command::new(tool).arg(path).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_lines_drive_the_stages() {
        let mut p = Progress::new(&[true; 5]);
        p.line("12:00:00  check: all files present. ok".into());
        p.line("12:00:01  shaders translate: 10/40 (25%), ~3 min left".into());
        p.line("12:00:02  assets: done".into());
        p.line("12:00:03  runtime: left out, the cross-build needs xwin".into());
        assert_eq!(p.state, [State::Done, State::Running, State::Done, State::Skipped, State::Waiting]);
        assert_eq!(p.percent[1], Some(25));
        assert_eq!(p.count(), (2, 4));
        p.line("ERROR: no space".into());
        p.finish(false);
        assert_eq!(p.state[1], State::Failed);
        assert_eq!(p.error, "no space");
    }

    #[test]
    fn find_output_keeps_folders_with_brackets() {
        let t = "/games/a (copy)  (Game 1.0)\nnot usable: /x  (packaged)\n1 found\n";
        assert_eq!(parse_find(t), vec![("/games/a (copy)".to_string(), "Game 1.0".to_string())]);
    }

    #[cfg(unix)]
    #[test]
    fn cancel_stops_the_build() {
        let dir = std::env::temp_dir().join(format!("leonida-builder-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("make_game.py"), "import subprocess\nprint('up')\nsubprocess.run(['sleep', '60'])\n").unwrap();
        let py = find_python().unwrap();
        let b = start(&py, &dir, &[], || {}).unwrap();
        assert!(matches!(b.rx.recv_timeout(Duration::from_secs(10)), Ok(Msg::Line(_))));
        b.cancel(false);
        let end = std::time::Instant::now() + Duration::from_secs(7);
        let stopped = std::iter::from_fn(|| b.rx.recv_timeout(end.saturating_duration_since(std::time::Instant::now())).ok())
            .any(|m| matches!(m, Msg::Done(c) if c != Some(0)));
        assert!(stopped);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
