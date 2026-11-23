// SPDX-License-Identifier: GPL-3.0-or-later

//! What the launcher offers, from what exists: the one primary action (Check, Build, Play, or
//! Cancel while something runs), whether it can be taken and why not, the secondary actions, and
//! the status line. Plain logic, from [`Facts`], so the tests cover every state.

/// What's done or running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Check,
    Build,
    Play,
}

impl Kind {
    /// The button's word.
    pub const fn verb(self) -> &'static str {
        match self {
            Self::Check => "Check",
            Self::Build => "Build",
            Self::Play => "Play",
        }
    }

    /// While it runs.
    pub const fn running(self) -> &'static str {
        match self {
            Self::Check => "Checking your copy",
            Self::Build => "Building the port",
            Self::Play => "The game is running",
        }
    }
}

/// An action the window offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Run(Kind),
    Cancel,
}

impl Action {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Run(kind) => kind.verb(),
            Self::Cancel => "Cancel",
        }
    }
}

/// How a finished command ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub kind: Kind,
    /// Its exit code; none if it was stopped by a signal, or didn't start.
    pub code: Option<i32>,
    /// Whether it was cancelled.
    pub cancelled: bool,
    /// What it said last, or why it didn't start.
    pub words: Option<String>,
}

impl Outcome {
    pub fn succeeded(&self) -> bool {
        self.code == Some(0) && !self.cancelled
    }
}

/// What exists, as the launcher sees it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Facts {
    /// The package folder, as set; empty if not.
    pub package: String,
    pub package_exists: bool,
    pub output: String,
    /// Whether the package set passed a check (this session or saved).
    pub checked: bool,
    /// Whether the output folder holds the built loader (`gta6rt.exe`, or `gta6rt.sh` on Linux).
    pub loader_exists: bool,
    /// The loader's file name, for the words.
    pub loader: String,
    /// Whether the package holds the game's executable, and its name.
    pub exe_exists: bool,
    pub exe: String,
    /// What runs now.
    pub running: Option<Kind>,
    /// How the last command ended, if one has.
    pub last: Option<Outcome>,
}

/// Why the primary action can't be taken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Blocked {
    NoPackage,
    PackageMissing(String),
    NoOutput,
    /// Built, but the package has no executable: the normal state today.
    NotBootable {
        exe: String,
    },
}

impl Blocked {
    /// Why, in words for the status line.
    pub fn words(&self) -> String {
        match self {
            Self::NoPackage => "Choose your game package folder in Settings to begin.".to_owned(),
            Self::PackageMissing(path) => format!("The package folder isn't there: {path}"),
            Self::NoOutput => "Choose an output folder in Settings, then build.".to_owned(),
            Self::NotBootable { exe } => format!(
                "Not bootable yet: the loader is built, but {exe} isn't in your package. \
                 Today's package doesn't include it, so there is nothing to run yet."
            ),
        }
    }
}

/// The primary action: what it is, and why it can't be taken, if it can't.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Primary {
    pub action: Action,
    pub blocked: Option<Blocked>,
}

impl Primary {
    pub fn enabled(&self) -> bool {
        self.blocked.is_none()
    }
}

/// The primary action for these facts.
pub fn primary(facts: &Facts) -> Primary {
    let blocked = |action, why| Primary {
        action,
        blocked: Some(why),
    };
    let ready = |action| Primary {
        action,
        blocked: None,
    };
    if facts.running.is_some() {
        return ready(Action::Cancel);
    }
    let check = Action::Run(Kind::Check);
    let build = Action::Run(Kind::Build);
    let play = Action::Run(Kind::Play);
    if facts.package.trim().is_empty() {
        return blocked(check, Blocked::NoPackage);
    }
    if !facts.package_exists {
        return blocked(check, Blocked::PackageMissing(facts.package.clone()));
    }
    if facts.loader_exists && facts.exe_exists {
        return ready(play);
    }
    if facts.loader_exists {
        return blocked(
            play,
            Blocked::NotBootable {
                exe: facts.exe.clone(),
            },
        );
    }
    if !facts.checked {
        return ready(check);
    }
    if facts.output.trim().is_empty() {
        return blocked(build, Blocked::NoOutput);
    }
    ready(build)
}

/// The other actions offered beside the primary one, each with whether it can be taken.
pub fn secondary(facts: &Facts) -> Vec<(Action, bool)> {
    if facts.running.is_some() {
        return Vec::new();
    }
    let package = !facts.package.trim().is_empty() && facts.package_exists;
    let output = !facts.output.trim().is_empty();
    match primary(facts).action {
        Action::Run(Kind::Check) => Vec::new(),
        Action::Run(Kind::Build) => vec![(Action::Run(Kind::Check), package)],
        Action::Run(Kind::Play) => vec![
            (Action::Run(Kind::Build), package && output),
            (Action::Run(Kind::Check), package),
        ],
        Action::Cancel => Vec::new(),
    }
}

/// The status line's kind: a status chip's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    Waiting,
    Running,
    Passed,
    Info,
    Failed,
}

/// The status line: its tone, the chip's word, and the line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusLine {
    pub tone: Tone,
    pub word: &'static str,
    pub text: String,
}

/// The status line for these facts, with what's under way (a stage) while something runs.
pub fn status(facts: &Facts, under_way: Option<&str>) -> StatusLine {
    let line = |tone, word, text: String| StatusLine { tone, word, text };
    if let Some(kind) = facts.running {
        let text = match under_way {
            Some(stage) => format!("{}: {stage}", kind.running()),
            None => format!("{}…", kind.running()),
        };
        return line(Tone::Running, "Running", text);
    }
    if let Some(last) = &facts.last {
        if last.cancelled {
            return line(
                Tone::Info,
                "Stopped",
                format!(
                    "{} was cancelled. A build continues where it left off next time.",
                    last.kind.verb()
                ),
            );
        }
        if !last.succeeded() {
            let how = match last.code {
                Some(code) => format!("exit code {code}"),
                None => "it didn't run".to_owned(),
            };
            let words = last
                .words
                .as_deref()
                .map(|words| format!(": {words}"))
                .unwrap_or_default();
            let what = match last.kind {
                Kind::Play => "The loader",
                kind => kind.verb(),
            };
            return line(
                Tone::Failed,
                "Failed",
                format!("{what} failed ({how}){words}"),
            );
        }
    }
    let primary = primary(facts);
    match (&primary.blocked, primary.action) {
        (Some(blocked @ Blocked::NotBootable { .. }), _) => {
            line(Tone::Info, "Not bootable yet", blocked.words())
        }
        (Some(blocked), _) => line(Tone::Waiting, "Setup", blocked.words()),
        (None, Action::Run(Kind::Check)) => line(
            Tone::Waiting,
            "Ready",
            "Check your copy of the game before building.".to_owned(),
        ),
        (None, Action::Run(Kind::Build)) => line(
            Tone::Passed,
            "Checked",
            format!(
                "Your copy is checked. Build the port; the loader goes to {}.",
                facts.output.trim()
            ),
        ),
        (None, Action::Run(Kind::Play)) => line(
            Tone::Passed,
            "Ready",
            format!("Built, and {} is in your package.", facts.exe),
        ),
        (None, Action::Cancel) => line(Tone::Running, "Running", String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> Facts {
        Facts {
            loader: "gta6rt.exe".to_owned(),
            exe: "game_scarlett_master.exe".to_owned(),
            ..Facts::default()
        }
    }

    fn set_up() -> Facts {
        Facts {
            package: "D:\\pkg".to_owned(),
            package_exists: true,
            output: "D:\\out".to_owned(),
            ..fresh()
        }
    }

    #[test]
    fn nothing_set_asks_for_the_package() {
        let facts = fresh();
        let primary = primary(&facts);
        assert_eq!(primary.action, Action::Run(Kind::Check));
        assert_eq!(primary.blocked, Some(Blocked::NoPackage));
        assert!(!primary.enabled());
        let status = status(&facts, None);
        assert_eq!(status.tone, Tone::Waiting);
        assert!(status.text.contains("package folder"));
        assert!(secondary(&facts).is_empty());
    }

    #[test]
    fn a_missing_package_folder_says_so() {
        let facts = Facts {
            package_exists: false,
            ..set_up()
        };
        assert_eq!(
            primary(&facts).blocked,
            Some(Blocked::PackageMissing("D:\\pkg".to_owned()))
        );
    }

    #[test]
    fn a_package_not_yet_checked_is_checked_first() {
        let facts = set_up();
        assert_eq!(
            primary(&facts),
            Primary {
                action: Action::Run(Kind::Check),
                blocked: None
            }
        );
    }

    #[test]
    fn a_checked_copy_builds_once_there_is_an_output_folder() {
        let checked = Facts {
            checked: true,
            ..set_up()
        };
        assert_eq!(primary(&checked).action, Action::Run(Kind::Build));
        assert!(primary(&checked).enabled());
        assert_eq!(secondary(&checked), vec![(Action::Run(Kind::Check), true)]);
        assert_eq!(status(&checked, None).tone, Tone::Passed);
        let no_output = Facts {
            output: String::new(),
            ..checked
        };
        assert_eq!(primary(&no_output).blocked, Some(Blocked::NoOutput));
    }

    #[test]
    fn built_without_the_executable_is_not_bootable_yet_and_reads_as_an_explanation() {
        let facts = Facts {
            checked: true,
            loader_exists: true,
            ..set_up()
        };
        let primary = primary(&facts);
        assert_eq!(primary.action, Action::Run(Kind::Play));
        assert!(!primary.enabled());
        let status = status(&facts, None);
        assert_eq!((status.tone, status.word), (Tone::Info, "Not bootable yet"));
        assert!(
            status
                .text
                .contains("game_scarlett_master.exe isn't in your package")
        );
        assert!(!status.text.to_lowercase().contains("error"));
        // Building again and checking stay offered.
        assert_eq!(
            secondary(&facts),
            vec![
                (Action::Run(Kind::Build), true),
                (Action::Run(Kind::Check), true)
            ]
        );
    }

    #[test]
    fn built_with_the_executable_plays() {
        let facts = Facts {
            loader_exists: true,
            exe_exists: true,
            ..set_up()
        };
        assert_eq!(primary(&facts).action, Action::Run(Kind::Play));
        assert!(primary(&facts).enabled());
        assert_eq!(status(&facts, None).tone, Tone::Passed);
    }

    #[test]
    fn while_running_the_one_action_is_cancel() {
        let facts = Facts {
            running: Some(Kind::Build),
            ..set_up()
        };
        assert_eq!(primary(&facts).action, Action::Cancel);
        assert!(primary(&facts).enabled());
        assert!(secondary(&facts).is_empty());
        let status = status(&facts, Some("Stage 2 of 5: shaders"));
        assert_eq!(status.tone, Tone::Running);
        assert_eq!(status.text, "Building the port: Stage 2 of 5: shaders");
        assert_eq!(Action::Cancel.label(), "Cancel");
    }

    #[test]
    fn a_failure_shows_its_code_and_last_words_until_the_next_run() {
        let facts = Facts {
            last: Some(Outcome {
                kind: Kind::Build,
                code: Some(1),
                cancelled: false,
                words: Some("3 file(s) missing or different.".to_owned()),
            }),
            checked: true,
            ..set_up()
        };
        let line = status(&facts, None);
        assert_eq!(line.tone, Tone::Failed);
        assert_eq!(
            line.text,
            "Build failed (exit code 1): 3 file(s) missing or different."
        );
        // It can be tried again.
        assert!(primary(&facts).enabled());
        let unstarted = Facts {
            last: Some(Outcome {
                kind: Kind::Play,
                code: None,
                cancelled: false,
                words: Some("gta6rt.sh: not found".to_owned()),
            }),
            ..set_up()
        };
        assert!(status_text(&unstarted).starts_with("The loader failed (it didn't run)"));
        let cancelled = Facts {
            last: Some(Outcome {
                kind: Kind::Build,
                code: None,
                cancelled: true,
                words: None,
            }),
            ..set_up()
        };
        assert_eq!(status(&cancelled, None).word, "Stopped");
    }

    fn status_text(facts: &Facts) -> String {
        status(facts, None).text
    }
}
