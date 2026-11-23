// SPDX-License-Identifier: GPL-3.0-or-later

//! Reading a command's output as it comes: the last lines for the log, and how far a build is.
//!
//! `make_game.py` prints free text: its `log()` lines are `HH:MM:SS  <text>`, and each stage's
//! begin with the stage's name (`shaders: carving shaders.rpf`). Its stages run in a fixed order
//! (check, shaders, assets, runtime, pipelines) after the line `building into …`, so a stage's
//! first line says it has begun and `<stage>: done` that it's over; that's what the determinate
//! bar follows. Lines before `building into` that name a stage (the GPU probe's `shaders: …`, the
//! runtime stage `left out` on Linux) only say what's planned. Its failures go to stderr as
//! `ERROR: <what>`.
//!
//! Machine-readable lines, if a later `make_game.py` prints them, take over from the free text:
//! `@@stage <id> <text>`, `@@progress <0..1>`, `@@eta <seconds>`, `@@check <ok|bad> <text>`,
//! `@@done <folder>` and `@@error <text>`. They aren't shown in the log as they are; unknown `@@`
//! words are ignored.

use std::collections::VecDeque;

/// How many of the last lines are kept for the log.
pub const KEPT: usize = 400;

/// The stages of a build, in the order make_game.py runs them.
pub const ORDER: [&str; 5] = ["check", "shaders", "assets", "runtime", "pipelines"];

/// A line, read.
#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    /// Text for the log.
    Text(String),
    /// `@@stage <id> <text>`.
    Stage { id: String, text: String },
    /// `@@progress <0..1>`.
    Progress(f32),
    /// `@@eta <seconds>`.
    Eta(f64),
    /// `@@check <ok|bad> <text>`.
    Check { ok: bool, text: String },
    /// `@@done <folder>`.
    Done(String),
    /// `@@error <text>`.
    Error(String),
    /// An `@@` line this launcher doesn't know, or one it can't read.
    Ignored,
}

/// Reads one line.
pub fn parse(line: &str) -> Line {
    let Some(rest) = line.strip_prefix("@@") else {
        return Line::Text(line.to_owned());
    };
    let (word, rest) = rest.split_once(' ').unwrap_or((rest, ""));
    let rest = rest.trim();
    match word {
        "stage" => {
            let (id, text) = rest.split_once(' ').unwrap_or((rest, ""));
            if id.is_empty() {
                return Line::Ignored;
            }
            Line::Stage {
                id: id.to_owned(),
                text: text.trim().to_owned(),
            }
        }
        "progress" => rest
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite())
            .map_or(Line::Ignored, |value| Line::Progress(value.clamp(0.0, 1.0))),
        "eta" => rest
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map_or(Line::Ignored, Line::Eta),
        "check" => {
            let (verdict, text) = rest.split_once(' ').unwrap_or((rest, ""));
            match verdict {
                "ok" | "bad" => Line::Check {
                    ok: verdict == "ok",
                    text: text.trim().to_owned(),
                },
                _ => Line::Ignored,
            }
        }
        "done" => Line::Done(rest.to_owned()),
        "error" => Line::Error(rest.to_owned()),
        _ => Line::Ignored,
    }
}

/// The text after a `log()` line's `HH:MM:SS  ` time, if it is one.
fn logged(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    let time = bytes.get(..10)?;
    let digits = [0, 1, 3, 4, 6, 7]
        .iter()
        .all(|at| time[*at].is_ascii_digit());
    (digits && time[2] == b':' && time[5] == b':' && &time[8..10] == b"  ").then(|| &line[10..])
}

/// What's known of a running command from its output.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tracker {
    /// The last lines, for the log.
    pub lines: VecDeque<String>,
    /// Whether `@@` lines have been seen: they take over from the free text.
    machine: bool,
    /// The build's stages, in order, as planned.
    planned: Vec<&'static str>,
    /// Whether the stages have begun (`building into …`).
    begun: bool,
    /// The stage under way, by its place in `planned`, and whether it said it's done.
    current: Option<usize>,
    finished_current: bool,
    finished: bool,
    /// From `@@` lines: the step and how far it is.
    step: Option<String>,
    fraction: Option<f32>,
    /// Seconds left, if said.
    pub eta: Option<f64>,
    /// What went wrong, if said: `ERROR: …` or `@@error`.
    pub error: Option<String>,
    /// The game copy's version, as the check says it: "Xbox Series package 1.0.2.1".
    pub game: Option<String>,
    /// `@@check` results that weren't ok.
    pub bad: Vec<String>,
    /// `@@done`'s folder.
    pub done: Option<String>,
}

impl Tracker {
    /// For a build of these stages (`None`: make_game.py's default, every stage). The check
    /// always runs first.
    pub fn build(stages: Option<&str>) -> Self {
        let chosen: Vec<&str> = stages
            .map(|stages| stages.split(',').map(str::trim).collect())
            .unwrap_or_else(|| ORDER.to_vec());
        Self {
            planned: ORDER
                .iter()
                .copied()
                .filter(|stage| *stage == "check" || chosen.contains(stage))
                .collect(),
            ..Self::default()
        }
    }

    /// For a command that isn't a build: no stages.
    pub fn plain() -> Self {
        Self::default()
    }

    /// Takes one line of output.
    pub fn feed(&mut self, line: &str) {
        let line = line.trim_end_matches(['\r', '\n']);
        match parse(line) {
            Line::Text(text) => {
                self.read_text(&text);
                if self.lines.len() == KEPT {
                    self.lines.pop_front();
                }
                self.lines.push_back(text);
            }
            other => {
                self.machine = true;
                self.read_machine(other);
            }
        }
    }

    fn read_machine(&mut self, line: Line) {
        match line {
            Line::Stage { id, text } => {
                self.step = Some(if text.is_empty() { id } else { text });
                self.fraction = Some(0.0);
                self.eta = None;
            }
            Line::Progress(fraction) => self.fraction = Some(fraction),
            Line::Eta(seconds) => self.eta = Some(seconds),
            Line::Check { ok, text } => {
                if !ok {
                    self.bad.push(text);
                }
            }
            Line::Done(folder) => {
                self.done = Some(folder);
                self.fraction = Some(1.0);
            }
            Line::Error(text) => self.error = Some(text),
            Line::Text(_) | Line::Ignored => {}
        }
    }

    fn read_text(&mut self, text: &str) {
        if let Some(error) = text.strip_prefix("ERROR: ") {
            self.error = Some(error.trim().to_owned());
        }
        let Some(said) = logged(text) else {
            return;
        };
        // The check's first line names the copy: "check: <title> <version>, <n> files".
        if let Some(rest) = said.strip_prefix("check: ")
            && self.game.is_none()
            && let Some((game, files)) = rest.rsplit_once(", ")
            && files.ends_with(" files")
        {
            self.game = Some(game.to_owned());
        }
        if self.machine {
            return;
        }
        if said.starts_with("building into ") {
            self.begun = true;
            self.current = self.current.max(Some(0));
            self.finished_current = true;
            return;
        }
        if said.starts_with("finished in ") {
            self.finished = true;
            return;
        }
        let Some((stage, rest)) = said.split_once(':') else {
            return;
        };
        if stage == "runtime" && rest.trim_start().starts_with("left out") {
            self.planned.retain(|planned| *planned != "runtime");
            return;
        }
        let Some(at) = self.planned.iter().position(|planned| *planned == stage) else {
            return;
        };
        // Before the stages begin, only the check speaks for itself.
        if !self.begun && stage != "check" {
            return;
        }
        if self.current.is_none_or(|current| at > current) {
            self.current = Some(at);
            self.finished_current = false;
        }
        if Some(at) == self.current && rest.trim() == "done" {
            self.finished_current = true;
        }
    }

    /// How far it is, from 0 to 1, if its output says: the `@@progress` of the step under way, or
    /// the build's stages done of those planned.
    pub fn progress(&self) -> Option<f32> {
        if self.machine {
            return self.fraction;
        }
        if self.finished {
            return Some(1.0);
        }
        let count = self.planned.len();
        let current = self.current?;
        if count == 0 {
            return None;
        }
        let done = current + usize::from(self.finished_current);
        Some(done.min(count) as f32 / count as f32)
    }

    /// What's under way, in words: the `@@stage`'s, or "Stage 2 of 5: shaders".
    pub fn stage(&self) -> Option<String> {
        if self.machine {
            return self.step.clone();
        }
        let current = self.current?;
        let count = self.planned.len();
        let index = if self.finished_current && current + 1 < count && self.begun {
            current + 1
        } else {
            current
        };
        let name = self.planned.get(index)?;
        Some(format!("Stage {} of {count}: {name}", index + 1))
    }

    /// The last line worth showing with a failure: the error if said, or the last line.
    pub fn last_words(&self) -> Option<String> {
        self.error.clone().or_else(|| {
            self.lines
                .iter()
                .rev()
                .find(|line| !line.trim().is_empty())
                .map(|line| line.trim().to_owned())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fed(tracker: &mut Tracker, lines: &[&str]) {
        for line in lines {
            tracker.feed(line);
        }
    }

    #[test]
    fn machine_lines_are_read() {
        assert_eq!(
            parse("@@stage shaders Translating shaders"),
            Line::Stage {
                id: "shaders".to_owned(),
                text: "Translating shaders".to_owned()
            }
        );
        assert_eq!(parse("@@progress 0.25"), Line::Progress(0.25));
        assert_eq!(parse("@@progress 7"), Line::Progress(1.0));
        assert_eq!(parse("@@progress NaN"), Line::Ignored);
        assert_eq!(parse("@@eta 90"), Line::Eta(90.0));
        assert_eq!(
            parse("@@check bad 3 files missing"),
            Line::Check {
                ok: false,
                text: "3 files missing".to_owned()
            }
        );
        assert_eq!(parse("@@check maybe x"), Line::Ignored);
        assert_eq!(parse("@@done /out"), Line::Done("/out".to_owned()));
        assert_eq!(
            parse("@@error disk full"),
            Line::Error("disk full".to_owned())
        );
        assert_eq!(parse("@@later 1 2"), Line::Ignored);
        assert_eq!(parse("@@stage"), Line::Ignored);
        assert_eq!(parse("plain"), Line::Text("plain".to_owned()));
    }

    #[test]
    fn machine_lines_drive_the_bar_and_stay_out_of_the_log() {
        let mut tracker = Tracker::build(None);
        fed(
            &mut tracker,
            &[
                "12:00:00  check: Made-up package 9.9, 10 files",
                "@@stage shaders Translating shaders",
                "@@progress 0.5",
                "@@eta 120",
                "a tool's own line",
                "@@check bad one file",
                "@@unknown word",
            ],
        );
        assert_eq!(tracker.stage().as_deref(), Some("Translating shaders"));
        assert_eq!(tracker.progress(), Some(0.5));
        assert_eq!(tracker.eta, Some(120.0));
        assert_eq!(tracker.bad, ["one file"]);
        assert_eq!(tracker.game.as_deref(), Some("Made-up package 9.9"));
        assert!(tracker.lines.iter().all(|line| !line.starts_with("@@")));
        assert_eq!(tracker.lines.len(), 2);
        // Free-text stage lines don't move it once @@ lines have come.
        tracker.feed("12:00:01  pipelines: done");
        assert_eq!(tracker.progress(), Some(0.5));
        tracker.feed("@@error the disk is full");
        assert_eq!(tracker.last_words().as_deref(), Some("the disk is full"));
        tracker.feed("@@done /out");
        assert_eq!(
            (tracker.progress(), tracker.done.as_deref()),
            (Some(1.0), Some("/out"))
        );
        // A new step starts again from its beginning.
        tracker.feed("@@stage assets Exporting");
        assert_eq!(tracker.progress(), Some(0.0));
    }

    #[test]
    fn the_free_text_log_gives_the_stages() {
        let mut tracker = Tracker::build(None);
        assert_eq!(tracker.progress(), None);
        fed(
            &mut tracker,
            &[
                "12:00:00  check: Made-up package 9.9, 10 files",
                "12:00:01  check: all files present. Executable found.",
            ],
        );
        assert_eq!(tracker.game.as_deref(), Some("Made-up package 9.9"));
        assert_eq!(tracker.progress(), Some(0.0));
        assert_eq!(tracker.stage().as_deref(), Some("Stage 1 of 5: check"));
        // Before the stages begin, a stage's name is only the plan: the GPU probe's line, and
        // the runtime stage left out.
        fed(
            &mut tracker,
            &[
                "12:00:02  shaders: smallest maximum subgroup size is 64: wave64 emulation off",
                "12:00:02  runtime: left out, the cross-build needs clang-cl (or pass --runtime-build)",
            ],
        );
        assert_eq!(tracker.progress(), Some(0.0));
        tracker.feed("12:00:03  building into /out (work files in /out/work)");
        assert_eq!(tracker.progress(), Some(0.25));
        assert_eq!(tracker.stage().as_deref(), Some("Stage 2 of 4: shaders"));
        fed(
            &mut tracker,
            &[
                "12:00:04  shaders: carving shaders.rpf",
                "  a tool's own output",
                "12:00:05  shaders: done",
            ],
        );
        assert_eq!(tracker.progress(), Some(0.5));
        assert_eq!(tracker.stage().as_deref(), Some("Stage 3 of 4: assets"));
        // Never backwards.
        fed(
            &mut tracker,
            &["12:00:06  assets: audio banks", "12:00:07  shaders: late"],
        );
        assert_eq!(tracker.progress(), Some(0.5));
        tracker.feed("12:00:08  pipelines: creating every shader module");
        assert_eq!(tracker.progress(), Some(0.75));
        tracker.feed("12:00:09  finished in 3 min");
        assert_eq!(tracker.progress(), Some(1.0));
        assert_eq!(tracker.lines.len(), 12);
    }

    #[test]
    fn chosen_stages_are_the_plan_and_errors_are_kept() {
        let mut tracker = Tracker::build(Some("shaders,pipelines"));
        fed(
            &mut tracker,
            &[
                "12:00:00  check: Made-up package 9.9, 10 files",
                "12:00:01  building into /out (work files in /w)",
                "12:00:02  pipelines: creating",
                "",
                "ERROR: vk_pipeline_check exited with 3",
            ],
        );
        assert_eq!(tracker.stage().as_deref(), Some("Stage 3 of 3: pipelines"));
        assert_eq!(
            tracker.last_words().as_deref(),
            Some("vk_pipeline_check exited with 3")
        );
        // A command that isn't a build has no stages, and its last line says what happened.
        let mut plain = Tracker::plain();
        fed(
            &mut plain,
            &["first", "Traceback (most recent call last):", "  boom", ""],
        );
        assert_eq!(plain.progress(), None);
        assert_eq!(plain.stage(), None);
        assert_eq!(plain.last_words().as_deref(), Some("boom"));
    }

    #[test]
    fn only_the_last_lines_are_kept() {
        let mut tracker = Tracker::plain();
        for n in 0..KEPT + 10 {
            tracker.feed(&format!("line {n}\r"));
        }
        assert_eq!(tracker.lines.len(), KEPT);
        assert_eq!(tracker.lines.front().unwrap(), "line 10");
        assert_eq!(logged("12:34:56  x"), Some("x"));
        assert_eq!(logged("12:34:56 x"), None);
        assert_eq!(logged("short"), None);
    }
}
