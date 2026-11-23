// SPDX-License-Identifier: GPL-3.0-or-later

//! Running a command off the interface's thread: its output read line by line on background
//! threads and sent on as it comes, then its exit code. Cancelling ends it and what it started
//! (make_game.py's tools, Wine's processes).

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Stdio};
use std::thread;

use iced::futures::Stream;
use iced::futures::channel::mpsc;

use crate::commands::Command;

/// What a running command says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// It started, as this process.
    Started(u32),
    /// A line of its output, stdout or stderr, as text (lossily, if it isn't UTF-8).
    Line(String),
    /// It ended, with its exit code (none if a signal ended it).
    Exited(Option<i32>),
    /// It couldn't be started, and why.
    Failed(String),
}

/// Starts `command` and returns what it says, ending with [`Event::Exited`] or [`Event::Failed`].
/// Its input is empty, so a question it asks (make_game.py's "which copy?") ends it instead of
/// waiting forever.
pub fn run(command: &Command) -> impl Stream<Item = Event> + use<> {
    let (sender, receiver) = mpsc::unbounded();
    match spawn(command) {
        Ok(child) => {
            let _ = sender.unbounded_send(Event::Started(child.id()));
            watch(child, sender);
        }
        Err(error) => {
            let _ = sender.unbounded_send(Event::Failed(format!(
                "{} couldn't be started: {error}",
                command.program
            )));
        }
    }
    receiver
}

fn spawn(command: &Command) -> std::io::Result<Child> {
    let mut process = std::process::Command::new(&command.program);
    process
        .args(&command.args)
        .envs(command.env.iter().map(|(name, value)| (name, value)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if !command.cwd.is_empty() {
        process.current_dir(&command.cwd);
    }
    #[cfg(unix)]
    {
        // Its own process group, so cancelling reaches what it starts too.
        use std::os::unix::process::CommandExt;
        process.process_group(0);
    }
    #[cfg(windows)]
    {
        // No console window over the launcher's.
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        process.creation_flags(CREATE_NO_WINDOW);
    }
    process.spawn()
}

/// Reads the child's output on two threads and waits for it on a third.
fn watch(mut child: Child, sender: mpsc::UnboundedSender<Event>) {
    let readers: Vec<_> = [
        child
            .stdout
            .take()
            .map(|out| Box::new(out) as Box<dyn Read + Send>),
        child
            .stderr
            .take()
            .map(|err| Box::new(err) as Box<dyn Read + Send>),
    ]
    .into_iter()
    .flatten()
    .map(|stream| {
        let sender = sender.clone();
        thread::spawn(move || read_lines(stream, &sender))
    })
    .collect();
    thread::spawn(move || {
        let status = child.wait();
        for reader in readers {
            let _ = reader.join();
        }
        let _ = sender.unbounded_send(match status {
            Ok(status) => Event::Exited(status.code()),
            Err(error) => Event::Failed(format!("waiting for it failed: {error}")),
        });
    });
}

fn read_lines(stream: impl Read, sender: &mpsc::UnboundedSender<Event>) {
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let text = String::from_utf8_lossy(&line);
                let text = text.trim_end_matches(['\n', '\r']);
                if sender.unbounded_send(Event::Line(text.to_owned())).is_err() {
                    break;
                }
            }
        }
    }
}

/// Ends the process `pid` and everything it started.
pub fn cancel(pid: u32) {
    let mut killer = cancel_command(pid);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        killer.creation_flags(0x0800_0000);
    }
    let _ = killer
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// The command that ends a process tree: `taskkill /T /F` on Windows, and elsewhere `kill` on the
/// process group the command was started in.
fn cancel_command(pid: u32) -> std::process::Command {
    if cfg!(windows) {
        let mut command = std::process::Command::new("taskkill");
        command.args(["/PID", &pid.to_string(), "/T", "/F"]);
        command
    } else {
        let mut command = std::process::Command::new("kill");
        command.args(["-TERM", "--", &format!("-{pid}")]);
        command
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use iced::futures::StreamExt;
    use iced::futures::executor::block_on;

    /// A stand-in for make_game.py: prints lines to stdout and stderr and exits with a code.
    fn stand_in(script: &str) -> Command {
        Command {
            program: "sh".to_owned(),
            args: vec!["-c".to_owned(), script.to_owned()],
            env: vec![("STAND_IN".to_owned(), "yes".to_owned())],
            cwd: std::env::temp_dir().display().to_string(),
        }
    }

    fn events(command: &Command) -> Vec<Event> {
        block_on(run(command).collect::<Vec<_>>())
    }

    #[test]
    fn lines_come_then_the_exit_code() {
        let events = events(&stand_in(
            "echo '12:00:00  check: Made-up package 9.9, 10 files'; echo \"env $STAND_IN\"; \
             printf 'no newline'; echo 'ERROR: it broke' >&2; exit 3",
        ));
        assert!(matches!(events[0], Event::Started(_)));
        let lines: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::Line(line) => Some(line.as_str()),
                _ => None,
            })
            .collect();
        assert!(lines.contains(&"12:00:00  check: Made-up package 9.9, 10 files"));
        assert!(lines.contains(&"env yes"));
        assert!(lines.contains(&"no newline"));
        assert!(lines.contains(&"ERROR: it broke"));
        assert_eq!(events.last(), Some(&Event::Exited(Some(3))));
    }

    #[test]
    fn a_question_on_stdin_ends_instead_of_waiting() {
        let events = events(&stand_in("read answer || exit 9; exit 0"));
        assert_eq!(events.last(), Some(&Event::Exited(Some(9))));
    }

    #[test]
    fn a_program_that_isnt_there_fails_to_start() {
        let command = Command {
            program: "/nonexistent/leonida-stand-in".to_owned(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: String::new(),
        };
        let events = events(&command);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::Failed(why) if why.contains("couldn't be started")));
    }

    #[test]
    fn cancelling_ends_it_and_what_it_started() {
        let mut stream = Box::pin(run(&stand_in("sleep 30 & sleep 30; exit 0")));
        let Some(Event::Started(pid)) = block_on(stream.next()) else {
            panic!("not started");
        };
        let started = std::time::Instant::now();
        cancel(pid);
        let rest: Vec<_> = block_on(stream.collect::<Vec<_>>());
        assert_eq!(rest.last(), Some(&Event::Exited(None)));
        // The background sleep held stdout open too: it ended with the group.
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
    }
}
