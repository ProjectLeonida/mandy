// SPDX-License-Identifier: GPL-3.0-or-later

//! Cancel pressed before the running command has said it started: there is no process to stop
//! yet, so the launcher stops it when its `Started` arrives. A stand-in that would run for half a
//! minute plays the command, started the way the launcher starts one (its own process group
//! elsewhere than on Windows, where `taskkill /T` ends its tree), and has to end within seconds.

mod common;

use std::process::{Child, Stdio};
use std::time::{Duration, Instant};

use leonida_launcher::app::{Message, Running};
use leonida_launcher::commands::Command;
use leonida_launcher::output::Tracker;
use leonida_launcher::process::Event;
use leonida_launcher::state::{Action, Kind};

/// A process that would run for about 30 seconds, started as the launcher starts a command.
fn stand_in() -> Child {
    let mut command = if cfg!(windows) {
        let mut command = std::process::Command::new("cmd");
        command.args(["/C", "ping -n 31 127.0.0.1 >NUL"]);
        command
    } else {
        let mut command = std::process::Command::new("sh");
        command.args(["-c", "sleep 30"]);
        command
    };
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("the stand-in starts")
}

/// Whether the child has ended within `within`.
fn ends_within(child: &mut Child, within: Duration) -> bool {
    let until = Instant::now() + within;
    while Instant::now() < until {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

#[test]
fn cancel_before_started_stops_the_process_once_it_has_started() {
    let mut child = stand_in();
    let pid = child.id();
    let mut launcher = common::launcher();
    launcher.running = Some(Running {
        kind: Kind::Build,
        command: Command {
            program: "stand-in".to_owned(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: String::new(),
        },
        pid: None,
        tracker: Tracker::build(None),
        cancelled: false,
    });

    // Cancel first: nothing to stop yet, so the process runs on.
    let _ = launcher.update(Message::Act(Action::Cancel));
    assert!(
        launcher
            .running
            .as_ref()
            .is_some_and(|running| running.cancelled)
    );
    assert!(
        !ends_within(&mut child, Duration::from_millis(300)),
        "the stand-in ended before it was stopped"
    );

    // Then the process says it started: the launcher stops it now.
    let _ = launcher.update(Message::Process(Event::Started(pid)));
    let ended = ends_within(&mut child, Duration::from_secs(10));
    if !ended {
        let _ = child.kill();
        let _ = child.wait();
    }
    assert!(ended, "a Cancel before Started left the process running");
}
