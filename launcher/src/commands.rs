// SPDX-License-Identifier: GPL-3.0-or-later

//! The command lines the launcher runs: `make_game.py`'s check and build, and the loader, as
//! `README.md` ("How to build", "Running the loader on Linux") and `docs/boot.md` (section 2)
//! give them. Each is built for a [`Platform`], so the tests check Windows' and Linux's alike.

use crate::config::{STAGES, Settings, Unimplemented, Wave64};

/// Which system the commands are for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    Linux,
}

impl Platform {
    /// The one the launcher is built for.
    pub const fn current() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else {
            Self::Linux
        }
    }

    /// The Python the repository runs its tools with: `python` on Windows (as the README's
    /// examples), `python3` elsewhere.
    pub const fn python(self) -> &'static str {
        match self {
            Self::Windows => "python",
            Self::Linux => "python3",
        }
    }

    /// The loader the runtime stage writes into the output folder: `gta6rt.exe`, or on Linux the
    /// `gta6rt.sh` beside it that starts it with Wine (make_game.py's `WINE_LAUNCHER`).
    pub const fn loader(self) -> &'static str {
        match self {
            Self::Windows => "gta6rt.exe",
            Self::Linux => "gta6rt.sh",
        }
    }

    fn separator(self) -> char {
        match self {
            Self::Windows => '\\',
            Self::Linux => '/',
        }
    }

    /// `folder` and `name`, joined as this system joins paths.
    pub fn join(self, folder: &str, name: &str) -> String {
        let separator = self.separator();
        if folder.ends_with(separator) || (self == Self::Windows && folder.ends_with('/')) {
            format!("{folder}{name}")
        } else {
            format!("{folder}{separator}{name}")
        }
    }
}

/// A command to run: the program, its arguments, the environment it adds, and where it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: String,
}

impl Command {
    /// As a shell would show it, for the log: each part quoted where it has a space.
    pub fn display(&self) -> String {
        let quote = |part: &str| {
            if part.is_empty() || part.contains(char::is_whitespace) {
                format!("\"{part}\"")
            } else {
                part.to_owned()
            }
        };
        let env = self
            .env
            .iter()
            .map(|(name, value)| format!("{name}={} ", quote(value)));
        env.chain(std::iter::once(quote(&self.program)))
            .chain(self.args.iter().map(|arg| format!(" {}", quote(arg))))
            .collect()
    }
}

/// What's needed and missing to make a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Missing {
    /// The game package folder isn't set.
    Package,
    /// The output folder isn't set.
    Output,
}

/// `make_game.py` with the arguments after it: `python -I -u make_game.py …`, in the repository.
/// `-I` as the repository runs it; `-u` so its lines arrive as they're printed, not when a pipe's
/// buffer fills (its `log()` flushes, its other prints don't).
fn make_game(
    platform: Platform,
    repo: &str,
    args: Vec<String>,
    env: Vec<(String, String)>,
) -> Command {
    let mut all = vec![
        "-I".to_owned(),
        "-u".to_owned(),
        platform.join(repo, "make_game.py"),
    ];
    all.extend(args);
    Command {
        program: platform.python().to_owned(),
        args: all,
        env,
        cwd: repo.to_owned(),
    }
}

/// `LEONIDA_VK_DEVICE`, when a GPU name filter is set: the runtime and make_game.py's GPU probe
/// (`tools/vk_subgroup_info.py`) take the first Vulkan device whose name holds it.
fn gpu_env(settings: &Settings) -> Vec<(String, String)> {
    let gpu = settings.gpu.trim();
    if gpu.is_empty() {
        Vec::new()
    } else {
        vec![("LEONIDA_VK_DEVICE".to_owned(), gpu.to_owned())]
    }
}

/// Checks the copy: `make_game.py --game <package> --check`, and `--full-check` to compare every
/// file's SHA-256.
pub fn check(platform: Platform, repo: &str, settings: &Settings) -> Result<Command, Missing> {
    let package = settings.package.trim();
    if package.is_empty() {
        return Err(Missing::Package);
    }
    let mut args = vec![
        "--game".to_owned(),
        package.to_owned(),
        "--check".to_owned(),
    ];
    if settings.full_check {
        args.push("--full-check".to_owned());
    }
    Ok(make_game(platform, repo, args, Vec::new()))
}

/// The stages to ask for: none (make_game.py's default, every stage) when all are chosen.
pub fn stages(settings: &Settings) -> Option<String> {
    let chosen: Vec<&str> = STAGES
        .iter()
        .copied()
        .filter(|stage| settings.stages.iter().any(|chosen| chosen == stage))
        .collect();
    if chosen.len() == STAGES.len() {
        None
    } else if chosen.is_empty() {
        // Only the check, which runs before every build anyway.
        Some("check".to_owned())
    } else {
        Some(chosen.join(","))
    }
}

/// The jobs to ask for, if a whole number above 0 is set.
pub fn jobs(settings: &Settings) -> Option<u32> {
    settings
        .jobs
        .trim()
        .parse::<u32>()
        .ok()
        .filter(|jobs| *jobs > 0)
}

/// Builds the port: `make_game.py --game <package> --out <output> [--work <work>]` and the
/// build's options.
pub fn build(platform: Platform, repo: &str, settings: &Settings) -> Result<Command, Missing> {
    let package = settings.package.trim();
    let output = settings.output.trim();
    if package.is_empty() {
        return Err(Missing::Package);
    }
    if output.is_empty() {
        return Err(Missing::Output);
    }
    let mut args = vec![
        "--game".to_owned(),
        package.to_owned(),
        "--out".to_owned(),
        output.to_owned(),
    ];
    let work = settings.work.trim();
    if !work.is_empty() {
        args.extend(["--work".to_owned(), work.to_owned()]);
    }
    if let Some(stages) = stages(settings) {
        args.extend(["--stages".to_owned(), stages]);
    }
    if settings.wav {
        args.push("--wav".to_owned());
    }
    if let Some(jobs) = jobs(settings) {
        args.extend(["--jobs".to_owned(), jobs.to_string()]);
    }
    if settings.wave64 != Wave64::Auto {
        args.extend([
            "--wave64-emulate".to_owned(),
            settings.wave64.flag().to_owned(),
        ]);
    }
    Ok(make_game(platform, repo, args, gpu_env(settings)))
}

/// A Linux path as Wine sees it, on its `Z:` drive (the whole file system).
pub fn wine_path(path: &str) -> String {
    format!("Z:{}", path.replace('/', "\\"))
}

/// Starts the game: the loader with `--run <executable> --game-dir <package>` and
/// `LEONIDA_D3D12=vulkan`, as `docs/boot.md` section 2 has it; on Linux through the output
/// folder's `gta6rt.sh`, in the package folder, the executable by its name there and the package
/// as Wine's `Z:` path.
pub fn play(platform: Platform, exe_name: &str, settings: &Settings) -> Result<Command, Missing> {
    let package = settings.package.trim();
    let output = settings.output.trim();
    if package.is_empty() {
        return Err(Missing::Package);
    }
    if output.is_empty() {
        return Err(Missing::Output);
    }
    let mut env = vec![("LEONIDA_D3D12".to_owned(), "vulkan".to_owned())];
    env.extend(gpu_env(settings));
    if settings.validation {
        env.push(("LEONIDA_VK_VALIDATION".to_owned(), "1".to_owned()));
    }
    if settings.unimplemented != Unimplemented::Log {
        env.push((
            "LEONIDA_UNIMPLEMENTED".to_owned(),
            settings.unimplemented.value().to_owned(),
        ));
    }
    let loader = platform.join(output, platform.loader());
    let overlay = settings.overlay.trim();
    let mut command = match platform {
        Platform::Windows => Command {
            program: loader,
            args: vec![
                "--run".to_owned(),
                platform.join(package, exe_name),
                "--game-dir".to_owned(),
                package.to_owned(),
            ],
            env,
            cwd: output.to_owned(),
        },
        Platform::Linux => Command {
            program: loader,
            args: vec![
                "--run".to_owned(),
                format!("./{exe_name}"),
                "--game-dir".to_owned(),
                wine_path(package),
            ],
            env,
            cwd: package.to_owned(),
        },
    };
    if !overlay.is_empty() {
        command.args.push("--overlay".to_owned());
        command.args.push(match platform {
            Platform::Windows => overlay.to_owned(),
            Platform::Linux => wine_path(overlay),
        });
    }
    if settings.trace_d3d12 {
        command.args.push("--trace-d3d12".to_owned());
    }
    Ok(command)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Wave64;

    fn windows() -> Settings {
        Settings {
            package: "D:\\Games\\GTA6 package".to_owned(),
            output: "D:\\Games\\Leonida".to_owned(),
            ..Settings::default()
        }
    }

    fn linux() -> Settings {
        Settings {
            package: "/games/package".to_owned(),
            output: "/home/player/Leonida".to_owned(),
            ..Settings::default()
        }
    }

    fn strings(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| (*part).to_owned()).collect()
    }

    #[test]
    fn check_runs_make_game_with_the_package() {
        let command = check(Platform::Windows, "C:\\src\\Leonida", &windows()).unwrap();
        assert_eq!(command.program, "python");
        assert_eq!(
            command.args,
            strings(&[
                "-I",
                "-u",
                "C:\\src\\Leonida\\make_game.py",
                "--game",
                "D:\\Games\\GTA6 package",
                "--check"
            ])
        );
        assert_eq!(command.cwd, "C:\\src\\Leonida");
        assert!(command.env.is_empty());
        let full = Settings {
            full_check: true,
            ..linux()
        };
        let command = check(Platform::Linux, "/src/Leonida", &full).unwrap();
        assert_eq!(command.program, "python3");
        assert_eq!(command.args[2], "/src/Leonida/make_game.py");
        assert_eq!(command.args.last().unwrap(), "--full-check");
        assert_eq!(
            check(Platform::Linux, "/src", &Settings::default()),
            Err(Missing::Package)
        );
    }

    #[test]
    fn build_takes_the_three_folders() {
        let settings = Settings {
            work: "E:\\leonida-work".to_owned(),
            ..windows()
        };
        let command = build(Platform::Windows, "C:\\src\\Leonida", &settings).unwrap();
        assert_eq!(
            command.args[3..],
            strings(&[
                "--game",
                "D:\\Games\\GTA6 package",
                "--out",
                "D:\\Games\\Leonida",
                "--work",
                "E:\\leonida-work"
            ])
        );
        // No work folder: make_game.py's own, <out>/work.
        let command = build(Platform::Linux, "/src/Leonida", &linux()).unwrap();
        assert!(!command.args.contains(&"--work".to_owned()));
        assert_eq!(command.args.len(), 7);
        let no_output = Settings {
            output: " ".to_owned(),
            ..linux()
        };
        assert_eq!(
            build(Platform::Linux, "/src", &no_output),
            Err(Missing::Output)
        );
    }

    #[test]
    fn build_options_become_make_games_flags_and_the_gpu_its_variable() {
        let settings = Settings {
            stages: strings(&["pipelines", "shaders"]),
            wav: true,
            jobs: " 6 ".to_owned(),
            wave64: Wave64::Off,
            gpu: "radeon".to_owned(),
            ..linux()
        };
        let command = build(Platform::Linux, "/src/Leonida", &settings).unwrap();
        let tail = &command.args[7..];
        assert_eq!(
            tail,
            strings(&[
                "--stages",
                "shaders,pipelines",
                "--wav",
                "--jobs",
                "6",
                "--wave64-emulate",
                "off"
            ])
        );
        assert_eq!(
            command.env,
            vec![("LEONIDA_VK_DEVICE".to_owned(), "radeon".to_owned())]
        );
        // Every stage is the default; none is the check alone; a bad job count is left out.
        assert_eq!(stages(&Settings::default()), None);
        let none = Settings {
            stages: Vec::new(),
            jobs: "many".to_owned(),
            ..Settings::default()
        };
        assert_eq!(stages(&none), Some("check".to_owned()));
        assert_eq!(jobs(&none), None);
        assert_eq!(
            jobs(&Settings {
                jobs: "0".to_owned(),
                ..Settings::default()
            }),
            None
        );
        // Auto is make_game.py's default: not passed.
        let auto = build(Platform::Windows, "C:\\src", &windows()).unwrap();
        assert!(!auto.args.contains(&"--wave64-emulate".to_owned()));
    }

    #[test]
    fn play_on_windows_runs_the_loader_on_the_executable() {
        let settings = Settings {
            validation: true,
            gpu: "5070".to_owned(),
            ..windows()
        };
        let command = play(Platform::Windows, "game_scarlett_master.exe", &settings).unwrap();
        assert_eq!(command.program, "D:\\Games\\Leonida\\gta6rt.exe");
        assert_eq!(
            command.args,
            strings(&[
                "--run",
                "D:\\Games\\GTA6 package\\game_scarlett_master.exe",
                "--game-dir",
                "D:\\Games\\GTA6 package"
            ])
        );
        assert_eq!(
            command.env,
            vec![
                ("LEONIDA_D3D12".to_owned(), "vulkan".to_owned()),
                ("LEONIDA_VK_DEVICE".to_owned(), "5070".to_owned()),
                ("LEONIDA_VK_VALIDATION".to_owned(), "1".to_owned()),
            ]
        );
        assert_eq!(command.cwd, "D:\\Games\\Leonida");
    }

    #[test]
    fn diagnostics_become_the_loaders_options_and_variables() {
        let settings = Settings {
            trace_d3d12: true,
            unimplemented: Unimplemented::Stop,
            overlay: "/scratch/overlay".to_owned(),
            ..linux()
        };
        let command = play(Platform::Linux, "game_scarlett_master.exe", &settings).unwrap();
        assert_eq!(
            command.args[4..],
            strings(&["--overlay", "Z:\\scratch\\overlay", "--trace-d3d12"])
        );
        assert!(
            command
                .env
                .contains(&("LEONIDA_UNIMPLEMENTED".to_owned(), "stop".to_owned()))
        );
        let on_windows = Settings {
            overlay: "E:\\overlay".to_owned(),
            unimplemented: Unimplemented::Fail,
            ..windows()
        };
        let command = play(Platform::Windows, "g.exe", &on_windows).unwrap();
        assert_eq!(command.args[4..], strings(&["--overlay", "E:\\overlay"]));
        assert!(
            command
                .env
                .contains(&("LEONIDA_UNIMPLEMENTED".to_owned(), "fail".to_owned()))
        );
        // The loader's defaults are left out.
        let plain = play(Platform::Windows, "g.exe", &windows()).unwrap();
        assert_eq!(plain.args.len(), 4);
        assert!(
            plain
                .env
                .iter()
                .all(|(name, _)| name != "LEONIDA_UNIMPLEMENTED")
        );
    }

    #[test]
    fn play_on_linux_goes_through_wine() {
        let command = play(Platform::Linux, "game_scarlett_master.exe", &linux()).unwrap();
        assert_eq!(command.program, "/home/player/Leonida/gta6rt.sh");
        assert_eq!(
            command.args,
            strings(&[
                "--run",
                "./game_scarlett_master.exe",
                "--game-dir",
                "Z:\\games\\package"
            ])
        );
        assert_eq!(command.cwd, "/games/package");
        assert_eq!(
            command.env,
            vec![("LEONIDA_D3D12".to_owned(), "vulkan".to_owned())]
        );
        assert_eq!(
            play(Platform::Linux, "x.exe", &Settings::default()),
            Err(Missing::Package)
        );
    }

    #[test]
    fn paths_join_as_each_system_does() {
        assert_eq!(Platform::Windows.join("C:\\a", "b"), "C:\\a\\b");
        assert_eq!(Platform::Windows.join("C:\\", "b"), "C:\\b");
        assert_eq!(Platform::Linux.join("/a/", "b"), "/a/b");
        assert_eq!(Platform::Linux.loader(), "gta6rt.sh");
        assert_eq!(wine_path("/a b/c"), "Z:\\a b\\c");
        let command = check(Platform::Linux, "/src", &linux()).unwrap();
        assert_eq!(
            command.display(),
            "python3 -I -u /src/make_game.py --game /games/package --check"
        );
        let spaced = play(Platform::Windows, "g.exe", &windows()).unwrap();
        assert!(spaced.display().starts_with("LEONIDA_D3D12=vulkan "));
        assert!(spaced.display().contains("\"D:\\Games\\GTA6 package\""));
    }
}
