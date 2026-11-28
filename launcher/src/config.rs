// SPDX-License-Identifier: GPL-3.0-or-later

//! The launcher's settings: one small JSON file in the platform's per-user configuration folder
//! (`%APPDATA%\Leonida\launcher.json` on Windows, `$XDG_CONFIG_HOME/leonida/launcher.json` or
//! `~/.config/leonida/launcher.json` on Linux), never in the repository.
//!
//! The file carries a `version`. A missing, unreadable or broken file gives the defaults; so does
//! one from a newer launcher, which is then left alone until something is changed and saved.
//! Fields the file lacks take their defaults, and fields it has that this version doesn't know are
//! dropped on the next save.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The settings file's version this launcher writes.
pub const VERSION: u32 = 1;

/// Where the settings live, under the platform's per-user configuration folder.
pub fn path() -> Option<PathBuf> {
    let folder = if cfg!(windows) { "Leonida" } else { "leonida" };
    dirs::config_dir().map(|config| config.join(folder).join("launcher.json"))
}

/// The wave64 translation mode, `make_game.py --wave64-emulate`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Wave64 {
    /// make_game.py's default: it asks the GPU.
    #[default]
    Auto,
    On,
    Off,
}

impl Wave64 {
    /// The option's value.
    pub const fn flag(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

/// What the loader does with a call to an import it doesn't implement
/// (`LEONIDA_UNIMPLEMENTED`, runtime/README.md).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Unimplemented {
    /// The loader's default: log it once and return 0.
    #[default]
    Log,
    /// Return `E_NOTIMPL` from the ones that return an `HRESULT`.
    Fail,
    /// End the process with exit code 3 at the first.
    Stop,
}

impl Unimplemented {
    /// The variable's value.
    pub const fn value(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Fail => "fail",
            Self::Stop => "stop",
        }
    }
}

/// The light or dark palette.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Shade {
    #[default]
    Dark,
    Light,
}

/// The stages `make_game.py --stages` chooses among, in order, after its check (which always
/// runs first).
pub const STAGES: [&str; 4] = ["shaders", "assets", "runtime", "pipelines"];

/// Every setting, as the file holds it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub version: u32,
    /// The game package: the folder that holds `MicrosoftGame.Config` (`--game`).
    pub package: String,
    /// Where the finished port goes (`--out`).
    pub output: String,
    /// Intermediate files (`--work`); empty is make_game.py's `<out>/work`.
    pub work: String,
    /// Which build stages run, by name from [`STAGES`]; all of them is make_game.py's default.
    pub stages: Vec<String>,
    /// Also decode the audio to WAV (`--wav`).
    pub wav: bool,
    /// Compare every file's SHA-256 when checking (`--full-check`).
    pub full_check: bool,
    /// Part of the GPU's name (`LEONIDA_VK_DEVICE`); empty takes the runtime's own choice.
    pub gpu: String,
    /// `--wave64-emulate`.
    pub wave64: Wave64,
    /// `--jobs`; empty is make_game.py's default, half the CPU threads.
    pub jobs: String,
    /// The Khronos validation layer at play (`LEONIDA_VK_VALIDATION=1`).
    pub validation: bool,
    /// The loader logs every D3D12 call and its outcome (`gta6rt --trace-d3d12`).
    pub trace_d3d12: bool,
    /// What the loader does with an unimplemented import (`LEONIDA_UNIMPLEMENTED`).
    pub unimplemented: Unimplemented,
    /// Where the game's writes go instead of the package (`gta6rt --overlay`); empty is the
    /// loader's default, `<package>\overlay`.
    pub overlay: String,
    /// Light or dark.
    pub shade: Shade,
    /// The package last checked without a fault, and the version make_game.py said it is.
    pub checked: Option<Checked>,
}

/// A package that passed `make_game.py --check`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checked {
    pub package: String,
    /// As make_game.py says it: "Xbox Series package 1.0.2.1".
    pub game: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: VERSION,
            package: String::new(),
            output: String::new(),
            work: String::new(),
            stages: STAGES.iter().map(|stage| (*stage).to_owned()).collect(),
            wav: false,
            full_check: false,
            gpu: String::new(),
            wave64: Wave64::Auto,
            jobs: String::new(),
            validation: false,
            trace_d3d12: false,
            unimplemented: Unimplemented::Log,
            overlay: String::new(),
            shade: Shade::Dark,
            checked: None,
        }
    }
}

/// What reading the settings found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Loaded {
    /// The file, read.
    Read(Box<Settings>),
    /// No file yet: the defaults.
    Missing,
    /// A file that couldn't be read or parsed, or isn't the settings: the defaults, and why.
    Unreadable(String),
    /// A newer launcher's file: the defaults, and the file is kept as it is until a save.
    Newer(u32),
}

impl Loaded {
    /// The settings to use.
    pub fn settings(&self) -> Settings {
        match self {
            Self::Read(settings) => (**settings).clone(),
            _ => Settings::default(),
        }
    }

    /// What to say of it, if anything.
    pub fn note(&self) -> Option<String> {
        match self {
            Self::Read(_) | Self::Missing => None,
            Self::Unreadable(why) => Some(format!(
                "Settings couldn't be read ({why}); using defaults."
            )),
            Self::Newer(version) => Some(format!(
                "Settings are from a newer launcher (version {version}); using defaults."
            )),
        }
    }
}

/// Reads settings from text.
pub fn parse(text: &str) -> Loaded {
    let value: serde_json::Value = match serde_json::from_str(text) {
        Ok(value) => value,
        Err(error) => return Loaded::Unreadable(error.to_string()),
    };
    let Some(version) = value.get("version").and_then(serde_json::Value::as_u64) else {
        return Loaded::Unreadable("no version".to_owned());
    };
    let version = u32::try_from(version).unwrap_or(u32::MAX);
    if version > VERSION {
        return Loaded::Newer(version);
    }
    if version == 0 {
        return Loaded::Unreadable("version 0".to_owned());
    }
    match serde_json::from_value::<Settings>(value) {
        Ok(settings) => Loaded::Read(Box::new(Settings {
            version: VERSION,
            ..settings
        })),
        Err(error) => Loaded::Unreadable(error.to_string()),
    }
}

/// Reads the settings file at `path`.
pub fn load(path: &Path) -> Loaded {
    match fs::read_to_string(path) {
        Ok(text) => parse(&text),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Loaded::Missing,
        Err(error) => Loaded::Unreadable(error.to_string()),
    }
}

/// The settings as the file holds them.
pub fn to_text(settings: &Settings) -> String {
    let settings = Settings {
        version: VERSION,
        ..settings.clone()
    };
    let mut text = serde_json::to_string_pretty(&settings).unwrap_or_default();
    text.push('\n');
    text
}

/// Writes the settings to `path`, through a file beside it, so a crash leaves the old one.
pub fn save(path: &Path, settings: &Settings) -> io::Result<()> {
    if let Some(folder) = path.parent() {
        fs::create_dir_all(folder)?;
    }
    let partial = path.with_extension("json.partial");
    fs::write(&partial, to_text(settings))?;
    fs::rename(&partial, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!(
            "leonida-launcher-test-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&folder);
        folder
    }

    #[test]
    fn settings_go_round_the_file() {
        let folder = scratch("round");
        let path = folder.join("leonida").join("launcher.json");
        let settings = Settings {
            package: "D:\\Games\\package".to_owned(),
            output: "/home/player/Leonida".to_owned(),
            work: String::new(),
            stages: vec!["shaders".to_owned(), "runtime".to_owned()],
            wav: true,
            full_check: true,
            gpu: "radeon".to_owned(),
            wave64: Wave64::On,
            jobs: "6".to_owned(),
            validation: true,
            shade: Shade::Light,
            checked: Some(Checked {
                package: "D:\\Games\\package".to_owned(),
                game: "Xbox Series package 1.0.2.1".to_owned(),
            }),
            ..Settings::default()
        };
        assert_eq!(load(&path), Loaded::Missing);
        save(&path, &settings).unwrap();
        assert_eq!(load(&path), Loaded::Read(Box::new(settings)));
        assert!(!path.with_extension("json.partial").exists());
        let _ = fs::remove_dir_all(folder);
    }

    #[test]
    fn a_missing_or_broken_file_gives_the_defaults() {
        assert_eq!(Loaded::Missing.settings(), Settings::default());
        for text in [
            "",
            "{",
            "[]",
            "{\"package\": \"x\"}",
            "{\"version\": \"one\"}",
        ] {
            let loaded = parse(text);
            assert!(
                matches!(loaded, Loaded::Unreadable(_)),
                "{text}: {loaded:?}"
            );
            assert_eq!(loaded.settings(), Settings::default());
            assert!(loaded.note().is_some());
        }
        // A field of the wrong kind is a broken file, not half a file.
        let loaded = parse("{\"version\": 1, \"wav\": \"yes\"}");
        assert!(matches!(loaded, Loaded::Unreadable(_)));
    }

    #[test]
    fn a_newer_file_gives_the_defaults_and_says_so() {
        let loaded = parse("{\"version\": 7, \"package\": \"x\"}");
        assert_eq!(loaded, Loaded::Newer(7));
        assert_eq!(loaded.settings(), Settings::default());
        assert!(loaded.note().unwrap().contains("version 7"));
        assert!(matches!(parse("{\"version\": 0}"), Loaded::Unreadable(_)));
    }

    #[test]
    fn missing_fields_take_defaults_and_unknown_ones_are_dropped() {
        let loaded = parse("{\"version\": 1, \"package\": \"/games/pkg\", \"later\": 3}");
        let Loaded::Read(settings) = loaded else {
            panic!("{loaded:?}")
        };
        assert_eq!(settings.package, "/games/pkg");
        assert_eq!(settings.stages, Settings::default().stages);
        assert_eq!(settings.wave64, Wave64::Auto);
        assert!(!to_text(&settings).contains("later"));
        assert!(to_text(&settings).contains("\"version\": 1"));
    }

    #[test]
    fn the_file_is_in_the_users_configuration_folder() {
        if let Some(path) = path() {
            assert!(path.ends_with("launcher.json"));
            assert!(path.parent().unwrap().ends_with(if cfg!(windows) {
                "Leonida"
            } else {
                "leonida"
            }));
        }
        assert_eq!(Wave64::default().flag(), "auto");
    }
}
