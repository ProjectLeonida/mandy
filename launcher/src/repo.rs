// SPDX-License-Identifier: GPL-3.0-or-later

//! Finding the Leonida repository the launcher runs from, and what it says of the game.

use std::fs;
use std::path::{Path, PathBuf};

/// The file that marks the repository's root.
pub const MARKER: &str = "make_game.py";

/// The game's executable when no manifest says otherwise (`game/xbox-1.0.2.1.json`'s
/// `expected_exe`).
pub const DEFAULT_EXE: &str = "game_scarlett_master.exe";

/// The repository: `--repo PATH` if given (it must hold `make_game.py`), or the first folder
/// holding it walking up from the launcher's own executable, then from the current folder.
pub fn find(
    given: Option<&Path>,
    exe: Option<&Path>,
    cwd: Option<&Path>,
) -> Result<PathBuf, String> {
    if let Some(given) = given {
        return if given.join(MARKER).is_file() {
            Ok(given.to_path_buf())
        } else {
            Err(format!("--repo {}: no {MARKER} there", given.display()))
        };
    }
    exe.and_then(Path::parent)
        .and_then(walk_up)
        .or_else(|| cwd.and_then(walk_up))
        .ok_or_else(|| {
            format!(
                "no Leonida repository found above the launcher or the current folder: run it \
                 from inside one, or pass --repo PATH (the folder that holds {MARKER})"
            )
        })
}

/// The first folder from `start` up that holds `make_game.py`.
fn walk_up(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|folder| folder.join(MARKER).is_file())
        .map(Path::to_path_buf)
}

/// The executable the loader runs, from the repository's version manifests (`game/*.json`,
/// each with an `expected_exe`); the 1.0.2.1 name if none says.
pub fn expected_exe(repo: &Path) -> String {
    let Ok(entries) = fs::read_dir(repo.join("game")) else {
        return DEFAULT_EXE.to_owned();
    };
    let mut manifests: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    manifests.sort();
    manifests
        .iter()
        .rev()
        .find_map(|manifest| {
            let text = fs::read_to_string(manifest).ok()?;
            let value: serde_json::Value = serde_json::from_str(&text).ok()?;
            value.get("expected_exe")?.as_str().map(str::to_owned)
        })
        .unwrap_or_else(|| DEFAULT_EXE.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "leonida-launcher-repo-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("launcher/target/release")).unwrap();
        fs::write(root.join(MARKER), "").unwrap();
        root
    }

    #[test]
    fn the_repository_is_found_above_the_executable() {
        let root = tree("exe");
        let exe = root.join("launcher/target/release/leonida-launcher");
        assert_eq!(find(None, Some(&exe), None), Ok(root.clone()));
        // Not above the executable: from the current folder.
        let elsewhere = std::env::temp_dir().join("leonida-launcher-nowhere/bin/x");
        let cwd = root.join("launcher");
        assert_eq!(find(None, Some(&elsewhere), Some(&cwd)), Ok(root.clone()));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repo_given_must_hold_make_game() {
        let root = tree("given");
        assert_eq!(find(Some(&root), None, None), Ok(root.clone()));
        let wrong = root.join("launcher");
        let error = find(Some(&wrong), None, None).unwrap_err();
        assert!(error.contains("no make_game.py"), "{error}");
        let none = find(None, None, None).unwrap_err();
        assert!(none.contains("--repo"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn the_executable_comes_from_the_manifest() {
        let root = tree("manifest");
        assert_eq!(expected_exe(&root), DEFAULT_EXE);
        fs::create_dir_all(root.join("game")).unwrap();
        fs::write(
            root.join("game/xbox-9.json"),
            "{\"title\": \"t\", \"expected_exe\": \"other.exe\", \"files\": {}}",
        )
        .unwrap();
        assert_eq!(expected_exe(&root), "other.exe");
        let _ = fs::remove_dir_all(root);
    }
}
