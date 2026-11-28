// SPDX-License-Identifier: GPL-3.0-or-later

//! Type: three font families, the faces the launcher ships, and the roles text takes.
//!
//! Chakra Petch is the display face, for titles and spaced-out uppercase labels; IBM Plex Sans is
//! for interface text; JetBrains Mono for metadata, paths, status and the log.

/// A font family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Display,
    Body,
    Mono,
}

impl Family {
    /// The family's name, as the fonts give it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Display => "Chakra Petch",
            Self::Body => "IBM Plex Sans",
            Self::Mono => "JetBrains Mono",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slant {
    Upright,
}

/// One font file: a family at a weight and slant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Face {
    pub family: Family,
    /// From 100 to 900, as in CSS: 400 is regular, 600 semibold.
    pub weight: u16,
    pub slant: Slant,
}

impl Face {
    /// The face's file in `launcher/fonts`; `None` for a face the launcher doesn't ship.
    pub const fn file(self) -> Option<&'static str> {
        Some(match (self.family, self.weight, self.slant) {
            (Family::Display, 600, Slant::Upright) => "ChakraPetch-SemiBold.ttf",
            (Family::Display, 700, Slant::Upright) => "ChakraPetch-Bold.ttf",
            (Family::Body, 400, Slant::Upright) => "IBMPlexSans-Regular.ttf",
            (Family::Mono, 400, Slant::Upright) => "JetBrainsMono-Regular.ttf",
            _ => return None,
        })
    }

    const fn upright(family: Family, weight: u16) -> Self {
        Self {
            family,
            weight,
            slant: Slant::Upright,
        }
    }
}

/// The faces the launcher ships, and so the only ones text may use.
pub const FACES: [Face; 4] = [
    Face::upright(Family::Display, 600),
    Face::upright(Family::Display, 700),
    Face::upright(Family::Body, 400),
    Face::upright(Family::Mono, 400),
];

/// A role text plays, with how it is set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Role {
    pub name: &'static str,
    pub face: Face,
    /// In logical pixels.
    pub size: f32,
    /// A multiple of the size.
    pub line_height: f32,
    /// Extra space after each letter, in ems.
    pub letter_spacing: f32,
    /// Set in capitals, whatever the text's case.
    pub uppercase: bool,
}

impl Role {
    const fn new(name: &'static str, face: Face, size: f32, line_height: f32) -> Self {
        Self {
            name,
            face,
            size,
            line_height,
            letter_spacing: 0.0,
            uppercase: false,
        }
    }

    /// The extra space after each letter, in logical pixels.
    pub fn letter_spacing_px(&self) -> f32 {
        self.letter_spacing * self.size
    }
}

/// Page and window titles.
pub const TITLE: Role = Role::new("title", Face::upright(Family::Display, 600), 34.0, 1.15);
/// Section headings.
pub const HEADING: Role = Role::new("heading", Face::upright(Family::Display, 600), 21.0, 1.3);
/// Small uppercase labels, spaced out: section names, eyebrows.
pub const LABEL: Role = Role {
    letter_spacing: 0.14,
    uppercase: true,
    ..Role::new("label", Face::upright(Family::Display, 600), 11.0, 1.3)
};
/// An app's name in its header bar: the display face, bold, a step larger than a label and more
/// spaced.
pub const BRAND: Role = Role {
    letter_spacing: 0.24,
    uppercase: true,
    ..Role::new("brand", Face::upright(Family::Display, 700), 13.0, 1.3)
};
/// Interface text: controls, lists, messages. The default.
pub const INTERFACE: Role = Role::new("interface", Face::upright(Family::Body, 400), 14.0, 1.3);
/// Metadata, counts, paths and the status line.
pub const META: Role = Role::new("meta", Face::upright(Family::Mono, 400), 11.0, 1.3);

pub const ROLES: [Role; 6] = [TITLE, HEADING, LABEL, BRAND, INTERFACE, META];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_role_uses_a_shipped_face() {
        for role in ROLES {
            assert!(FACES.contains(&role.face), "{}", role.name);
        }
    }

    #[test]
    fn every_shipped_face_has_a_file_of_its_own() {
        let files: Vec<_> = FACES.iter().map(|face| face.file().unwrap()).collect();
        for (at, file) in files.iter().enumerate() {
            assert!(!files[..at].contains(file), "{file}");
        }
        assert_eq!(Face::upright(Family::Body, 700).file(), None);
    }

    #[test]
    fn labels_are_spaced_out_capitals() {
        let capitals: Vec<_> = ROLES.iter().filter(|r| r.uppercase).collect();
        assert_eq!(capitals, [&LABEL, &BRAND]);
        assert!((LABEL.letter_spacing_px() - 1.54).abs() < 1e-6);
        const { assert!(BRAND.size > LABEL.size && BRAND.face.weight > LABEL.face.weight) };
    }

    #[test]
    fn sizes_from_the_design_language() {
        assert_eq!(TITLE.size, 34.0);
        assert_eq!(HEADING.size, 21.0);
        assert_eq!(LABEL.size, 11.0);
        assert_eq!(INTERFACE.size, 14.0);
        assert_eq!(META.face.family, Family::Mono);
    }
}
