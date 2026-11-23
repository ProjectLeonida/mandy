// SPDX-License-Identifier: GPL-3.0-or-later

//! The launcher's fonts: the faces its text roles use, compiled in, so its text looks the same
//! on any system. [`load`] puts them in iced's font system before anything is drawn.

use std::sync::{Arc, Once, PoisonError};

use crate::ui::tokens::typography::{Family, Slant};
use crate::ui::tokens::{Face, Role};
use iced::advanced::graphics::text::cosmic_text::fontdb;
use iced::font::{self, Font, Weight};

/// The default: interface text, IBM Plex Sans at 400.
pub const DEFAULT: Font = of(crate::ui::tokens::typography::INTERFACE.face);

/// The iced font for a face.
pub const fn of(face: Face) -> Font {
    Font {
        family: font::Family::Name(face.family.name()),
        weight: weight(face.weight),
        stretch: font::Stretch::Normal,
        style: match face.slant {
            Slant::Upright => font::Style::Normal,
        },
    }
}

/// The iced font for a role.
pub const fn role(role: &Role) -> Font {
    of(role.face)
}

/// CSS weights to iced's, to the nearest hundred.
const fn weight(css: u16) -> Weight {
    match css {
        0..150 => Weight::Thin,
        150..250 => Weight::ExtraLight,
        250..350 => Weight::Light,
        350..450 => Weight::Normal,
        450..550 => Weight::Medium,
        550..650 => Weight::Semibold,
        650..750 => Weight::Bold,
        750..850 => Weight::ExtraBold,
        _ => Weight::Black,
    }
}

/// The font files, in the order of [`FACES`], under the SIL Open Font License 1.1 (their licence
/// texts are beside them in launcher/fonts, with SHA256SUMS).
pub const SHIPPED: [&[u8]; 4] = [
    include_bytes!("../../fonts/ChakraPetch-SemiBold.ttf"),
    include_bytes!("../../fonts/ChakraPetch-Bold.ttf"),
    include_bytes!("../../fonts/IBMPlexSans-Regular.ttf"),
    include_bytes!("../../fonts/JetBrainsMono-Regular.ttf"),
];

/// Loads the fonts into iced's font system, before anything is drawn. The generic families point
/// at the launcher's: sans-serif at IBM Plex Sans, monospace at JetBrains Mono. Only the first
/// call loads.
pub fn load() {
    static LOADED: Once = Once::new();
    LOADED.call_once(|| {
        let mut system = iced::advanced::graphics::text::font_system()
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        load_into(system.raw().db_mut());
    });
}

fn load_into(db: &mut fontdb::Database) {
    for bytes in SHIPPED {
        db.load_font_source(fontdb::Source::Binary(Arc::new(bytes)));
    }
    db.set_sans_serif_family(Family::Body.name());
    db.set_monospace_family(Family::Mono.name());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tokens::typography::FACES;

    #[test]
    fn every_shipped_face_is_its_file() {
        let mut db = fontdb::Database::new();
        load_into(&mut db);
        for face in FACES {
            let query = fontdb::Query {
                families: &[fontdb::Family::Name(face.family.name())],
                weight: fontdb::Weight(face.weight),
                stretch: fontdb::Stretch::Normal,
                style: fontdb::Style::Normal,
            };
            let id = db.query(&query).expect("the face is loaded");
            let found = db.face(id).unwrap();
            assert_eq!(found.weight, fontdb::Weight(face.weight), "{face:?}");
        }
        assert_eq!(db.len(), FACES.len());
    }

    #[test]
    fn weights_round_to_iceds() {
        assert_eq!(weight(400), Weight::Normal);
        assert_eq!(weight(600), Weight::Semibold);
        assert_eq!(weight(700), Weight::Bold);
        assert_eq!(DEFAULT.weight, Weight::Normal);
    }
}
