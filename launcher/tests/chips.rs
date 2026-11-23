// SPDX-License-Identifier: GPL-3.0-or-later

//! Every status chip, side by side on the Play page's background in each mode: over the text
//! column (the page's background at 0.97 over the scene) and over the scene itself. Written as a
//! PNG for the eye, to `$LEONIDA_LAUNCHER_SCREENSHOTS/chips-<mode>.png`; checked here for its
//! words' contrast on the chip, over the column.

mod common;

use std::path::PathBuf;

use iced::Size;
use iced::widget::{Stack, column, container, image, row};

use leonida_launcher::app::{COLUMN, Message};
use leonida_launcher::scene::{self, Colors, Input};
use leonida_launcher::ui::theme::Status;
use leonida_launcher::ui::tokens::color::contrast;
use leonida_launcher::ui::tokens::{Palette, Surface, space};
use leonida_launcher::ui::{Element, Theme, status};

const STATES: [(Status, &str); 6] = [
    (Status::Waiting, "Setup"),
    (Status::Running, "Running"),
    (Status::Passed, "Checked"),
    (Status::Info, "Not bootable yet"),
    (Status::Warning, "Warning"),
    (Status::Failed, "Failed"),
];

#[test]
fn every_chip_on_the_play_pages_background() {
    let size = Size::new(1040.0, 360.0);
    let folder = std::env::var_os("LEONIDA_LAUNCHER_SCREENSHOTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screenshots"));
    for palette in Palette::ALL {
        let theme = Theme::new(palette);
        let picture = scene::generate(&Input {
            colors: Colors::of(&palette),
            width: size.width as u32,
            height: size.height as u32,
            scale: 1.0,
            column: COLUMN,
            corner: None,
        })
        .unwrap();
        let chips = || {
            column(STATES.map(|(state, word)| Element::<Message>::from(status(state, word))))
                .spacing(space::M)
        };
        let view: Element<'_, Message> = Stack::with_children([
            image(image::Handle::from_rgba(
                picture.width,
                picture.height,
                picture.rgba,
            ))
            .into(),
            container(row![chips(), iced::widget::space().width(600), chips()])
                .padding(space::XXL)
                .into(),
        ])
        .into();
        let rgba = common::render_view(view, &theme, size);
        common::write_png(
            &folder.join(format!("chips-{}.png", palette.name)),
            (size.width as u32, size.height as u32),
            &rgba,
        );
        // Each chip's word on its own tint, over the column's background, keeps 4.5:1.
        let bg = palette.surface(Surface::Bg);
        for (state, _) in STATES {
            let look = theme.status(state);
            let fill = look.fill.map_or(bg, |fill| fill.over(bg));
            let ratio = contrast(look.ink, fill);
            assert!(ratio >= 4.5, "{}: {state:?}: {ratio:.2}", palette.name);
        }
    }
}
