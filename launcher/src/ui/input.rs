// SPDX-License-Identifier: GPL-3.0-or-later

//! What every widget's input handling shares: whether a key's press is a plain one, and asking
//! for a redraw only when what a widget would draw has changed.

/// Asks for a redraw when what a widget would draw differs from what it last drew; a redraw
/// records what it draws. So nothing draws while nothing changes.
pub(crate) fn redraw_on_change<T: PartialEq + Copy, Message>(
    drawn: &mut Option<T>,
    now: T,
    event: &iced::Event,
    shell: &mut iced::advanced::Shell<'_, Message>,
) {
    if let iced::Event::Window(iced::window::Event::RedrawRequested(_)) = event {
        *drawn = Some(now);
    } else if *drawn != Some(now) {
        shell.request_redraw();
    }
}

/// Whether a key press is plain: no Ctrl, Alt or Super with it.
pub(crate) fn plain(modifiers: iced::keyboard::Modifiers) -> bool {
    !(modifiers.control() || modifiers.alt() || modifiers.logo())
}
