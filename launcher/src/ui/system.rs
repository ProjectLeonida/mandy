// SPDX-License-Identifier: GPL-3.0-or-later

//! What the system asks of the interface: motion reduced, for those it bothers, and battery saver,
//! for the battery. Each is read once, from an environment variable, until Settings has the
//! switch (the design language's Motion).
//!
//! - **Reduced motion** turns off every animation that isn't needed: a wheel's steps jump, the
//!   ends don't give, a dialog or a page is there at once, a loader holds still.
//! - **Battery saver** shortens animations to half ([`duration`]), and the theme it's set on
//!   ([`Theme::saving_battery`](crate::ui::Theme::saving_battery)) shrinks glow to half its blur and
//!   leaves out the void's radial tints, the one gradient a shader draws over a whole surface.

use std::sync::OnceLock;
use std::time::Duration;

/// The environment variable that turns non-essential motion off: `LEONIDA_REDUCE_MOTION=1`.
pub const REDUCE_MOTION_VARIABLE: &str = "LEONIDA_REDUCE_MOTION";

/// The environment variable that saves battery: `LEONIDA_BATTERY_SAVER=1`.
pub const BATTERY_SAVER_VARIABLE: &str = "LEONIDA_BATTERY_SAVER";

/// Whether a variable is set to 1.
fn on(variable: &str) -> bool {
    std::env::var_os(variable).is_some_and(|value| value == "1")
}

/// Whether motion is reduced ([`REDUCE_MOTION_VARIABLE`]), read once.
pub fn reduced_motion() -> bool {
    static REDUCED: OnceLock<bool> = OnceLock::new();
    *REDUCED.get_or_init(|| on(REDUCE_MOTION_VARIABLE))
}

/// Whether battery saver is on ([`BATTERY_SAVER_VARIABLE`]), read once.
pub fn battery_saver() -> bool {
    static SAVING: OnceLock<bool> = OnceLock::new();
    *SAVING.get_or_init(|| on(BATTERY_SAVER_VARIABLE))
}

/// How long an animation the design gives `whole` lasts: half as long on battery saver, so half as
/// many frames are drawn.
pub fn duration(whole: Duration) -> Duration {
    shortened(whole, battery_saver())
}

fn shortened(whole: Duration, saving: bool) -> Duration {
    if saving { whole / 2 } else { whole }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn battery_saver_halves_animations() {
        let slow = crate::ui::tokens::motion::SLOW;
        assert_eq!(shortened(slow, false), slow);
        assert_eq!(shortened(slow, true), slow / 2);
    }
}
