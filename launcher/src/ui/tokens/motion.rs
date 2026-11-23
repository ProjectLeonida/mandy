// SPDX-License-Identifier: GPL-3.0-or-later

//! Motion: short and decisive, 150–400ms on one easing curve, and nothing looping at idle but
//! what shows that work is under way.

use std::time::Duration;

/// Small changes: a hover, a press, a fade.
pub const FAST: Duration = Duration::from_millis(150);
/// Most transitions: focus and selection.
pub const NORMAL: Duration = Duration::from_millis(200);
/// Large movements.
pub const SLOW: Duration = Duration::from_millis(400);

/// The text caret's blink: shown this long, then hidden this long.
pub const CARET_BLINK: Duration = Duration::from_millis(500);
/// The caret blinks only this long after the last input, then stays shown, so nothing draws at
/// idle.
pub const CARET_BLINKS_FOR: Duration = Duration::from_secs(10);
/// An app draws nothing this long after the last input, unless work is under way.
pub const IDLE: Duration = Duration::from_secs(12);

/// A CSS `cubic-bezier()` easing curve, from (0, 0) to (1, 1) through two control points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicBezier {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl CubicBezier {
    /// How far along the motion is at `t`, the share of its time gone, both from 0 to 1: the
    /// curve's y where its x is `t`, as CSS works it out.
    pub fn at(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        // One axis of the curve, its ends at 0 and 1: at s, and its slope.
        let axis = |p1: f32, p2: f32, s: f32| {
            let u = 1.0 - s;
            3.0 * u * u * s * p1 + 3.0 * u * s * s * p2 + s * s * s
        };
        let slope = |p1: f32, p2: f32, s: f32| {
            let u = 1.0 - s;
            3.0 * u * u * p1 + 6.0 * u * s * (p2 - p1) + 3.0 * s * s * (1.0 - p2)
        };
        // The s whose x is t: Newton's method, and halving where it stalls.
        let mut s = t;
        for _ in 0..8 {
            let error = axis(self.x1, self.x2, s) - t;
            let d = slope(self.x1, self.x2, s);
            if error.abs() < 1e-5 || d.abs() < 1e-6 {
                break;
            }
            s = (s - error / d).clamp(0.0, 1.0);
        }
        if (axis(self.x1, self.x2, s) - t).abs() > 1e-4 {
            let (mut low, mut high) = (0.0_f32, 1.0_f32);
            for _ in 0..30 {
                s = (low + high) / 2.0;
                if axis(self.x1, self.x2, s) < t {
                    low = s;
                } else {
                    high = s;
                }
            }
        }
        axis(self.y1, self.y2, s)
    }
}

/// `cubic-bezier(.2, .7, .2, 1)`: quick to start, gentle to settle.
pub const EASE: CubicBezier = CubicBezier {
    x1: 0.2,
    y1: 0.7,
    x2: 0.2,
    y2: 1.0,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_are_short_and_decisive() {
        for duration in [FAST, NORMAL, SLOW] {
            assert!((150..=400).contains(&duration.as_millis()), "{duration:?}");
        }
        assert!(FAST < NORMAL && NORMAL < SLOW);
    }

    #[test]
    fn the_caret_stops_blinking_well_before_idle() {
        assert!(CARET_BLINKS_FOR + CARET_BLINK < IDLE);
    }

    #[test]
    fn the_curve_runs_from_start_to_end_and_never_back() {
        assert!(EASE.at(0.0).abs() < 1e-4);
        assert!((EASE.at(1.0) - 1.0).abs() < 1e-4);
        let mut last = 0.0;
        for step in 1..=100 {
            let y = EASE.at(step as f32 / 100.0);
            assert!(y >= last - 1e-5, "{step}: {y} after {last}");
            last = y;
        }
        // Quick to start: well past halfway at a quarter of the time.
        assert!(EASE.at(0.25) > 0.5, "{}", EASE.at(0.25));
        let linear = CubicBezier {
            x1: 0.25,
            y1: 0.25,
            x2: 0.75,
            y2: 0.75,
        };
        assert!((linear.at(0.3) - 0.3).abs() < 1e-3);
    }
}
