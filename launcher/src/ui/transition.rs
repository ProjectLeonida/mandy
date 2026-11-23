// SPDX-License-Identifier: GPL-3.0-or-later

//! Something on its way from where it is to where it's going, on the design language's curve: a checkbox's
//! mark fading in, a switch's knob sliding across, a dialog coming up, a page sliding in.
//!
//! Turned back part of the way, it goes back from where it is. Starting again from where it would
//! be had it begun at the far end jumps: the design language's curve is quick to start and slow to settle, so
//! a quarter of the time in is three quarters of the way, and the same time back is hardly any.

use std::time::{Duration, Instant};

use crate::ui::tokens::motion;
use iced::Rectangle;
use iced::advanced::Shell;

/// A value on its way from one place to another: a share, from 0 to 1, or anywhere between.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Transition {
    from: f32,
    to: f32,
    /// How long it takes: the whole way, from 0 to 1, the time it was given, less of the way as
    /// much less.
    takes: Duration,
    /// When it started: the first frame drawn after it was asked for.
    start: Option<Instant>,
    /// Where it was at the last frame.
    at: f32,
}

impl Transition {
    /// From `from` to `to`, the whole way from 0 to 1 taking `whole`; none with motion reduced,
    /// or when it's already there, as it's then where it's going at once.
    pub fn start(from: f32, to: f32, whole: Duration) -> Option<Self> {
        (from != to && !crate::ui::system::reduced_motion())
            .then(|| Self::new(from, to, crate::ui::system::duration(whole)))
    }

    fn new(from: f32, to: f32, whole: Duration) -> Self {
        Self {
            from,
            to,
            takes: whole.mul_f32((to - from).abs().min(1.0)),
            start: None,
            at: from,
        }
    }

    /// Where it is, as of the last frame.
    pub fn at(&self) -> f32 {
        self.at
    }

    /// A frame at `now`: where it is then, and whether it's still on its way.
    pub fn frame(&mut self, now: Instant) -> bool {
        let start = *self.start.get_or_insert(now);
        let t = if self.takes.is_zero() {
            1.0
        } else {
            now.saturating_duration_since(start).as_secs_f32() / self.takes.as_secs_f32()
        };
        if t >= 1.0 {
            self.at = self.to;
            false
        } else {
            self.at = self.from + (self.to - self.from) * motion::EASE.at(t);
            true
        }
    }

    /// At a frame, the transition in `slot` moves on: the next frame's asked for while it's on
    /// its way, and it's gone once it's there.
    pub fn advance<Message>(slot: &mut Option<Self>, now: Instant, shell: &mut Shell<'_, Message>) {
        if let Some(transition) = slot {
            if transition.frame(now) {
                shell.request_redraw();
            } else {
                *slot = None;
            }
        }
    }

    /// Turns what's in `slot` toward `to`, from where it's got to, or, if nothing's moving, from
    /// `was`, and asks for a frame. Nothing moves with motion reduced: it's at `to` at once.
    pub fn turn<Message>(
        slot: &mut Option<Self>,
        was: f32,
        to: f32,
        whole: Duration,
        shell: &mut Shell<'_, Message>,
    ) {
        let from = slot.map_or(was, |transition| transition.at);
        *slot = Self::start(from, to, whole);
        shell.request_redraw();
    }
}

/// A control's glow coming and going: under the pointer it fades in, and away from it, out, over
/// [`motion::FAST`], so light comes from interaction without a jump.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Lit {
    /// How lit it is, from 0 to 1, as of the last frame.
    share: f32,
    /// Whether it's going to be lit, once it has been told.
    aim: Option<bool>,
    fade: Option<Transition>,
}

impl Lit {
    /// It should be lit, or not: it fades there from where it is.
    pub fn aim<Message>(&mut self, lit: bool, shell: &mut Shell<'_, Message>) {
        if self.aim == Some(lit) {
            return;
        }
        let to = f32::from(u8::from(lit));
        if self.aim.is_none() {
            // Its first look: as it is, no fade.
            self.share = to;
        } else {
            Transition::turn(&mut self.fade, self.share, to, motion::FAST, shell);
            if self.fade.is_none() {
                self.share = to;
            }
        }
        self.aim = Some(lit);
    }

    /// A frame: where the fade is, asking for the next while it's on its way.
    pub fn frame<Message>(&mut self, event: &iced::Event, shell: &mut Shell<'_, Message>) {
        if let iced::Event::Window(iced::window::Event::RedrawRequested(now)) = event {
            Transition::advance(&mut self.fade, *now, shell);
            self.share = match &self.fade {
                Some(fade) => fade.at(),
                None => f32::from(u8::from(self.aim.unwrap_or(false))),
            };
        }
    }

    /// How lit it is: 0 to 1.
    pub fn share(&self) -> f32 {
        self.share
    }
}

/// How far into a cycle `now` is, from 0 to 1, cycles having begun at `start`. In whole
/// microseconds, so it's as smooth after a week as after a second: a second's fraction in an
/// `f32` loses its smoothness after a few days.
pub(crate) fn phase(start: Instant, now: Instant, cycle: Duration) -> f32 {
    let into = now.saturating_duration_since(start).as_micros() % cycle.as_micros().max(1);
    into as f32 / cycle.as_micros().max(1) as f32
}

/// Whether something animating forever should ask for its next frame: not with motion reduced,
/// nor while it's out of sight, scrolled away; scrolled back, the redraws that scrolling it
/// brings bring it back to life.
pub(crate) fn wants_frames(bounds: Rectangle, viewport: &Rectangle) -> bool {
    !crate::ui::system::reduced_motion() && bounds.intersects(viewport)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_eases_there_and_stops() {
        let mut transition = Transition::new(0.0, 1.0, motion::FAST);
        let start = Instant::now();
        assert!(transition.frame(start));
        assert_eq!(transition.at(), 0.0);
        assert!(transition.frame(start + motion::FAST / 2));
        let mid = transition.at();
        assert!(mid > 0.5 && mid < 1.0, "quick to start: {mid}");
        assert!(!transition.frame(start + motion::FAST));
        assert_eq!(transition.at(), 1.0);
    }

    #[test]
    fn turned_back_it_goes_back_from_where_it_is() {
        let mut slot = Some(Transition::new(0.0, 1.0, motion::SLOW));
        let start = Instant::now();
        let mut messages: Vec<()> = Vec::new();
        let mut shell = Shell::new(&mut messages);
        Transition::advance(&mut slot, start, &mut shell);
        Transition::advance(&mut slot, start + motion::SLOW / 4, &mut shell);
        let there = slot.unwrap().at();
        assert!(
            there > 0.6,
            "a quarter of the time is most of the way: {there}"
        );
        Transition::turn(&mut slot, 1.0, 0.0, motion::SLOW, &mut shell);
        let mut back = slot.unwrap();
        let again = start + motion::SLOW / 4;
        assert!(back.frame(again));
        assert_eq!(back.at(), there, "no jump");
        // Less of the way takes as much less time.
        assert!(!back.frame(again + motion::SLOW.mul_f32(there)));
        assert_eq!(back.at(), 0.0);
    }
}
