// SPDX-License-Identifier: GPL-3.0-or-later

//! Leonida's launcher: one window to check the player's copy of the game, build the port with
//! `make_game.py`, and start the loader. Everything it does runs the repository's own commands;
//! nothing goes online.
//!
//! Its look comes from [`ui`], the launcher's own design system on stock iced 0.14, in two
//! palettes taken from Leonida's logo.

pub mod commands;
pub mod config;
pub mod output;
pub mod process;
pub mod repo;
pub mod scene;
pub mod state;
pub mod ui;
