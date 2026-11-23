// SPDX-License-Identifier: GPL-3.0-or-later

//! Single-line text editing: what a key does to a text and its selection. It knows nothing of
//! iced, so it's property-tested here and fuzzed on its own (fuzz/, which builds this file and
//! `secret.rs` by path). Offsets are bytes, always on grapheme boundaries, so an accent never parts
//! from its letter.

use std::ops::Range;

use unicode_segmentation::UnicodeSegmentation;

/// Where a text is edited: a `String`, or a secret's fixed buffer.
pub trait Buffer {
    fn text(&self) -> &str;

    /// Replaces a range, on char boundaries, with text. False, changing nothing, if the result
    /// wouldn't fit.
    fn replace(&mut self, range: Range<usize>, with: &str) -> bool;

    /// How many more bytes it takes.
    fn room(&self) -> usize;
}

impl Buffer for String {
    fn text(&self) -> &str {
        self
    }

    fn replace(&mut self, range: Range<usize>, with: &str) -> bool {
        self.replace_range(range, with);
        true
    }

    fn room(&self) -> usize {
        usize::MAX
    }
}

/// A selection: from the anchor, where it started, to the head, where the caret is. Empty when
/// they meet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Selection {
    pub anchor: usize,
    pub head: usize,
}

impl Selection {
    /// Just a caret.
    pub fn caret(at: usize) -> Self {
        Self {
            anchor: at,
            head: at,
        }
    }

    pub fn range(&self) -> Range<usize> {
        self.anchor.min(self.head)..self.anchor.max(self.head)
    }

    pub fn is_empty(&self) -> bool {
        self.anchor == self.head
    }

    /// The selection within `text`, on its grapheme boundaries: for a text changed elsewhere.
    pub fn clamp(self, text: &str) -> Self {
        Self {
            anchor: boundary_at_or_after(text, self.anchor),
            head: boundary_at_or_after(text, self.head),
        }
    }
}

/// Where the caret goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Motion {
    Left,
    Right,
    WordLeft,
    WordRight,
    Home,
    End,
}

/// An edit, as keys ask for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Edit<'a> {
    /// Typed or pasted text, replacing the selection, as much as fits. The fields are
    /// single-line: a line break or a tab between words is a space, and other control characters
    /// are dropped.
    Insert(&'a str),
    /// Backspace: the selection, or the letter (or word) before the caret.
    DeleteBack {
        word: bool,
    },
    /// Delete: the selection, or the letter (or word) after the caret.
    DeleteForward {
        word: bool,
    },
    /// Moves the caret, extending the selection with `select`.
    Move {
        motion: Motion,
        select: bool,
    },
    SelectAll,
}

/// What of a typed or pasted text goes into a line with `room` bytes left, as slices of it, so no
/// copy of it is made: what's typed into a secret lands only in its buffer. Control characters
/// are left out; a line break or a tab between words is a space, and at its ends nothing, so a
/// copied line's break doesn't come along. What doesn't fit is cut at a letter's end.
fn fitting(typed: &str, mut room: usize) -> Vec<&str> {
    let typed = typed.trim_matches(char::is_control);
    // Each piece, and whether it's a space put for a break.
    let mut pieces = Vec::new();
    let (mut start, mut gap, mut spaced) = (0, false, false);
    for (at, c) in typed.char_indices() {
        if c.is_control() {
            if !gap {
                pieces.push((&typed[start..at], false));
                (gap, spaced) = (true, false);
            }
            spaced |= matches!(c, '\n' | '\r' | '\t' | '\u{b}' | '\u{c}');
        } else if gap {
            if spaced {
                pieces.push((" ", true));
            }
            (gap, start) = (false, at);
        }
    }
    if !gap {
        pieces.push((&typed[start..], false));
    }
    let mut kept: Vec<(&str, bool)> = Vec::with_capacity(pieces.len());
    for (piece, space) in pieces.into_iter().filter(|(piece, _)| !piece.is_empty()) {
        if piece.len() <= room {
            room -= piece.len();
            kept.push((piece, space));
            continue;
        }
        let end = piece
            .grapheme_indices(true)
            .map(|(at, letter)| at + letter.len())
            .take_while(|&end| end <= room)
            .last()
            .unwrap_or(0);
        if end > 0 {
            kept.push((&piece[..end], false));
        }
        // Cut short after a break: not its space alone at the end.
        if kept.last().is_some_and(|(_, space)| *space) {
            kept.pop();
        }
        break;
    }
    kept.into_iter().map(|(piece, _)| piece).collect()
}

/// Applies an edit, and says whether the text changed. With `words` off, as for a hidden secret,
/// word motions and deletions reach the ends, so they show nothing of where its words are.
pub fn apply<B: Buffer + ?Sized>(
    buffer: &mut B,
    selection: &mut Selection,
    edit: Edit<'_>,
    words: bool,
) -> bool {
    *selection = selection.clamp(buffer.text());
    let text = buffer.text();
    let at = selection.head;
    let changed = match edit {
        Edit::Insert(typed) => {
            let range = selection.range();
            let pieces = fitting(typed, buffer.room().saturating_add(range.len()));
            if pieces.is_empty() {
                return false;
            }
            let mut at = range.start;
            let mut done = buffer.replace(range, "");
            for piece in pieces {
                done &= buffer.replace(at..at, piece);
                at += piece.len();
            }
            *selection = Selection::caret(at);
            done
        }
        Edit::DeleteBack { word } => {
            let range = if selection.is_empty() {
                let from = match (word, words) {
                    (false, _) => previous(text, at),
                    (true, true) => word_start(text, at),
                    (true, false) => 0,
                };
                from..at
            } else {
                selection.range()
            };
            delete(buffer, selection, range)
        }
        Edit::DeleteForward { word } => {
            let range = if selection.is_empty() {
                let to = match (word, words) {
                    (false, _) => next(text, at),
                    (true, true) => word_end(text, at),
                    (true, false) => text.len(),
                };
                at..to
            } else {
                selection.range()
            };
            delete(buffer, selection, range)
        }
        Edit::Move { motion, select } => {
            let range = selection.range();
            let head = match (motion, words) {
                (Motion::Left, _) if !select && !selection.is_empty() => range.start,
                (Motion::Right, _) if !select && !selection.is_empty() => range.end,
                (Motion::Left, _) => previous(text, at),
                (Motion::Right, _) => next(text, at),
                (Motion::WordLeft, true) => word_start(text, at),
                (Motion::WordRight, true) => word_end(text, at),
                (Motion::Home, _) | (Motion::WordLeft, false) => 0,
                (Motion::End, _) | (Motion::WordRight, false) => text.len(),
            };
            *selection = if select {
                Selection {
                    anchor: selection.anchor,
                    head,
                }
            } else {
                Selection::caret(head)
            };
            false
        }
        Edit::SelectAll => {
            *selection = Selection {
                anchor: 0,
                head: text.len(),
            };
            false
        }
    };
    *selection = selection.clamp(buffer.text());
    changed
}

fn delete<B: Buffer + ?Sized>(
    buffer: &mut B,
    selection: &mut Selection,
    range: Range<usize>,
) -> bool {
    if range.is_empty() {
        *selection = Selection::caret(range.start);
        return false;
    }
    let start = range.start;
    // Removing never grows the text, so it always fits.
    let changed = buffer.replace(range, "");
    *selection = Selection::caret(start);
    changed
}

/// The grapheme boundary at or after a byte offset, or the end.
fn boundary_at_or_after(text: &str, at: usize) -> usize {
    if at >= text.len() {
        return text.len();
    }
    text.grapheme_indices(true)
        .map(|(start, _)| start)
        .find(|start| *start >= at)
        .unwrap_or(text.len())
}

/// The boundary before `at`, or the start.
fn previous(text: &str, at: usize) -> usize {
    text.grapheme_indices(true)
        .map(|(start, _)| start)
        .take_while(|start| *start < at)
        .last()
        .unwrap_or(0)
}

/// The boundary after `at`, or the end.
fn next(text: &str, at: usize) -> usize {
    text.grapheme_indices(true)
        .map(|(start, grapheme)| start + grapheme.len())
        .find(|end| *end > at)
        .unwrap_or(text.len())
}

fn is_word(segment: &str) -> bool {
    segment.chars().any(char::is_alphanumeric)
}

/// The start of the word before `at`, skipping the spaces and punctuation in between.
fn word_start(text: &str, at: usize) -> usize {
    text.split_word_bound_indices()
        .rev()
        .find(|(start, segment)| *start < at && is_word(segment))
        .map_or(0, |(start, _)| start)
}

/// The end of the word after `at`.
fn word_end(text: &str, at: usize) -> usize {
    text.split_word_bound_indices()
        .map(|(start, segment)| (start + segment.len(), segment))
        .find(|(end, segment)| *end > at && is_word(segment))
        .map_or(text.len(), |(end, _)| end)
}

/// How many graphemes come before a byte offset: a caret's place among the letters.
#[cfg(test)]
fn graphemes_before(text: &str, at: usize) -> usize {
    text.grapheme_indices(true)
        .take_while(|(start, _)| *start < at)
        .count()
}

/// The byte offset of the grapheme boundary nearest after a char offset, as a click finds it.
#[cfg(test)]
fn boundary_of_char(text: &str, chars: usize) -> usize {
    let byte = text
        .char_indices()
        .nth(chars)
        .map_or(text.len(), |(at, _)| at);
    boundary_at_or_after(text, byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str, at: usize, edits: &[Edit<'_>]) -> (String, Selection) {
        let mut text = text.to_owned();
        let mut selection = Selection::caret(at);
        for edit in edits {
            apply(&mut text, &mut selection, *edit, true);
        }
        (text, selection)
    }

    #[test]
    fn typing_replaces_the_selection() {
        let (text, selection) = run("harbour", 0, &[Edit::SelectAll, Edit::Insert("Neon")]);
        assert_eq!((text.as_str(), selection), ("Neon", Selection::caret(4)));
        let (text, _) = run("ab", 1, &[Edit::Insert("x\r\ny\t\u{7}")]);
        assert_eq!(
            text, "ax yb",
            "single line: a break a space, at the end nothing"
        );
        let (text, _) = run("", 0, &[Edit::Insert("a\u{7}b\n")]);
        assert_eq!(text, "ab", "other control characters dropped");
        let mut text = String::from("ab");
        let mut selection = Selection::caret(1);
        assert!(!apply(&mut text, &mut selection, Edit::Insert("\n"), true));
    }

    #[test]
    fn a_paste_that_doesnt_fit_is_cut_at_a_letters_end() {
        assert_eq!(fitting("harbour", 4), ["harb"]);
        // é as e and its accent: not parted.
        assert_eq!(fitting("cafe\u{301}s", 5), ["caf"]);
        assert_eq!(
            fitting("one\ntwo", 4),
            ["one"],
            "not a break's space alone at the end"
        );
        assert_eq!(fitting(" ", 4), [" "], "a typed space is a space");
        assert_eq!(fitting("one\ntwo", 6), ["one", " ", "tw"]);
        assert!(fitting("x", 0).is_empty());
    }

    #[test]
    fn deleting_takes_whole_graphemes() {
        // e and a combining acute accent are one letter.
        let (text, selection) = run("cafe\u{301}!", 6, &[Edit::DeleteBack { word: false }]);
        assert_eq!((text.as_str(), selection), ("caf!", Selection::caret(3)));
        let (text, _) = run("cafe\u{301}!", 3, &[Edit::DeleteForward { word: false }]);
        assert_eq!(text, "caf!");
        let (text, selection) = run("", 0, &[Edit::DeleteBack { word: false }]);
        assert_eq!((text.as_str(), selection), ("", Selection::caret(0)));
    }

    #[test]
    fn words_skip_spaces_and_punctuation() {
        let text = "nvme0n1 — 512 GB";
        let mut selection = Selection::caret(text.len());
        let mut buffer = text.to_owned();
        let mut starts = Vec::new();
        for _ in 0..4 {
            let left = Edit::Move {
                motion: Motion::WordLeft,
                select: false,
            };
            apply(&mut buffer, &mut selection, left, true);
            starts.push(selection.head);
        }
        assert_eq!(starts, [text.len() - 2, 12, 0, 0]);
        let (rest, _) = run("one two  three", 14, &[Edit::DeleteBack { word: true }]);
        assert_eq!(rest, "one two  ");
        let (rest, _) = run("one two  three", 3, &[Edit::DeleteForward { word: true }]);
        assert_eq!(rest, "one  three");
    }

    #[test]
    fn hidden_words_reach_the_ends() {
        let mut text = String::from("two words");
        let mut selection = Selection::caret(6);
        let right = Edit::Move {
            motion: Motion::WordRight,
            select: false,
        };
        apply(&mut text, &mut selection, right, false);
        assert_eq!(selection, Selection::caret(9));
        apply(
            &mut text,
            &mut selection,
            Edit::DeleteBack { word: true },
            false,
        );
        assert_eq!(text, "");
    }

    #[test]
    fn moving_collapses_or_extends_the_selection() {
        let select = |motion| Edit::Move {
            motion,
            select: true,
        };
        let go = |motion| Edit::Move {
            motion,
            select: false,
        };
        let (_, selection) = run("harbour", 2, &[select(Motion::Right), select(Motion::End)]);
        assert_eq!(selection, Selection { anchor: 2, head: 7 });
        let (_, selection) = run("harbour", 2, &[select(Motion::End), go(Motion::Left)]);
        assert_eq!(selection, Selection::caret(2));
        let (_, selection) = run("harbour", 2, &[select(Motion::Home), go(Motion::Right)]);
        assert_eq!(selection, Selection::caret(2));
        let (text, selection) = run(
            "harbour",
            2,
            &[select(Motion::End), Edit::DeleteBack { word: false }],
        );
        assert_eq!((text.as_str(), selection), ("ha", Selection::caret(2)));
        assert_eq!(Selection { anchor: 5, head: 1 }.range(), 1..5);
    }

    #[test]
    fn offsets_land_on_boundaries() {
        let text = "e\u{301}x";
        assert_eq!(Selection::caret(1).clamp(text), Selection::caret(3));
        assert_eq!(Selection::caret(99).clamp(text), Selection::caret(4));
        assert_eq!(graphemes_before(text, 3), 1);
        assert_eq!(graphemes_before(text, 4), 2);
        assert_eq!(boundary_of_char(text, 1), 3);
        assert_eq!(boundary_of_char(text, 9), 4);
    }

    mod properties {
        use super::*;
        use proptest::prelude::*;

        fn edit() -> impl Strategy<Value = (u8, String)> {
            (0u8..12, "[a-z é\u{301}🙂\n—-]{0,4}")
        }

        fn to_edit((kind, text): &(u8, String)) -> Edit<'_> {
            let motion = [
                Motion::Left,
                Motion::Right,
                Motion::WordLeft,
                Motion::WordRight,
                Motion::Home,
                Motion::End,
            ][usize::from(*kind) % 6];
            match kind {
                0..=2 => Edit::Insert(text),
                3 => Edit::DeleteBack { word: false },
                4 => Edit::DeleteBack { word: true },
                5 => Edit::DeleteForward { word: false },
                6 => Edit::DeleteForward { word: true },
                7 => Edit::SelectAll,
                8 | 9 => Edit::Move {
                    motion,
                    select: true,
                },
                _ => Edit::Move {
                    motion,
                    select: false,
                },
            }
        }

        fn is_boundary(text: &str, at: usize) -> bool {
            at == text.len() || text.grapheme_indices(true).any(|(start, _)| start == at)
        }

        proptest! {
            #[test]
            fn the_selection_stays_on_boundaries(
                start in "[a-z é\u{301}🙂]{0,8}",
                edits in proptest::collection::vec(edit(), 0..24),
                words: bool,
            ) {
                let mut text = start;
                let mut selection = Selection::caret(text.len());
                for edit in &edits {
                    apply(&mut text, &mut selection, to_edit(edit), words);
                    prop_assert!(is_boundary(&text, selection.anchor));
                    prop_assert!(is_boundary(&text, selection.head));
                    prop_assert!(!text.chars().any(char::is_control));
                }
            }

            #[test]
            fn typing_then_backspacing_restores_the_text(
                start in "[a-z é🙂]{0,8}",
                typed in "[a-z🙂]{1,6}",
                at in 0usize..10,
            ) {
                let mut text = start.clone();
                let mut selection = Selection::caret(at).clamp(&text);
                for letter in typed.chars() {
                    let letter = letter.to_string();
                    apply(&mut text, &mut selection, Edit::Insert(&letter), true);
                }
                for _ in typed.chars() {
                    apply(&mut text, &mut selection, Edit::DeleteBack { word: false }, true);
                }
                prop_assert_eq!(text, start);
            }

            #[test]
            fn moves_never_change_the_text(
                start in "[a-z é\u{301}🙂—]{0,8}",
                moves in proptest::collection::vec(8u8..12, 0..16),
            ) {
                let mut text = start.clone();
                let mut selection = Selection::caret(0);
                for kind in moves {
                    let edit = (kind, String::new());
                    prop_assert!(!apply(&mut text, &mut selection, to_edit(&edit), true));
                }
                prop_assert_eq!(text, start);
            }
        }
    }
}
