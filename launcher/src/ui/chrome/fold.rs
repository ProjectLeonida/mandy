// SPDX-License-Identifier: GPL-3.0-or-later

//! A folded window's two panes, its sidebar and its page, one at a time, as on a phone: choosing a
//! place slides the page in from the right over the sidebar, which drifts a little to the left
//! beneath it; going back slides the page out to the right, the sidebar coming back from the left.

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer;
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, overlay};
use iced::{Event, Length, Rectangle, Size, Vector, window};

use crate::ui::tokens::motion;

use crate::ui::transition::Transition;
use crate::ui::{Element, Renderer, Theme};

/// How far the sidebar drifts as the page covers it, a share of the width.
const DRIFT: f32 = 0.3;

/// The sidebar and the page, `page` saying which shows. The page should have a fill of its own:
/// it passes over the sidebar.
pub fn folded<'a, Message>(
    sidebar: impl Into<Element<'a, Message>>,
    page: impl Into<Element<'a, Message>>,
    page_shown: bool,
) -> Folded<'a, Message> {
    Folded {
        panes: [sidebar.into(), page.into()],
        page_shown,
    }
}

/// A folded window's panes ([`folded`]).
pub struct Folded<'a, Message> {
    panes: [Element<'a, Message>; 2],
    page_shown: bool,
}

/// Where the sidebar and the page are across a `width`, `at` the eased share of the way to the
/// page: the sidebar from 0 to drifted left, the page from off the right edge to 0.
fn offsets(at: f32, width: f32) -> (f32, f32) {
    let at = at.clamp(0.0, 1.0);
    (-width * DRIFT * at, width * (1.0 - at))
}

#[derive(Default)]
struct State {
    /// Whether the page showed when last seen, and the slide under way: how far the page has
    /// come in, from 0 to 1.
    seen: Option<bool>,
    slide: Option<Transition>,
}

impl<Message> Folded<'_, Message> {
    /// The pane that shows, or that's coming: the one that takes events, focus and menus.
    fn shown(&self) -> usize {
        usize::from(self.page_shown)
    }
}

impl<Message> Widget<Message, Theme, Renderer> for Folded<'_, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn children(&self) -> Vec<Tree> {
        self.panes.iter().map(Tree::new).collect()
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(&self.panes);
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.resolve(Length::Fill, Length::Fill, Size::ZERO);
        let limits = layout::Limits::new(Size::ZERO, size);
        // The pane that shows is laid out; the other only while it's drawn, sliding, or about to
        // be, the other pane asked for.
        let state = tree.state.downcast_ref::<State>();
        let both = state.slide.is_some() || state.seen != Some(self.page_shown);
        let shown = self.shown();
        let nodes = self
            .panes
            .iter_mut()
            .zip(&mut tree.children)
            .enumerate()
            .map(|(at, (pane, tree))| {
                if both || at == shown {
                    pane.as_widget_mut().layout(tree, renderer, &limits)
                } else {
                    layout::Node::new(Size::ZERO)
                }
            })
            .collect();
        layout::Node::with_children(size, nodes)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        let shown = self.shown();
        if let Some(pane_layout) = layout.children().nth(shown) {
            self.panes[shown].as_widget_mut().operate(
                &mut tree.children[shown],
                pane_layout,
                renderer,
                operation,
            );
        }
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<State>();
        // The other pane asked for: it slides in; turned back part of the way, it goes back from
        // where it got to.
        if let Some(seen) = state.seen
            && seen != self.page_shown
        {
            let (was, to) = (
                f32::from(u8::from(seen)),
                f32::from(u8::from(self.page_shown)),
            );
            Transition::turn(&mut state.slide, was, to, motion::SLOW, shell);
        }
        state.seen = Some(self.page_shown);
        if let Event::Window(window::Event::RedrawRequested(now)) = event {
            Transition::advance(&mut state.slide, *now, shell);
        }
        let shown = self.shown();
        // Sliding, the other pane's still drawn: it has the window's events too, with no pointer,
        // so its own animations, such as the sidebar's mark, go on rather than freeze.
        let sliding = tree.state.downcast_ref::<State>().slide.is_some();
        if sliding && matches!(event, Event::Window(_)) {
            let other = 1 - shown;
            if let Some(other_layout) = layout.children().nth(other) {
                self.panes[other].as_widget_mut().update(
                    &mut tree.children[other],
                    event,
                    other_layout,
                    mouse::Cursor::Unavailable,
                    renderer,
                    clipboard,
                    shell,
                    viewport,
                );
            }
        }
        // Sliding, nothing is where it's laid out, so the pointer is nowhere on it: a click lands
        // once the pane has come.
        let cursor = if sliding {
            mouse::Cursor::Unavailable
        } else {
            cursor
        };
        if let Some(pane_layout) = layout.children().nth(shown) {
            self.panes[shown].as_widget_mut().update(
                &mut tree.children[shown],
                event,
                pane_layout,
                cursor,
                renderer,
                clipboard,
                shell,
                viewport,
            );
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_ref::<State>();
        let Some(slide) = state.slide else {
            let shown = self.shown();
            if let Some(pane_layout) = layout.children().nth(shown) {
                self.panes[shown].as_widget().draw(
                    &tree.children[shown],
                    renderer,
                    theme,
                    style,
                    pane_layout,
                    cursor,
                    viewport,
                );
            }
            return;
        };
        // Sliding: the sidebar beneath, the page over it, neither under the pointer. Each in a
        // layer of its own, the page's last: iced draws layers in the order they're made, so the
        // page's fill, in a layer of its own, covers the layers the sidebar made, its list's.
        let (sidebar, page) = offsets(slide.at(), layout.bounds().width);
        for ((at, x), pane_layout) in [(0, sidebar), (1, page)].into_iter().zip(layout.children()) {
            iced::advanced::Renderer::with_layer(renderer, layout.bounds(), |renderer| {
                iced::advanced::Renderer::with_translation(
                    renderer,
                    Vector::new(x, 0.0),
                    |renderer| {
                        self.panes[at].as_widget().draw(
                            &tree.children[at],
                            renderer,
                            theme,
                            style,
                            pane_layout,
                            mouse::Cursor::Unavailable,
                            viewport,
                        );
                    },
                );
            });
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        if tree.state.downcast_ref::<State>().slide.is_some() {
            return mouse::Interaction::default();
        }
        let shown = self.shown();
        layout
            .children()
            .nth(shown)
            .map(|pane_layout| {
                self.panes[shown].as_widget().mouse_interaction(
                    &tree.children[shown],
                    pane_layout,
                    cursor,
                    viewport,
                    renderer,
                )
            })
            .unwrap_or_default()
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'b, Message, Theme, Renderer>> {
        let shown = self.shown();
        let pane_layout = layout.children().nth(shown)?;
        let [sidebar, page] = &mut self.panes;
        let pane = if shown == 1 { page } else { sidebar };
        pane.as_widget_mut().overlay(
            &mut tree.children[shown],
            pane_layout,
            renderer,
            viewport,
            translation,
        )
    }
}

impl<'a, Message: 'a> From<Folded<'a, Message>> for Element<'a, Message> {
    fn from(folded: Folded<'a, Message>) -> Self {
        Element::new(folded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_comes_in_from_the_right_as_the_sidebar_drifts_left() {
        let near =
            |(a, b): (f32, f32), (c, d): (f32, f32)| (a - c).abs() < 1e-3 && (b - d).abs() < 1e-3;
        assert!(near(offsets(0.0, 400.0), (0.0, 400.0)));
        assert!(near(offsets(1.0, 400.0), (-120.0, 0.0)));
        assert!(near(offsets(0.5, 400.0), (-60.0, 200.0)));
    }
}
