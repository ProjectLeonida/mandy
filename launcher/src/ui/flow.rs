// SPDX-License-Identifier: GPL-3.0-or-later

//! A flow: children side by side while they fit, one under another when they don't, as a setting's
//! words and its control are on a phone. A child that fills its room is never narrower than the
//! flow's least width beside the others.

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer;
use iced::advanced::widget::{Operation, Tree, Widget};
use iced::advanced::{Clipboard, Shell, mouse, overlay};
use iced::{Alignment, Event, Length, Point, Rectangle, Size, Vector};

use crate::ui::tokens::{size, space};

use crate::ui::{Element, Renderer, Theme};

/// Children side by side while they fit, stacked when they don't.
pub fn flow<'a, Message>(
    children: impl IntoIterator<Item = Element<'a, Message>>,
) -> Flow<'a, Message> {
    Flow {
        children: children.into_iter().collect(),
        spacing: space::L,
        least: size::FLOW_LEAST,
        align: Alignment::Center,
    }
}

/// A flow ([`flow`]).
pub struct Flow<'a, Message> {
    children: Vec<Element<'a, Message>>,
    spacing: f32,
    least: f32,
    align: Alignment,
}

impl<'a, Message> Flow<'a, Message> {
    /// The room between children, across and down.
    pub fn spacing(self, spacing: f32) -> Self {
        Self { spacing, ..self }
    }

    /// The least a child that fills its room may be beside the others: narrower, they stack.
    pub fn least(self, least: f32) -> Self {
        Self { least, ..self }
    }

    /// How children line up side by side: centered by default.
    pub fn align_y(self, align: Alignment) -> Self {
        Self { align, ..self }
    }
}

/// Whether children of these widths fit side by side in `room`: the fixed ones as wide as they
/// are, the filling ones (None) at least `least`.
fn side_by_side(widths: &[Option<f32>], room: f32, spacing: f32, least: f32) -> bool {
    let gaps = spacing * widths.len().saturating_sub(1) as f32;
    let need: f32 = widths.iter().map(|width| width.unwrap_or(least)).sum();
    need + gaps <= room
}

fn fills(length: Length) -> bool {
    matches!(length, Length::Fill | Length::FillPortion(_))
}

impl<Message> Widget<Message, Theme, Renderer> for Flow<'_, Message> {
    fn children(&self) -> Vec<Tree> {
        self.children.iter().map(Tree::new).collect()
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(&self.children);
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Shrink)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let room = limits.max().width;
        let tall = limits.max().height;
        let loose = layout::Limits::new(Size::ZERO, Size::new(room, tall));
        // Each child is laid out once, at the width it ends up with, as laying out text at a
        // second width wraps it again: the fixed ones first, whose widths decide.
        let mut nodes: Vec<Option<layout::Node>> = self
            .children
            .iter_mut()
            .zip(&mut tree.children)
            .map(|(child, tree)| {
                (!fills(child.as_widget().size().width))
                    .then(|| child.as_widget_mut().layout(tree, renderer, &loose))
            })
            .collect();
        let widths: Vec<Option<f32>> = nodes
            .iter()
            .map(|node| node.as_ref().map(|node| node.size().width))
            .collect();
        // With no bound on the width, as across a sideways scroll, they're side by side, the
        // filling ones as narrow as they may be.
        let beside = !room.is_finite() || side_by_side(&widths, room, self.spacing, self.least);
        let fixed: f32 = widths.iter().flatten().sum();
        let gaps = self.spacing * self.children.len().saturating_sub(1) as f32;
        let count = widths.iter().filter(|width| width.is_none()).count().max(1);
        let share = if room.is_finite() {
            ((room - fixed - gaps) / count as f32).max(0.0)
        } else {
            self.least
        };
        // Side by side, the filling children share what the others leave; stacked, each has
        // the whole room.
        let filled = if beside {
            layout::Limits::new(Size::ZERO, Size::new(share, tall))
        } else {
            loose
        };
        let nodes: Vec<layout::Node> = nodes
            .iter_mut()
            .zip(self.children.iter_mut().zip(&mut tree.children))
            .map(|(node, (child, tree))| {
                node.take()
                    .unwrap_or_else(|| child.as_widget_mut().layout(tree, renderer, &filled))
            })
            .collect();
        if beside {
            let height = nodes
                .iter()
                .map(|node| node.size().height)
                .fold(0.0, f32::max);
            let mut x = 0.0;
            let nodes: Vec<layout::Node> = nodes
                .into_iter()
                .map(|node| {
                    let size = node.size();
                    let y = match self.align {
                        Alignment::Start => 0.0,
                        Alignment::Center => (height - size.height) / 2.0,
                        Alignment::End => height - size.height,
                    };
                    let node = node.move_to(Point::new(x, y));
                    x += size.width + self.spacing;
                    node
                })
                .collect();
            let width = if room.is_finite() {
                room
            } else {
                (x - self.spacing).max(0.0)
            };
            layout::Node::with_children(Size::new(width, height), nodes)
        } else {
            // Stacked, each from the start, as wide as it is or, filling, the whole room.
            let mut y = 0.0;
            let nodes: Vec<layout::Node> = nodes
                .into_iter()
                .map(|node| {
                    let height = node.size().height;
                    let node = node.move_to(Point::new(0.0, y));
                    y += height + self.spacing;
                    node
                })
                .collect();
            let height = (y - self.spacing).max(0.0);
            layout::Node::with_children(Size::new(room, height), nodes)
        }
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        for ((child, tree), layout) in self
            .children
            .iter_mut()
            .zip(&mut tree.children)
            .zip(layout.children())
        {
            child
                .as_widget_mut()
                .operate(tree, layout, renderer, operation);
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
        for ((child, tree), layout) in self
            .children
            .iter_mut()
            .zip(&mut tree.children)
            .zip(layout.children())
        {
            child.as_widget_mut().update(
                tree, event, layout, cursor, renderer, clipboard, shell, viewport,
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
        for ((child, tree), layout) in self
            .children
            .iter()
            .zip(&tree.children)
            .zip(layout.children())
        {
            child
                .as_widget()
                .draw(tree, renderer, theme, style, layout, cursor, viewport);
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
        self.children
            .iter()
            .zip(&tree.children)
            .zip(layout.children())
            .map(|((child, tree), layout)| {
                child
                    .as_widget()
                    .mouse_interaction(tree, layout, cursor, viewport, renderer)
            })
            .max()
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
        overlay::from_children(
            &mut self.children,
            tree,
            layout,
            renderer,
            viewport,
            translation,
        )
    }
}

impl<'a, Message: 'a> From<Flow<'a, Message>> for Element<'a, Message> {
    fn from(flow: Flow<'a, Message>) -> Self {
        Element::new(flow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_go_side_by_side_while_they_fit() {
        // Words that fill, at least 160, and a 220 field, 16 between.
        let widths = [None, Some(220.0)];
        assert!(side_by_side(&widths, 396.0, 16.0, 160.0));
        assert!(!side_by_side(&widths, 395.0, 16.0, 160.0));
        // Fixed alone: as wide as they are.
        assert!(side_by_side(
            &[Some(100.0), Some(100.0)],
            216.0,
            16.0,
            160.0
        ));
        assert!(side_by_side(&[], 0.0, 16.0, 160.0));
    }
}
