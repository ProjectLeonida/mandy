// SPDX-License-Identifier: GPL-3.0-or-later

//! Shapes iced's quads can't draw, as one custom primitive drawn by signed distance fields,
//! antialiased (`shape.wgsl`):
//!
//! - **Rounding:** a window's corners. iced clips only to rectangles, so content reaching a corner
//!   would show square. Drawn in a layer after everything else, it multiplies what's drawn by
//!   the coverage of the rounded rectangle inside the line (premultiplied, so every channel), and
//!   then adds the line in the ring that leaves: content clipped to the inside of the line and the
//!   line drawn over the edge, as CSS's `border` with `overflow: hidden` draws.
//! - **Strokes:** a line icon's round-capped strokes, such as the close button's cross. Meshes
//!   would be jagged, as windows draw without multisampling.
//!
//! A primitive draws after its layer's quads and before its text.

use std::sync::atomic::{AtomicU32, Ordering};

use iced::advanced::graphics::geometry::Renderer as _;
use iced::advanced::renderer::{self as quads, Renderer as _};
use iced::widget::canvas;
use iced::{Background, Border, Color, Point, Rectangle, Shadow, Size, Vector};
use iced_wgpu::graphics::{Viewport, color};
use iced_wgpu::primitive::{self, Renderer as _};
use iced_wgpu::wgpu;

use crate::ui::tokens::size;

use crate::ui::Renderer;

/// The most strokes an icon has.
pub(crate) const MAX_STROKES: usize = 4;

/// A shape to draw in its bounds, in logical pixels.
#[derive(Debug)]
pub(crate) struct Shape {
    kind: Kind,
    /// How wide its bounds were drawn: drawn under a transformation that scales, the bounds it's
    /// prepared with are wider by as much, and so is it.
    width: f32,
    /// Where its parameters are in the pipeline's buffer this frame.
    slot: AtomicU32,
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    /// Rounded corners of the bounds less [`REACH`] all round, and a line inside their edge
    /// `width` wide.
    Rounding {
        radius: f32,
        width: f32,
        line: Color,
    },
    /// Round-capped strokes, `count` of them, from one point to another, from the bounds' top
    /// left; or, if `centered`, from the top left of an icon's box in the middle of the bounds.
    Strokes {
        segments: [[f32; 4]; MAX_STROKES],
        count: usize,
        width: f32,
        ink: Color,
        centered: bool,
    },
}

/// Whether the renderer draws custom primitives: wgpu does; its fallback, tiny-skia, doesn't.
fn primitives(renderer: &Renderer) -> bool {
    matches!(renderer, Renderer::Primary(_))
}

/// How much further each way than the corners it rounds a rounding's drawn: where they end
/// mid-pixel, that pixel's center is on the drawing's edge and nothing's drawn there, so the
/// corner stayed square. The layer's own bounds keep it to the window.
const REACH: f32 = 1.0;

/// Rounds the corners of everything drawn so far in `bounds` if `clear`, and draws a line along
/// their edge. Draw it in a layer of its own, after everything it rounds. On a renderer without
/// custom primitives, only the line is drawn: what's under the corners has to keep out of them
/// itself (the frame's fill and the header bar's are rounded quads).
pub(crate) fn round(
    renderer: &mut Renderer,
    bounds: Rectangle,
    radius: f32,
    width: f32,
    line: Color,
    clear: bool,
) {
    if clear && primitives(renderer) {
        let reach = bounds.expand(REACH);
        renderer.draw_primitive(
            reach,
            Shape::new(
                Kind::Rounding {
                    radius,
                    width,
                    line,
                },
                reach,
            ),
        );
        return;
    }
    renderer.fill_quad(
        quads::Quad {
            bounds,
            border: Border {
                color: line,
                width,
                radius: radius.into(),
            },
            shadow: Shadow::default(),
            snap: true,
        },
        Background::Color(Color::TRANSPARENT),
    );
}

/// An icon's strokes as a shape keeps them: at most [`MAX_STROKES`].
fn kept(segments: &[[f32; 4]]) -> ([[f32; 4]; MAX_STROKES], usize) {
    debug_assert!(
        segments.len() <= MAX_STROKES,
        "an icon of {} strokes: shape.wgsl draws {MAX_STROKES}",
        segments.len()
    );
    let mut kept = [[0.0; 4]; MAX_STROKES];
    let count = segments.len().min(MAX_STROKES);
    kept[..count].copy_from_slice(&segments[..count]);
    (kept, count)
}

/// Draws an icon's strokes, its box in the middle of `within`, such as a button: the middle of
/// the whole pixels iced draws a quad filling `within` on, as it snaps a quad's edges to them, so
/// the icon is exactly in the middle of the button at any scale.
pub(crate) fn icon(
    renderer: &mut Renderer,
    within: Rectangle,
    segments: &[[f32; 4]],
    width: f32,
    ink: Color,
) {
    if !primitives(renderer) {
        canvas_strokes(renderer, within, segments, width, ink);
        return;
    }
    let (segments, count) = kept(segments);
    renderer.draw_primitive(
        within,
        Shape::new(
            Kind::Strokes {
                segments,
                count,
                width,
                ink,
                centered: true,
            },
            within,
        ),
    );
}

/// The strokes as round-capped paths, for a renderer without custom primitives: the icon's box
/// in the middle of `within`.
fn canvas_strokes(
    renderer: &mut Renderer,
    within: Rectangle,
    segments: &[[f32; 4]],
    width: f32,
    ink: Color,
) {
    let origin = Point::new(
        (within.center_x() - size::ICON / 2.0).round(),
        (within.center_y() - size::ICON / 2.0).round(),
    );
    let mut frame = canvas::Frame::new(renderer, Size::new(size::ICON, size::ICON));
    for &[x0, y0, x1, y1] in segments {
        let path = canvas::Path::line(Point::new(x0, y0), Point::new(x1, y1));
        frame.stroke(
            &path,
            canvas::Stroke::default()
                .with_color(ink)
                .with_width(width)
                .with_line_cap(canvas::LineCap::Round),
        );
    }
    let geometry = frame.into_geometry();
    renderer.with_translation(Vector::new(origin.x, origin.y), |renderer| {
        renderer.draw_geometry(geometry);
    });
}
impl Shape {
    fn new(kind: Kind, bounds: Rectangle) -> Self {
        Self {
            kind,
            width: bounds.width,
            slot: AtomicU32::new(0),
        }
    }

    /// Its parameters as the shader takes them, in physical pixels, its bounds as prepared, at
    /// the window's `scale`.
    fn uniforms(&self, bounds: &Rectangle, scale: f32) -> Uniforms {
        // As much larger as a transformation made its bounds.
        let grown = if self.width > 0.0 {
            bounds.width / self.width
        } else {
            1.0
        };
        let sized = scale * grown;
        let mut uniforms = Uniforms {
            origin: [bounds.x * scale, bounds.y * scale],
            size: [bounds.width * scale, bounds.height * scale],
            ..Uniforms::default()
        };
        match &self.kind {
            Kind::Rounding {
                radius,
                width,
                line,
            } => {
                let area = bounds.shrink(REACH * grown);
                uniforms.origin = [area.x * scale, area.y * scale];
                uniforms.size = [area.width * scale, area.height * scale];
                uniforms.color = premultiplied(*line);
                uniforms.radius = radius * sized;
                uniforms.width = width * sized;
            }
            Kind::Strokes {
                segments,
                count,
                width,
                ink,
                centered,
            } => {
                if *centered {
                    // As iced's quads snap their edges to whole pixels.
                    let snap = |at: f32| (at * scale + 0.001).round();
                    let middle_x = (snap(bounds.x) + snap(bounds.x + bounds.width)) / 2.0;
                    let middle_y = (snap(bounds.y) + snap(bounds.y + bounds.height)) / 2.0;
                    let half = size::ICON * sized / 2.0;
                    uniforms.origin = [middle_x - half, middle_y - half];
                }
                uniforms.color = premultiplied(*ink);
                uniforms.width = width * sized;
                uniforms.count = *count as u32;
                for (to, from) in uniforms.segments.iter_mut().zip(&segments[..*count]) {
                    *to = from.map(|c| c * sized);
                }
            }
        }
        uniforms
    }
}

/// Where a rounding draws, in a target `target` physical pixels big: the corners its radius
/// reaches into and strips along the edges its line covers, each pixel in one of them only, as
/// the mask multiplies what's drawn. Away from them it changes nothing, so the window's middle
/// isn't drawn over twice a frame. A window too small for them is one region, its whole.
fn regions(uniforms: &Uniforms, target: (u32, u32)) -> Regions {
    let [x, y] = uniforms.origin;
    let [width, height] = uniforms.size;
    // A pixel more each way, for the reach and antialiasing.
    let (x0, y0) = ((x - REACH).floor(), (y - REACH).floor());
    let (x1, y1) = ((x + width + REACH).ceil(), (y + height + REACH).ceil());
    let edge = uniforms.width.ceil() + 2.0;
    let corner = uniforms.radius.ceil().max(uniforms.width.ceil()) + 2.0;
    let whole = [[x0, y0, x1, y1]];
    let parts = [
        // The corners.
        [x0, y0, x0 + corner, y0 + corner],
        [x1 - corner, y0, x1, y0 + corner],
        [x0, y1 - corner, x0 + corner, y1],
        [x1 - corner, y1 - corner, x1, y1],
        // The edges between them.
        [x0 + corner, y0, x1 - corner, y0 + edge],
        [x0 + corner, y1 - edge, x1 - corner, y1],
        [x0, y0 + corner, x0 + edge, y1 - corner],
        [x1 - edge, y0 + corner, x1, y1 - corner],
    ];
    let rects: &[[f32; 4]] = if x1 - x0 <= 2.0 * corner || y1 - y0 <= 2.0 * corner {
        &whole
    } else {
        &parts
    };
    let (w, h) = (target.0 as f32, target.1 as f32);
    let mut regions = Regions::default();
    for &[left, top, right, bottom] in rects {
        let (left, right) = (left.clamp(0.0, w), right.clamp(0.0, w));
        let (top, bottom) = (top.clamp(0.0, h), bottom.clamp(0.0, h));
        if right > left && bottom > top {
            regions.rects[regions.count] = [
                left as u32,
                top as u32,
                (right - left) as u32,
                (bottom - top) as u32,
            ];
            regions.count += 1;
        }
    }
    regions
}

/// Scissor rectangles a shape draws in, `[x, y, width, height]` in physical pixels: none, a
/// shape that draws in its whole bounds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Regions {
    rects: [[u32; 4]; 8],
    count: usize,
}

/// A color as iced's shaders take it, premultiplied.
fn premultiplied(color: Color) -> [f32; 4] {
    let [r, g, b, a] = color::pack(color).components();
    [r * a, g * a, b * a, a]
}

/// `Shape` in `shape.wgsl`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Uniforms {
    origin: [f32; 2],
    size: [f32; 2],
    color: [f32; 4],
    radius: f32,
    width: f32,
    count: u32,
    segments: [[f32; 4]; MAX_STROKES],
}

/// Its size in the shader's layout: 48 bytes, then the segments.
const UNIFORMS: u64 = 48 + 16 * MAX_STROKES as u64;

impl Uniforms {
    /// Its bytes, in the shader's layout. Written out by hand: `unsafe` is forbidden, so no
    /// casting.
    fn bytes(&self) -> Bytes {
        let mut bytes = [0; UNIFORMS as usize];
        let line = [self.radius, self.width];
        let words = self
            .origin
            .iter()
            .chain(&self.size)
            .chain(&self.color)
            .chain(&line)
            .map(|value| value.to_ne_bytes())
            .chain([self.count.to_ne_bytes(), 0.0_f32.to_ne_bytes()])
            .chain(
                self.segments
                    .iter()
                    .flatten()
                    .map(|value| value.to_ne_bytes()),
            );
        for (to, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            to.copy_from_slice(&word);
        }
        bytes
    }
}

/// A shape's parameters, as the buffer holds them.
type Bytes = [u8; UNIFORMS as usize];

impl primitive::Primitive for Shape {
    type Pipeline = Pipeline;

    fn prepare(
        &self,
        pipeline: &mut Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        let uniforms = self.uniforms(bounds, viewport.scale_factor());
        let regions = match self.kind {
            Kind::Rounding { .. } => {
                let target = viewport.physical_size();
                regions(&uniforms, (target.width, target.height))
            }
            Kind::Strokes { .. } => Regions::default(),
        };
        for &stage in Stage::of(&self.kind) {
            pipeline.stages.make(device, stage);
        }
        let slot = pipeline.push(device, queue, uniforms.bytes(), regions);
        self.slot.store(slot, Ordering::Relaxed);
    }

    fn draw(&self, pipeline: &Pipeline, pass: &mut wgpu::RenderPass<'_>) -> bool {
        let slot = self.slot.load(Ordering::Relaxed);
        let offset = slot * pipeline.stride as u32;
        let regions = pipeline
            .regions
            .get(slot as usize)
            .copied()
            .unwrap_or_default();
        for &stage in Stage::of(&self.kind) {
            let Some(stage) = pipeline.stages.get(stage) else {
                continue;
            };
            pass.set_pipeline(stage);
            pass.set_bind_group(0, &pipeline.bind_group, &[offset]);
            if regions.count == 0 {
                pass.draw(0..3, 0..1);
            }
            // In its regions alone: the renderer sets the next primitive's scissor, and the whole
            // target's after the last.
            for &[x, y, width, height] in &regions.rects[..regions.count] {
                pass.set_scissor_rect(x, y, width, height);
                pass.draw(0..3, 0..1);
            }
        }
        true
    }
}

/// The shapes' render pipelines, and a buffer of every shape's parameters this frame, one slot
/// each, bound at its offset.
pub(crate) struct Pipeline {
    stages: Stages,
    layout: wgpu::BindGroupLayout,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// A slot's size: the uniforms, aligned as the device wants dynamic offsets.
    stride: u64,
    capacity: u64,
    /// This frame's slots, kept to copy into a larger buffer; and last frame's, which the buffer
    /// still holds, so a slot that's the same isn't written again.
    slots: Vec<Bytes>,
    last: Vec<Bytes>,
    /// Where each slot's shape draws, if not its whole bounds.
    regions: Vec<Regions>,
}

/// One of the shader's fragment stages, each its own render pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Multiplies what's drawn by the coverage inside a window's line (blended Zero, SrcAlpha).
    Mask,
    /// The line, added (One, One).
    Line,
    /// Round-capped strokes over what's drawn, premultiplied.
    Strokes,
}

impl Stage {
    fn entry(self) -> &'static str {
        match self {
            Self::Mask => "mask",
            Self::Line => "line",
            Self::Strokes => "strokes",
        }
    }

    fn blend(self) -> wgpu::BlendState {
        // What's drawn, times the coverage: premultiplied, so color and alpha alike.
        let multiply = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::SrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        match self {
            Self::Mask => wgpu::BlendState {
                color: multiply,
                alpha: multiply,
            },
            Self::Line => wgpu::BlendState {
                color: add,
                alpha: add,
            },
            Self::Strokes => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
        }
    }

    /// The stages a shape of this kind is drawn with.
    fn of(kind: &Kind) -> &'static [Self] {
        match kind {
            Kind::Rounding { width, .. } if *width > 0.0 => &[Self::Mask, Self::Line],
            Kind::Rounding { .. } => &[Self::Mask],
            Kind::Strokes { .. } => &[Self::Strokes],
        }
    }
}

/// The render pipelines, each made the first time a shape needs it: a
/// pipeline is compiled when it's made, tens of milliseconds each on lavapipe, and most windows
/// draw one or two kinds of shape, or none.
struct Stages {
    shader: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    made: [Option<wgpu::RenderPipeline>; 4],
}

impl Stages {
    fn make(&mut self, device: &wgpu::Device, stage: Stage) {
        let made = &mut self.made[stage as usize];
        if made.is_some() {
            return;
        }
        *made = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("launcher shapes"),
                layout: Some(&self.layout),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some(stage.entry()),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.format,
                        blend: Some(stage.blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            }),
        );
    }

    fn get(&self, stage: Stage) -> Option<&wgpu::RenderPipeline> {
        self.made[stage as usize].as_ref()
    }
}

impl Pipeline {
    /// Takes a slot for a shape's parameters, and writes them, unless the slot holds them from
    /// the last frame, as a still window's shapes do.
    fn push(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: Bytes,
        regions: Regions,
    ) -> u32 {
        let slot = self.slots.len();
        self.slots.push(bytes);
        self.regions.push(regions);
        if slot as u64 >= self.capacity {
            self.capacity = (self.capacity * 2).max(slot as u64 + 1);
            (self.buffer, self.bind_group) =
                buffer(device, &self.layout, self.stride, self.capacity);
            // A new buffer holds nothing: what it held last frame is gone too.
            self.last.clear();
            for (at, slot) in self.slots.iter().enumerate() {
                queue.write_buffer(&self.buffer, at as u64 * self.stride, slot);
            }
        } else if self.last.get(slot) != Some(&bytes) {
            queue.write_buffer(&self.buffer, slot as u64 * self.stride, &bytes);
        }
        slot as u32
    }
}

fn buffer(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    stride: u64,
    capacity: u64,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("launcher shapes"),
        size: stride * capacity,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("launcher shapes"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: wgpu::BufferSize::new(UNIFORMS),
            }),
        }],
    });
    (buffer, bind_group)
}

impl primitive::Pipeline for Pipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("launcher shapes"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(UNIFORMS),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("launcher shapes"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("launcher shapes"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shape.wgsl").into()),
        });
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        let stride = UNIFORMS.div_ceil(alignment) * alignment;
        let capacity = 4;
        let (buffer, bind_group) = buffer(device, &layout, stride, capacity);
        Self {
            stages: Stages {
                shader,
                layout: pipeline_layout,
                format,
                made: [None, None, None, None],
            },
            layout,
            buffer,
            bind_group,
            stride,
            capacity,
            slots: Vec::new(),
            last: Vec::new(),
            regions: Vec::new(),
        }
    }

    /// A frame is drawn: its slots are free again, and what they held is what the buffer holds.
    fn trim(&mut self) {
        std::mem::swap(&mut self.slots, &mut self.last);
        self.slots.clear();
        self.regions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kind_is_drawn_with_only_the_stages_it_needs() {
        let rounding = |width| Kind::Rounding {
            radius: 14.0,
            width,
            line: Color::WHITE,
        };
        assert_eq!(Stage::of(&rounding(1.0)), [Stage::Mask, Stage::Line]);
        assert_eq!(Stage::of(&rounding(0.0)), [Stage::Mask]);
        let (segments, count) = kept(&[[0.0, 0.0, 1.0, 1.0]]);
        let strokes = Kind::Strokes {
            segments,
            count,
            width: 1.4,
            ink: Color::WHITE,
            centered: false,
        };
        assert_eq!(Stage::of(&strokes), [Stage::Strokes]);
    }

    #[test]
    fn uniforms_are_in_physical_pixels_and_premultiplied() {
        let area = Rectangle::new(iced::Point::new(10.0, 20.0), iced::Size::new(100.0, 50.0));
        let line = Color::from_rgba(1.0, 0.5, 0.0, 0.5);
        let bounds = area.expand(REACH);
        let rounding = Shape::new(
            Kind::Rounding {
                radius: 14.0,
                width: 1.0,
                line,
            },
            bounds,
        );
        let uniforms = rounding.uniforms(&bounds, 1.5);
        assert_eq!(uniforms.origin, [15.0, 30.0]);
        assert_eq!(uniforms.size, [150.0, 75.0]);
        assert_eq!((uniforms.radius, uniforms.width), (21.0, 1.5));
        assert_eq!(uniforms.color, [0.5, 0.25, 0.0, 0.5]);
        assert_eq!(uniforms.count, 0);

        let (segments, count) = kept(&[[4.5, 4.5, 11.5, 11.5], [11.5, 4.5, 4.5, 11.5]]);
        let cross = Shape::new(
            Kind::Strokes {
                segments,
                count,
                width: 1.4,
                ink: Color::WHITE,
                centered: false,
            },
            area,
        );
        let uniforms = cross.uniforms(&area, 2.0);
        assert_eq!(uniforms.count, 2);
        assert_eq!(uniforms.segments[1], [23.0, 9.0, 9.0, 23.0]);
        assert_eq!(uniforms.segments[2], [0.0; 4]);
        assert_eq!(uniforms.width, 2.8);
        assert_eq!(uniforms.color, [1.0; 4]);
        // Under a transformation that makes it twice as large, as a dialog coming up does, its
        // strokes are twice as large too.
        let grown = Rectangle::new(area.position(), area.size() * 2.0);
        let uniforms = cross.uniforms(&grown, 1.0);
        assert_eq!(uniforms.segments[1], [23.0, 9.0, 9.0, 23.0]);
        assert_eq!(uniforms.width, 2.8);
    }

    #[test]
    fn an_icon_is_in_the_middle_of_the_whole_pixels_its_button_fills() {
        // A 32 button at 7.5 across, as a header bar's margin puts one at 1.5×: iced snaps the
        // button's quad to whole pixels, and the icon's box is in the middle of those.
        let button = Rectangle::new(iced::Point::new(7.5, 7.0), iced::Size::new(32.0, 32.0));
        let (segments, count) = kept(&crate::ui::tokens::window::CLOSE_CROSS);
        let cross = Shape::new(
            Kind::Strokes {
                segments,
                count,
                width: size::ICON_STROKE,
                ink: Color::WHITE,
                centered: true,
            },
            button,
        );
        for (scale, origin) in [
            (1.0, [16.0, 15.0]),
            (1.5, [23.0, 23.0]),
            (2.0, [31.0, 30.0]),
        ] {
            let uniforms = cross.uniforms(&button, scale);
            assert_eq!(uniforms.origin, origin, "at {scale}×");
            // The box is the icon's size at that scale, its middle the snapped button's.
            let half = size::ICON * scale / 2.0;
            let snap = |at: f32| (at * scale + 0.001).round();
            let middle = (snap(button.x) + snap(button.x + button.width)) / 2.0;
            assert_eq!(uniforms.origin[0] + half, middle);
        }
    }

    #[test]
    fn a_rounding_draws_only_where_its_corners_and_line_are() {
        let uniforms = Uniforms {
            origin: [0.0, 0.0],
            size: [800.0, 600.0],
            radius: 14.0,
            width: 1.0,
            ..Uniforms::default()
        };
        let regions = regions(&uniforms, (800, 600));
        assert_eq!(regions.count, 8);
        let rects = &regions.rects[..regions.count];
        // The corners, 16 square from a pixel outside, the radius and antialiasing, and the
        // edges between them, 3 deep; kept to the target.
        assert_eq!(rects[0], [0, 0, 15, 15]);
        assert_eq!(rects[3], [785, 585, 15, 15]);
        assert_eq!(rects[4], [15, 0, 770, 2]);
        assert_eq!(rects[7], [798, 15, 2, 570]);
        // Every pixel on the edge in one of them, none in two: the mask multiplies.
        let inside = |x: u32, y: u32| {
            rects
                .iter()
                .filter(|[rx, ry, w, h]| x >= *rx && x < rx + w && y >= *ry && y < ry + h)
                .count()
        };
        for x in 0..800 {
            assert_eq!(inside(x, 0), 1, "top, at {x}");
            assert_eq!(inside(x, 599), 1, "bottom, at {x}");
        }
        for y in 0..600 {
            assert_eq!(inside(0, y), 1, "left, at {y}");
            assert_eq!(inside(799, y), 1, "right, at {y}");
        }
        assert_eq!(inside(400, 300), 0, "the middle isn't drawn over");
        // Too small for them: the whole, kept to the target.
        let small = Uniforms {
            size: [20.0, 20.0],
            ..uniforms
        };
        let regions = super::regions(&small, (800, 600));
        assert_eq!(&regions.rects[..regions.count], [[0, 0, 21, 21]]);
    }

    #[test]
    fn uniforms_take_the_shaders_layout() {
        let uniforms = Uniforms {
            origin: [1.0, 2.0],
            count: 3,
            segments: [[0.0, 0.0, 0.0, 9.0]; MAX_STROKES],
            ..Uniforms::default()
        };
        let bytes = uniforms.bytes();
        assert_eq!(bytes.len() as u64, UNIFORMS);
        let word = |at: usize| <[u8; 4]>::try_from(&bytes[at * 4..at * 4 + 4]).unwrap();
        assert_eq!(f32::from_ne_bytes(word(1)), 2.0);
        // origin, size, color, radius, width, then the count at byte 40.
        assert_eq!(u32::from_ne_bytes(word(10)), 3);
        assert_eq!(f32::from_ne_bytes(word(15)), 9.0);
    }
}
