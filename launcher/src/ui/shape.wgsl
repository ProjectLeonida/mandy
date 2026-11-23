// SPDX-License-Identifier: GPL-3.0-or-later

// Shapes iced's quads can't draw, by their signed distance fields, antialiased (shape.rs): a
// window's rounded corners and line, line icons, and the void's radial tints.
//
// The render pass's viewport is the primitive's bounds, so the vertex shader covers it with one
// triangle; the fragment shaders measure from the shape's origin in physical pixels. Colors are
// premultiplied, as iced draws.

struct Shape {
    // The shape's top left and size, in physical pixels.
    origin: vec2<f32>,
    size: vec2<f32>,
    // The line's or the strokes' color, premultiplied.
    color: vec4<f32>,
    // The corners' radius, and the line's or a stroke's width.
    radius: f32,
    width: f32,
    // How many of the segments are strokes.
    count: u32,
    _padding: f32,
    // Strokes, from xy to zw, from the origin. The tinted void's: two tints, each its ellipse
    // (its middle, xy, and radii, zw) and its color, premultiplied; their fades are in radius and
    // width.
    segments: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> shape: Shape;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

// The distance from a point to a rounded rectangle centered on the origin: negative inside.
fn rounded(p: vec2<f32>, half: vec2<f32>, radius: f32) -> f32 {
    let r = min(radius, min(half.x, half.y));
    let q = abs(p) - half + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

// How much of a pixel a distance covers, as iced's quads measure it.
fn coverage(distance: f32) -> f32 {
    return clamp(0.5 - distance, 0.0, 1.0);
}

// Inside the line: what the content keeps.
fn inner(at: vec2<f32>) -> f32 {
    let half = shape.size * 0.5;
    let p = at - shape.origin - half;
    let inset = max(half - vec2<f32>(shape.width), vec2<f32>(0.0));
    return coverage(rounded(p, inset, max(shape.radius - shape.width, 0.0)));
}

// The window's whole shape, its line included.
fn outer(at: vec2<f32>) -> f32 {
    let half = shape.size * 0.5;
    return coverage(rounded(at - shape.origin - half, half, shape.radius));
}

// Multiplies what's drawn by the coverage inside the line (blended Zero, SrcAlpha): the corners
// and the line's ring go transparent.
@fragment
fn mask(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, inner(at.xy));
}

// The line in the ring the mask cleared, added (blended One, One).
@fragment
fn line(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    return shape.color * max(outer(at.xy) - inner(at.xy), 0.0);
}

// Round-capped strokes, their union, over what's drawn (premultiplied).
@fragment
fn strokes(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let p = at.xy - shape.origin;
    var distance = 1.0e6;
    for (var i = 0u; i < shape.count; i++) {
        let segment = shape.segments[i];
        let pa = p - segment.xy;
        let ba = segment.zw - segment.xy;
        let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-6), 0.0, 1.0);
        distance = min(distance, length(pa - ba * h));
    }
    return shape.color * coverage(distance - shape.width * 0.5);
}
