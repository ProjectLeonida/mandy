// SPDX-License-Identifier: GPL-3.0-or-later

//! The Play page's background: a dusk (dark) or a morning (light) over water, in the theme's
//! gradient, with a shore and two palms in silhouette, generated on the CPU into one RGBA picture
//! of the page's size. Nothing of it is an image file: a theme's colours go in, pixels come out.
//!
//! The scene lives in a 16:10 frame, its horizon at 0.66 of its height and its sun at 0.805 of
//! its width, and covers the page anchored to the page's bottom right. Over the text column on
//! the left the page's background lies at 0.97 opacity, then falls away smoothly, so the text
//! there keeps its contrast whatever the scene does. However narrow the page, the scene stays:
//! the column's background covers what lies under the text.
//!
//! [`generate`] is a pure function: the same input gives the same bytes. It's slow enough (tens
//! of milliseconds) that the window runs it off its thread, once per size, scale, theme, mode and
//! column width, and shows the cached picture.

use tiny_skia::{FillRule, Mask, Path, PathBuilder, Stroke, Transform};

use crate::ui::tokens::{Color, Palette};

/// The scene's frame: 16:10.
pub const ASPECT: f32 = 16.0 / 10.0;
/// The horizon, from the frame's top, as a share of its height.
pub const HORIZON: f32 = 0.66;
/// The sun's centre, as a share of the frame's width.
pub const SUN_X: f32 = 0.805;
/// The background's opacity over the text column.
pub const SOLID: f32 = 0.97;
/// The least the fade takes beside the column, in logical pixels, on a page too narrow for more.
pub const MIN_FADE: f32 = 48.0;
/// The share of the page's width the fade takes at most.
const FADE_SPAN: f32 = 0.40;
/// The sun's radius, as a share of the frame's height.
const SUN_RADIUS: f32 = 0.112;

/// What the scene takes from a theme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Colors {
    pub dark: bool,
    /// The page's background.
    pub bg: Color,
    /// The theme's gradient, top to bottom: blue, violet, magenta, pink, coral, orange in Sunset.
    pub gradient: [Color; 6],
    /// The theme's outline color (clouds at dusk, the morning's silhouettes).
    pub outline: Color,
    /// The theme's deepest color (the dusk's silhouettes).
    pub shadow: Color,
}

impl Colors {
    /// A palette's.
    pub fn of(palette: &Palette) -> Self {
        Self {
            dark: palette.mode == crate::ui::tokens::Mode::Dark,
            bg: palette.surface(crate::ui::tokens::Surface::Bg),
            gradient: palette.gradient(),
            outline: palette.outline(),
            shadow: palette.shadow(),
        }
    }
}

/// Everything a picture depends on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Input {
    pub colors: Colors,
    /// The page, in physical pixels.
    pub width: u32,
    pub height: u32,
    /// Physical pixels to a logical one.
    pub scale: f32,
    /// The text column's width from the page's left edge, in logical pixels.
    pub column: f32,
    /// The window's rounded corner at the page's bottom right, its radius in logical pixels, if
    /// the window floats with drawn corners: the picture is clear outside it.
    pub corner: Option<f32>,
}

/// A generated picture: RGBA8, `width` by `height`, straight alpha.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

type Rgb = [f32; 3];

/// Points of a polygon or a line.
type Points = Vec<(f32, f32)>;

fn srgb(color: Color) -> Rgb {
    color.rgb8().map(|c| f32::from(c) / 255.0)
}

fn to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn lin(c: Rgb) -> Rgb {
    c.map(to_linear)
}

fn oklab(c: Rgb) -> Rgb {
    let l = (0.412_221_47 * c[0] + 0.536_332_54 * c[1] + 0.051_445_99 * c[2]).cbrt();
    let m = (0.211_903_5 * c[0] + 0.680_699_5 * c[1] + 0.107_396_96 * c[2]).cbrt();
    let s = (0.088_302_46 * c[0] + 0.281_718_85 * c[1] + 0.629_978_7 * c[2]).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn from_oklab(c: Rgb) -> Rgb {
    let l = (c[0] + 0.396_337_78 * c[1] + 0.215_803_76 * c[2]).powi(3);
    let m = (c[0] - 0.105_561_346 * c[1] - 0.063_854_17 * c[2]).powi(3);
    let s = (c[0] - 0.089_484_18 * c[1] - 1.291_485_5 * c[2]).powi(3);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

fn lerp3(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Between two sRGB colours, in OKLab; the result is sRGB.
fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    from_oklab(lerp3(oklab(lin(a)), oklab(lin(b)), t)).map(to_srgb)
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn add(a: &mut Rgb, b: Rgb, k: f32) {
    for (a, b) in a.iter_mut().zip(b) {
        *a += b * k;
    }
}

fn blend(a: &mut Rgb, b: Rgb, k: f32) {
    for (a, b) in a.iter_mut().zip(b) {
        *a += (b - *a) * k;
    }
}

/// A colour ramp along stops, in OKLab, sampled into a table of linear RGB.
struct Ramp {
    table: Vec<Rgb>,
}

impl Ramp {
    const STEPS: usize = 1024;

    fn new(stops: &[(f32, Rgb)]) -> Self {
        let labs: Vec<(f32, Rgb)> = stops.iter().map(|(p, c)| (*p, oklab(lin(*c)))).collect();
        let table = (0..=Self::STEPS)
            .map(|i| {
                let t = i as f32 / Self::STEPS as f32;
                let at = labs
                    .windows(2)
                    .position(|pair| t <= pair[1].0)
                    .unwrap_or(labs.len() - 2);
                let ((p0, c0), (p1, c1)) = (labs[at], labs[at + 1]);
                let k = ((t - p0) / (p1 - p0).max(1e-6)).clamp(0.0, 1.0);
                from_oklab(lerp3(c0, c1, k))
            })
            .collect();
        Self { table }
    }

    fn at(&self, t: f32) -> Rgb {
        self.table[(t.clamp(0.0, 1.0) * Self::STEPS as f32).round() as usize]
    }
}

/// A small, fixed sequence of numbers in 0..1, the same on every machine.
struct Sequence(u32);

impl Sequence {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / 16_777_216.0
    }

    fn between(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }
}

/// A pixel's noise, from its place and channel alone: -1..1, triangular, against banding.
fn dither(x: u32, y: u32, channel: u32) -> f32 {
    let hash = |mut h: u32| {
        h ^= h >> 16;
        h = h.wrapping_mul(0x7feb_352d);
        h ^= h >> 15;
        h = h.wrapping_mul(0x846c_a68b);
        h ^= h >> 16;
        h
    };
    let seed = x.wrapping_mul(0x9e37_79b1)
        ^ y.wrapping_mul(0x85eb_ca77)
        ^ channel.wrapping_mul(0xc2b2_ae3d);
    let a = hash(seed);
    let b = hash(a ^ 0x68e3_1da4);
    ((a >> 8) as f32 + (b >> 8) as f32) / 16_777_216.0 - 1.0
}

/// The six low clouds: centre x and y, spread x and y (shares of the frame), and opacity.
const CLOUDS: [(f32, f32, f32, f32, f32); 6] = [
    (0.8664, 0.4668, 0.0829, 0.00557, 0.3507),
    (0.9621, 0.3525, 0.0918, 0.00858, 0.3141),
    (0.6991, 0.3908, 0.0764, 0.00727, 0.2803),
    (0.6834, 0.4405, 0.1384, 0.00812, 0.3685),
    (0.4609, 0.4701, 0.1242, 0.00926, 0.4078),
    (0.4577, 0.5154, 0.0953, 0.00812, 0.3729),
];

/// How the scene's frame lies on the page, in physical pixels: the frame's size and where its
/// top left is (at or above and left of the page's), covering the page from its bottom right.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub width: f32,
    pub height: f32,
    pub left: f32,
    pub top: f32,
}

/// The frame covering a page `width` by `height`, anchored to its bottom right.
pub fn placement(width: f32, height: f32) -> Placement {
    let frame_width = width.max(height * ASPECT);
    let frame_height = frame_width / ASPECT;
    Placement {
        width: frame_width,
        height: frame_height,
        left: width - frame_width,
        top: height - frame_height,
    }
}

/// The sun's centre on the page, in physical pixels.
pub fn sun(width: f32, height: f32) -> (f32, f32) {
    let frame = placement(width, height);
    (
        frame.left + SUN_X * frame.width,
        frame.top + HORIZON * frame.height,
    )
}

/// The background's opacity at `x` logical pixels from the page's left, on a page `page`
/// logical pixels wide with a text column `column` wide. However narrow the page, the scene
/// stays: the column's background covers what of it lies under the text, and the fade takes what
/// room is left beside it, at least [`MIN_FADE`].
pub fn fade(x: f32, page: f32, column: f32) -> Option<f32> {
    let span = (FADE_SPAN * page).min(page - column).max(MIN_FADE);
    Some(SOLID * (1.0 - smoothstep(column, column + span, x)))
}

/// The palms' and the shore's coverage of the page, 0..255 a pixel.
fn silhouettes(frame: &Placement, width: u32, height: u32) -> Option<Mask> {
    let mut mask = Mask::new(width, height)?;
    let to_page = Transform::from_row(frame.width, 0.0, 0.0, frame.height, frame.left, frame.top);
    // The frame's x and y in shares of its width and height; lengths in shares of its width.
    let aspect = ASPECT;
    let mut fill = |path: Option<Path>| {
        if let Some(path) = path {
            mask.fill_path(&path, FillRule::Winding, true, to_page);
        }
    };
    // The shore: a low rise from the middle to the right.
    let mut shore = PathBuilder::new();
    shore.move_to(0.40, 1.0);
    for i in 0..=160 {
        let x = 0.40 + 0.60 * i as f32 / 160.0;
        let y = 1.0
            - 0.118 * smoothstep(0.40, 1.0, x).powf(1.25)
            - 0.006 * (i as f32 * 0.11).sin() * smoothstep(0.45, 0.6, x);
        shore.line_to(x, y);
    }
    shore.line_to(1.0, 1.0);
    shore.close();
    fill(shore.finish());
    let mut sequence = Sequence(0x2545_f491);
    for (base, top, bend, size) in [
        ((0.672, 1.02), (0.632, 0.330), 0.034, 0.118),
        ((0.945, 1.02), (0.972, 0.205), -0.028, 0.140),
    ] {
        for path in palm(base, top, bend, size, aspect, &mut sequence) {
            fill(Some(path));
        }
    }
    Some(mask)
}

/// A quadratic Bézier's points, `n` of them.
fn bezier(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), n: usize) -> Vec<(f32, f32)> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let u = 1.0 - t;
            (
                u * u * p0.0 + 2.0 * u * t * p1.0 + t * t * p2.0,
                u * u * p0.1 + 2.0 * u * t * p1.1 + t * t * p2.1,
            )
        })
        .collect()
}

fn polygon(points: &[(f32, f32)]) -> Option<Path> {
    let mut path = PathBuilder::new();
    let (first, rest) = points.split_first()?;
    path.move_to(first.0, first.1);
    for (x, y) in rest {
        path.line_to(*x, *y);
    }
    path.close();
    path.finish()
}

/// One palm, in frame shares (x of the width, y of the height; `size` and `bend` of the width):
/// a tapering, curved trunk, three coconuts and a crown of arching fronds.
fn palm(
    base: (f32, f32),
    top: (f32, f32),
    bend: f32,
    size: f32,
    aspect: f32,
    sequence: &mut Sequence,
) -> Vec<Path> {
    // Work in a square space (both axes in shares of the width), then scale y back.
    let square = |(x, y): (f32, f32)| (x, y / aspect);
    let (base, top) = (square(base), square(top));
    let control = ((base.0 + top.0) / 2.0 + bend, (base.1 + top.1) / 2.0);
    let trunk = bezier(base, control, top, 48);
    let widths: Vec<f32> = (0..48)
        .map(|i| (0.085 * (1.0 - i as f32 / 47.0).powf(1.6) + 0.046) * size)
        .collect();
    let mut outline: Vec<(f32, f32)> = trunk
        .iter()
        .zip(&widths)
        .map(|((x, y), w)| (x - w / 2.0, *y))
        .collect();
    outline.extend(
        trunk
            .iter()
            .zip(&widths)
            .rev()
            .map(|((x, y), w)| (x + w / 2.0, *y)),
    );
    let back = |points: Vec<(f32, f32)>| -> Vec<(f32, f32)> {
        points.into_iter().map(|(x, y)| (x, y * aspect)).collect()
    };
    let mut paths: Vec<Path> = polygon(&back(outline)).into_iter().collect();
    for (ox, oy, r) in [
        (-0.045, 0.050, 0.034),
        (0.030, 0.062, 0.036),
        (0.0, 0.085, 0.030),
    ] {
        let (cx, cy) = (top.0 + ox * size, top.1 + oy * size);
        let points: Vec<(f32, f32)> = (0..24)
            .map(|i| {
                let a = std::f32::consts::TAU * i as f32 / 24.0;
                (cx + a.cos() * r * size, cy + a.sin() * r * size)
            })
            .collect();
        paths.extend(polygon(&back(points)));
    }
    for degrees in [
        -198.0_f32, -170.0, -143.0, -116.0, -90.0, -64.0, -37.0, -10.0, 18.0,
    ] {
        let angle = (degrees + sequence.between(-7.0, 7.0)).to_radians();
        let flat = angle.cos().abs();
        let length = size * sequence.between(0.86, 1.06) * (0.74 + 0.30 * flat);
        let hang = 0.16 + 0.34 * flat + if angle.sin() > 0.0 { 0.20 } else { 0.0 };
        let droop = hang + sequence.between(-0.04, 0.05);
        let (feather, rib) = frond(top, angle, length, droop);
        paths.extend(polygon(&back(feather)));
        let mut midrib = PathBuilder::new();
        let rib = back(rib);
        midrib.move_to(rib[0].0, rib[0].1);
        for (x, y) in &rib[1..] {
            midrib.line_to(*x, *y);
        }
        if let Some(line) = midrib.finish() {
            // The midrib, stroked in the frame's square space: its width is a share of the width,
            // scaled back as the rest. Stroked in shares, which the page's transform scales.
            let stroke = Stroke {
                width: length * 0.011,
                ..Stroke::default()
            };
            paths.extend(line.stroke(&stroke, 1.0));
        }
    }
    paths
}

/// One frond: a feather (an arching midrib with a saw-toothed edge either side for the
/// leaflets, the lower ones hanging) and its midrib, in the square space.
fn frond(crown: (f32, f32), angle: f32, length: f32, droop: f32) -> (Points, Points) {
    let (dx, dy) = (angle.cos(), angle.sin());
    let tip = (
        crown.0 + dx * length,
        crown.1 + dy * length * 0.62 + droop * length,
    );
    let control = (
        crown.0 + dx * length * 0.50,
        crown.1 + dy * length * 0.95 - 0.16 * length,
    );
    let rib = bezier(crown, control, tip, 44);
    let last = rib.len() - 1;
    let mut upper = Vec::with_capacity(rib.len());
    let mut lower = Vec::with_capacity(rib.len());
    for (i, (x, y)) in rib.iter().enumerate() {
        let t = i as f32 / last as f32;
        let (a, b) = (rib[i.saturating_sub(1)], rib[(i + 1).min(last)]);
        let (mut tx, mut ty) = (b.0 - a.0, b.1 - a.1);
        let norm = tx.hypot(ty).max(1e-9);
        (tx, ty) = (tx / norm, ty / norm);
        // The normal that points up (smaller y).
        let (nx, ny) = if -tx <= 0.0 { (ty, -tx) } else { (-ty, tx) };
        let envelope = (std::f32::consts::PI * (t.powf(0.85) * 1.04).min(1.0))
            .sin()
            .max(0.0)
            .powf(0.75)
            * (1.0 - 0.25 * t);
        let jag = if i % 2 == 0 { 1.0 } else { 0.34 };
        let wu = 0.082 * length * envelope * jag;
        let wl = 0.118 * length * envelope * jag;
        upper.push((x + nx * wu + tx * wu * 0.55, y + ny * wu + ty * wu * 0.55));
        lower.push((
            x - nx * wl + tx * wl * 0.55,
            y - ny * wl + ty * wl * 0.55 + wl * 0.45,
        ));
    }
    upper.extend(lower.into_iter().rev());
    (upper, rib)
}

/// The picture for `input`, or none if the page is too narrow for a scene.
pub fn generate(input: &Input) -> Option<Picture> {
    let (width, height) = (input.width, input.height);
    if width == 0 || height == 0 || input.scale <= 0.0 {
        return None;
    }
    let page = width as f32 / input.scale;
    let colors = &input.colors;
    let dark = colors.dark;
    let white = [1.0, 1.0, 1.0];
    let bg = srgb(colors.bg);
    let [blue, violet, magenta, pink, coral, orange] = colors.gradient.map(srgb);
    let outline = srgb(colors.outline);
    let sky = Ramp::new(&if dark {
        [
            (0.00, bg),
            (0.18, mix(blue, bg, 0.66)),
            (0.40, mix(violet, bg, 0.36)),
            (0.62, mix(magenta, bg, 0.14)),
            (0.80, pink),
            (0.92, coral),
            (1.00, orange),
        ]
    } else {
        [
            (0.00, bg),
            (0.22, mix(blue, white, 0.80)),
            (0.44, mix(violet, white, 0.74)),
            (0.64, mix(magenta, white, 0.68)),
            (0.82, mix(pink, white, 0.58)),
            (0.93, mix(coral, white, 0.50)),
            (1.00, mix(orange, white, 0.42)),
        ]
    });
    let (sun_top, sun_bottom) = if dark {
        (lin(mix(orange, white, 0.72)), lin(mix(coral, orange, 0.35)))
    } else {
        (lin(white), lin(mix(orange, white, 0.62)))
    };
    let silhouette = lin(if dark {
        mix(srgb(colors.shadow), [0.0; 3], 0.35)
    } else {
        mix(outline, bg, 0.34)
    });
    let silhouette_alpha = if dark { 1.0 } else { 0.88 };
    let cloud = lin(mix(outline, magenta, 0.30));
    let pink_light = lin(pink);
    let streak_color = lin(mix(orange, white, if dark { 0.38 } else { 0.8 }));
    let line_color = lin(mix(orange, white, 0.5));
    let water = lin(if dark { bg } else { mix(bg, blue, 0.10) });
    let streak_amount = if dark { 1.25 } else { 0.80 };

    let frame = placement(width as f32, height as f32);
    let mask = silhouettes(&frame, width, height)?;
    let coverage = mask.data();

    // The reflection's per-row gain: smoothed noise along the frame's height.
    let rows = frame.height.ceil() as usize + 3;
    let mut sequence = Sequence(0x9e37_79b9);
    let raw: Vec<f32> = (0..rows).map(|_| sequence.next()).collect();
    let row_gain = |row: usize| -> f32 {
        let at = row.min(rows - 3);
        ((raw[at] + raw[at + 1] + raw[at + 2]) / 3.0).powi(2)
    };
    let encode: Vec<f32> = (0..=4096).map(|i| to_srgb(i as f32 / 4096.0)).collect();
    let encode = |c: f32| encode[(c.clamp(0.0, 1.0) * 4096.0) as usize];

    let fades: Vec<f32> = (0..width)
        .map(|x| fade((x as f32 + 0.5) / input.scale, page, input.column).unwrap_or(0.0))
        .collect();
    let corner = input.corner.map(|radius| radius * input.scale);

    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    for y in 0..height {
        let frame_y = y as f32 + 0.5 - frame.top;
        let fy = frame_y / frame.height;
        let above = fy < HORIZON;
        let depth = ((fy - HORIZON) / (1.0 - HORIZON)).clamp(0.0, 1.0);
        let base = if above {
            sky.at(fy / HORIZON)
        } else {
            let mut c = sky.at(1.0 - 0.55 * depth.powf(0.75));
            blend(&mut c, water, 0.30 + 0.62 * depth.powf(0.6));
            c
        };
        let dy = fy - HORIZON;
        let line_y = (-(dy / 0.0016).powi(2)).exp();
        let streak_width = 0.034 + 0.17 * depth;
        let streak_y = if above {
            0.0
        } else {
            (1.0 - depth).powf(1.3) * row_gain(frame_y.max(0.0) as usize) * streak_amount
        };
        let clouds: Vec<_> = CLOUDS
            .iter()
            .filter(|cloud| (fy - cloud.1).abs() < 4.0 * cloud.3 + 0.01)
            .collect();
        let row = y as usize * width as usize;
        for x in 0..width {
            let fx = (x as f32 + 0.5 - frame.left) / frame.width;
            let dx = (fx - SUN_X) * ASPECT;
            let r = (dx * dx + dy * dy).sqrt();
            let mut c = base;
            for &&(cx, cy, sx, sy, amount) in &clouds {
                let ex = ((fx - cx) / sx).powi(2);
                let m = (-(ex + ((fy - cy) / sy).powi(2))).exp() * amount;
                if dark {
                    let ly = (fy - cy - sy * 1.2) / (sy * 0.6);
                    blend(&mut c, cloud, m);
                    add(&mut c, pink_light, (-(ex + ly * ly)).exp() * 0.16);
                } else {
                    blend(&mut c, white, m * 0.8);
                }
            }
            if above && r < SUN_RADIUS + 0.003 {
                let disc = smoothstep(SUN_RADIUS + 0.0022, SUN_RADIUS - 0.0022, r);
                let k = ((HORIZON - fy) / SUN_RADIUS).clamp(0.0, 1.0).powf(0.8);
                blend(&mut c, lerp3(sun_bottom, sun_top, k), disc);
            }
            if streak_y > 0.0 {
                add(
                    &mut c,
                    streak_color,
                    (-(dx / streak_width).powi(2)).exp() * streak_y,
                );
            }
            if line_y > 0.001 {
                add(
                    &mut c,
                    line_color,
                    line_y * (-(dx / 0.60).powi(2)).exp() * 0.55,
                );
            }
            let a = f32::from(coverage[row + x as usize]) / 255.0 * silhouette_alpha;
            if a > 0.0 {
                blend(&mut c, silhouette, a);
            }
            let mut s = c.map(encode);
            blend(&mut s, bg, fades[x as usize]);
            let at = (row + x as usize) * 4;
            for (channel, value) in s.iter().enumerate() {
                let noise = dither(x, y, channel as u32) * 0.9 / 255.0;
                rgba[at + channel] = ((value + noise).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            rgba[at + 3] = 255;
        }
    }
    if let Some(radius) = corner {
        clear_corner(&mut rgba, width, height, radius);
    }
    Some(Picture {
        width,
        height,
        rgba,
    })
}

/// Clears what lies outside a rounded bottom-right corner of `radius` physical pixels,
/// antialiased.
fn clear_corner(rgba: &mut [u8], width: u32, height: u32, radius: f32) {
    let reach = radius.ceil() as u32 + 1;
    let (cx, cy) = (width as f32 - radius, height as f32 - radius);
    for y in height.saturating_sub(reach)..height {
        for x in width.saturating_sub(reach)..width {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            if px < cx || py < cy {
                continue;
            }
            let distance = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt() - radius;
            let coverage = (0.5 - distance).clamp(0.0, 1.0);
            let at = ((y * width + x) * 4 + 3) as usize;
            rgba[at] = (f32::from(rgba[at]) * coverage).round() as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tokens::color::contrast;
    use crate::ui::tokens::{Ink, Mode, Palette};

    fn input(palette: &Palette, size: (u32, u32), scale: f32) -> Input {
        Input {
            colors: Colors::of(palette),
            width: size.0,
            height: size.1,
            scale,
            column: 468.0,
            corner: None,
        }
    }

    fn pixel(picture: &Picture, x: u32, y: u32) -> Color {
        let at = ((y * picture.width + x) * 4) as usize;
        let [r, g, b] = [0, 1, 2].map(|c| u32::from(picture.rgba[at + c]));
        Color::hex((r << 16) | (g << 8) | b)
    }

    #[test]
    fn muted_text_reads_on_every_pixel_under_the_column() {
        for palette in Palette::ALL {
            let muted = palette.ink(Ink::Muted);
            for (size, scale) in [((1040, 754), 1.0), ((720, 594), 1.0), ((2320, 1394), 2.0)] {
                let input = input(&palette, size, scale);
                let picture = generate(&input).expect("wide enough");
                let column = (input.column * scale) as u32;
                let mut worst = f64::MAX;
                for y in 0..picture.height {
                    for x in 0..column {
                        worst = worst.min(contrast(muted, pixel(&picture, x, y)));
                    }
                }
                assert!(worst >= 4.5, "{}: {size:?}: {worst:.2}", palette.name);
            }
        }
    }

    #[test]
    fn a_narrow_page_keeps_its_scene_beside_the_column() {
        let palette = Palette::SUNSET_DARK;
        let bg = palette.surface(crate::ui::tokens::Surface::Bg);
        // Narrower than the column and a fade's room: the scene stays, the column covered.
        for (width, scale) in [(600_u32, 1.0), (520, 1.0), (1200, 2.0)] {
            let picture = generate(&input(&palette, (width, 600), scale)).expect("always a scene");
            let right = pixel(&picture, width - 4, 560);
            assert_ne!(
                right, bg,
                "{width} at {scale}x: the scene at the right edge"
            );
        }
        // The fade still covers the column, and takes at least its least room beside it.
        assert_eq!(fade(0.0, 600.0, 468.0), Some(SOLID));
        assert!(fade(468.0 + MIN_FADE, 500.0, 468.0).unwrap() < 0.01);
    }

    #[test]
    fn the_same_input_gives_the_same_bytes() {
        let input = input(&Palette::SUNSET_LIGHT, (800, 500), 1.0);
        assert_eq!(generate(&input), generate(&input));
    }

    #[test]
    fn the_theme_decides_the_sky_and_the_modes_differ() {
        let dark = input(&Palette::SUNSET_DARK, (1040, 754), 1.0);
        let base = generate(&dark).unwrap();
        let mut changed = dark;
        changed.colors.gradient[2] = Color::hex(0x00ff00);
        let other = generate(&changed).unwrap();
        // The sky at the magenta stop's height, at the right edge, clear of the fade, changes
        // with it.
        let frame = placement(1040.0, 754.0);
        let (x, y) = (1030, (frame.top + 0.62 * HORIZON * frame.height) as u32);
        assert_ne!(pixel(&base, x, y), pixel(&other, x, y));
        let light = generate(&input(&Palette::SUNSET_LIGHT, (1040, 754), 1.0)).unwrap();
        assert_ne!(base.rgba, light.rgba);
        assert!(!Colors::of(&Palette::SUNSET_LIGHT).dark);
        assert_eq!(Palette::SUNSET_DARK.mode, Mode::Dark);
    }

    #[test]
    fn the_sun_lands_where_the_mapping_says() {
        for (width, height) in [(1040_u32, 754_u32), (1600, 500), (700, 800)] {
            let palette = Palette::SUNSET_DARK;
            let picture = generate(&input(&palette, (width, height), 1.0)).unwrap();
            let frame = placement(width as f32, height as f32);
            // Covering: the frame is at least the page, and its bottom right is the page's.
            assert!(frame.width >= width as f32 - 0.01 && frame.height >= height as f32 - 0.01);
            assert!((frame.left + frame.width - width as f32).abs() < 0.01);
            assert!((frame.top + frame.height - height as f32).abs() < 0.01);
            let (sx, sy) = sun(width as f32, height as f32);
            assert!((sx - (frame.left + 0.805 * frame.width)).abs() < 0.01);
            // Just above the horizon at the sun's centre, the disc is bright; well left of the
            // disc, at the same height, the sky is darker.
            let radius = SUN_RADIUS * frame.height;
            let y = (sy - radius * 0.5) as u32;
            let disc = pixel(&picture, sx as u32, y).luminance();
            let beside = pixel(&picture, (sx - radius * 2.2).max(0.0) as u32, y).luminance();
            assert!(disc > beside, "{width}x{height}: {disc} <= {beside}");
        }
    }

    #[test]
    fn silhouettes_are_darker_than_the_sky_beside_them_at_dusk() {
        let picture = generate(&input(&Palette::SUNSET_DARK, (1040, 754), 1.0)).unwrap();
        let frame = placement(1040.0, 754.0);
        // The right palm's trunk, low down, and the sky a little left of it.
        let trunk_x = frame.left + 0.955 * frame.width;
        let y = frame.top + 0.62 * frame.height;
        let on = pixel(&picture, trunk_x as u32, y as u32).luminance();
        let beside = pixel(&picture, (trunk_x - 0.06 * frame.width) as u32, y as u32).luminance();
        assert!(on < beside, "{on} >= {beside}");
    }

    /// `cargo test --release --lib scene::tests::timing -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement, for an optimised build"]
    fn timing() {
        for (size, scale) in [((1040, 754), 1.0), ((2320, 1394), 2.0)] {
            for palette in Palette::ALL {
                let input = input(&palette, size, scale);
                let _ = generate(&input);
                let start = std::time::Instant::now();
                let runs = 5;
                for _ in 0..runs {
                    let _ = generate(&input);
                }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(runs);
                println!("{}x{} {}: {ms:.1} ms", size.0, size.1, palette.name);
            }
        }
    }

    #[test]
    fn the_corner_is_clear_outside_its_radius() {
        let mut input = input(&Palette::SUNSET_DARK, (900, 600), 1.0);
        input.corner = Some(13.0);
        let picture = generate(&input).unwrap();
        let alpha = |x: u32, y: u32| picture.rgba[((y * 900 + x) * 4 + 3) as usize];
        assert_eq!(alpha(899, 599), 0);
        assert_eq!(alpha(880, 580), 255);
        assert_eq!(alpha(0, 0), 255);
    }
}
