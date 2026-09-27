//! The colour-matrix subset of the CSS `filter` shorthand functions.
//!
//! Seven shorthand filter functions expand to a single `feColorMatrix`, or to
//! an `feComponentTransfer` whose `feFuncR`/`feFuncG`/`feFuncB` are
//! `type="linear"` or a two-entry `type="table"`:
//!
//! | function | equivalent primitive |
//! |---|---|
//! | `grayscale(a)`   | `feColorMatrix type="matrix"` |
//! | `sepia(a)`       | `feColorMatrix type="matrix"` |
//! | `saturate(a)`    | `feColorMatrix type="saturate"` |
//! | `hue-rotate(θ)`  | `feColorMatrix type="hueRotate"` |
//! | `invert(a)`      | `feComponentTransfer` / `type="table" tableValues="a (1 - a)"` |
//! | `brightness(a)`  | `feComponentTransfer` / `type="linear" slope="a"` |
//! | `contrast(a)`    | `feComponentTransfer` / `type="linear" slope="a" intercept="-(0.5 * a) + 0.5"` |
//!
//! Each of those seven is an affine map of the **non-premultiplied** RGB triple
//! that leaves alpha alone, so all seven are represented here by the same 4×5
//! matrix `feColorMatrix type="matrix"` uses:
//!
//! ```text
//! | R' |   | a00 a01 a02 a03 a04 |   | R |
//! | G' |   | a10 a11 a12 a13 a14 |   | G |
//! | B' | = | a20 a21 a22 a23 a24 | · | B |
//! | A' |   | a30 a31 a32 a33 a34 |   | A |
//! | 1  |   |  0   0   0   0   1  |   | 1 |
//! ```
//!
//! The remaining three shorthands are **not** in this module and fall through to
//! the [`crate::filters`] `Filter` graph instead: `blur()` and `drop-shadow()`
//! are spatial (they sample neighbouring pixels), and `opacity()` scales alpha
//! — see [`ColorMatrixChain::from_filters`] for why alpha-scaling cannot join
//! this path.
//!
//! # Composited-group filtering
//!
//! The normative requirement is: "All the elements descendants are rendered
//! together as a group with the filter effect applied to the group as a whole."
//! <https://drafts.csswg.org/filter-effects-1/#FilterProperty>
//!
//! With `vello-cpu-filters`, the subtree is recorded, rasterised into a bounded
//! premultiplied RGBA buffer, transformed one pixel at a time, and then
//! composited into the parent. The clamp belongs here, after the subtree has
//! composited. Applying and clamping the matrix to each source paint is not
//! equivalent: a translucent white source can clamp before its coverage is
//! mixed with a dark destination, producing a different result.
//!
//! The pinned Vello CPU renderer does not execute colour-matrix primitives and
//! its multithreaded dispatcher rejects every filter layer, so `blitz-paint`
//! supplies the offscreen pass and reuses one renderer's scratch allocations.
//! Builds without that feature retain the older per-paint rewrite as a
//! compatibility fallback for backends without an offscreen seam.

use anyrender::Paint;
use anyrender::recording::{RenderCommand, Scene};
#[cfg(feature = "vello-cpu-filters")]
use anyrender::{ImageRenderer as _, PaintScene as _};
#[cfg(feature = "vello-cpu-filters")]
use anyrender_vello_cpu::VelloCpuImageRenderer;
use color::{ColorSpaceTag, DynamicColor, Srgb};
#[cfg(feature = "vello-cpu-filters")]
use kurbo::{Affine, Rect};
use peniko::{
    Blob, ColorStop, ColorStops, Gradient, ImageAlphaType, ImageBrush, ImageData, ImageFormat,
    InterpolationAlphaSpace,
};
#[cfg(feature = "vello-cpu-filters")]
use peniko::{Extend, ImageQuality, ImageSampler};
use smallvec::SmallVec;
#[cfg(feature = "vello-cpu-filters")]
use std::cell::RefCell;
use std::sync::Arc;

use crate::color::Color;
use crate::filters::StyloFilter;

/// The largest number of stops one authored gradient segment may be subdivided
/// into. Each chain stage can only add one breakpoint per channel per bound, so
/// a realistic `filter` never approaches this; it exists so a pathological
/// chain cannot grow the ramp without bound.
const MAX_SEGMENT_STOPS: usize = 32;

#[cfg(feature = "vello-cpu-filters")]
thread_local! {
    /// Keep one renderer's scratch allocations warm without retaining one
    /// framebuffer for every distinct animated element size ever observed.
    static FILTER_RENDERER: RefCell<Option<(u32, u32, VelloCpuImageRenderer)>> =
        const { RefCell::new(None) };
}

/// Published luminance coefficients for `feColorMatrix type="saturate"` and
/// `type="hueRotate"`.
const LUMA_SATURATE: [f32; 3] = [0.213, 0.715, 0.072];
/// `grayscale()` uses slightly different published luminance coefficients. Deliberately
/// *not* [`LUMA_SATURATE`]: this module reproduces each set as published.
const LUMA_GRAYSCALE: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// A single `feColorMatrix type="matrix"` operation: four rows of
/// `[R, G, B, A, offset]` coefficients, in the row-major order the `values`
/// attribute lists them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColorMatrix([f32; 20]);

impl ColorMatrix {
    /// The default identity matrix for `type="matrix"`.
    pub(crate) const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, 0.0,
    ]);

    /// Build from the three RGB rows of `feColorMatrix`, leaving alpha alone.
    ///
    /// `rgb[row]` is `[a_r, a_g, a_b, offset]`; the `A` column is zero for every
    /// matrix in this module, because none of the seven shorthands lets alpha
    /// contribute to a colour channel.
    const fn from_rgb_rows(rgb: [[f32; 4]; 3]) -> Self {
        Self([
            rgb[0][0], rgb[0][1], rgb[0][2], 0.0, rgb[0][3], //
            rgb[1][0], rgb[1][1], rgb[1][2], 0.0, rgb[1][3], //
            rgb[2][0], rgb[2][1], rgb[2][2], 0.0, rgb[2][3], //
            0.0, 0.0, 0.0, 1.0, 0.0,
        ])
    }

    /// The same affine transfer function on each of R, G and B, as the
    /// `feFuncR`/`feFuncG`/`feFuncB` `type="linear"` form
    /// `C' = slope * C + intercept`.
    const fn linear_transfer(slope: f32, intercept: f32) -> Self {
        Self::from_rgb_rows([
            [slope, 0.0, 0.0, intercept],
            [0.0, slope, 0.0, intercept],
            [0.0, 0.0, slope, intercept],
        ])
    }

    /// `brightness(amount)`: `feFuncR/G/B type="linear" slope="[amount]"`.
    pub(crate) fn brightness(amount: f32) -> Self {
        Self::linear_transfer(amount, 0.0)
    }

    /// `contrast(amount)`: `type="linear" slope="[amount]"
    /// intercept="-(0.5 * [amount]) + 0.5"`.
    pub(crate) fn contrast(amount: f32) -> Self {
        Self::linear_transfer(amount, -(0.5 * amount) + 0.5)
    }

    /// `invert(amount)`: `type="table" tableValues="[amount] (1 - [amount])"`.
    ///
    /// A two-entry table is one interpolation region, so
    /// `C' = v_k + (C - k/n) * n * (v_{k+1} - v_k)` with `n = 1, k = 0` reduces
    /// to the affine `C' = amount + C * (1 - 2 * amount)`.
    ///
    /// Amounts above one clamp to one.
    pub(crate) fn invert(amount: f32) -> Self {
        let amount = amount.clamp(0.0, 1.0);
        Self::linear_transfer(1.0 - 2.0 * amount, amount)
    }

    /// `saturate(amount)` via `feColorMatrix type="saturate"`.
    pub(crate) fn saturate(amount: f32) -> Self {
        let [lr, lg, lb] = LUMA_SATURATE;
        let s = amount;
        Self::from_rgb_rows([
            [lr + (1.0 - lr) * s, lg - lg * s, lb - lb * s, 0.0],
            [lr - lr * s, lg + (1.0 - lg) * s, lb - lb * s, 0.0],
            [lr - lr * s, lg - lg * s, lb + (1.0 - lb) * s, 0.0],
        ])
    }

    /// `grayscale(amount)`, written as a `type="matrix"` whose
    /// coefficients are expressed in terms of `[1 - amount]`.
    ///
    /// Amounts above one clamp to one.
    pub(crate) fn grayscale(amount: f32) -> Self {
        let k = 1.0 - amount.clamp(0.0, 1.0);
        let [lr, lg, lb] = LUMA_GRAYSCALE;
        Self::from_rgb_rows([
            [lr + (1.0 - lr) * k, lg - lg * k, lb - lb * k, 0.0],
            [lr - lr * k, lg + (1.0 - lg) * k, lb - lb * k, 0.0],
            [lr - lr * k, lg - lg * k, lb + (1.0 - lb) * k, 0.0],
        ])
    }

    /// The published `sepia(amount)` matrix.
    ///
    /// Amounts above one clamp to one.
    pub(crate) fn sepia(amount: f32) -> Self {
        let k = 1.0 - amount.clamp(0.0, 1.0);
        Self::from_rgb_rows([
            [0.393 + 0.607 * k, 0.769 - 0.769 * k, 0.189 - 0.189 * k, 0.0],
            [0.349 - 0.349 * k, 0.686 + 0.314 * k, 0.168 - 0.168 * k, 0.0],
            [0.272 - 0.272 * k, 0.534 - 0.534 * k, 0.131 + 0.869 * k, 0.0],
        ])
    }

    /// `hue-rotate(angle)` via `feColorMatrix type="hueRotate"`, whose 3×3 is
    /// a constant matrix plus `cos` and `sin` terms. `angle` is in radians and
    /// is deliberately not normalised.
    pub(crate) fn hue_rotate(angle_radians: f32) -> Self {
        let (sin, cos) = angle_radians.sin_cos();
        let [lr, lg, lb] = LUMA_SATURATE;
        // Published `type="hueRotate"` matrix.
        #[rustfmt::skip]
        let base = [
            [lr, lg, lb],
            [lr, lg, lb],
            [lr, lg, lb],
        ];
        #[rustfmt::skip]
        let cos_term = [
            [ 0.787, -0.715, -0.072],
            [-0.213,  0.285, -0.072],
            [-0.213, -0.715,  0.928],
        ];
        #[rustfmt::skip]
        let sin_term = [
            [-0.213, -0.715,  0.928],
            [ 0.143,  0.140, -0.283],
            [-0.787,  0.715,  0.072],
        ];
        let mut rows = [[0.0_f32; 4]; 3];
        for r in 0..3 {
            for c in 0..3 {
                rows[r][c] = base[r][c] + cos * cos_term[r][c] + sin * sin_term[r][c];
            }
        }
        Self::from_rgb_rows(rows)
    }

    /// Apply the matrix to one non-premultiplied sRGB colour.
    ///
    /// The filter arithmetic uses non-premultiplied, sRGB-encoded components;
    /// results are clamped to the closed interval `[0,1]`.
    /// Apply the matrix to a vector in the space the backend interpolates
    /// gradients in, **without** clamping.
    ///
    /// For plain components this is the matrix directly. For premultiplied
    /// components, scaling `C' = M·C + t` by alpha gives
    /// `P' = M·P + t·A`, which is affine in the vector — the reason this path
    /// requires a zero alpha column, since `A * (m·A)` would be quadratic.
    fn apply_vector(&self, vector: [f32; 4], premultiplied: bool) -> [f32; 4] {
        let m = &self.0;
        let a = vector[3];
        let mut out = [0.0_f32; 4];
        for (row, slot) in out[..3].iter_mut().enumerate() {
            let base = row * 5;
            let linear = m[base] * vector[0] + m[base + 1] * vector[1] + m[base + 2] * vector[2];
            let alpha_column = m[base + 3] * a;
            let offset = m[base + 4];
            *slot = if premultiplied {
                linear + offset * a
            } else {
                linear + alpha_column + offset
            };
        }
        out[3] = a;
        out
    }

    pub(crate) fn apply(&self, color: Color) -> Color {
        let [r, g, b, a] = color.components;
        let m = &self.0;
        let out = |row: usize| {
            (m[row * 5] * r
                + m[row * 5 + 1] * g
                + m[row * 5 + 2] * b
                + m[row * 5 + 3] * a
                + m[row * 5 + 4])
                .clamp(0.0, 1.0)
        };
        Color::new([out(0), out(1), out(2), out(3)])
    }
}

/// The ordered list of colour matrices a `filter` value expands to.
///
/// Matrices are applied one at a time in author order with a `[0,1]` clamp
/// between them. Pre-multiplying them would skip the intermediate clamps and
/// diverge whenever an intermediate result leaves the unit range.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ColorMatrixChain(SmallVec<[ColorMatrix; 2]>);

impl ColorMatrixChain {
    /// Expand a computed `filter` list into colour matrices.
    ///
    /// Returns `None` unless **every** function in the list is one of the seven
    /// alpha-preserving colour-matrix shorthands. A list containing `blur()`,
    /// `drop-shadow()`, `opacity()` or a `url()` reference is left entirely to
    /// the [`crate::filters`] `Filter` graph. `blur()` and `drop-shadow()` are
    /// spatial, `opacity()` changes alpha, and mixing these with the pixel
    /// matrix path would require an ordered multi-stage filter graph that the
    /// pinned renderer does not support.
    pub(crate) fn from_filters(filters: &[StyloFilter]) -> Option<Self> {
        if filters.is_empty() {
            return None;
        }
        let mut chain = SmallVec::with_capacity(filters.len());
        for filter in filters {
            chain.push(match filter {
                StyloFilter::Brightness(amount) => ColorMatrix::brightness(amount.0),
                StyloFilter::Contrast(amount) => ColorMatrix::contrast(amount.0),
                StyloFilter::Grayscale(amount) => ColorMatrix::grayscale(amount.0),
                StyloFilter::HueRotate(angle) => ColorMatrix::hue_rotate(angle.radians()),
                StyloFilter::Invert(amount) => ColorMatrix::invert(amount.0),
                StyloFilter::Saturate(amount) => ColorMatrix::saturate(amount.0),
                StyloFilter::Sepia(amount) => ColorMatrix::sepia(amount.0),
                StyloFilter::Blur(_)
                | StyloFilter::DropShadow(_)
                | StyloFilter::Opacity(_)
                | StyloFilter::Url(_) => return None,
            });
        }
        Some(Self(chain))
    }

    /// Whether the chain leaves every colour unchanged.
    ///
    /// `hue-rotate(0deg)` is exactly the identity (`cos 0 = 1`, `sin 0 = 0`).
    /// Skipping such a chain keeps a no-op filter byte-identical to no filter.
    pub(crate) fn is_identity(&self) -> bool {
        self.0.iter().all(|m| *m == ColorMatrix::IDENTITY)
    }

    fn apply(&self, color: Color) -> Color {
        self.0.iter().fold(color, |acc, m| m.apply(acc))
    }

    fn apply_dynamic(&self, color: DynamicColor) -> DynamicColor {
        DynamicColor::from_alpha_color(self.apply(color.to_alpha_color::<Srgb>()))
    }

    /// Rasterise the recorded subtree, apply this chain once to its composited
    /// premultiplied pixels, and return an image placed in device space.
    ///
    /// The normative requirement is: "All the elements descendants are rendered
    /// together as a group with the filter effect applied to the group as a whole."
    /// <https://drafts.csswg.org/filter-effects-1/#FilterProperty>
    #[cfg(feature = "vello-cpu-filters")]
    pub(crate) fn rasterize_composited_scene(
        &self,
        scene: Scene,
        bounds: Rect,
    ) -> Result<(ImageBrush, Affine), Scene> {
        let x0 = bounds.x0.floor();
        let y0 = bounds.y0.floor();
        let x1 = bounds.x1.ceil();
        let y1 = bounds.y1.ceil();
        let width = x1 - x0;
        let height = y1 - y0;
        if ![x0, y0, width, height].iter().all(|v| v.is_finite())
            || width <= 0.0
            || height <= 0.0
            || width > f64::from(u16::MAX)
            || height > f64::from(u16::MAX)
        {
            return Err(scene);
        }
        let width = width as u32;
        let height = height as u32;
        let offset = Affine::translate((-x0, -y0));
        let mut renderer = FILTER_RENDERER
            .with(|cached| cached.borrow_mut().take())
            .filter(|(cached_width, cached_height, _)| {
                *cached_width == width && *cached_height == height
            })
            .map(|(_, _, renderer)| renderer)
            .unwrap_or_else(|| VelloCpuImageRenderer::new(width, height));
        renderer.reset();
        let mut pixels = Vec::new();
        renderer.render_to_vec(
            move |target| target.append_scene(scene, offset),
            &mut pixels,
        );
        FILTER_RENDERER.with(|cached| {
            cached.borrow_mut().replace((width, height, renderer));
        });
        self.apply_to_premultiplied_rgba8(&mut pixels);

        let image = ImageBrush {
            image: ImageData {
                data: Blob::new(Arc::new(pixels)),
                format: ImageFormat::Rgba8,
                alpha_type: ImageAlphaType::AlphaPremultiplied,
                width,
                height,
            },
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        };
        Ok((image, Affine::translate((x0, y0))))
    }

    #[cfg(feature = "vello-cpu-filters")]
    fn apply_to_premultiplied_rgba8(&self, pixels: &mut [u8]) {
        for pixel in pixels.as_chunks_mut::<4>().0 {
            let alpha = f32::from(pixel[3]) / 255.0;
            if alpha == 0.0 {
                continue;
            }
            let unpremultiply = 1.0 / (255.0 * alpha);
            let source = Color::new([
                f32::from(pixel[0]) * unpremultiply,
                f32::from(pixel[1]) * unpremultiply,
                f32::from(pixel[2]) * unpremultiply,
                alpha,
            ]);
            let filtered = self.apply(source);
            for (channel, value) in pixel[..3].iter_mut().zip(filtered.components[..3].iter()) {
                *channel = (value * alpha * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }

    /// Compatibility fallback for a backend without an offscreen pass.
    ///
    /// This rewrites individual paints and therefore cannot reproduce a clamp
    /// that occurs after translucent sources have composited. The Vello CPU
    /// feature takes [`Self::rasterize_composited_scene`] instead.
    pub(crate) fn apply_to_scene(&self, scene: &mut Scene) {
        if self.is_identity() {
            return;
        }
        for command in &mut scene.commands {
            match command {
                RenderCommand::Fill(cmd) => self.apply_to_paint(&mut cmd.brush),
                RenderCommand::Stroke(cmd) => self.apply_to_paint(&mut cmd.brush),
                RenderCommand::GlyphRun(cmd) => self.apply_to_paint(&mut cmd.brush),
                RenderCommand::BoxShadow(cmd) => cmd.brush = self.apply(cmd.brush),
                // A nested layer carries no colour of its own: its blend mode,
                // group alpha and clip are all alpha-domain, and any nested
                // `filter` on it is a graph this chain deliberately does not own.
                RenderCommand::PushLayer(_)
                | RenderCommand::PushClipLayer(_)
                | RenderCommand::PopLayer => {}
            }
        }
    }

    fn apply_to_paint(&self, paint: &mut Paint) {
        match paint {
            Paint::Solid(color) => *color = self.apply(*color),
            Paint::Gradient(gradient) => self.apply_to_gradient(gradient),
            Paint::Image(brush) => self.apply_to_image(brush),
            // A backend-owned paint whose pixels this crate cannot see. Leaving
            // it alone is wrong, but inventing a colour for it would be worse;
            // `blitz-paint` never emits either variant today.
            Paint::Resource(_) | Paint::Custom(_) => {}
        }
    }

    /// Rewrite a gradient's colour ramp.
    ///
    /// A gradient is one paint but many pixel colours, and its `[0,1]` clamp is
    /// applied per *pixel*, after the ramp is interpolated. Filtering only
    /// the authored stops and letting the backend interpolate between the
    /// already-clamped results is therefore **not** the same function: the
    /// affine part commutes with interpolation, but the clamp does not. For
    /// `brightness(1.3)` over a `0.9 -> 0.5` ramp the unclamped value crosses 1
    /// a third of the way along; clamping first and interpolating gives 0.885
    /// there where the per-pixel answer is 1.0, a divergence of ~29/255.
    ///
    /// The fix is to give the backend the extra stops it needs to reproduce the
    /// per-pixel function exactly. Between two adjacent stops
    /// `vello_common::encode::encode_stops` builds a single linear ramp
    /// (`bias + x * scale`) over the interpolation components, so along one
    /// segment the source vector is affine in `x`. Each chain stage maps it by
    /// an affine map and then clamps, so the *result* is piecewise-affine in
    /// `x`, and its only breakpoints are where some stage's channel crosses 0
    /// or 1. Splitting the segment at exactly those crossings leaves a function
    /// that is affine on every piece, which is precisely what the backend's
    /// per-segment linear ramp reproduces.
    ///
    /// This models the configuration `blitz-paint` builds and peniko defaults
    /// to: sRGB stops, linearly interpolated. A gradient asking to interpolate
    /// in another space goes through the backend's own resampling first, whose
    /// stop positions this crate cannot see, so those keep the endpoint-only
    /// rewrite and stay approximate — as does any segment that would need more
    /// than [`MAX_SEGMENT_STOPS`] stops.
    fn apply_to_gradient(&self, gradient: &mut Gradient) {
        let premultiplied = match gradient.interpolation_alpha_space {
            InterpolationAlphaSpace::Premultiplied => true,
            InterpolationAlphaSpace::Unpremultiplied => false,
        };
        // Premultiplied interpolation is only affine in the interpolated vector
        // while alpha does not feed a colour channel: `A * (m·A)` is quadratic
        // in alpha. Every matrix this module builds zeroes that column, so this
        // is a guard, not a live path.
        let alpha_column_is_zero = self
            .0
            .iter()
            .all(|m| m.0[3] == 0.0 && m.0[8] == 0.0 && m.0[13] == 0.0);
        let exact = gradient.interpolation_cs == ColorSpaceTag::Srgb
            && (!premultiplied || alpha_column_is_zero);

        if !exact {
            for stop in gradient.stops.iter_mut() {
                stop.color = self.apply_dynamic(stop.color);
            }
            return;
        }

        let source: SmallVec<[ColorStop; 4]> = gradient.stops.0.clone();
        let mut out: SmallVec<[ColorStop; 4]> = SmallVec::new();
        for (index, pair) in source.windows(2).enumerate() {
            let (left, right) = (pair[0], pair[1]);
            if index == 0 {
                out.push(ColorStop {
                    offset: left.offset,
                    color: self.apply_dynamic(left.color),
                });
            }
            let span = right.offset - left.offset;
            if span > 0.0 {
                let start = to_vector(left.color, premultiplied);
                let end = to_vector(right.color, premultiplied);
                for (x, vector) in self.segment_breakpoints(start, end, premultiplied) {
                    out.push(ColorStop {
                        offset: left.offset + x * span,
                        color: from_vector(vector, premultiplied),
                    });
                }
            }
            out.push(ColorStop {
                offset: right.offset,
                color: self.apply_dynamic(right.color),
            });
        }
        if out.is_empty() {
            // Zero or one stop: nothing to interpolate between.
            for stop in gradient.stops.iter_mut() {
                stop.color = self.apply_dynamic(stop.color);
            }
            return;
        }
        gradient.stops = ColorStops(out);
    }

    /// The interior breakpoints of one gradient segment, as `(x, vector)` pairs
    /// with `x` strictly inside `(0, 1)` and the vector already filtered.
    ///
    /// Each stage is walked over the pieces the previous stages produced. On a
    /// piece the input is affine in `x`, so a channel meets its bound at most
    /// once, and the root is the linear solution of `out_c(x) = bound(x)`.
    fn segment_breakpoints(
        &self,
        start: [f32; 4],
        end: [f32; 4],
        premultiplied: bool,
    ) -> SmallVec<[(f32, [f32; 4]); 4]> {
        const EPS: f32 = 1e-6;

        let mut pieces: SmallVec<[(f32, [f32; 4]); 8]> =
            SmallVec::from_slice(&[(0.0, start), (1.0, end)]);
        for matrix in &self.0 {
            let mut next: SmallVec<[(f32, [f32; 4]); 8]> = SmallVec::new();
            for window in pieces.windows(2) {
                let (xa, va) = window[0];
                let (xb, vb) = window[1];
                let qa = matrix.apply_vector(va, premultiplied);
                let qb = matrix.apply_vector(vb, premultiplied);
                next.push((xa, clamp_vector(qa, premultiplied)));

                let mut roots: SmallVec<[f32; 6]> = SmallVec::new();
                for channel in 0..3 {
                    for bound in 0..2 {
                        // `f(s) = q_c(s) - bound(s)`, both affine in `s`.
                        let (ba, bb) = if bound == 0 {
                            (0.0, 0.0)
                        } else if premultiplied {
                            (qa[3], qb[3])
                        } else {
                            (1.0, 1.0)
                        };
                        let fa = qa[channel] - ba;
                        let fb = qb[channel] - bb;
                        let delta = fa - fb;
                        if delta.abs() <= EPS || (fa > 0.0) == (fb > 0.0) {
                            continue;
                        }
                        let s = fa / delta;
                        if s > EPS && s < 1.0 - EPS {
                            roots.push(s);
                        }
                    }
                }
                roots.sort_unstable_by(f32::total_cmp);
                roots.dedup_by(|a, b| (*a - *b).abs() <= EPS);
                for s in roots {
                    let x = xa + s * (xb - xa);
                    let q = std::array::from_fn(|i| qa[i] + s * (qb[i] - qa[i]));
                    next.push((x, clamp_vector(q, premultiplied)));
                }
            }
            if let Some(&(xb, vb)) = pieces.last() {
                let last = matrix.apply_vector(vb, premultiplied);
                next.push((xb, clamp_vector(last, premultiplied)));
            }
            if next.len() > MAX_SEGMENT_STOPS {
                // Refuse to grow the ramp without bound; the pieces found so
                // far are still a strictly better reconstruction than the two
                // endpoints alone.
                pieces = next;
                break;
            }
            pieces = next;
        }
        pieces
            .into_iter()
            .filter(|(x, _)| *x > EPS && *x < 1.0 - EPS)
            .collect()
    }

    /// Rewrite an image brush's pixels.
    ///
    /// Matrix calculation uses non-premultiplied values, so premultiplied
    /// source pixels are divided out first and multiplied back afterwards.
    /// Fully transparent pixels have no colour to filter and are left alone.
    fn apply_to_image(&self, brush: &mut ImageBrush) {
        let image = &mut brush.image;
        // `ImageFormat` is `#[non_exhaustive]`: a variant this code has never
        // seen has an unknown byte layout, and guessing at one would corrupt
        // the image. Leave such a brush unfiltered.
        let (ri, gi, bi) = match image.format {
            ImageFormat::Rgba8 => (0, 1, 2),
            ImageFormat::Bgra8 => (2, 1, 0),
            _ => return,
        };
        let premultiplied = match image.alpha_type {
            ImageAlphaType::Alpha => false,
            ImageAlphaType::AlphaPremultiplied => true,
        };
        let src: &[u8] = image.data.data();
        let mut out = Vec::with_capacity(src.len());
        for px in src.as_chunks::<4>().0 {
            let a = f32::from(px[3]) / 255.0;
            if a == 0.0 {
                out.extend_from_slice(px);
                continue;
            }
            let unmul = if premultiplied { 1.0 / a } else { 1.0 };
            let mut channel = [0.0_f32; 4];
            channel[0] = f32::from(px[ri]) / 255.0 * unmul;
            channel[1] = f32::from(px[gi]) / 255.0 * unmul;
            channel[2] = f32::from(px[bi]) / 255.0 * unmul;
            channel[3] = a;
            let filtered = self.apply(Color::new(channel));
            let remul = if premultiplied { a } else { 1.0 };
            let byte = |v: f32| (v * remul * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            let mut dst = [0_u8; 4];
            dst[ri] = byte(filtered.components[0]);
            dst[gi] = byte(filtered.components[1]);
            dst[bi] = byte(filtered.components[2]);
            dst[3] = px[3];
            out.extend_from_slice(&dst);
        }
        *image = ImageData {
            data: Blob::new(Arc::new(out)),
            format: image.format,
            alpha_type: image.alpha_type,
            width: image.width,
            height: image.height,
        };
    }
}

/// The vector the backend interpolates: premultiplied `[R, G, B, A]` or plain
/// `[R, G, B, A]`, matching `vello_common::encode::encode_stops`.
fn to_vector(color: DynamicColor, premultiplied: bool) -> [f32; 4] {
    let [r, g, b, a] = color.to_alpha_color::<Srgb>().components;
    if premultiplied {
        [r * a, g * a, b * a, a]
    } else {
        [r, g, b, a]
    }
}

/// The inverse of [`to_vector`].
fn from_vector(vector: [f32; 4], premultiplied: bool) -> DynamicColor {
    let a = vector[3];
    let components = if premultiplied && a > 0.0 {
        [vector[0] / a, vector[1] / a, vector[2] / a, a]
    } else if premultiplied {
        [0.0, 0.0, 0.0, 0.0]
    } else {
        vector
    };
    DynamicColor::from_alpha_color(Color::new(components))
}

/// Clamp colour channels to `[0,1]`, expressed in whichever space the vector
/// uses: premultiplied colour is bounded by alpha.
fn clamp_vector(mut vector: [f32; 4], premultiplied: bool) -> [f32; 4] {
    let upper = if premultiplied { vector[3] } else { 1.0 };
    for channel in &mut vector[..3] {
        *channel = channel.clamp(0.0, upper);
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(color: Color) -> [u8; 3] {
        let c = color.to_rgba8();
        [c.r, c.g, c.b]
    }

    const MID: Color = Color::from_rgb8(128, 64, 192);

    #[test]
    fn hue_rotate_zero_is_the_identity_matrix() {
        // Zero rotation collapses the matrix to identity.
        let m = ColorMatrix::hue_rotate(0.0);
        for (got, want) in m.0.iter().zip(ColorMatrix::IDENTITY.0.iter()) {
            assert!((got - want).abs() < 1e-6, "{:?} != identity", m.0);
        }
        assert_eq!(rgb(m.apply(MID)), rgb(MID));
    }

    #[test]
    fn hue_rotate_360_degrees_is_not_normalised_away() {
        // Do not normalize the authored angle; a full turn is still identity.
        let m = ColorMatrix::hue_rotate(std::f32::consts::TAU);
        let out = m.apply(MID);
        for i in 0..3 {
            assert!((out.components[i] - MID.components[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn hue_rotate_matches_the_spec_matrix_at_90_degrees() {
        // a00 = 0.213 + cos·0.787 - sin·0.213; at 90°, cos = 0, sin = 1.
        let m = ColorMatrix::hue_rotate(std::f32::consts::FRAC_PI_2);
        assert!((m.0[0] - (0.213 - 0.213)).abs() < 1e-6);
        assert!((m.0[1] - (0.715 - 0.715)).abs() < 1e-6);
        assert!((m.0[2] - (0.072 + 0.928)).abs() < 1e-6);
    }

    #[test]
    fn brightness_is_a_linear_slope() {
        // slope = amount, intercept = 0.
        assert_eq!(rgb(ColorMatrix::brightness(0.5).apply(MID)), [64, 32, 96]);
        // Amounts over one are allowed and the result clamps at the top.
        assert_eq!(
            rgb(ColorMatrix::brightness(4.0).apply(MID)),
            [255, 255, 255]
        );
    }

    #[test]
    fn contrast_uses_the_specified_intercept() {
        // C' = a·C + (-(0.5a) + 0.5). At a = 0 every channel is 0.5.
        assert_eq!(rgb(ColorMatrix::contrast(0.0).apply(MID)), [128, 128, 128]);
        // a = 1 is the identity.
        assert_eq!(rgb(ColorMatrix::contrast(1.0).apply(MID)), rgb(MID));
    }

    #[test]
    fn invert_reads_the_two_entry_table() {
        // At amount = 1: C' = 1 - C.
        assert_eq!(rgb(ColorMatrix::invert(1.0).apply(MID)), [127, 191, 63]);
        // amount = 0.5 collapses every channel onto 0.5.
        assert_eq!(rgb(ColorMatrix::invert(0.5).apply(MID)), [128, 128, 128]);
        // Amounts above one clamp to one.
        assert_eq!(
            rgb(ColorMatrix::invert(2.0).apply(MID)),
            rgb(ColorMatrix::invert(1.0).apply(MID))
        );
    }

    #[test]
    fn saturate_one_and_grayscale_zero_are_the_identity() {
        assert_eq!(rgb(ColorMatrix::saturate(1.0).apply(MID)), rgb(MID));
        assert_eq!(rgb(ColorMatrix::grayscale(0.0).apply(MID)), rgb(MID));
        assert_eq!(rgb(ColorMatrix::sepia(0.0).apply(MID)), rgb(MID));
    }

    #[test]
    fn grayscale_one_uses_the_published_luma_coefficients() {
        // At amount = 1 every row is (0.2126, 0.7152, 0.0722).
        let out = ColorMatrix::grayscale(1.0).apply(MID);
        let luma = 0.2126 * (128.0 / 255.0) + 0.7152 * (64.0 / 255.0) + 0.0722 * (192.0 / 255.0);
        for i in 0..3 {
            assert!((out.components[i] - luma).abs() < 1e-5, "{out:?}");
        }
    }

    #[test]
    fn alpha_is_never_touched() {
        let translucent = Color::new([0.5, 0.25, 0.75, 0.4]);
        for m in [
            ColorMatrix::brightness(0.3),
            ColorMatrix::contrast(2.0),
            ColorMatrix::invert(1.0),
            ColorMatrix::hue_rotate(1.2),
            ColorMatrix::saturate(0.0),
            ColorMatrix::grayscale(1.0),
            ColorMatrix::sepia(1.0),
        ] {
            assert_eq!(m.apply(translucent).components[3], 0.4);
        }
    }

    /// Evaluate a stop list the way `vello_common::encode::encode_stops` does
    /// for an sRGB gradient: one linear ramp per adjacent pair, over
    /// premultiplied components.
    fn sample_ramp(stops: &[ColorStop], x: f32) -> [f32; 4] {
        let pair = stops
            .windows(2)
            .find(|w| x >= w[0].offset && x <= w[1].offset)
            .unwrap_or(&stops[stops.len() - 2..]);
        let (left, right) = (pair[0], pair[1]);
        let span = right.offset - left.offset;
        let s = if span > 0.0 {
            (x - left.offset) / span
        } else {
            0.0
        };
        let a = to_vector(left.color, true);
        let b = to_vector(right.color, true);
        std::array::from_fn(|i| a[i] + s * (b[i] - a[i]))
    }

    fn gradient_with(stops: &[(f32, Color)]) -> Gradient {
        let mut gradient = Gradient::new_linear((0.0, 0.0), (1.0, 0.0));
        gradient.stops = ColorStops(
            stops
                .iter()
                .map(|(offset, color)| ColorStop {
                    offset: *offset,
                    color: DynamicColor::from_alpha_color(*color),
                })
                .collect(),
        );
        gradient
    }

    /// The per-pixel answer: interpolate the SOURCE ramp, then filter.
    fn reference_at(chain: &ColorMatrixChain, source: &Gradient, x: f32) -> [f32; 4] {
        let v = sample_ramp(&source.stops.0, x);
        let a = v[3];
        let unpremul = if a > 0.0 {
            Color::new([v[0] / a, v[1] / a, v[2] / a, a])
        } else {
            Color::new([0.0, 0.0, 0.0, 0.0])
        };
        let filtered = chain.apply(unpremul);
        let fa = filtered.components[3];
        [
            filtered.components[0] * fa,
            filtered.components[1] * fa,
            filtered.components[2] * fa,
            fa,
        ]
    }

    #[track_caller]
    fn assert_ramp_matches_per_pixel(chain: &ColorMatrixChain, stops: &[(f32, Color)]) {
        let source = gradient_with(stops);
        let mut filtered = source.clone();
        chain.apply_to_gradient(&mut filtered);
        for i in 0..=400 {
            let x = i as f32 / 400.0;
            let got = sample_ramp(&filtered.stops.0, x);
            let want = reference_at(chain, &source, x);
            for c in 0..4 {
                assert!(
                    (got[c] - want[c]).abs() < 1.5 / 255.0,
                    "x={x} channel={c}: ramp {got:?} vs per-pixel {want:?}\nstops {:?}",
                    filtered.stops.0
                );
            }
        }
    }

    /// The clamp is per pixel, and clamping does not commute with
    /// interpolation. Filtering only the authored stops would leave the ramp
    /// ~29/255 away from the per-pixel answer around the crossing at
    /// x = 0.326923; subdividing there makes it exact.
    #[test]
    fn a_clamping_gradient_ramp_matches_the_per_pixel_filter() {
        let chain = ColorMatrixChain(SmallVec::from_slice(&[ColorMatrix::brightness(1.3)]));
        let stops = [
            (0.0, Color::new([0.9, 0.9, 0.9, 1.0])),
            (1.0, Color::new([0.5, 0.5, 0.5, 1.0])),
        ];
        assert_ramp_matches_per_pixel(&chain, &stops);

        // The subdivision is what makes it exact: a stop was inserted.
        let mut filtered = gradient_with(&stops);
        chain.apply_to_gradient(&mut filtered);
        assert!(
            filtered.stops.0.len() > 2,
            "expected a breakpoint stop, got {:?}",
            filtered.stops.0
        );
        // 1.3 * (0.9 - 0.4x) = 1  =>  x = 0.13077 / 0.4 = 0.326923
        let inserted = filtered.stops.0[1].offset;
        assert!(
            (inserted - 0.326_923).abs() < 1e-4,
            "breakpoint at {inserted}, expected 0.326923"
        );
    }

    #[test]
    fn gradient_ramps_stay_exact_for_chains_alpha_and_hard_stops() {
        // Two stages, both clamping, on a translucent ramp with a hard stop.
        let chain = ColorMatrixChain(SmallVec::from_slice(&[
            ColorMatrix::hue_rotate(1.9),
            ColorMatrix::contrast(1.8),
        ]));
        assert_ramp_matches_per_pixel(
            &chain,
            &[
                (0.0, Color::new([0.95, 0.10, 0.20, 1.0])),
                (0.4, Color::new([0.05, 0.85, 0.30, 0.25])),
                (0.4, Color::new([0.20, 0.20, 0.90, 0.75])),
                (1.0, Color::new([0.60, 0.05, 0.05, 0.40])),
            ],
        );
    }

    #[test]
    fn a_non_clamping_gradient_keeps_its_authored_stops() {
        // brightness(0.5) cannot leave [0,1] from inputs in [0,1], so there is
        // no breakpoint to insert and the ramp keeps its two stops.
        let chain = ColorMatrixChain(SmallVec::from_slice(&[ColorMatrix::brightness(0.5)]));
        let mut filtered = gradient_with(&[
            (0.0, Color::new([0.9, 0.2, 0.4, 1.0])),
            (1.0, Color::new([0.1, 0.8, 0.6, 1.0])),
        ]);
        chain.apply_to_gradient(&mut filtered);
        assert_eq!(filtered.stops.0.len(), 2);
    }

    #[test]
    fn the_chain_clamps_between_matrices() {
        // brightness(4) saturates every channel to 1 before contrast sees it,
        // so the pair is NOT the same as the composed matrix 4·0.5·C + 0.25.
        let chain = ColorMatrixChain(SmallVec::from_buf([
            ColorMatrix::brightness(4.0),
            ColorMatrix::contrast(0.5),
        ]));
        let dark = Color::from_rgb8(64, 64, 64);
        // Sequential: 4·0.251 = 1.0 (clamped), then 0.5·1 + 0.25 = 0.75.
        assert_eq!(rgb(chain.apply(dark)), [191, 191, 191]);
        // Composed without the clamp would be 2·0.251 + 0.25 = 0.752 -> 192.
        let composed = ColorMatrix::linear_transfer(2.0, 0.25);
        assert_eq!(rgb(composed.apply(dark)), [192, 192, 192]);
    }
}
