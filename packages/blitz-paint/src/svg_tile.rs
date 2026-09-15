//! The seam through which an embedder lends `blitz-paint` a rasteriser for repeating SVG
//! background tiles.
//!
//! # Why this exists
//!
//! A repeating SVG `background-image` is painted by replaying the tile's vector scene once
//! per tile, because -- unlike a raster image, whose brush tiles itself through
//! [`peniko::Extend::Repeat`] in a single fill -- an SVG has no brush to lean on. That cost
//! scales with the painted area: a full-screen 4x4 stipple at 1280x720 is 57,600 tiles, and
//! measured **50.9 ms** per frame in release against a 16.7 ms 60 fps budget.
//!
//! Measurement rules out every cheaper fix (see `tests/blitz-tests/tests/svg_tiling_cost.rs`).
//! Recording the tile scene once and replaying the commands removes the usvg tree walk and
//! buys 28%. Collapsing all 115,600 draw calls into two by batching the geometry into one
//! fill per command buys **nothing** -- the cost is the *volume of vector geometry* handed to
//! the rasteriser, not the walk and not the call count. The same stipple as a raster tile,
//! through the single-fill repeating brush, costs **2.0 ms**. So the only route inside the
//! budget is to stop submitting per-tile geometry and rasterise the tile once.
//!
//! # Why the embedder has to supply it
//!
//! `blitz-paint` is backend-agnostic: it depends on [`anyrender`]'s traits and holds no
//! renderer. [`anyrender::ImageRenderer`] is a real offscreen rasterisation abstraction, but
//! only a backend crate implements it. Routing pixels through
//! [`anyrender::RenderContext::try_register_custom_resource`] and `Paint::Resource` is not an
//! alternative: the CPU backend maps `Paint::Resource` to transparent, so such a tile paints
//! nothing.
//!
//! The rasteriser must be **the same one that draws the scene**. Antialiasing has to match
//! what the backend would have produced drawing that geometry in place; a bolt-on rasteriser
//! such as `resvg`/`tiny-skia` makes even a pixel-aligned tile a coin flip.
//!
//! # Exactness
//!
//! Substituting a rasterised tile for a vector replay is only pixel-exact when no resampling
//! occurs, which is when the tile is **an integral number of device pixels**. `blitz-paint`
//! enforces that itself ([`is_exact_tile_size`]); an implementor does not have to. Measured
//! either side of the predicate: 0 differing pixels for a 4x4 and a 2x2 tile and under a
//! fractional `background-position`, against 46.96% of the surface differing at
//! `background-size: 4.5px`.

use std::sync::Arc;

use anyrender::Scene;
use blitz_dom::node::RasterImageData;

/// One tile `blitz-paint` would like rasterised.
pub struct SvgTileRequest<'a> {
    /// The tile's source tree.
    ///
    /// This is the cache key. Two requests naming the same `Arc` at the same size are the
    /// same tile, and a cache that holds an `Arc` clone keeps the address unique for as long
    /// as the entry lives -- so `Arc::as_ptr(tree) as usize` is a sound key *provided* the
    /// entry owns a clone.
    pub tree: &'a Arc<usvg::Tree>,
    /// The tile's drawing commands, already scaled so that the tree fills
    /// `width` x `height` exactly.
    ///
    /// Rasterise this rather than the tree: it is what `blitz-paint` would have drawn, at the
    /// size it would have drawn it, so an implementor cannot get the scaling subtly different
    /// from the vector path it is replacing. Replay it with
    /// [`anyrender::PaintScene::append_scene`] at [`kurbo::Affine::IDENTITY`].
    pub scene: &'a Scene,
    /// Tile width in whole device pixels.
    pub width: u32,
    /// Tile height in whole device pixels.
    pub height: u32,
}

/// Rasterises a repeating SVG background tile on `blitz-paint`'s behalf.
///
/// Implement this over the same renderer that consumes the scene -- with
/// [`anyrender::ImageRenderer`], typically `render_to_vec` into an RGBA8 buffer.
///
/// Implementations are expected to **cache**: the same tile is requested for every frame it
/// is visible in, and `blitz-paint` keeps no state between frames. Keying on the tile's
/// source identity together with `width` and `height` is sufficient, and
/// [`RasterImageData`] is cheap to clone (its pixels sit behind an `Arc`).
pub trait SvgTileRasterizer {
    /// Rasterise a tile into a `request.width` x `request.height` RGBA8 buffer, in the same
    /// layout [`blitz_dom::node::RasterImageData`] carries elsewhere.
    ///
    /// Return `None` to decline, for any reason -- a tile too large to be worth caching, a
    /// renderer unavailable on this thread, anything. Declining is always safe: the caller
    /// falls back to replaying the vector per tile, which is what it did before this trait
    /// existed, so an implementation that always returns `None` is a correct one.
    fn rasterize_svg_tile(&self, request: SvgTileRequest<'_>) -> Option<RasterImageData>;
}

/// Whether a resolved tile size in device pixels can be served by a rasterised tile without
/// resampling.
///
/// A repeating image brush samples one rasterisation of the tile at every repeat, so it can
/// only reproduce a per-tile vector replay when the sampling is an identity -- when the tile
/// occupies a whole number of device pixels. A fractional size makes the brush interpolate a
/// texture that the replay would have drawn crisply at each tile: measured 46.96% of the
/// surface differing at `background-size: 4.5px`.
///
/// This is deliberately expressed in **device** pixels, not CSS pixels: an integral CSS size
/// under a non-unit `scale` is fractional on the device, and that is the case that looks safe
/// and is not.
pub(crate) fn is_exact_tile_size(width: f64, height: f64) -> bool {
    integral(width) && integral(height)
}

fn integral(v: f64) -> bool {
    v.is_finite() && v > 0.0 && v.fract() == 0.0
}

/// Whether the lattice is placed such that the tile lands on the device pixel grid.
///
/// The brush repeats in the fill's own coordinate space, so unless that space maps to the
/// surface by a whole-pixel translation the texels do not line up with pixels and the sampler
/// interpolates. Any rotation, skew or scale on the element disqualifies it for the same
/// reason, so the linear part has to be the identity exactly.
pub(crate) fn is_pixel_aligned_placement(placed: kurbo::Affine) -> bool {
    let [a, b, c, d, tx, ty] = placed.as_coeffs();
    a == 1.0 && b == 0.0 && c == 0.0 && d == 1.0 && integral_or_zero(tx) && integral_or_zero(ty)
}

fn integral_or_zero(v: f64) -> bool {
    v.is_finite() && v.fract() == 0.0
}

/// Whether every texel of the rasterised tile is fully opaque or fully transparent.
///
/// This is the condition that is easy to miss, and it cost a round here. Even a
/// pixel-aligned, unscaled tile is **not** exact if it carries partial alpha, because
/// rasterising into the tile buffer quantises coverage to 8 bits and compositing the tile
/// then quantises again, where the vector replay composites once. A half-covered black edge
/// over white resolves to 128 drawn directly and 127 through an 8-bit tile. Measured on a
/// tile with an antialiased circle and a `0.6`-opacity triangle: **17,812 of 30,000 pixels**
/// differ, at an integral tile size.
///
/// With binary alpha there is no intermediate blend to quantise: each destination pixel
/// either takes the texel's colour verbatim or is left alone, which is exactly what the
/// replay does with full or zero coverage. It also makes the straight-vs-premultiplied
/// question moot, since the two agree at alpha 0 and 255.
///
/// The scan is O(texels) once per cache miss, against a rasterisation of the same tile.
pub(crate) fn has_binary_alpha(image: &RasterImageData) -> bool {
    image
        .data
        .as_ref()
        .as_chunks::<4>()
        .0
        .iter()
        .all(|texel| texel[3] == 0 || texel[3] == 255)
}

#[cfg(test)]
mod tests {
    use super::{has_binary_alpha, is_exact_tile_size, is_pixel_aligned_placement};
    use blitz_dom::node::RasterImageData;
    use kurbo::Affine;
    use std::sync::Arc;

    fn image(alphas: &[u8]) -> RasterImageData {
        let data: Vec<u8> = alphas.iter().flat_map(|a| [0u8, 0, 0, *a]).collect();
        RasterImageData::new(alphas.len() as u32, 1, Arc::new(data))
    }

    #[test]
    fn a_tile_is_blittable_only_with_binary_alpha() {
        assert!(has_binary_alpha(&image(&[0, 255, 255, 0])));
        assert!(has_binary_alpha(&image(&[255])));
        // One antialiased texel is enough to make the substitution inexact.
        assert!(!has_binary_alpha(&image(&[0, 255, 128, 0])));
        assert!(!has_binary_alpha(&image(&[1])));
        assert!(!has_binary_alpha(&image(&[254])));
    }

    #[test]
    fn only_whole_pixel_translations_are_aligned() {
        assert!(is_pixel_aligned_placement(Affine::translate((12.0, -4.0))));
        assert!(is_pixel_aligned_placement(Affine::IDENTITY));
        assert!(!is_pixel_aligned_placement(Affine::translate((12.5, 0.0))));
        assert!(!is_pixel_aligned_placement(Affine::translate((0.0, 0.25))));
        // Any linear part at all: the brush would no longer blit.
        assert!(!is_pixel_aligned_placement(Affine::scale(2.0)));
        assert!(!is_pixel_aligned_placement(Affine::rotate(0.5)));
        assert!(!is_pixel_aligned_placement(Affine::scale_non_uniform(
            1.0, 1.001
        )));
    }

    #[test]
    fn only_whole_device_pixel_tiles_are_exact() {
        assert!(is_exact_tile_size(4.0, 4.0));
        assert!(is_exact_tile_size(1.0, 64.0));
        // An integral CSS size under a non-unit scale need not be integral on the device:
        // 5 CSS px at scale 1.5 is 7.5. (4 px at 1.5 is 6.0, and stays eligible -- the
        // predicate is about the device size, not about the scale being whole.)
        assert!(!is_exact_tile_size(5.0 * 1.5, 5.0 * 1.5));
        assert!(is_exact_tile_size(4.0 * 1.5, 4.0 * 1.5));
        assert!(!is_exact_tile_size(4.5, 4.0));
        assert!(!is_exact_tile_size(4.0, 4.5));
        // Degenerate sizes are never served from a raster tile.
        assert!(!is_exact_tile_size(0.0, 4.0));
        assert!(!is_exact_tile_size(-4.0, 4.0));
        assert!(!is_exact_tile_size(f64::NAN, 4.0));
        assert!(!is_exact_tile_size(f64::INFINITY, 4.0));
    }
}
