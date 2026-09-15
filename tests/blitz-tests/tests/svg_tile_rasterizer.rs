//! A rasterised repeating SVG tile must paint exactly what replaying the tile's vector scene
//! per tile paints -- and must be declined wherever it cannot.
//!
//! `paint_scene_with_tiles` swaps N vector replays for one fill of a repeating image brush.
//! That is only pixel-exact when the brush has nothing to resample, i.e. when the tile is a
//! whole number of *device* pixels. These tests pin both halves: the substitution happens
//! where it is exact, and does not happen where it would not be.
//!
//! The guard is the load-bearing part, so the cases here sit either side of the predicate
//! rather than comfortably inside it. The one that matters most is a tile that is integral in
//! CSS pixels but fractional on the device under a non-unit scale: it looks eligible and is
//! not.

use anyrender::{PaintScene, Scene, render_to_buffer};
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_dom::node::{ImageData, RasterImageData, SvgImageData};
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_paint::{
    SvgTileKey, SvgTileRasterizer, SvgTileRequest, paint_scene, paint_scene_with_tiles,
};
use blitz_traits::shell::{ColorScheme, Viewport};
use kurbo::Affine;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// The 4x4 desktop stipple: one black pixel at (0,0) and one at (2,2).
const DITHER: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><rect width="1" height="1" fill="black"/><rect x="2" y="2" width="1" height="1" fill="black"/></svg>"#;

/// A pixel-aligned, fully opaque multi-colour tile: eligible, but not as trivial as the
/// single-colour stipple.
const BARS: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="6" height="10"><rect width="3" height="10" fill="#204080"/><rect x="3" width="3" height="5" fill="#f0c040"/></svg>"##;

/// A tile with curved, off-grid, partially transparent geometry. Its rasterisation carries
/// partial alpha, so it must be refused however neatly it is sized and placed.
const BLOB: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><circle cx="3.3" cy="4.7" r="2.4" fill="#c03070"/><path d="M0 0 L8 3 L2 8 Z" fill="#20a0d0" opacity="0.6"/></svg>"##;

/// The embedder side of the seam: rasterise through the same renderer that draws the scene,
/// and cache across calls. Counts requests and rasterisations so a test can distinguish
/// "`blitz-paint` never asked" from "it asked and then rejected the tile".
#[derive(Default)]
struct CachingRasterizer {
    cache: RefCell<HashMap<SvgTileKey, RasterImageData>>,
    requests: RefCell<u32>,
    rasterizations: RefCell<u32>,
}

fn rasterize(request: &SvgTileRequest<'_>) -> RasterImageData {
    let scene = request.scene.clone();
    let pixels = render_to_buffer::<VelloCpuImageRenderer, _>(
        move |target| target.append_scene(scene, Affine::IDENTITY),
        request.width,
        request.height,
    );
    RasterImageData::new(request.width, request.height, Arc::new(pixels))
}

impl SvgTileRasterizer for CachingRasterizer {
    fn rasterize_svg_tile(&self, request: SvgTileRequest<'_>) -> Option<RasterImageData> {
        *self.requests.borrow_mut() += 1;
        let key = request.cache_key();
        if let Some(hit) = self.cache.borrow().get(&key) {
            return Some(hit.clone());
        }
        *self.rasterizations.borrow_mut() += 1;
        let image = rasterize(&request);
        self.cache.borrow_mut().insert(key, image.clone());
        Some(image)
    }
}

/// Refuses everything. Declining must always be safe.
struct RefusingRasterizer;
impl SvgTileRasterizer for RefusingRasterizer {
    fn rasterize_svg_tile(&self, _request: SvgTileRequest<'_>) -> Option<RasterImageData> {
        None
    }
}

/// Rasterises the real tile, then repaints every texel magenta **keeping its alpha**.
///
/// The frame therefore changes if and only if `blitz-paint` actually painted with the tile.
/// Preserving alpha is what makes it a faithful probe: a solid opaque tile would sail through
/// the binary-alpha check that the real antialiased tile fails, and would report a
/// substitution that does not happen in practice.
struct TintingProbe;
impl SvgTileRasterizer for TintingProbe {
    fn rasterize_svg_tile(&self, request: SvgTileRequest<'_>) -> Option<RasterImageData> {
        let mut data = rasterize(&request).data.as_ref().to_vec();
        for texel in data.as_chunks_mut::<4>().0 {
            [texel[0], texel[1], texel[2]] = [255, 0, 255];
        }
        Some(RasterImageData::new(
            request.width,
            request.height,
            Arc::new(data),
        ))
    }
}

fn document(svg_src: &str, style: &str, w: u32, h: u32, scale: f64) -> HtmlDocument {
    let body = format!(r#"<div class="bg" style="{style}"></div>"#);
    let html = format!(r#"<html><body style="margin:0; background:#ffffff;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(w, h, scale as f32, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    let svg =
        SvgImageData::from_data(svg_src.as_bytes(), &usvg::Options::default()).expect("valid SVG");
    let ids: Vec<_> = doc
        .query_selector_all(".bg")
        .expect("valid selector")
        .into_iter()
        .collect();
    assert!(!ids.is_empty(), "test body must contain a .bg element");
    for id in ids {
        let el = doc.get_node_mut(id).unwrap().element_data_mut().unwrap();
        for layer in el.background_images.iter_mut().flatten() {
            layer.status = blitz_dom::node::Status::Ok;
            layer.image = ImageData::Svg(svg.clone());
        }
    }
    doc.resolve(0.0);
    doc
}

const W: u32 = 200;
const H: u32 = 150;

fn render_plain(doc: &mut HtmlDocument, scale: f64) -> Vec<u8> {
    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene(scene, doc.as_mut(), scale, W, H, 0, 0),
        W,
        H,
    )
}

fn render_with(doc: &mut HtmlDocument, scale: f64, rasterizer: &dyn SvgTileRasterizer) -> Vec<u8> {
    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene_with_tiles(scene, doc.as_mut(), scale, W, H, 0, 0, rasterizer),
        W,
        H,
    )
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0.iter())
        .filter(|(p, q)| p != q)
        .count()
}

struct Case {
    name: &'static str,
    svg: &'static str,
    style: &'static str,
    scale: f64,
    /// Whether `blitz-paint`'s pre-rasterisation guard lets the tile be requested at all.
    requested: bool,
    /// Whether the substitution is actually used. Strictly weaker than `requested`: a tile
    /// carrying partial alpha is asked for and then rejected, because that is the one
    /// condition that cannot be evaluated until the tile exists.
    substituted: bool,
}

const BASE: &str = "width:200px; height:150px; background-image:url('https://x/y.svg');";

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "4x4 stipple, scale 1",
            svg: DITHER,
            style: "background-repeat:repeat;",
            scale: 1.0,
            requested: true,
            substituted: true,
        },
        Case {
            name: "opaque multi-colour tile, non-square",
            svg: BARS,
            style: "background-size:6px 10px; background-repeat:repeat;",
            scale: 1.0,
            requested: true,
            substituted: true,
        },
        Case {
            name: "integral CSS px, scale 2 (still integral on device)",
            svg: BARS,
            style: "background-size:3px 5px; background-repeat:repeat;",
            scale: 2.0,
            requested: true,
            substituted: true,
        },
        Case {
            name: "integral tile at a whole-pixel position",
            svg: DITHER,
            style: "background-repeat:repeat; background-position:2px 1px;",
            scale: 1.0,
            requested: true,
            substituted: true,
        },
        // Boundary: 5 CSS px at scale 1.5 is 7.5 device px. Looks eligible, is not.
        Case {
            name: "integral CSS px, scale 1.5 (fractional on device)",
            svg: BARS,
            style: "background-size:5px 5px; background-repeat:repeat;",
            scale: 1.5,
            requested: false,
            substituted: false,
        },
        // Boundary: sized and placed perfectly, but the tile antialiases, so rasterising it
        // quantises coverage to 8 bits twice where the replay quantises once. Asked for,
        // then rejected.
        Case {
            name: "antialiased tile, integral size",
            svg: BLOB,
            style: "background-size:8px 8px; background-repeat:repeat;",
            scale: 1.0,
            requested: true,
            substituted: false,
        },
        // Boundary: whole-pixel tile, fractional placement, so the brush would interpolate.
        Case {
            name: "integral tile at a fractional position",
            svg: DITHER,
            style: "background-repeat:repeat; background-position:1.5px 0.5px;",
            scale: 1.0,
            requested: false,
            substituted: false,
        },
        Case {
            name: "fractional CSS size",
            svg: BARS,
            style: "background-size:4.5px 4.5px; background-repeat:repeat;",
            scale: 1.0,
            requested: false,
            substituted: false,
        },
        Case {
            name: "background-repeat: space",
            svg: BARS,
            style: "background-size:6px 10px; background-repeat:space;",
            scale: 1.0,
            requested: false,
            substituted: false,
        },
        Case {
            name: "repeat on one axis only",
            svg: BARS,
            style: "background-size:6px 10px; background-repeat:repeat no-repeat;",
            scale: 1.0,
            requested: false,
            substituted: false,
        },
        Case {
            name: "no-repeat, a single tile",
            svg: BARS,
            style: "background-size:6px 10px; background-repeat:no-repeat;",
            scale: 1.0,
            requested: false,
            substituted: false,
        },
        Case {
            name: "one tile because it fills the element",
            svg: BARS,
            style: "background-size:200px 150px; background-repeat:repeat;",
            scale: 1.0,
            requested: false,
            substituted: false,
        },
        // A transform on the element puts a linear part in the placement, so the brush would
        // no longer blit even though the tile itself is perfectly sized.
        Case {
            name: "integral tile on a scaled element",
            svg: DITHER,
            style: "background-repeat:repeat; transform:scale(3);",
            scale: 1.0,
            requested: false,
            substituted: false,
        },
        Case {
            name: "integral tile on a rotated element",
            svg: DITHER,
            style: "background-repeat:repeat; transform:rotate(20deg);",
            scale: 1.0,
            requested: false,
            substituted: false,
        },
    ]
}

/// The substitution must never move a pixel, in any configuration -- the ones it takes and
/// the ones it declines alike.
#[test]
fn a_rasterized_tile_paints_what_the_vector_replay_paints() {
    for case in cases() {
        let style = format!("{BASE} {}", case.style);
        let plain = render_plain(
            &mut document(case.svg, &style, W, H, case.scale),
            case.scale,
        );

        let rasterizer = CachingRasterizer::default();
        let fast = render_with(
            &mut document(case.svg, &style, W, H, case.scale),
            case.scale,
            &rasterizer,
        );

        assert_eq!(
            *rasterizer.requests.borrow() > 0,
            case.requested,
            "{}: tile requested = {}, expected {}",
            case.name,
            *rasterizer.requests.borrow() > 0,
            case.requested
        );
        let diff = differing(&plain, &fast);
        assert_eq!(
            diff,
            0,
            "{}: rasterised tile differs from vector replay on {diff} of {} px",
            case.name,
            W * H
        );
    }
}

/// The byte-identity above would also hold if the substitution never happened, so this pins
/// which cases actually take it -- by answering with a tile that is deliberately the wrong
/// colour and checking whether the frame changes.
///
/// It is the guard against the whole suite passing vacuously.
#[test]
fn the_substitution_happens_exactly_where_it_is_claimed_to() {
    for case in cases() {
        let style = format!("{BASE} {}", case.style);
        let plain = render_plain(
            &mut document(case.svg, &style, W, H, case.scale),
            case.scale,
        );
        let probed = render_with(
            &mut document(case.svg, &style, W, H, case.scale),
            case.scale,
            &TintingProbe,
        );
        let changed = differing(&plain, &probed) > 0;
        assert_eq!(
            changed,
            case.substituted,
            "{}: a deliberately wrong tile {} the frame, so the substitution {} taken;              expected it {}",
            case.name,
            if changed { "changed" } else { "did not change" },
            if changed { "was" } else { "was not" },
            if case.substituted {
                "taken"
            } else {
                "not taken"
            }
        );
    }
}

/// A rasterizer that declines everything must leave every pixel exactly as it was, which is
/// what makes `None` a safe answer for an implementor.
#[test]
fn declining_every_tile_changes_nothing() {
    for case in cases() {
        let style = format!("{BASE} {}", case.style);
        let plain = render_plain(
            &mut document(case.svg, &style, W, H, case.scale),
            case.scale,
        );
        let refused = render_with(
            &mut document(case.svg, &style, W, H, case.scale),
            case.scale,
            &RefusingRasterizer,
        );
        assert_eq!(
            differing(&plain, &refused),
            0,
            "{}: declining moved a pixel",
            case.name
        );
    }
}

/// The cache has to be reachable across frames: the same document repainted must rasterise
/// once. A wrong key would still render correctly, just slowly -- which nothing else here
/// would catch.
#[test]
fn the_same_tile_is_rasterized_once_across_frames() {
    let style = format!("{BASE} background-repeat:repeat;");
    let mut doc = document(DITHER, &style, W, H, 1.0);
    let rasterizer = CachingRasterizer::default();
    for _ in 0..4 {
        let _ = render_with(&mut doc, 1.0, &rasterizer);
    }
    assert_eq!(*rasterizer.requests.borrow(), 4, "one request per frame");
    assert_eq!(
        *rasterizer.rasterizations.borrow(),
        1,
        "the tile must be rasterised once and served from cache thereafter"
    );
}

/// `Scene` is re-exported through `anyrender`; this keeps the import honest if that moves.
#[allow(dead_code)]
fn _scene_type_is_reachable(_: &Scene) {}
