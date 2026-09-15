//! A repeated background is tiled in the element's own coordinate space, and the CSS
//! transform then applies to the painted result (css-backgrounds-3 s3.1: the image is
//! painted into the element's boxes; css-transforms-1 s3: the transform maps that paint).
//!
//! The lattice therefore has to step along the element's own axes. Placing tile `i` with
//! `then_translate` instead adds the step to the fill transform's *output* translation, so
//! the lattice advances `tile_len` surface pixels per tile while each tile is
//! `scale * tile_len` surface pixels wide: the pattern spans the element's *unscaled*
//! extent and the background stops short.
//!
//! These tests pin the coverage, using a single-tile background of the same element as the
//! control -- both must cover exactly the element's clip box, whatever the transform.

use anyrender::render_to_buffer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_dom::node::{ImageData, RasterImageData, SvgImageData};
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;

const SURFACE: u32 = 200;

/// A saturated page background no tile colour here collides with, so "painted" is exactly
/// "not this colour".
const PAGE_BG: [u8; 3] = [0x00, 0xff, 0x00];

/// A solid blue 15x15 SVG tile, so its coverage is unambiguous.
const BLUE_TILE: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15"><rect width="15" height="15" fill="blue"/></svg>"#;

/// Renders `body` onto a `SURFACE`x`SURFACE` buffer. When `svg_src` is given, every
/// background layer of the `.bg` element is replaced by that SVG, the way
/// `svg_background_tiling.rs` does -- injected rather than fetched, so the test needs no
/// network provider while the layer's `background-repeat`/`-size` still come from the
/// stylesheet.
fn render(body: &str, svg_src: Option<&str>) -> Vec<u8> {
    let html = format!(r#"<html><body style="margin:0; background:#00ff00;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(SURFACE, SURFACE, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);

    if let Some(svg_src) = svg_src {
        let svg = SvgImageData::from_data(svg_src.as_bytes(), &usvg::Options::default())
            .expect("valid test SVG");
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
    }

    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene(scene, doc.as_mut(), 1.0, SURFACE, SURFACE, 0, 0),
        SURFACE,
        SURFACE,
    )
}

/// The same, with a solid blue 15x15 *raster* image injected instead. `Blob`-backed RGBA8 is
/// what a decoded `url()` image reaches the painter as.
fn render_raster(body: &str) -> Vec<u8> {
    let html = format!(r#"<html><body style="margin:0; background:#00ff00;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(SURFACE, SURFACE, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);

    let pixels: Vec<u8> = [0u8, 0, 255, 255].repeat(15 * 15);
    let raster = RasterImageData::new(15, 15, Arc::new(pixels));
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
            layer.image = ImageData::Raster(raster.clone());
        }
    }
    doc.resolve(0.0);

    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene(scene, doc.as_mut(), 1.0, SURFACE, SURFACE, 0, 0),
        SURFACE,
        SURFACE,
    )
}

fn painted_pixels(buf: &[u8]) -> usize {
    buf.as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0..3] != PAGE_BG)
        .count()
}

/// The inclusive span of surface columns carrying any painted pixel, ignoring columns whose
/// only coverage is a single antialiased edge sample.
fn painted_x_span(buf: &[u8]) -> Option<(u32, u32)> {
    let painted: Vec<u32> = (0..SURFACE)
        .filter(|x| {
            (0..SURFACE)
                .filter(|y| {
                    let i = ((y * SURFACE + x) * 4) as usize;
                    buf[i..i + 3] != PAGE_BG
                })
                .count()
                > 1
        })
        .collect();
    Some((*painted.first()?, *painted.last()?))
}

const WHOLE_SURFACE: usize = (SURFACE * SURFACE) as usize;

/// The element of the bug report, parameterised on the image, the `background-size`
/// declaration and the transform. A 150x150 box at (50,50) under `scale(3)` spans page
/// [-100, 350] on both axes, so it covers the whole 200x200 surface.
fn div(image: &str, size_decl: &str, repeat: &str, transform: &str) -> String {
    format!(
        r#"<div class="bg" style="position:absolute; left:50px; top:50px;
             width:150px; height:150px; background-image: {image};
             {size_decl} background-repeat: {repeat}; {transform}"></div>"#
    )
}

const GRADIENT: &str = "linear-gradient(red, blue)";
const SVG_URL: &str = "url('https://example.com/tile.svg')";
const RASTER_URL: &str = "url('https://example.com/tile.png')";

/// The exact configuration from the bug report: a 10x10 gradient tile lattice has to cover
/// the scaled element just as a single tile does.
#[test]
fn a_multi_tile_gradient_covers_a_scaled_element() {
    let single = painted_pixels(&render(
        &div(GRADIENT, "", "repeat", "transform: scale(3);"),
        None,
    ));
    assert_eq!(
        single, WHOLE_SURFACE,
        "control: a single-tile background must already cover the scaled element"
    );

    let multi = painted_pixels(&render(
        &div(
            GRADIENT,
            "background-size: 15px 15px;",
            "repeat",
            "transform: scale(3);",
        ),
        None,
    ));
    assert_eq!(
        multi, WHOLE_SURFACE,
        "a 10x10 tile lattice covered {multi} px of the scaled element where a single tile \
         covers {single}; the lattice is stepping along the surface axes instead of the \
         element's own"
    );
}

/// The same for an SVG `background-image`, which reaches the surface through a different
/// painter (`draw_svg_image_layer`) but the same tiling arithmetic.
#[test]
fn a_multi_tile_svg_covers_a_scaled_element() {
    let painted = painted_pixels(&render(
        &div(
            SVG_URL,
            "background-size: 15px 15px;",
            "repeat",
            "transform: scale(3);",
        ),
        Some(BLUE_TILE),
    ));
    assert_eq!(
        painted, WHOLE_SURFACE,
        "an SVG tile lattice covered {painted} px of the scaled element, not the whole surface"
    );
}

/// A raster `background-repeat: space` lattice, which is the raster painter's one looping
/// path (`Repeat`/`Round` become a single fill that the brush's own `Extend::Repeat` tiles
/// in the shader), stepped by the same element-space stride.
#[test]
fn a_spaced_raster_lattice_covers_a_scaled_element() {
    let painted = painted_pixels(&render_raster(&div(
        RASTER_URL,
        "background-size: 15px 15px;",
        "space",
        "transform: scale(3);",
    )));
    assert_eq!(
        painted, WHOLE_SURFACE,
        "a spaced raster lattice covered {painted} px of the scaled element"
    );
}

/// The raster painter's single-fill `Repeat` has no lattice to get wrong, but its *placement*
/// is the same element-space translation, and `background-origin` makes that translation
/// non-zero: the layer starts at the content box, not at the element's own origin.
///
/// A 60x60 border box with 10px padding puts the content box at element-local [10,50]. Under
/// `scale(3)` about the box centre (`p -> 3p - 60`) and a 50px page offset the layer must
/// land on surface [20,140]; adding the 10px content-box offset to the transform's *output*
/// instead lands it on [0,120] -- the same size, shifted by 20px and painting outside the
/// content box at one end while leaving it bare at the other.
#[test]
fn a_repeating_raster_follows_its_origin_box_under_a_scale() {
    let body = format!(
        r#"<div class="bg" style="position:absolute; left:50px; top:50px; box-sizing:border-box;
             width:60px; height:60px; padding:10px; background-image: {RASTER_URL};
             background-origin: content-box; background-size: 10px 10px;
             background-repeat: repeat; transform: scale(3);"></div>"#
    );
    let buf = render_raster(&body);
    let (lo, hi) = painted_x_span(&buf).expect("the layer must paint something");
    assert_eq!(
        (lo, hi),
        (20, 139),
        "the repeating raster layer painted surface x [{lo},{hi}], not the content box's          [20,139]; its `background-origin` offset is being applied in surface space"
    );
}

/// Under any transform, a repeated background covers the same area as a single-tile one --
/// the element's clip box -- because tiling happens before the transform applies.
///
/// The tolerance absorbs edge antialiasing only: each tile is filled separately, so the
/// outer boundary of a rotated lattice is composed of per-tile coverage values rather than
/// one, and a tile edge that lands mid-pixel can tip that pixel either way.
#[test]
fn a_tile_lattice_covers_what_a_single_tile_covers() {
    for transform in [
        "",
        "transform: scale(3);",
        "transform: scale(0.5);",
        "transform: rotate(30deg);",
        "transform: rotate(30deg) scale(1.5);",
        "transform: skew(20deg, 10deg);",
        "transform: translate(-20px, 30px) rotate(-15deg) scale(2);",
    ] {
        let single = painted_pixels(&render(&div(GRADIENT, "", "repeat", transform), None));
        let multi = painted_pixels(&render(
            &div(GRADIENT, "background-size: 15px 15px;", "repeat", transform),
            None,
        ));
        let tolerance = (single / 100).max(4);
        assert!(
            multi.abs_diff(single) <= tolerance,
            "{transform:?}: a 15px tile lattice painted {multi} px where a single tile paints \
             {single} px (tolerance {tolerance}); tiling is not following the element's own axes"
        );
    }
}

/// `background-repeat: space` computes its own stride and is placed by the same code, so it
/// has to follow the element's axes too. 150 % 15 == 0 leaves no gap to distribute, so the
/// ten tiles abut and cover the whole clip box.
#[test]
fn a_spaced_lattice_covers_a_scaled_element() {
    let painted = painted_pixels(&render(
        &div(
            GRADIENT,
            "background-size: 15px 15px;",
            "space",
            "transform: scale(3);",
        ),
        None,
    ));
    assert_eq!(
        painted, WHOLE_SURFACE,
        "`background-repeat: space` covered {painted} px of the scaled element"
    );
}
