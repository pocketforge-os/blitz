//! A repeating SVG `background-image` is tiled, like every other image kind.
//!
//! `draw_svg_image_layer` used to resolve one position and one size and replay the tree
//! exactly once, with no reference to `background-repeat` anywhere in it, so a small tile
//! painted as a single patch in the corner of its box and the rest of the element was left
//! bare. A 4x4 dither used as a page-wide stipple therefore disappeared entirely.
//!
//! An SVG layer is replayed into the scene rather than filled through a brush, so -- unlike
//! a raster image, whose `Extend::Repeat` tiles in the shader in one fill -- it needs
//! explicit tiles, and those tiles need bounding. These tests pin the tiling, the bound, and
//! the two things the bound is easy to get wrong: the paint offset, and a transformed
//! element, where the lattice steps in the element's own space rather than the surface's.

use anyrender::render_to_buffer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_dom::node::{ImageData, SvgImageData};
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;

/// The 4x4 dither tile this bug was found with: one black pixel at (0,0) and one at (2,2),
/// the other fourteen transparent.
const DITHER: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><rect width="1" height="1" fill="black"/><rect x="2" y="2" width="1" height="1" fill="black"/></svg>"#;

/// A 50x50 solid green SVG with a `viewBox` and no intrinsic size -- the image WPT's
/// `background-size-near-zero-svg.html` uses.
const GREEN_50: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50"><rect fill="green" width="50" height="50"/></svg>"#;

/// Renders `body` with every background layer's image replaced by `svg_src`, onto a
/// `canvas_w` x `canvas_h` buffer painted at the given document offset.
///
/// The image is injected rather than fetched so the test needs no network provider; the
/// layer's `background-repeat`/`-size`/`-position` still come from the stylesheet.
#[allow(clippy::too_many_arguments)]
fn render(
    body: &str,
    svg_src: &str,
    doc_w: u32,
    doc_h: u32,
    canvas_w: u32,
    canvas_h: u32,
    x_offset: u32,
    y_offset: u32,
) -> Vec<u8> {
    let html = format!(r#"<html><body style="margin:0; background:#ffffff;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(doc_w, doc_h, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);

    let svg = SvgImageData::from_data(svg_src.as_bytes(), &usvg::Options::default())
        .expect("valid test SVG");
    let ids: Vec<_> = doc
        .query_selector_all(".bg")
        .expect("valid selector")
        .into_iter()
        .collect();
    assert!(!ids.is_empty(), "test body must contain a .bg element");
    for id in ids {
        let node = doc.get_node_mut(id).unwrap();
        let el = node.element_data_mut().unwrap();
        for layer in el.background_images.iter_mut().flatten() {
            layer.status = blitz_dom::node::Status::Ok;
            layer.image = ImageData::Svg(svg.clone());
        }
    }
    doc.resolve(0.0);

    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene(scene, doc.as_mut(), 1.0, doc_w, doc_h, x_offset, y_offset),
        canvas_w,
        canvas_h,
    )
}

fn pixel(buf: &[u8], canvas_w: u32, x: u32, y: u32) -> [u8; 3] {
    let idx = ((y * canvas_w + x) * 4) as usize;
    [buf[idx], buf[idx + 1], buf[idx + 2]]
}

const BLACK: [u8; 3] = [0, 0, 0];
const WHITE: [u8; 3] = [255, 255, 255];

/// The defect itself: the whole element carries the dither, not just its top-left 4x4.
///
/// Asserted as the lattice the pattern defines -- every pixel whose coordinates are
/// (0,0) or (2,2) mod 4 is black and every other pixel is white -- so a tile drawn at the
/// wrong phase fails just as loudly as a tile that is missing.
#[test]
fn a_repeating_svg_tile_covers_the_whole_element() {
    const SIZE: u32 = 64;
    let buf = render(
        r#"<div class="bg" style="width:64px; height:64px;
             background-image:url('https://example.com/x.svg'); background-repeat:repeat;"></div>"#,
        DITHER,
        SIZE,
        SIZE,
        SIZE,
        SIZE,
        0,
        0,
    );

    let mut wrong = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            let on_dot = (x % 4, y % 4) == (0, 0) || (x % 4, y % 4) == (2, 2);
            let expected = if on_dot { BLACK } else { WHITE };
            if pixel(&buf, SIZE, x, y) != expected {
                wrong.push((x, y));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} pixels do not match the 4x4 dither lattice; first ten: {:?}",
        wrong.len(),
        SIZE * SIZE,
        &wrong[..wrong.len().min(10)]
    );
}

/// `background-repeat: no-repeat` still draws exactly one tile at the background position.
#[test]
fn no_repeat_still_draws_a_single_tile() {
    const SIZE: u32 = 32;
    let buf = render(
        r#"<div class="bg" style="width:32px; height:32px;
             background-image:url('https://example.com/x.svg'); background-repeat:no-repeat;"></div>"#,
        DITHER,
        SIZE,
        SIZE,
        SIZE,
        SIZE,
        0,
        0,
    );

    assert_eq!(pixel(&buf, SIZE, 0, 0), BLACK, "the single tile must paint");
    assert_eq!(pixel(&buf, SIZE, 2, 2), BLACK, "the single tile must paint");
    for (x, y) in [(4, 4), (6, 6), (8, 0), (0, 8), (30, 30)] {
        assert_eq!(
            pixel(&buf, SIZE, x, y),
            WHITE,
            "no-repeat must not tile: ({x},{y}) is outside the single 4x4 tile"
        );
    }
}

/// `blitz-shell` passes `insets.left`/`insets.top` as the paint offset for safe-area insets,
/// and every sub-document render offsets by the iframe's absolute page position. The surface
/// the tiles are bounded against moves with that offset; bounding against a zero-based rect
/// instead blanks the far edge of the element.
///
/// The WPT runner hardcodes both offsets to 0, so this path has no upstream coverage.
#[test]
fn tiles_reach_the_far_edge_under_a_nonzero_paint_offset() {
    const DOC_W: u32 = 100;
    const DOC_H: u32 = 40;
    const X_OFFSET: u32 = 40;
    const Y_OFFSET: u32 = 12;
    const CANVAS_W: u32 = DOC_W + X_OFFSET;
    const CANVAS_H: u32 = DOC_H + Y_OFFSET;

    let buf = render(
        r#"<div class="bg" style="width:100px; height:40px;
             background-image:url('https://example.com/x.svg'); background-repeat:repeat;"></div>"#,
        DITHER,
        DOC_W,
        DOC_H,
        CANVAS_W,
        CANVAS_H,
        X_OFFSET,
        Y_OFFSET,
    );

    // The element occupies screen x in [40,140) and y in [12,52). Its last tile column and
    // row must still carry dots.
    let missing: Vec<(u32, u32)> = (X_OFFSET..CANVAS_W)
        .flat_map(|x| (Y_OFFSET..CANVAS_H).map(move |y| (x, y)))
        .filter(|(x, y)| {
            let local = (x - X_OFFSET, y - Y_OFFSET);
            let on_dot =
                (local.0 % 4, local.1 % 4) == (0, 0) || (local.0 % 4, local.1 % 4) == (2, 2);
            on_dot && pixel(&buf, CANVAS_W, *x, *y) != BLACK
        })
        .collect();
    assert!(
        missing.is_empty(),
        "{} dots missing under a ({X_OFFSET},{Y_OFFSET}) paint offset; first ten: {:?}",
        missing.len(),
        &missing[..missing.len().min(10)]
    );
}

/// Which tiles land on a given pixel must not depend on how big the surface is.
///
/// This is the invariant the bound has to preserve, and the one that catches a bound
/// computed in the wrong space. The SVG lattice is placed *before* the element's transform,
/// so it scales and rotates with the element; the bound is therefore the surface pulled back
/// through that transform. Rendering the same document onto a larger surface and comparing
/// the shared region catches a mismatch between the two with no golden image and no toggle.
#[test]
fn culling_does_not_depend_on_the_surface_size() {
    const SMALL: u32 = 200;
    const LARGE: u32 = 600;

    for transform in [
        "",
        "transform:rotate(30deg);",
        "transform:rotate(30deg) scale(1.5);",
        "transform:scale(3);",
        "transform:skew(20deg, 10deg);",
        "transform:translate(-100px, -100px) rotate(30deg);",
        // Degenerate and extreme transforms: pulling the surface back through a singular or
        // overflowing matrix must disable the bound, not silently paint nothing on one
        // surface size and something on the other.
        "transform:scale(0);",
        "transform:matrix(1e300, 0, 0, 1e300, 0, 0);",
        "transform:scale(1e-300);",
    ] {
        let body = format!(
            r#"<div class="bg" style="position:absolute; left:25px; top:25px;
                 width:150px; height:150px; background-image:url('https://example.com/x.svg');
                 background-repeat:repeat; background-size:15px 15px; {transform}"></div>"#
        );
        let small = render(&body, DITHER, SMALL, SMALL, SMALL, SMALL, 0, 0);
        let large = render(&body, DITHER, LARGE, LARGE, LARGE, LARGE, 0, 0);

        let mismatched = (0..SMALL)
            .flat_map(|y| (0..SMALL).map(move |x| (x, y)))
            .filter(|(x, y)| {
                let s = ((y * SMALL + x) * 4) as usize;
                let l = ((y * LARGE + x) * 4) as usize;
                small[s..s + 3] != large[l..l + 3]
            })
            .count();
        assert_eq!(
            mismatched, 0,
            "transform:{transform:?} renders differently on a {SMALL}x{SMALL} surface than in \
             the same region of a {LARGE}x{LARGE} one ({mismatched} px differ); tiles are being \
             dropped that the lattice still places on the surface"
        );
    }
}

/// A `background-size` far below one device pixel resolves none of the image's own detail,
/// so the tile is widened to a device pixel rather than replayed an unbounded number of
/// times -- and the image has to be scaled to the widened tile, or the lattice leaves gaps.
///
/// This is WPT `css/css-backgrounds/background-size/background-size-near-zero-svg.html`,
/// whose reference is a solid 100x100 green square.
#[test]
fn a_near_zero_background_size_fills_the_element() {
    const SIZE: u32 = 100;
    let buf = render(
        r#"<div class="bg" style="width:100px; height:100px;
             background-image:url('https://example.com/x.svg'); background-size:0.2px 0.2px;"></div>"#,
        GREEN_50,
        SIZE,
        SIZE,
        SIZE,
        SIZE,
        0,
        0,
    );

    let unfilled: Vec<(u32, u32)> = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .filter(|(x, y)| {
            let [r, g, b] = pixel(&buf, SIZE, *x, *y);
            // usvg resolves `green` to #008000; accept any pixel that is predominantly it.
            !(r < 40 && (100..160).contains(&g) && b < 40)
        })
        .collect();
    assert!(
        unfilled.is_empty(),
        "{} of {} pixels are not green; first ten: {:?}",
        unfilled.len(),
        SIZE * SIZE,
        &unfilled[..unfilled.len().min(10)]
    );
}

/// A repeating tile on an element far taller than the surface must still paint the visible
/// rows, and must not cost time proportional to the element's height.
#[test]
fn a_tall_element_paints_its_visible_rows() {
    const W: u32 = 120;
    const H: u32 = 80;

    let buf = render(
        r#"<div class="bg" style="width:120px; height:20000px;
             background-image:url('https://example.com/x.svg'); background-repeat:repeat;"></div>"#,
        DITHER,
        W,
        H,
        W,
        H,
        0,
        0,
    );

    // Column 60 is 0 mod 4, so every row 0 mod 4 carries a dot there.
    let unpainted: Vec<u32> = (0..H)
        .filter(|y| y % 4 == 0 && pixel(&buf, W, 60, *y) != BLACK)
        .collect();
    assert!(
        unpainted.is_empty(),
        "tall element left screen rows {unpainted:?} without their dither dot"
    );
}
