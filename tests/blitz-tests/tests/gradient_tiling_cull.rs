//! A repeated gradient background is tiled with one fill per tile, and those tiles are
//! culled to the render surface. The cull must be computed in the space the tiles are
//! actually placed in -- the element's own, since the lattice is applied before the
//! element's transform -- and the surface it is bounded against sits at the
//! `initial_x`/`initial_y` offset `paint_scene` was given.
//!
//! Both of those were got wrong once. These tests pin them.

use anyrender::render_to_buffer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;

/// A plain two-stop gradient, so nothing here depends on repeating-gradient colour-stop
/// handling -- only on which tiles get drawn.
const GRADIENT: &str =
    "background-image: linear-gradient(to right, #ff0000 0 50%, #0000ff 50% 100%);";

fn render(
    body: &str,
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
    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene(scene, doc.as_mut(), 1.0, doc_w, doc_h, x_offset, y_offset),
        canvas_w,
        canvas_h,
    )
}

fn is_white(buf: &[u8], canvas_w: u32, x: u32, y: u32) -> bool {
    let idx = ((y * canvas_w + x) * 4) as usize;
    buf[idx] == 255 && buf[idx + 1] == 255 && buf[idx + 2] == 255
}

/// `blitz-shell` passes `insets.left`/`insets.top` as the paint offset for safe-area
/// insets, and every sub-document render offsets by the iframe's absolute page position.
/// The surface the tiles are culled against has to move with that offset; bounding against
/// a zero-based rect instead blanks the far edge of the element.
#[test]
fn tiles_reach_the_far_edge_under_a_nonzero_paint_offset() {
    const DOC_W: u32 = 100;
    const DOC_H: u32 = 40;
    const X_OFFSET: u32 = 40;
    const CANVAS_W: u32 = DOC_W + X_OFFSET;

    let body = format!(
        r#"<div style="width:100px; height:40px; {GRADIENT}
             background-size: 4px 4px; background-repeat: repeat;"></div>"#
    );
    let buf = render(&body, DOC_W, DOC_H, CANVAS_W, DOC_H, X_OFFSET, 0);

    // The div spans local x in [0,100), so on a 140px canvas offset by 40 it must be
    // painted all the way to x=139 with no white gap.
    let unpainted: Vec<u32> = (X_OFFSET..CANVAS_W)
        .filter(|x| is_white(&buf, CANVAS_W, *x, 20))
        .collect();
    assert!(
        unpainted.is_empty(),
        "background was culled away at screen x {unpainted:?}; expected paint through {}",
        CANVAS_W - 1
    );
}

/// The same check on the block axis, which is the one a tall page stresses.
#[test]
fn tiles_reach_the_far_edge_under_a_nonzero_vertical_offset() {
    const DOC_W: u32 = 40;
    const DOC_H: u32 = 100;
    const Y_OFFSET: u32 = 40;
    const CANVAS_H: u32 = DOC_H + Y_OFFSET;

    let body = format!(
        r#"<div style="width:40px; height:100px; {GRADIENT}
             background-size: 4px 4px; background-repeat: repeat;"></div>"#
    );
    let buf = render(&body, DOC_W, DOC_H, DOC_W, CANVAS_H, 0, Y_OFFSET);

    let unpainted: Vec<u32> = (Y_OFFSET..CANVAS_H)
        .filter(|y| is_white(&buf, DOC_W, 20, *y))
        .collect();
    assert!(
        unpainted.is_empty(),
        "background was culled away at screen y {unpainted:?}; expected paint through {}",
        CANVAS_H - 1
    );
}

/// The tiles that land on a given pixel must not depend on how big the surface is.
///
/// This is the invariant culling has to preserve, and the one that catches a cull computed
/// in the wrong space. The lattice is placed *before* the element's transform, so it scales
/// and rotates with the element; the bound is therefore the surface pulled back through that
/// transform. Bounding in surface space instead describes a different set of tiles as soon as
/// the transform's linear part is not the identity, and silently drops some of them.
/// Rendering the same document onto a larger surface and comparing the shared region catches
/// exactly that, with no golden image and no toggle.
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
        // Degenerate and extreme transforms: the cull must not turn a non-finite or
        // zero-determinant intermediate into a saturating cast that silently paints nothing
        // on one surface size but not the other.
        "transform:scale(0);",
        "transform:matrix(1e300, 0, 0, 1e300, 0, 0);",
        "transform:scale(1e-300);",
    ] {
        let body = format!(
            r#"<div style="position:absolute; left:25px; top:25px; width:150px; height:150px;
                 {GRADIENT} background-size:15px 15px; background-repeat:repeat;
                 {transform}"></div>"#
        );
        let small = render(&body, SMALL, SMALL, SMALL, SMALL, 0, 0);
        let large = render(&body, LARGE, LARGE, LARGE, LARGE, 0, 0);

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
             culled that the lattice still places on the surface"
        );
    }
}

/// A repeated gradient on an element far taller than the surface must still paint the
/// visible part, and must not cost time proportional to the element's height.
#[test]
fn a_tall_element_paints_its_visible_rows() {
    const W: u32 = 120;
    const H: u32 = 80;

    let body = format!(
        r#"<div style="width:120px; height:20000px; {GRADIENT}
             background-size:4px 4px; background-repeat:repeat;"></div>"#
    );
    let buf = render(&body, W, H, W, H, 0, 0);
    let unpainted: Vec<u32> = (0..H).filter(|y| is_white(&buf, W, 60, *y)).collect();
    assert!(
        unpainted.is_empty(),
        "tall element left screen rows {unpainted:?} unpainted"
    );
}
