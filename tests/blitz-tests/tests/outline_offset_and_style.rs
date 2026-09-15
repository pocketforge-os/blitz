//! `outline-offset` positions the outline, and `outline-style` patterns it.
//!
//! css-ui-4 §3.5: "If the computed value of `outline-offset` is anything other than 0,
//! then the outline is outset from the border edge by that amount. Negative values must
//! cause the outline to shrink into the border box. Both the height and the width of the
//! outside of the shape drawn by the outline should not become smaller than twice the
//! computed value of the `outline-width` property [...] User agents should apply this
//! constraint independently in each dimension."
//!
//! css-ui-4 §3.3: "`<outline-line-style>` accepts the same values as `<line-style>`
//! [CSS Backgrounds 3 §3.2] with the same meaning, except that `hidden` is not a legal
//! outline style", so `dashed` means "a series of square-ended dashes", not a solid ring.
//!
//! css-ui-4 §3.1: "The outline created with the outline properties is drawn 'over' a box,
//! i.e., the outline is always on top" -- which is what makes a negative offset visible at
//! all, since the ring then lies inside the border box on top of the element's background.
//!
//! The upstream WPT ref-test runner renders every test at `x_offset = y_offset = 0`
//! (`wpt/runner/src/test_runners/ref_test.rs`), so it cannot see where in the frame a
//! paint lands. These tests assert the painted columns directly.

use anyrender::render_to_buffer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;

const SIZE: u32 = 100;
/// The element under test: a 60x60 black box at (20, 20), so its border box spans
/// `[20, 80)` on both axes with 20px of white page on every side.
const BOX_START: i32 = 20;
const BOX_END: i32 = 80;
const OUTLINE_WIDTH: i32 = 4;
/// The row scanned by [`ring_columns`]: through the middle of the box, clear of both
/// horizontal edges of the ring.
const MID: u32 = 50;

fn render(style: &str) -> Vec<u8> {
    let html = format!(
        r#"<html><body style="margin:0; background:#ffffff;">
             <div style="position:absolute; left:20px; top:20px; width:60px; height:60px;
                         background:#000000; {style}"></div>
           </body></html>"#
    );
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(SIZE, SIZE, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene(scene, doc.as_mut(), 1.0, SIZE, SIZE, 0, 0),
        SIZE,
        SIZE,
    )
}

/// Whether the pixel is the outline colour (`#ff0000`), allowing for antialiasing.
fn is_ring(buf: &[u8], x: u32, y: u32) -> bool {
    let i = ((y * SIZE + x) * 4) as usize;
    buf[i] > 180 && buf[i + 1] < 80 && buf[i + 2] < 80
}

/// The x columns painted in the outline colour on row `y`.
fn ring_columns(buf: &[u8], y: u32) -> Vec<i32> {
    (0..SIZE)
        .filter(|x| is_ring(buf, *x, y))
        .map(|x| x as i32)
        .collect()
}

/// The columns the ring must cover on a row that crosses only its left and right sides,
/// for an outline of [`OUTLINE_WIDTH`] at `offset`: the border box displaced outwards by
/// `offset` gives the ring's inner edge, and that edge grown by the width gives its outer
/// edge.
fn expected_columns(offset: i32) -> Vec<i32> {
    let left_inner = BOX_START - offset;
    let right_inner = BOX_END + offset;
    (left_inner - OUTLINE_WIDTH..left_inner)
        .chain(right_inner..right_inner + OUTLINE_WIDTH)
        .collect()
}

/// A solid outline lands exactly where `outline-offset` puts it, in both directions.
///
/// A fix that hardcodes one direction passes half of this and fails the other half, which
/// is the point of testing both signs against the same expectation.
#[test]
fn a_solid_outline_sits_at_its_offset() {
    for offset in [-4, -3, -1, 0, 1, 2, 5] {
        let buf = render(&format!(
            "outline: {OUTLINE_WIDTH}px solid #ff0000; outline-offset: {offset}px;"
        ));
        assert_eq!(
            ring_columns(&buf, MID),
            expected_columns(offset),
            "outline-offset: {offset}px put the ring in the wrong columns"
        );
    }
}

/// The ring moves outwards as the offset grows, monotonically and by exactly the change in
/// offset. This is the property a renderer that ignores the declaration violates: it draws
/// every offset in the same place.
#[test]
fn the_ring_moves_with_the_offset() {
    let left_edge = |offset: i32| -> i32 {
        let buf = render(&format!(
            "outline: {OUTLINE_WIDTH}px solid #ff0000; outline-offset: {offset}px;"
        ));
        *ring_columns(&buf, MID)
            .first()
            .expect("no outline was painted")
    };

    let zero = left_edge(0);
    for offset in [-3, -1, 1, 3, 6] {
        assert_eq!(
            left_edge(offset) - zero,
            -offset,
            "the outline's outer edge did not move by {offset}px when outline-offset did"
        );
    }
}

/// A negative offset draws the outline inside the border box, where it must still be
/// visible: css-ui-4 §3.1 puts the outline on top of the box. Painted under the element's
/// own background instead, an inset focus ring vanishes completely.
#[test]
fn a_negative_offset_paints_over_the_elements_own_background() {
    let buf = render(&format!(
        "outline: {OUTLINE_WIDTH}px solid #ff0000; outline-offset: -{OUTLINE_WIDTH}px;"
    ));
    // At offset -4 with width 4 the ring occupies exactly the first 4px inside the border
    // box, every pixel of it over the element's black background.
    let inside: Vec<i32> = (BOX_START..BOX_START + OUTLINE_WIDTH).collect();
    let columns = ring_columns(&buf, MID);
    assert!(
        columns.starts_with(&inside),
        "an inset outline was not drawn over the element's background: columns {columns:?}"
    );
}

/// css-ui-4 §3.5's minimum-size rule: a large negative offset must not invert the ring.
/// The outside of the shape stops shrinking at twice the outline width.
#[test]
fn a_large_negative_offset_is_clamped_not_inverted() {
    let buf = render(&format!(
        "outline: {OUTLINE_WIDTH}px solid #ff0000; outline-offset: -1000px;"
    ));
    let painted: Vec<(u32, u32)> = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .filter(|(x, y)| is_ring(&buf, *x, *y))
        .collect();
    assert!(!painted.is_empty(), "the clamped outline vanished entirely");

    let (xs, ys): (Vec<u32>, Vec<u32>) = painted.into_iter().unzip();
    let (x0, x1) = (*xs.iter().min().unwrap(), *xs.iter().max().unwrap());
    let (y0, y1) = (*ys.iter().min().unwrap(), *ys.iter().max().unwrap());
    let min_side = 2 * OUTLINE_WIDTH as u32;
    assert_eq!(
        (x1 - x0 + 1, y1 - y0 + 1),
        (min_side, min_side),
        "the clamped outline is {}x{}; §3.5 floors it at {min_side}x{min_side}",
        x1 - x0 + 1,
        y1 - y0 + 1
    );
    let centre = ((BOX_START + BOX_END) / 2) as u32;
    assert_eq!(
        (x0 + min_side / 2, y0 + min_side / 2),
        (centre, centre),
        "the clamped outline is not centred on the box"
    );
}

/// `outline-style: dashed` is a series of dashes, not a solid ring (css-ui-4 §3.3 ->
/// CSS Backgrounds 3 §3.2). Scanned along the top side, a dashed outline leaves gaps; a
/// solid one does not.
#[test]
fn a_dashed_outline_is_drawn_as_dashes() {
    let top_row = (BOX_START - OUTLINE_WIDTH / 2) as u32;
    let painted = |style: &str| -> usize {
        let buf = render(&format!(
            "outline: {OUTLINE_WIDTH}px {style} #ff0000; outline-offset: 0;"
        ));
        (BOX_START..BOX_END)
            .filter(|x| is_ring(&buf, *x as u32, top_row))
            .count()
    };

    let solid = painted("solid");
    let dashed = painted("dashed");
    let span = (BOX_END - BOX_START) as usize;
    assert_eq!(
        solid, span,
        "the solid outline did not cover its whole side"
    );
    assert!(
        dashed > 0,
        "the dashed outline painted nothing along the top side"
    );
    assert!(
        dashed < solid,
        "the dashed outline covered all {solid} px of its side -- it was painted solid"
    );
    // A 2:1 dash:gap ratio covers about two thirds of the side. Allow a wide band around
    // that: the point is that it is patterned, not solid and not a dotted-looking sliver.
    assert!(
        (span / 2..=(span * 8) / 10).contains(&dashed),
        "dashed coverage {dashed}/{span} is not a 2:1 dash pattern"
    );
}

/// `dotted` is likewise patterned rather than solid, and distinguishable from `dashed`.
#[test]
fn a_dotted_outline_is_drawn_as_dots() {
    let top_row = (BOX_START - OUTLINE_WIDTH / 2) as u32;
    let painted = |style: &str| -> usize {
        let buf = render(&format!(
            "outline: {OUTLINE_WIDTH}px {style} #ff0000; outline-offset: 0;"
        ));
        (BOX_START..BOX_END)
            .filter(|x| is_ring(&buf, *x as u32, top_row))
            .count()
    };
    let dotted = painted("dotted");
    let dashed = painted("dashed");
    assert!(dotted > 0, "the dotted outline painted nothing");
    assert!(
        dotted < dashed,
        "dots ({dotted} px) covered as much of the side as dashes ({dashed} px)"
    );
}

/// The offset applies to a patterned outline too: the two fixes have to compose, since a
/// dashed ring is drawn by a different path from a solid one.
#[test]
fn a_dashed_outline_also_honours_the_offset() {
    let top_of_ring = |offset: i32| -> u32 {
        let buf = render(&format!(
            "outline: {OUTLINE_WIDTH}px dashed #ff0000; outline-offset: {offset}px;"
        ));
        (0..SIZE)
            .find(|y| (0..SIZE).any(|x| is_ring(&buf, x, *y)))
            .expect("no dashed outline was painted")
    };

    let zero = top_of_ring(0);
    for offset in [-3, -1, 2, 5] {
        assert_eq!(
            top_of_ring(offset) as i32 - zero as i32,
            -offset,
            "a dashed outline ignored outline-offset: {offset}px"
        );
    }
}

/// A square box keeps square outline corners, whatever the style and wherever the offset
/// puts the ring. There is no `border-radius` here for the outline to follow (css-ui-4
/// §3.1), so every edge of the ring is a rectangle and its outermost corner pixel is
/// painted.
///
/// The patterned styles are drawn over a frame derived from this box, which is where a
/// radius can be invented: deriving the ring's corner radius as `border-radius + distance`
/// unconditionally gives a square box a rounded ring as soon as it has any outline at all.
/// `dotted` is excluded because its dots are circles -- one is anchored in each corner, but
/// a circle does not reach the square corner of its own bounding box.
#[test]
fn a_square_box_keeps_square_outline_corners() {
    for offset in [0, 2, 6] {
        let outer = (BOX_START - offset - OUTLINE_WIDTH) as u32;
        for style in ["solid", "dashed", "double"] {
            let buf = render(&format!(
                "outline: {OUTLINE_WIDTH}px {style} #ff0000; outline-offset: {offset}px;"
            ));
            assert!(
                is_ring(&buf, outer, outer),
                "outline: {OUTLINE_WIDTH}px {style} at offset {offset}px left the ring's \
                 outermost corner ({outer}, {outer}) unpainted -- the corner was rounded"
            );
        }
    }
}
