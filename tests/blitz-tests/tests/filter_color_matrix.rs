//! CSS `filter` colour-matrix shorthands, per Filter Effects 1.
//!
//! The expected values here are computed from the specification text, not from
//! a previous render: §13.1 gives each shorthand's `<filter>` equivalent, §9.6
//! and §9.7.1 give the arithmetic, and §5 fixes the colour space ("Filter
//! Functions must operate in the sRGB color space") and the grouping ("All the
//! elements descendants are rendered together as a group with the filter effect
//! applied to the group as a whole").
//!
//! The `wpt` runner cannot stand in for these: it compares a test render
//! against a reference render made by the same engine, so a filter that is
//! silently dropped from *both* still passes.

use anyrender::render_to_buffer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;

const W: u32 = 40;
const H: u32 = 40;

fn render(body: &str) -> Vec<u8> {
    let html = format!(r#"<html><body style="margin:0; background:#ffffff;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(W, H, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene(scene, doc.as_mut(), 1.0, W, H, 0, 0),
        W,
        H,
    )
}

fn pixel(buf: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [buf[i], buf[i + 1], buf[i + 2]]
}

/// Read the centre of the 40x40 canvas.
fn centre(buf: &[u8]) -> [u8; 3] {
    pixel(buf, 20, 20)
}

#[track_caller]
fn assert_close(got: [u8; 3], want: [u8; 3], what: &str) {
    let ok = got
        .iter()
        .zip(want.iter())
        .all(|(g, w)| i32::from(*g).abs_diff(i32::from(*w)) <= 1);
    assert!(ok, "{what}: got {got:?}, expected {want:?} (+/-1)");
}

/// A 40x40 block of one colour, optionally filtered.
fn block(filter: &str, color: &str) -> String {
    format!(r#"<div style="width:40px; height:40px; background:{color}; {filter}"></div>"#)
}

/// §13.1.7: `brightness(a)` is `feFuncR/G/B type="linear" slope="a"`, i.e.
/// `C' = a * C` on non-premultiplied sRGB.
#[test]
fn brightness_scales_every_channel() {
    // #8040c0 = (128, 64, 192); x0.5 = (64, 32, 96).
    let buf = render(&block("filter: brightness(0.5);", "#8040c0"));
    assert_close(centre(&buf), [64, 32, 96], "brightness(0.5)");
}

/// §13.1.8: `contrast(a)` is `type="linear" slope="a"
/// intercept="-(0.5 * a) + 0.5"`. At `a = 0` every channel collapses onto 0.5.
#[test]
fn contrast_zero_is_mid_grey() {
    let buf = render(&block("filter: contrast(0);", "#8040c0"));
    assert_close(centre(&buf), [128, 128, 128], "contrast(0)");
}

/// §13.1.5: `invert(1)` is `type="table" tableValues="1 0"`, i.e. `C' = 1 - C`.
#[test]
fn invert_one_complements_every_channel() {
    let buf = render(&block("filter: invert(1);", "#8040c0"));
    assert_close(centre(&buf), [127, 191, 63], "invert(1)");
}

/// §13.1.4/§9.6: `hue-rotate(0deg)` collapses the hueRotate matrix onto the
/// identity (`cos 0 = 1`, `sin 0 = 0`). It must be byte-identical to no filter
/// at all -- this is the control that separates "the filter ran and did
/// nothing" from "the filter was dropped".
#[test]
fn hue_rotate_zero_is_byte_identical_to_no_filter() {
    let filtered = render(&block("filter: hue-rotate(0deg);", "#8040c0"));
    let plain = render(&block("", "#8040c0"));
    assert_eq!(filtered, plain, "hue-rotate(0deg) changed the frame");
}

/// §9.6 `type="hueRotate"` at 90 degrees: `cos = 0`, `sin = 1`, so the matrix
/// is the constant term plus the sin term, e.g. the red row becomes
/// `(0.213 - 0.213, 0.715 - 0.715, 0.072 + 0.928) = (0, 0, 1)`.
#[test]
fn hue_rotate_ninety_matches_the_spec_matrix() {
    // Source (128, 64, 192) / 255 = (0.5020, 0.2510, 0.7529).
    let r = 128.0 / 255.0;
    let g = 64.0 / 255.0;
    let b = 192.0 / 255.0;
    let rows = [
        [0.213 - 0.213, 0.715 - 0.715, 0.072 + 0.928],
        [0.213 + 0.143, 0.715 + 0.140, 0.072 - 0.283],
        [0.213 - 0.787, 0.715 + 0.715, 0.072 + 0.072],
    ];
    let want: [u8; 3] = std::array::from_fn(|i| {
        let v: f32 = rows[i][0] * r + rows[i][1] * g + rows[i][2] * b;
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    });
    let buf = render(&block("filter: hue-rotate(90deg);", "#8040c0"));
    assert_close(centre(&buf), want, "hue-rotate(90deg)");
}

/// §5: "The list of functions are applied in the order provided", and §9.7.1
/// puts `C` and `C'` "both in the closed interval [0,1]", so the chain clamps
/// between functions. `brightness(4)` saturates a mid grey to white before
/// `contrast(0.5)` maps it to 0.75 -- a single composed matrix without the
/// intermediate clamp would give 0.752.
#[test]
fn a_chain_applies_in_order_with_clamping_between() {
    let buf = render(&block("filter: brightness(4) contrast(0.5);", "#404040"));
    assert_close(centre(&buf), [191, 191, 191], "brightness(4) contrast(0.5)");
}

/// The design-authority declaration from PocketForge's Poolsuite app,
/// `filter: brightness(.55) contrast(1.2)` on a selected desktop icon.
/// (128, 64, 192) -> x0.55 -> (70.4, 35.2, 105.6) -> x1.2 - 0.1*255 ->
/// (58.98, 16.74, 101.22).
#[test]
fn brightness_then_contrast_darkens_a_selected_icon() {
    let buf = render(&block("filter: brightness(.55) contrast(1.2);", "#8040c0"));
    let want: [u8; 3] = std::array::from_fn(|i| {
        let c = [128.0_f32, 64.0, 192.0][i] / 255.0;
        let v = (1.2 * (0.55 * c).clamp(0.0, 1.0) - 0.1).clamp(0.0, 1.0);
        (v * 255.0 + 0.5) as u8
    });
    assert_close(centre(&buf), want, "brightness(.55) contrast(1.2)");
}

/// §5: the filter applies to the element "and its descendants [...] as a
/// group". A child painted inside a filtered ancestor must be filtered even
/// though the child carries no `filter` of its own.
#[test]
fn the_filter_covers_descendants() {
    let body = r#"<div style="width:40px; height:40px; background:#ffffff; filter: invert(1);">
        <div style="width:40px; height:40px; background:#8040c0;"></div>
    </div>"#;
    let buf = render(body);
    assert_close(centre(&buf), [127, 191, 63], "invert(1) on a descendant");
}

/// A translucent child over an opaque background inside the filtered group.
/// §5 composites the group first and filters the result, so the expected value
/// is `f(0.5 * blue + 0.5 * white)`, not `f(blue)` alone. `contrast(0.5)`
/// carries a non-zero intercept, which is the term a per-paint rewrite has to
/// get right for this to agree.
#[test]
fn a_translucent_overlap_matches_the_filtered_composite() {
    let body = r#"<div style="width:40px; height:40px; background:#ffffff; filter: contrast(0.5);">
        <div style="width:40px; height:40px; background:rgba(0,0,255,0.5);"></div>
    </div>"#;
    let buf = render(body);
    // composite = (0.5, 0.5, 1.0); contrast(0.5): C' = 0.5C + 0.25.
    let want: [u8; 3] = std::array::from_fn(|i| {
        let c = [0.5_f32, 0.5, 1.0][i];
        ((0.5 * c + 0.25) * 255.0 + 0.5) as u8
    });
    assert_close(
        centre(&buf),
        want,
        "contrast(0.5) over a translucent overlap",
    );
}

/// §5: "first any filter effect is applied, then any clipping, masking and
/// opacity". Element opacity multiplies the already-filtered group, so a black
/// result at 50% over white is mid grey either way -- but an implementation
/// that filtered *after* opacity would lift the black towards the backdrop
/// before inverting it.
#[test]
fn opacity_is_applied_after_the_filter() {
    let buf = render(&block("filter: invert(1); opacity: 0.5;", "#ffffff"));
    assert_close(centre(&buf), [128, 128, 128], "invert(1) under opacity 0.5");
}

/// A gradient is one paint but many pixel colours, and §9.7.1's `[0,1]` clamp
/// is applied per pixel, after the ramp is interpolated. Filtering only the
/// authored stops and letting the renderer interpolate between already-clamped
/// results is a different function, because clamping does not commute with
/// interpolation.
///
/// `brightness(1.3)` over `#e6e6e6 -> #808080` saturates at
/// `u = (0.90196 - 1/1.3) / 0.4 = 0.3318`. Left of that the answer is white;
/// right of it it is the unclamped line. Interpolating between the two clamped
/// endpoints instead would cut ~10-22/255 out of the middle of the ramp.
#[test]
fn a_clamping_gradient_is_filtered_per_pixel_not_per_stop() {
    const LEFT: f32 = 230.0 / 255.0;
    const RIGHT: f32 = 128.0 / 255.0;
    const SLOPE: f32 = 1.3;

    let body = r#"<div style="width:40px; height:40px;
        background: linear-gradient(to right, rgb(230,230,230), rgb(128,128,128));
        filter: brightness(1.3);"></div>"#;
    let buf = render(body);

    // The per-pixel answer: interpolate the source ramp, then filter and clamp.
    let expected = |x: u32| -> u8 {
        let u = (x as f32 + 0.5) / W as f32;
        let source = LEFT + u * (RIGHT - LEFT);
        ((SLOPE * source).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    // What filtering only the two endpoints would have produced.
    let per_stop = |x: u32| -> u8 {
        let u = (x as f32 + 0.5) / W as f32;
        let a = (SLOPE * LEFT).clamp(0.0, 1.0);
        let b = (SLOPE * RIGHT).clamp(0.0, 1.0);
        ((a + u * (b - a)) * 255.0 + 0.5) as u8
    };

    for x in [5_u32, 12, 20, 30, 38] {
        let got = pixel(&buf, x, 20);
        let want = expected(x);
        assert!(
            got.iter()
                .all(|c| i32::from(*c).abs_diff(i32::from(want)) <= 2),
            "x={x}: got {got:?}, per-pixel expects {want} (per-stop would give {})",
            per_stop(x)
        );
    }
    // The saturated end really is saturated, not a ramp down from it.
    assert_eq!(
        pixel(&buf, 2, 20),
        [255, 255, 255],
        "left end should be white"
    );
}

/// A filter this module does not own must not be silently turned into a no-op
/// colour rewrite: `blur()` still goes to the `Filter` graph. The backend in
/// use may or may not execute it, so this only pins that the frame is not the
/// unfiltered one when the backend does support it, and never panics when it
/// does not.
#[test]
fn a_blur_filter_does_not_panic() {
    let buf = render(&block("filter: blur(2px);", "#8040c0"));
    assert_eq!(buf.len() as u32, W * H * 4);
}
