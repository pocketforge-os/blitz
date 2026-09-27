//! CSS `filter` colour-matrix shorthands.
//!
//! The expected values here are computed from the specification text, not from
//! a previous render. The grouping requirement is: "All the elements descendants
//! are rendered together as a group with the filter effect applied to the group
//! as a whole." <https://drafts.csswg.org/filter-effects-1/#FilterProperty>
//!
//! The `wpt` runner cannot stand in for these: it compares a test render
//! against a reference render made by the same engine, so a filter that is
//! silently dropped from *both* still passes.

use anyrender::{ImageRenderer, render_to_buffer};
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;
use std::time::Instant;

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

fn document(body: &str, width: u32, height: u32) -> HtmlDocument {
    let html = format!(r#"<html><body style="margin:0; background:#fff;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(width, height, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    doc
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

/// `brightness(a)` is `feFuncR/G/B type="linear" slope="a"`, i.e.
/// `C' = a * C` on non-premultiplied sRGB.
#[test]
fn brightness_scales_every_channel() {
    // #8040c0 = (128, 64, 192); x0.5 = (64, 32, 96).
    let buf = render(&block("filter: brightness(0.5);", "#8040c0"));
    assert_close(centre(&buf), [64, 32, 96], "brightness(0.5)");
}

/// `contrast(a)` is `type="linear" slope="a"
/// intercept="-(0.5 * a) + 0.5"`. At `a = 0` every channel collapses onto 0.5.
#[test]
fn contrast_zero_is_mid_grey() {
    let buf = render(&block("filter: contrast(0);", "#8040c0"));
    assert_close(centre(&buf), [128, 128, 128], "contrast(0)");
}

/// `invert(1)` is `type="table" tableValues="1 0"`, i.e. `C' = 1 - C`.
#[test]
fn invert_one_complements_every_channel() {
    let buf = render(&block("filter: invert(1);", "#8040c0"));
    assert_close(centre(&buf), [127, 191, 63], "invert(1)");
}

/// `hue-rotate(0deg)` collapses the hueRotate matrix onto the
/// identity (`cos 0 = 1`, `sin 0 = 0`). It must be byte-identical to no filter
/// at all -- this is the control that separates "the filter ran and did
/// nothing" from "the filter was dropped".
#[test]
fn hue_rotate_zero_is_byte_identical_to_no_filter() {
    let filtered = render(&block("filter: hue-rotate(0deg);", "#8040c0"));
    let plain = render(&block("", "#8040c0"));
    assert_eq!(filtered, plain, "hue-rotate(0deg) changed the frame");
}

/// `type="hueRotate"` at 90 degrees: `cos = 0`, `sin = 1`, so the matrix
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

/// The chain applies functions in author order and clamps to `[0,1]` between
/// them. `brightness(4)` saturates a mid grey to white before
/// `contrast(0.5)` maps it to 0.75 -- a single composed matrix without the
/// intermediate clamp would give 0.752.
#[test]
fn a_chain_applies_in_order_with_clamping_between() {
    let buf = render(&block("filter: brightness(4) contrast(0.5);", "#404040"));
    assert_close(centre(&buf), [191, 191, 191], "brightness(4) contrast(0.5)");
}

/// A representative `filter: brightness(.55) contrast(1.2)` on a selected
/// desktop icon.
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

/// A filter applies to the element and its descendants as one group. A child
/// painted inside a filtered ancestor must be filtered even
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
/// The group composites first and filters the result, so the expected value
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

/// The filter sees the composited group, so the half-transparent white child first
/// produces mid-grey over black and `brightness(2)` then clamps that result to white.
#[test]
fn brightness_clamps_after_a_translucent_child_is_composited() {
    let body = r#"<div style="width:40px; height:40px; background:#000; filter:brightness(2);">
        <div style="width:40px; height:40px; background:rgba(255,255,255,0.5);"></div>
    </div>"#;
    let buf = render(body);
    assert_close(
        centre(&buf),
        [255, 255, 255],
        "brightness(2) after translucent compositing",
    );
}

/// Antialiasing is partial coverage and therefore another translucent composite.
/// Find a non-trivial edge texel in an unfiltered reference and verify that the
/// filtered group doubles that composited value, rather than filtering opaque white
/// before its coverage is applied.
#[test]
fn brightness_filters_an_antialiased_edge_after_coverage() {
    let plain = render(
        r#"<div style="width:40px; height:40px; background:#000;">
            <div style="width:24px; height:24px; border-radius:50%; background:#fff;"></div>
        </div>"#,
    );
    let filtered = render(
        r#"<div style="width:40px; height:40px; background:#000; filter:brightness(2);">
            <div style="width:24px; height:24px; border-radius:50%; background:#fff;"></div>
        </div>"#,
    );

    let (index, source) = plain
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .find(|(_, px)| px[0] >= 32 && px[0] <= 160 && px[0] == px[1] && px[1] == px[2])
        .map(|(index, px)| (index, px[0]))
        .expect("the rounded edge should contain a partially covered texel");
    let got = filtered.as_chunks::<4>().0[index][0];
    let want = source.saturating_mul(2);
    assert!(
        got.abs_diff(want) <= 2,
        "antialiased edge: source coverage {source}, got {got}, expected {want} (+/-2)"
    );
}

/// `contrast(2)` maps values below 0.25 below zero. The translucent white child
/// composites to 0.2 over black, so filtering the group clamps it to black.
#[test]
fn contrast_clamps_below_zero_after_translucent_compositing() {
    let body = r#"<div style="width:40px; height:40px; background:#000; filter:contrast(2);">
        <div style="width:40px; height:40px; background:rgba(255,255,255,0.2);"></div>
    </div>"#;
    let buf = render(body);
    assert_close(
        centre(&buf),
        [0, 0, 0],
        "contrast(2) below-zero clamp after translucent compositing",
    );
}

/// A filter establishes a group, not an overflow clip. Descendant ink outside
/// the filtered element's border box remains visible when overflow is visible,
/// and the ancestor filter applies to that ink.
#[test]
fn a_group_filter_preserves_visible_descendant_overflow() {
    let body = r#"<div style="position:relative; width:10px; height:10px;
            overflow:visible; filter:brightness(2)">
        <div style="position:absolute; left:20px; top:0; width:10px; height:10px;
            background:#404040"></div>
    </div>"#;
    let buf = render(body);
    assert_close(
        pixel(&buf, 25, 5),
        [128, 128, 128],
        "filtered descendant outside a visible-overflow border box",
    );
}

/// Descendant ink overflow includes effects that layout overflow alone does not
/// know about. The outer filter must retain and transform that effect too.
#[test]
fn a_group_filter_bounds_visible_descendant_effect_overflow() {
    let body = r#"<div style="position:relative; width:5px; height:5px;
            overflow:visible; filter:brightness(2)">
        <div style="position:absolute; left:10px; top:0; width:5px; height:5px;
            box-shadow:10px 0 0 #404040"></div>
    </div>"#;
    let buf = render(body);
    assert_close(
        pixel(&buf, 22, 2),
        [128, 128, 128],
        "filtered descendant effect outside layout overflow",
    );
}

/// The same geometry remains clipped when overflow independently requires it.
#[test]
fn a_group_filter_keeps_hidden_descendant_overflow_clipped() {
    let body = r#"<div style="position:relative; width:10px; height:10px;
            overflow:hidden; filter:brightness(2)">
        <div style="position:absolute; left:20px; top:0; width:10px; height:10px;
            background:#404040"></div>
    </div>"#;
    let buf = render(body);
    assert_close(
        pixel(&buf, 25, 5),
        [255, 255, 255],
        "hidden overflow outside a filtered border box",
    );
}

/// Element opacity multiplies the already-filtered group, so a black
/// result at 50% over white is mid grey either way -- but an implementation
/// that filtered *after* opacity would lift the black towards the backdrop
/// before inverting it.
#[test]
fn opacity_is_applied_after_the_filter() {
    let buf = render(&block("filter: invert(1); opacity: 0.5;", "#ffffff"));
    assert_close(centre(&buf), [128, 128, 128], "invert(1) under opacity 0.5");
}

/// A gradient is one paint but many pixel colours, and the `[0,1]` clamp
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

/// Manual release-mode cost probe for the actual filter exposure dimensions at
/// 1280x720. The content is generic, but the filter chains and boxes match the
/// product reference: whole-frame invert, one 48x48 icon, and 1218x200 photos.
/// The full-screen hue rotation remains as a deliberately pessimistic control.
/// This is ignored during the normal test suite.
#[test]
#[ignore = "manual 1280x720 frame-time benchmark"]
fn filtered_subtree_frame_cost_1280x720() {
    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;
    const WARMUPS: usize = 5;
    const RUNS: usize = 30;

    let full_screen_scene = |filter: &str| {
        format!(
            r#"<main style="width:1280px; height:720px; {filter}
        background:linear-gradient(135deg,#d04480,#40a0dc);">
        <div style="width:900px; height:520px; background:rgba(255,255,255,.35);"></div>
        <div style="width:700px; height:420px; margin:-360px 0 0 420px;
            border-radius:80px; background:rgba(20,30,60,.55);"></div>
    </main>"#
        )
    };

    let focused_icon = r##"<main style="width:1280px; height:720px; background:#246; padding:52px;">
        <svg viewBox="0 0 48 48" style="width:48px; height:48px;
            filter:brightness(.55) contrast(1.2)">
            <rect x="2" y="8" width="44" height="34" rx="5" fill="#ef8a62"/>
            <circle cx="17" cy="24" r="9" fill="#67a9cf" fill-opacity=".65"/>
            <path d="M26 14 L42 36 L18 36 Z" fill="#f7f7f7"/>
        </svg>
    </main>"##;

    let photo = |angle: u32| {
        format!(
            r#"<main style="width:1280px; height:720px; background:#ececec; padding:31px;">
        <div style="box-sizing:border-box; width:1218px; height:200px;
            filter:hue-rotate({angle}deg);
            background:linear-gradient(135deg,#e56b8a,#f2c46d 45%,#54a7cb);">
            <div style="width:760px; height:150px; border-radius:90px;
                background:rgba(255,255,255,.38)"></div>
            <div style="width:600px; height:100px; margin:-105px 0 0 560px;
                background:rgba(12,42,74,.58)"></div>
        </div>
    </main>"#
        )
    };

    let clipped = r#"<main style="width:1280px; height:720px; background:#246; padding:40px;">
        <div style="width:320px; height:180px; overflow:hidden;">
            <div style="width:1280px; height:720px; filter:hue-rotate(92deg);
                background:linear-gradient(135deg,#d04480,#40a0dc)"></div>
        </div>
    </main>"#;

    for (name, body) in [
        ("full_screen_invert", full_screen_scene("filter:invert(1);")),
        ("focused_icon_48x48", focused_icon.to_owned()),
        ("photo_1218x200_hue_rotate_0", photo(0)),
        ("photo_1218x200_hue_rotate_61", photo(61)),
        ("photo_1218x200_hue_rotate_122", photo(122)),
        (
            "worst_case_full_screen_hue_rotate_92",
            full_screen_scene("filter:hue-rotate(92deg);"),
        ),
        ("tuning_partially_clipped_hue_rotate", clipped.to_owned()),
    ] {
        let mut doc = document(&body, WIDTH, HEIGHT);
        let mut renderer = VelloCpuImageRenderer::new(WIDTH, HEIGHT);
        let mut buffer = Vec::new();

        for _ in 0..WARMUPS {
            renderer.reset();
            renderer.render_to_vec(
                |scene| paint_scene(scene, doc.as_mut(), 1.0, WIDTH, HEIGHT, 0, 0),
                &mut buffer,
            );
        }

        let mut samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            renderer.reset();
            let start = Instant::now();
            renderer.render_to_vec(
                |scene| paint_scene(scene, doc.as_mut(), 1.0, WIDTH, HEIGHT, 0, 0),
                &mut buffer,
            );
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "FILTER_BENCH {name} runs={RUNS} min_ms={:.3} median_ms={:.3} max_ms={:.3}",
            samples[0],
            samples[RUNS / 2],
            samples[RUNS - 1],
        );
    }
}
