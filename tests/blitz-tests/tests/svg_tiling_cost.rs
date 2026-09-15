//! Cost of tiling a repeated SVG background, measured the way the 54.8 ms figure on
//! `tsp-f3fm.146` was: one `render_to_buffer` of a 1280x720 surface entirely covered by the
//! 4x4 desktop dither, in release.
//!
//! Run with:
//!     cargo test --release -p blitz-tests --test svg_tiling_cost -- --nocapture --ignored
//!
//! `#[ignore]` so the timing never runs as part of the ordinary suite, where it would be
//! both slow and meaningless (a debug number is not comparable).

use anyrender::render_to_buffer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_dom::node::{ImageData, RasterImageData, SvgImageData};
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;
use std::time::Instant;

/// The desktop wallpaper stipple: a 4x4 tile with one black pixel at (0,0) and one at (2,2).
const DITHER: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><rect width="1" height="1" fill="black"/><rect x="2" y="2" width="1" height="1" fill="black"/></svg>"#;

/// The same stipple as a 4x4 RGBA raster. `draw_raster_image_layer` takes the single-fill
/// `Extend::Repeat` path for `background-repeat: repeat`, so this is the floor the one-fill
/// approach could reach at best -- the identical visible result in one `scene.fill`.
fn dither_raster() -> RasterImageData {
    let mut px = vec![0u8; 4 * 4 * 4];
    for (x, y) in [(0usize, 0usize), (2, 2)] {
        let i = (y * 4 + x) * 4;
        px[i..i + 4].copy_from_slice(&[0, 0, 0, 255]);
    }
    RasterImageData::new(4, 4, Arc::new(px))
}

fn build_raster(w: u32, h: u32, el_h: u32) -> HtmlDocument {
    let body = format!(
        r#"<div class="bg" style="width:{w}px; height:{el_h}px;
             background-image:url('https://example.com/x.png'); background-repeat:repeat;"></div>"#
    );
    let html = format!(r#"<html><body style="margin:0; background:#ffffff;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(w, h, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    let raster = dither_raster();
    let ids: Vec<_> = doc
        .query_selector_all(".bg")
        .expect("valid selector")
        .into_iter()
        .collect();
    for id in ids {
        let el = doc.get_node_mut(id).unwrap().element_data_mut().unwrap();
        for layer in el.background_images.iter_mut().flatten() {
            layer.status = blitz_dom::node::Status::Ok;
            layer.image = ImageData::Raster(raster.clone());
        }
    }
    doc.resolve(0.0);
    doc
}

/// The same page with no background layer at all: everything in the frame except tiling.
fn build_bare(w: u32, h: u32, el_h: u32) -> HtmlDocument {
    let body = format!(r#"<div class="bg" style="width:{w}px; height:{el_h}px;"></div>"#);
    let html = format!(r#"<html><body style="margin:0; background:#ffffff;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(w, h, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    doc
}

fn render_doc(mut doc: HtmlDocument, w: u32, h: u32) -> Vec<u8> {
    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene(scene, doc.as_mut(), 1.0, w, h, 0, 0),
        w,
        h,
    )
}

/// The same stipple and the same element, painted either as an SVG layer (vector replay per
/// tile) or as a raster layer (one fill, `Extend::Repeat`).
fn build_styled(w: u32, h: u32, size_decl: &str, raster: bool) -> HtmlDocument {
    let body = format!(
        r#"<div class="bg" style="width:{w}px; height:{h}px;
             background-image:url('https://example.com/x'); background-repeat:repeat;
             {size_decl}"></div>"#
    );
    let html = format!(r#"<html><body style="margin:0; background:#ffffff;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(w, h, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    let svg =
        SvgImageData::from_data(DITHER.as_bytes(), &usvg::Options::default()).expect("valid SVG");
    let rst = dither_raster();
    let ids: Vec<_> = doc
        .query_selector_all(".bg")
        .expect("valid selector")
        .into_iter()
        .collect();
    for id in ids {
        let el = doc.get_node_mut(id).unwrap().element_data_mut().unwrap();
        for layer in el.background_images.iter_mut().flatten() {
            layer.status = blitz_dom::node::Status::Ok;
            layer.image = if raster {
                ImageData::Raster(rst.clone())
            } else {
                ImageData::Svg(svg.clone())
            };
        }
    }
    doc.resolve(0.0);
    doc
}

fn build(w: u32, h: u32, el_h: u32) -> HtmlDocument {
    let body = format!(
        r#"<div class="bg" style="width:{w}px; height:{el_h}px;
             background-image:url('https://example.com/x.svg'); background-repeat:repeat;"></div>"#
    );
    let html = format!(r#"<html><body style="margin:0; background:#ffffff;">{body}</body></html>"#);
    let mut doc = HtmlDocument::from_html(
        &html,
        DocumentConfig {
            viewport: Some(Viewport::new(w, h, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    let svg =
        SvgImageData::from_data(DITHER.as_bytes(), &usvg::Options::default()).expect("valid SVG");
    let ids: Vec<_> = doc
        .query_selector_all(".bg")
        .expect("valid selector")
        .into_iter()
        .collect();
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

fn time_doc(label: &str, mut doc: HtmlDocument, w: u32, h: u32, runs: u32) -> f64 {
    // One warm-up frame so allocation and any lazily-built state is not in the sample.
    let _ = render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| paint_scene(scene, doc.as_mut(), 1.0, w, h, 0, 0),
        w,
        h,
    );
    let mut best = f64::MAX;
    let mut total = 0.0;
    for _ in 0..runs {
        let start = Instant::now();
        let buf = render_to_buffer::<VelloCpuImageRenderer, _>(
            |scene| paint_scene(scene, doc.as_mut(), 1.0, w, h, 0, 0),
            w,
            h,
        );
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&buf);
        total += ms;
        best = best.min(ms);
    }
    let mean = total / runs as f64;
    println!("{label:<40} best={best:8.2}ms  mean={mean:8.2}ms");
    best
}

/// Does the single-fill image-brush route actually produce the same pixels as replaying the
/// vector per tile?
///
/// This is the question that decides the approach, and it is not the same question as "is it
/// faster". Replaying a vector per tile rasterises each tile in place, resolving antialiasing
/// against the surface grid at that tile's exact sub-pixel position. Rasterising once and
/// sampling through an `Extend::Repeat` brush resolves antialiasing once against the *tile's*
/// own grid and then resamples. Those can only agree when the tile is integer-sized and
/// integer-aligned at scale 1 -- which the 4x4 stipple is, and which a tile at a fractional
/// `background-size` is not.
///
/// So this reports the answer for both, rather than asserting the convenient one.
#[test]
#[ignore = "diagnostic; run explicitly"]
fn image_brush_vs_vector_replay_pixels() {
    for (label, w, h, size_decl) in [
        ("aligned 4x4 tile at scale 1", 1280u32, 720u32, ""),
        (
            "fractional background-size",
            320,
            200,
            "background-size: 4.5px 4.5px;",
        ),
        ("half-scale tile", 320, 200, "background-size: 2px 2px;"),
        (
            "fractional offset",
            320,
            200,
            "background-position: 1.5px 0.5px;",
        ),
    ] {
        let svg_buf = render_doc(build_styled(w, h, size_decl, false), w, h);
        let img_buf = render_doc(build_styled(w, h, size_decl, true), w, h);
        let differing = svg_buf
            .as_chunks::<4>()
            .0
            .iter()
            .zip(img_buf.as_chunks::<4>().0.iter())
            .filter(|(a, b)| a[0..3] != b[0..3])
            .count();
        println!(
            "{label:<32} {w}x{h}  differing px vector-replay vs image-brush: {differing}  \
             ({:.4}% of surface)",
            100.0 * differing as f64 / (w * h) as f64
        );
    }
}

#[test]
#[ignore = "timing harness; run explicitly in release"]
fn dither_frame_cost() {
    const W: u32 = 1280;
    const H: u32 = 720;
    const RUNS: u32 = 9;
    println!("\n1280x720, 57,600 tiles of a 4x4 dither. 60fps budget = 16.70 ms.\n");

    // The headline case from the bead.
    let svg = time_doc("SVG tile, repeat", build(W, H, H), W, H, RUNS);
    // Flat in element height, which is what the tsp-f3fm.122 culling bound bought.
    time_doc(
        "SVG tile, element 20000px tall",
        build(W, H, 20000),
        W,
        H,
        RUNS,
    );
    // The floor: the identical visible result through the single-fill raster brush path.
    let raster = time_doc(
        "same stipple as a raster tile (1 fill)",
        build_raster(W, H, H),
        W,
        H,
        RUNS,
    );
    // Everything in the frame except the tiling.
    let bare = time_doc(
        "no background layer at all",
        build_bare(W, H, H),
        W,
        H,
        RUNS,
    );

    println!(
        "\ntiling cost: SVG {:.2} ms vs raster {:.2} ms over a {:.2} ms bare frame -- \
         a {:.1}x gap to the single-fill floor\n",
        svg - bare,
        raster - bare,
        bare,
        (svg - bare) / (raster - bare).max(0.001)
    );
}
