//! Paint a [`blitz_dom::BaseDocument`] by pushing [`anyrender`] drawing commands into
//! an impl [`anyrender::PaintScene`].

#![allow(clippy::collapsible_if)]

mod color;
mod color_matrix;
mod debug_overlay;
mod filters;
mod gradient;
mod kurbo_css;
mod layers;
mod render;
mod sizing;
#[cfg(feature = "svg")]
mod svg_tile;
mod text;

use std::collections::HashMap;

use anyrender::{PaintScene, Scene};
use blitz_dom::{BaseDocument, NodeId, util::Color};
use render::BlitzDomPainter;

#[cfg(feature = "svg")]
pub use svg_tile::{SvgTileRasterizer, SvgTileRequest};

const FONT_EMBOLDEN_ENABLED: bool = cfg!(any(
    feature = "font-embolden",
    all(feature = "apple-font-embolden", target_os = "macos"),
    all(feature = "apple-font-embolden", target_os = "ios"),
));

/// The default color for text selection highlights
const SELECTION_COLOR: Color = Color::from_rgb8(180, 213, 255);

/// Pre-computed `Scene`s for each CustomWidget, keyed by `(document id, node id)`
type CustomWidgetSceneMap = HashMap<(usize, NodeId), Scene>;

/// Paint a [`blitz_dom::BaseDocument`] by pushing drawing commands into
/// an impl [`anyrender::PaintScene`].
///
/// This function assumes that the styles and layout in the [`BaseDocument`] are already
/// resolved. Please ensure that this is the case before trying to paint.
///
/// The implementation of [`PaintScene`] is responsible for handling the commands that are pushed into it.
/// Generally this will involve executing them to draw a rasterized image/texture. But in some cases it may choose to
/// transform them to a vector format (e.g. SVG/PDF) or serialize them in raw form for later use.
pub fn paint_scene(
    scene: &mut impl PaintScene,
    doc: &mut BaseDocument,
    scale: f64,
    width: u32,
    height: u32,
    x_offset: u32,
    y_offset: u32,
) {
    paint_scene_inner(
        scene,
        doc,
        scale,
        width,
        height,
        x_offset,
        y_offset,
        #[cfg(feature = "svg")]
        None,
    );
}

/// [`paint_scene`], with a rasteriser lent for repeating SVG background tiles.
///
/// A repeating SVG `background-image` is otherwise painted by replaying the tile's vector
/// scene once per tile, which costs time proportional to the painted area -- a full-screen
/// 4x4 stipple at 1280x720 measured 50.9 ms per frame in release, against a 16.7 ms 60 fps
/// budget. Given a rasteriser, a tile whose resolved size is a whole number of device pixels
/// is instead rasterised once and tiled by a repeating image brush in a single fill, which
/// measured 2.0 ms for the same frame.
///
/// The substitution is pixel-exact rather than merely close, and `blitz-paint` keeps it that
/// way: it is applied only where an image brush cannot resample (a whole-device-pixel tile,
/// more than one tile, both axes repeating), and `rasterizer` may decline any tile. Every
/// case that is declined or ineligible takes the vector path unchanged.
///
/// See [`SvgTileRasterizer`] for what an implementation owes the caller -- in particular that
/// it should rasterise with the same renderer that consumes this scene, and that it should
/// cache across frames.
#[cfg(feature = "svg")]
// One more than `paint_scene`, which is already at the limit; splitting the viewport
// parameters into a struct would be a breaking change to the existing entry point.
#[allow(clippy::too_many_arguments)]
pub fn paint_scene_with_tiles(
    scene: &mut impl PaintScene,
    doc: &mut BaseDocument,
    scale: f64,
    width: u32,
    height: u32,
    x_offset: u32,
    y_offset: u32,
    rasterizer: &dyn SvgTileRasterizer,
) {
    paint_scene_inner(
        scene,
        doc,
        scale,
        width,
        height,
        x_offset,
        y_offset,
        Some(rasterizer),
    );
}

#[allow(clippy::too_many_arguments)]
fn paint_scene_inner(
    scene: &mut impl PaintScene,
    doc: &mut BaseDocument,
    scale: f64,
    width: u32,
    height: u32,
    x_offset: u32,
    y_offset: u32,
    #[cfg(feature = "svg")] rasterizer: Option<&dyn SvgTileRasterizer>,
) {
    // Run `.paint()` on every custom widget in the document (and all subdocuments) ahead of time.
    // This helps us avoid borrow-checker issues as we recurse down the tree (`.paint()` require `&mut self`).
    //
    // TODO: Take widget and sub-document visibility into account
    #[allow(unused_mut)]
    let mut custom_widget_scenes: CustomWidgetSceneMap = HashMap::new();
    #[cfg(feature = "custom-widget")]
    build_custom_widget_scenes(&mut custom_widget_scenes, doc, scene, scale);

    let generator = BlitzDomPainter::new(
        doc,
        scale,
        width,
        height,
        x_offset as f64,
        y_offset as f64,
        &custom_widget_scenes,
        #[cfg(feature = "svg")]
        rasterizer,
    );
    generator.paint_scene(scene);

    // println!(
    //     "Rendered using {} clips (depth: {}) (wanted: {})",
    //     CLIPS_USED.load(atomic::Ordering::SeqCst),
    //     CLIP_DEPTH_USED.load(atomic::Ordering::SeqCst),
    //     CLIPS_WANTED.load(atomic::Ordering::SeqCst)
    // );
}

#[cfg(feature = "custom-widget")]
fn build_custom_widget_scenes(
    custom_widget_scenes: &mut CustomWidgetSceneMap,
    doc: &mut BaseDocument,
    render_ctx: &mut impl anyrender::RenderContext,
    scale: f64,
) {
    let doc_id = doc.id();

    // Process scenes for every custom widget in the document
    let custom_widget_node_ids = doc.custom_widget_node_ids();
    for node_id in custom_widget_node_ids.into_iter() {
        if let Some(scene) = process_custom_widget_node(doc, render_ctx, node_id, scale) {
            custom_widget_scenes.insert((doc_id, node_id), scene);
        }
    }

    // Recurse into sub documents
    let sub_document_node_ids = doc.sub_document_node_ids();
    for node_id in sub_document_node_ids.into_iter() {
        if let Some(sub_doc) = doc.get_node_mut(node_id).and_then(|node| node.subdoc_mut()) {
            let mut inner = sub_doc.inner_mut();
            build_custom_widget_scenes(custom_widget_scenes, &mut inner, render_ctx, scale);
        }
    }
}

#[cfg(feature = "custom-widget")]
fn process_custom_widget_node(
    doc: &mut BaseDocument,
    render_ctx: &mut impl anyrender::RenderContext,
    node_id: NodeId,
    scale: f64,
) -> Option<Scene> {
    use blitz_dom::node::{CustomWidgetStatus, ProxyRenderContext};

    let node = doc.get_node_mut(node_id)?;
    let width = (node.final_layout().size.width as f64 * scale) as u32;
    let height = (node.final_layout().size.height as f64 * scale) as u32;

    if width == 0 || height == 0 {
        return None;
    }

    let style = (*node.stylo_element_data().primary_styles()?).clone();
    let element = node.data.downcast_element_mut()?;
    let widget_data = element.custom_widget_data_mut()?;

    let mut render_ctx = ProxyRenderContext {
        inner: render_ctx,
        resource_ids: &mut widget_data.active_resource_ids,
    };

    if widget_data.status == CustomWidgetStatus::Suspended {
        widget_data.widget.can_create_surfaces(&mut render_ctx);
        widget_data.status = CustomWidgetStatus::Active;
    }

    let widget_scene = widget_data
        .widget
        .paint(&mut render_ctx, &style, width, height, scale);

    Some(widget_scene)
}
