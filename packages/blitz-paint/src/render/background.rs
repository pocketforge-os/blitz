use super::{ElementCx, PhysicalTracks, to_image_quality, to_peniko_image};
use crate::color::{Color, ToColorColor};
use crate::gradient::to_peniko_gradient;
use anyrender::{PaintScene, Scene, recording::RenderCommand};
use blitz_dom::node::{ImageData, ImageResourceData, SpecialElementData};
use kurbo::{self, Affine, BezPath, Point, Rect, Size, Vec2};
use peniko::{self, Fill};
use style::{
    properties::{
        generated::longhands::{
            background_attachment::single_value::computed_value::T as StyloBackgroundAttachment,
            background_clip::single_value::computed_value::T as StyloBackgroundClip,
            background_origin::single_value::computed_value::T as StyloBackgroundOrigin,
            mask_origin::single_value::computed_value::T as StyloMaskOrigin,
        },
        style_structs::{Background, SVG},
    },
    values::{
        computed::{
            BackgroundRepeat, Gradient as StyloGradient, Image as ComputedImage, LengthPercentage,
            background::BackgroundSize,
        },
        generics::image::GenericImage,
        specified::background::BackgroundRepeatKeyword,
    },
};

#[cfg(feature = "tracing")]
use tracing::warn;

/// A box from the CSS box model. Abstracts over the (structurally identical)
/// computed value types of the `background-clip`/`background-origin` and
/// `mask-clip`/`mask-origin` properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)] // The variants are named after the CSS keywords
pub(super) enum BoxModelBox {
    BorderBox,
    PaddingBox,
    ContentBox,
}

// Also covers MaskClip as the type is the same
impl From<StyloBackgroundClip> for BoxModelBox {
    fn from(value: StyloBackgroundClip) -> Self {
        match value {
            StyloBackgroundClip::BorderBox => Self::BorderBox,
            StyloBackgroundClip::PaddingBox => Self::PaddingBox,
            StyloBackgroundClip::ContentBox => Self::ContentBox,

            // TODO: support BorderArea
            StyloBackgroundClip::BorderArea => Self::BorderBox,
        }
    }
}

impl From<StyloBackgroundOrigin> for BoxModelBox {
    fn from(value: StyloBackgroundOrigin) -> Self {
        match value {
            StyloBackgroundOrigin::BorderBox => Self::BorderBox,
            StyloBackgroundOrigin::PaddingBox => Self::PaddingBox,
            StyloBackgroundOrigin::ContentBox => Self::ContentBox,
        }
    }
}

impl From<StyloMaskOrigin> for BoxModelBox {
    fn from(value: StyloMaskOrigin) -> Self {
        match value {
            StyloMaskOrigin::BorderBox => Self::BorderBox,
            StyloMaskOrigin::PaddingBox => Self::PaddingBox,
            StyloMaskOrigin::ContentBox => Self::ContentBox,
        }
    }
}

/// The styles and image data for a single layer of a CSS image layer list
/// (`background-image` or `mask-image`). The `background-*` and `mask-*`
/// properties share computed value types, which allows the layer painting code
/// to be shared.
pub(super) struct ImageLayerStyles<'a> {
    /// The computed value of the `background-image`/`mask-image` layer
    pub stylo_image: &'a ComputedImage,
    /// The loaded image resource if `stylo_image` is a `url()` image
    pub image_data: Option<&'a ImageResourceData>,
    pub position_x: &'a LengthPercentage,
    pub position_y: &'a LengthPercentage,
    pub repeat: &'a BackgroundRepeat,
    pub size: &'a BackgroundSize,
    pub clip: BoxModelBox,
    pub origin: BoxModelBox,
    pub attachment: StyloBackgroundAttachment,
}

impl<'a> ImageLayerStyles<'a> {
    pub(super) fn from_background(
        bg_styles: &'a Background,
        image_data: &'a [Option<ImageResourceData>],
        idx: usize,
    ) -> Self {
        Self {
            stylo_image: &bg_styles.background_image.0[idx],
            image_data: image_data.get(idx).and_then(Option::as_ref),
            position_x: get_cyclic(&bg_styles.background_position_x.0, idx),
            position_y: get_cyclic(&bg_styles.background_position_y.0, idx),
            repeat: get_cyclic(&bg_styles.background_repeat.0, idx),
            size: get_cyclic(&bg_styles.background_size.0, idx),
            clip: (*get_cyclic(&bg_styles.background_clip.0, idx)).into(),
            origin: (*get_cyclic(&bg_styles.background_origin.0, idx)).into(),
            attachment: *get_cyclic(&bg_styles.background_attachment.0, idx),
        }
    }

    pub(super) fn from_svg(
        svg_styles: &'a SVG,
        image_data: &'a [Option<ImageResourceData>],
        idx: usize,
    ) -> Self {
        Self {
            stylo_image: &svg_styles.mask_image.0[idx],
            image_data: image_data.get(idx).and_then(Option::as_ref),
            position_x: get_cyclic(&svg_styles.mask_position_x.0, idx),
            position_y: get_cyclic(&svg_styles.mask_position_y.0, idx),
            repeat: get_cyclic(&svg_styles.mask_repeat.0, idx),
            size: get_cyclic(&svg_styles.mask_size.0, idx),
            clip: (*get_cyclic(&svg_styles.mask_clip.0, idx)).into(),
            origin: (*get_cyclic(&svg_styles.mask_origin.0, idx)).into(),
            // There is no `mask-attachment` property
            attachment: StyloBackgroundAttachment::Scroll,
        }
    }
}

impl ElementCx<'_, '_> {
    pub(super) fn draw_background(&self, scene: &mut impl PaintScene) {
        let bg_styles = &self.style.get_background();
        let image_data = &self.element.background_images;
        let layer_count = bg_styles.background_image.0.len();

        // The background color is clipped by the clip of the last layer in the list
        let background_clip: BoxModelBox =
            (*get_cyclic(&bg_styles.background_clip.0, layer_count - 1)).into();
        let background_clip_path = self.box_path(background_clip);

        // Draw background color (if any)
        self.draw_solid_bg(scene, &background_clip_path);

        for idx in (0..layer_count).rev() {
            let layer = ImageLayerStyles::from_background(bg_styles, image_data, idx);
            if layer_paints_nothing(&layer) {
                continue;
            }
            let background_clip_path = self.box_path(layer.clip);

            self.context.layer_manager.maybe_with_layer(
                scene,
                true,
                1.0,
                self.transform,
                &background_clip_path,
                None,
                None,
                |scene| {
                    self.draw_image_layer(scene, &layer);
                },
            );
        }
    }

    /// The path of the given CSS box model box for this element
    pub(super) fn box_path(&self, css_box: BoxModelBox) -> BezPath {
        match css_box {
            BoxModelBox::BorderBox => self.frame.border_box_path(),
            BoxModelBox::PaddingBox => self.frame.padding_box_path(),
            BoxModelBox::ContentBox => self.frame.content_box_path(),
        }
    }

    /// The rect of the given CSS box model box for this element
    fn box_rect(&self, css_box: BoxModelBox) -> Rect {
        match css_box {
            BoxModelBox::BorderBox => self.frame.border_box,
            BoxModelBox::PaddingBox => self.frame.padding_box,
            BoxModelBox::ContentBox => self.frame.content_box,
        }
    }

    /// Draw a single layer of a CSS image layer list (`background-image` or `mask-image`)
    pub(super) fn draw_image_layer(&self, scene: &mut impl PaintScene, layer: &ImageLayerStyles) {
        match layer.stylo_image {
            GenericImage::None => {
                // Do nothing
            }
            GenericImage::Gradient(gradient) => self.draw_gradient_layer(scene, gradient, layer),
            GenericImage::Url(_) => {
                self.draw_raster_image_layer(scene, layer);
                #[cfg(feature = "svg")]
                self.draw_svg_image_layer(scene, layer);
            }
            GenericImage::LightDark(_) => {
                #[cfg(feature = "tracing")]
                warn!("Implement image layer drawing for ImageLightDark")
            }
            GenericImage::PaintWorklet(_) => {
                #[cfg(feature = "tracing")]
                warn!("Implement image layer drawing for Image::PaintWorklet")
            }
            GenericImage::CrossFade(_) => {
                #[cfg(feature = "tracing")]
                warn!("Implement image layer drawing for Image::CrossFade")
            }
            GenericImage::Image(_) => {
                #[cfg(feature = "tracing")]
                warn!("Implement image layer drawing for Image::Image")
            }
            GenericImage::ImageSet(_) => {
                #[cfg(feature = "tracing")]
                warn!("Implement image layer drawing for Image::ImageSet")
            }
        }
    }

    pub(super) fn draw_table_row_backgrounds(&self, scene: &mut impl PaintScene) {
        let SpecialElementData::TableRoot(table) = &self.element.special_data else {
            return;
        };
        let Some(grid_info) = &mut *table.computed_grid_info.borrow_mut() else {
            return;
        };

        let cols = PhysicalTracks::from_tracks(&grid_info.columns);
        let inner_width = cols.span() as f64;

        let rows = PhysicalTracks::from_tracks(&grid_info.rows);
        let row_origin = rows.origin();
        for (row, row_position) in table.rows.iter().zip(rows.iter()) {
            let row_node = &self.context.dom.get_node(row.node_id).unwrap();
            let Some(style) = row_node.primary_styles() else {
                continue;
            };

            let y = (row_position.start - row_origin) as f64;
            let height = (row_position.end - row_position.start) as f64;
            let shape = Rect::new(0.0, y, inner_width, y + height).scale_from_origin(self.scale);

            let current_color = style.clone_color();
            let background_color = &style.get_background().background_color;
            let bg_color = background_color
                .resolve_to_absolute(&current_color)
                .as_srgb_color();

            if bg_color != Color::TRANSPARENT {
                // Fill the color
                scene.fill(Fill::NonZero, self.transform, bg_color, None, &shape);
            }
        }
    }

    fn draw_solid_bg(&self, scene: &mut impl PaintScene, shape: &BezPath) {
        let current_color = self.style.clone_color();
        let background_color = &self.style.get_background().background_color;
        let bg_color = background_color
            .resolve_to_absolute(&current_color)
            .as_srgb_color();

        if bg_color != Color::TRANSPARENT {
            // Fill the color
            scene.fill(Fill::NonZero, self.transform, bg_color, None, shape);
        }
    }

    /// Whether the layer is positioned against the viewport
    /// (`background-attachment: fixed`) rather than the element's origin box.
    /// `fixed` behaves as `scroll` on elements affected by a CSS transform
    /// (the transformed element acts as the layer's containing block).
    fn layer_is_fixed(&self, layer: &ImageLayerStyles) -> bool {
        layer.attachment == StyloBackgroundAttachment::Fixed && !self.is_transformed()
    }

    /// The background positioning area and the transform from its coordinate
    /// space to the scene for a fixed layer: the viewport, unaffected by any
    /// scrolling.
    fn fixed_positioning_area(&self) -> (Rect, Affine) {
        let viewport_rect = Rect::new(
            0.0,
            0.0,
            self.context.width as f64,
            self.context.height as f64,
        );
        let transform = Affine::translate((self.context.initial_x, self.context.initial_y));
        (viewport_rect, transform)
    }

    /// The render surface, in the coordinates a fill's transform maps into.
    ///
    /// `paint_scene` offsets the whole document by `initial_x`/`initial_y`, so the surface
    /// occupies that offset plus the viewport size in this space -- the same correction
    /// `render_element` makes before its own cull check.
    fn surface_rect(&self) -> Rect {
        Rect::from_origin_size(
            (self.context.initial_x, self.context.initial_y),
            (self.context.width as f64, self.context.height as f64),
        )
    }

    /// Whether this element or any of its ancestors has a CSS transform
    ///
    /// TODO: this misses transformed elements whose resolved transform is the
    /// identity (e.g. `transform: translate(0)`), and elements with
    /// `will-change: transform`, both of which should also degrade `fixed`
    /// to `scroll` (see WPT css/css-transforms/transform-fixed-bg-005/008)
    fn is_transformed(&self) -> bool {
        let mut current = Some(self.node.id);
        while let Some(node) = current.and_then(|id| self.context.dom.get_node(id)) {
            if node.transform().is_some() {
                return true;
            }
            current = node.parent;
        }
        false
    }

    #[cfg(feature = "svg")]
    fn draw_svg_image_layer(&self, scene: &mut impl PaintScene, layer: &ImageLayerStyles) {
        let Some(bg_image) = layer.image_data else {
            return;
        };
        let ImageData::Svg(svg) = &bg_image.image else {
            return;
        };

        // A zero-sized `viewBox` disables rendering of the SVG
        if svg.intrinsic_dimensions.degenerate_view_box {
            return;
        }

        // For a fixed layer the positioning area (the viewport) already covers
        // everything visible, so it also serves as the clip rect (no extension
        // towards the clip box is needed).
        let (origin_rect, base_transform, clip_rect) = if self.layer_is_fixed(layer) {
            let (viewport_rect, transform) = self.fixed_positioning_area();
            (viewport_rect, transform, viewport_rect)
        } else {
            (
                self.box_rect(layer.origin),
                self.transform,
                self.box_rect(layer.clip),
            )
        };

        let svg_size = svg.tree.size();

        // Size the SVG per the CSS default sizing algorithm
        // (https://drafts.csswg.org/css-images/#default-sizing). An SVG image
        // may lack an intrinsic width, height, and/or aspect ratio, so each is
        // passed separately (usvg's resolved `Tree::size` is only used as the
        // source coordinate space of the rendered tree).
        let intrinsic_width = svg.intrinsic_width().filter(|w| w.is_finite() && *w > 0.0);
        let intrinsic_height = svg.intrinsic_height().filter(|h| h.is_finite() && *h > 0.0);
        let aspect_ratio = match (intrinsic_width, intrinsic_height) {
            (Some(w), Some(h)) => Some(w / h),
            _ => svg.viewbox_aspect_ratio(),
        }
        .filter(|r| r.is_finite() && *r > 0.0);

        let (bg_pos, bg_size) = compute_layer_position_and_size(
            layer,
            origin_rect.width() / self.scale,
            origin_rect.height() / self.scale,
            BackgroundSizeComputeMode::Intrinsic {
                width: intrinsic_width,
                height: intrinsic_height,
                ratio: aspect_ratio,
            },
        );

        let bg_pos = (bg_pos.to_vec2() * self.scale).to_point();
        let bg_size = bg_size * self.scale;

        // css-backgrounds-3 s3.9: if either dimension of the computed `background-size` is
        // zero, the image is not rendered. Returning here also keeps the tiling arithmetic
        // below away from a division by zero.
        if bg_size.width <= 0.0 || bg_size.height <= 0.0 {
            return;
        }

        let BackgroundRepeat(repeat_x, repeat_y) = layer.repeat;

        // An SVG layer is drawn by replaying its tree into the scene, so -- exactly like a
        // gradient, and unlike a raster image -- it has no brush whose own `Extend::Repeat`
        // could tile it in a single fill. `Repeat`/`Round` therefore need explicit tiles,
        // which is what `gradient_axis_tiling` produces; reusing it also gives this layer
        // kind the same near-zero `background-size` handling (a tile thinner than a device
        // pixel is widened to one, instead of asking for an unbounded number of replays).
        let mut x = gradient_axis_tiling(
            *repeat_x,
            origin_rect.x0,
            origin_rect.width(),
            clip_rect.x0,
            clip_rect.width(),
            bg_pos.x,
            bg_size.width,
        );
        let mut y = gradient_axis_tiling(
            *repeat_y,
            origin_rect.y0,
            origin_rect.height(),
            clip_rect.y0,
            clip_rect.height(),
            bg_pos.y,
            bg_size.height,
        );

        // Scale the tree to the tile actually being drawn, not to `bg_size`: the two differ
        // only where `gradient_axis_tiling` widened a sub-device-pixel tile, and there the
        // image has to grow with it or the lattice would leave gaps between the replays.
        let tile_rect = Rect::new(0.0, 0.0, x.rect_len, y.rect_len);
        let svg_transform = Affine::scale_non_uniform(
            tile_rect.width() / svg_size.width() as f64,
            tile_rect.height() / svg_size.height() as f64,
        );

        // Drop the tiles that cannot land on the render surface, so the replay count follows
        // the visible pixels rather than the element's own extent.
        //
        // The per-tile offset here is applied *before* `base_transform` (`pre_translate`), so
        // that the lattice scales and rotates with the element the way CSS requires, rather
        // than stepping along the surface axes. The bound has to be expressed in that same
        // space: the bounding box of the surface pulled back through `base_transform` contains
        // every layer-space point that can map onto the surface, so culling against it can
        // only ever drop a tile that is genuinely off-surface. A singular `base_transform`
        // makes that pullback non-finite, which `cull_axis_to_surface` reads as "do not cull".
        let surface = base_transform
            .inverse()
            .transform_rect_bbox(self.surface_rect());
        cull_axis_to_surface(&mut x, tile_rect.x0, tile_rect.x1, surface.x0, surface.x1);
        cull_axis_to_surface(&mut y, tile_rect.y0, tile_rect.y1, surface.y0, surface.y1);

        let placed = base_transform.pre_translate(Vec2 {
            x: x.translate,
            y: y.translate,
        });

        // Record the tile's vector scene once, then replay the recording per tile.
        //
        // `render_svg_tree` walks the usvg tree and rebuilds a `BezPath` for every path it
        // finds (`anyrender_svg`'s `util::to_bez_path`), so calling it once per tile makes a
        // full-screen 4x4 stipple pay ~57,600 tree walks and ~115,200 path allocations for a
        // frame whose visible result is two distinct 1x1 rects. `anyrender::Scene` implements
        // `PaintScene` itself, so the walk can be done once into a recording and the resulting
        // commands re-issued with each tile's transform, holding the paths by reference.
        //
        // This is byte-identical by construction rather than by measurement, and the grouping
        // is what makes it so. `render_svg_tree_with` calls `render_group` with
        // `Affine::IDENTITY` as the local transform and the caller's transform as
        // `global_transform`, and emits each path at `global_transform * local`. Recording with
        // `Affine::IDENTITY` as the global transform therefore stores exactly `local`, and
        // replaying at `tile_transform * local` reproduces the same product with the same
        // association -- which matters because f64 matrix multiplication is not associative,
        // so `(A * B) * C` and `A * (B * C)` can differ in the last bits and move a pixel.
        let mut tile_scene = Scene::new();
        anyrender_svg::render_svg_tree(&mut tile_scene, &svg.tree, Affine::IDENTITY);

        // `anyrender_svg` emits only fills, strokes and layers, with solid or gradient paints.
        // Anything else means it gained a command kind this replay does not reproduce
        // faithfully -- a glyph run, or a paint `Scene` records lossily -- so fall back to
        // walking the tree per tile, which is by definition the old behaviour.
        let replayable = tile_scene.commands.iter().all(|command| {
            matches!(
                command,
                RenderCommand::PushLayer(_)
                    | RenderCommand::PushClipLayer(_)
                    | RenderCommand::PopLayer
                    | RenderCommand::Fill(_)
                    | RenderCommand::Stroke(_)
            )
        });

        for hc in 0..y.count {
            for wc in 0..x.count {
                let transform = placed.pre_translate(Vec2 {
                    x: wc as f64 * x.stride,
                    y: hc as f64 * y.stride,
                }) * svg_transform;

                if replayable {
                    replay_recorded_tile(scene, &tile_scene, transform);
                } else {
                    anyrender_svg::render_svg_tree(scene, &svg.tree, transform);
                }
            }
        }
    }

    fn draw_raster_image_layer(&self, scene: &mut impl PaintScene, layer: &ImageLayerStyles) {
        let Some(bg_image) = layer.image_data else {
            return;
        };
        let ImageData::Raster(image_data) = &bg_image.image else {
            return;
        };

        let image_rendering = self.style.clone_image_rendering();
        let quality = to_image_quality(image_rendering);

        let (origin_rect, base_transform) = if self.layer_is_fixed(layer) {
            self.fixed_positioning_area()
        } else {
            (self.box_rect(layer.origin), self.transform)
        };

        let image_width = image_data.width as f64;
        let image_height = image_data.height as f64;

        let (bg_pos, bg_size) = compute_layer_position_and_size(
            layer,
            origin_rect.width() / self.scale,
            origin_rect.height() / self.scale,
            BackgroundSizeComputeMode::Size(image_width as f32, image_height as f32),
        );

        let bg_pos = (bg_pos.to_vec2() * self.scale).to_point();
        let bg_size = bg_size * self.scale;

        let x_ratio = bg_size.width / image_width;
        let y_ratio = bg_size.height / image_height;

        let BackgroundRepeat(repeat_x, repeat_y) = layer.repeat;

        let x = raster_axis_tiling(
            *repeat_x,
            origin_rect.x0,
            origin_rect.width(),
            bg_pos.x,
            bg_size.width,
            image_width,
            x_ratio,
        );
        let y = raster_axis_tiling(
            *repeat_y,
            origin_rect.y0,
            origin_rect.height(),
            bg_pos.y,
            bg_size.height,
            image_height,
            y_ratio,
        );

        // `translate` and `stride` are lengths in the element's own coordinate space, so they
        // are applied *before* `base_transform` and before the image scale: the fill rect is
        // in image pixels, `image_transform` takes it to element-local pixels, and only then
        // is the tile positioned. Adding the placement to the transform's output instead
        // (`then_translate`) steps the layer along the surface axes, which is a different
        // point as soon as the element carries a transform whose linear part is not the
        // identity -- and steps a `Space` lattice at the unscaled stride.
        let placed = base_transform.pre_translate(Vec2 {
            x: x.translate,
            y: y.translate,
        });
        let image_transform = Affine::scale_non_uniform(x_ratio, y_ratio);
        let tile_rect = Rect::new(0.0, 0.0, x.rect_len, y.rect_len);

        for hc in 0..y.count {
            for wc in 0..x.count {
                let transform = placed.pre_translate(Vec2 {
                    x: wc as f64 * x.stride,
                    y: hc as f64 * y.stride,
                }) * image_transform;

                scene.fill(
                    peniko::Fill::NonZero,
                    transform,
                    to_peniko_image(image_data, quality).as_ref(),
                    None,
                    &tile_rect,
                );
            }
        }
    }

    fn draw_gradient_layer(
        &self,
        scene: &mut impl PaintScene,
        gradient: &StyloGradient,
        layer: &ImageLayerStyles,
    ) {
        // For a fixed layer the positioning area (the viewport) already covers
        // everything visible, so it also serves as the clip rect (no extension
        // towards the clip box is needed).
        let (origin_rect, base_transform, clip_rect) = if self.layer_is_fixed(layer) {
            let (viewport_rect, transform) = self.fixed_positioning_area();
            (viewport_rect, transform, viewport_rect)
        } else {
            (
                self.box_rect(layer.origin),
                self.transform,
                self.box_rect(layer.clip),
            )
        };

        let (bg_pos, bg_size) = compute_layer_position_and_size(
            layer,
            origin_rect.width() / self.scale,
            origin_rect.height() / self.scale,
            BackgroundSizeComputeMode::Auto,
        );

        let bg_pos = (bg_pos.to_vec2() * self.scale).to_point();
        let bg_size = bg_size * self.scale;

        // css-backgrounds-3 s3.9: if either dimension of the computed `background-size` is
        // zero, the image is not rendered. Returning here also keeps the tiling arithmetic
        // below away from a division by zero.
        if bg_size.width <= 0.0 || bg_size.height <= 0.0 {
            return;
        }

        let BackgroundRepeat(repeat_x, repeat_y) = layer.repeat;

        let mut x = gradient_axis_tiling(
            *repeat_x,
            origin_rect.x0,
            origin_rect.width(),
            clip_rect.x0,
            clip_rect.width(),
            bg_pos.x,
            bg_size.width,
        );
        let mut y = gradient_axis_tiling(
            *repeat_y,
            origin_rect.y0,
            origin_rect.height(),
            clip_rect.y0,
            clip_rect.height(),
            bg_pos.y,
            bg_size.height,
        );
        let tile_rect = Rect::new(0.0, 0.0, x.rect_len, y.rect_len);

        // Drop the tiles that cannot land on the render surface. Without this the tile count
        // scales with the element's own extent, so a tall page costs time proportional to its
        // height rather than to the pixels actually being produced.
        //
        // The per-tile offset below is applied *before* `base_transform` (`pre_translate`), so
        // the lattice scales, rotates and skews with the element the way CSS requires. The
        // bound has to be expressed in that same space: the bounding box of the surface pulled
        // back through `base_transform` contains every layer-space point that can map onto the
        // surface, so culling against it can only ever drop a tile that is genuinely
        // off-surface. A singular `base_transform` makes that pullback non-finite, which
        // `cull_axis_to_surface` reads as "do not cull".
        let surface = base_transform
            .inverse()
            .transform_rect_bbox(self.surface_rect());
        cull_axis_to_surface(&mut x, tile_rect.x0, tile_rect.x1, surface.x0, surface.x1);
        cull_axis_to_surface(&mut y, tile_rect.y0, tile_rect.y1, surface.y0, surface.y1);

        let current_color = self.style.clone_color();

        let (gradient, gradient_transform) =
            to_peniko_gradient(gradient, tile_rect, self.scale, &current_color);
        let brush = anyrender::Paint::Gradient(&gradient);

        // The layer's placement and its tile lattice are both expressed in the element's own
        // coordinate space (`origin_rect` is the element-local box and `stride` a length in
        // it), so both are applied *before* `base_transform`. Adding them to the transform's
        // output instead -- `then_translate` -- steps the lattice along the surface axes while
        // each tile still carries the transform's linear part, so under `scale(s)` the pattern
        // advances `tile_len` surface pixels per tile while each tile is `s * tile_len` wide
        // and the background covers only the element's unscaled extent.
        let placed = base_transform.pre_translate(Vec2 {
            x: x.translate,
            y: y.translate,
        });

        for hc in 0..y.count {
            for wc in 0..x.count {
                let transform = placed.pre_translate(Vec2 {
                    x: wc as f64 * x.stride,
                    y: hc as f64 * y.stride,
                });

                scene.fill(
                    peniko::Fill::NonZero,
                    transform,
                    brush.clone(),
                    gradient_transform,
                    &tile_rect,
                );
            }
        }
    }
}

/// Re-issue a recorded tile scene into `scene`, with `transform` applied ahead of each
/// command's own.
///
/// The recording is held by reference throughout: `PaintScene`'s methods take the shape as
/// `&impl Shape` and the paint as `impl Into<PaintRef<'_>>`, so replaying a tile costs a
/// matrix multiply and a call per command, with no path rebuilt and no brush cloned.
///
/// Only the command kinds `anyrender_svg` emits are handled; the caller checks for anything
/// else once, before the tile loop, and falls back to walking the tree per tile.
#[cfg(feature = "svg")]
fn replay_recorded_tile(scene: &mut impl PaintScene, tile: &Scene, transform: Affine) {
    for command in &tile.commands {
        match command {
            RenderCommand::PushLayer(cmd) => scene.push_layer(
                cmd.blend,
                cmd.alpha,
                transform * cmd.transform,
                &cmd.clip,
                cmd.filter.clone(),
                cmd.backdrop_filter.clone(),
            ),
            RenderCommand::PushClipLayer(cmd) => {
                scene.push_clip_layer(transform * cmd.transform, &cmd.clip);
            }
            RenderCommand::PopLayer => scene.pop_layer(),
            RenderCommand::Fill(cmd) => scene.fill(
                cmd.fill,
                transform * cmd.transform,
                &cmd.brush,
                cmd.brush_transform,
                &cmd.shape,
            ),
            RenderCommand::Stroke(cmd) => scene.stroke(
                &cmd.style,
                transform * cmd.transform,
                &cmd.brush,
                cmd.brush_transform,
                &cmd.shape,
            ),
            // Filtered out before the tile loop; see `draw_svg_image_layer`.
            RenderCommand::GlyphRun(_) | RenderCommand::BoxShadow(_) => {}
        }
    }
}

fn compute_layer_position_and_size(
    layer: &ImageLayerStyles,
    container_w: f64,
    container_h: f64,
    size_mode: BackgroundSizeComputeMode,
) -> (Point, Size) {
    use BackgroundRepeatKeyword::*;

    let bg_size = compute_layer_size(layer, container_w as f32, container_h as f32, size_mode);

    let bg_pos = compute_layer_position(
        layer,
        (container_w - bg_size.width) as f32,
        (container_h - bg_size.height) as f32,
    );

    let BackgroundRepeat(repeat_x, repeat_y) = layer.repeat;

    let bg_size = if matches!(repeat_x, Round) && matches!(repeat_y, Round) {
        let count = (container_w / bg_size.width).round();
        let width = container_w / count;

        let count = (container_h / bg_size.height).round();
        let height = container_h / count;

        Size::new(width, height)
    } else if matches!(repeat_x, Round) {
        let count = (container_w / bg_size.width).round();
        let width = container_w / count;
        Size::new(width, bg_size.height)
    } else if matches!(repeat_y, Round) {
        let count = (container_h / bg_size.height).round();
        let height = container_h / count;
        Size::new(bg_size.width, height)
    } else {
        bg_size
    };

    (bg_pos, bg_size)
}

#[inline]
fn compute_layer_position(layer: &ImageLayerStyles, width: f32, height: f32) -> Point {
    use style::values::computed::Length;

    let bg_pos_x = layer.position_x.resolve(Length::new(width)).px() as f64;
    let bg_pos_y = layer.position_y.resolve(Length::new(height)).px() as f64;

    Point::new(bg_pos_x, bg_pos_y)
}

fn compute_layer_size(
    layer: &ImageLayerStyles,
    container_w: f32,
    container_h: f32,
    mode: BackgroundSizeComputeMode,
) -> kurbo::Size {
    use style::values::computed::Length;
    use style::values::generics::length::GenericLengthPercentageOrAuto as Lpa;

    let (width, height): (f32, f32) = match layer.size {
        BackgroundSize::ExplicitSize { width, height } => {
            let width = width.map(|w| w.0.resolve(Length::new(container_w)));
            let height = height.map(|h| h.0.resolve(Length::new(container_h)));

            match (width, height) {
                (Lpa::LengthPercentage(width), Lpa::LengthPercentage(height)) => {
                    let width = width.px();
                    let height = height.px();
                    match mode {
                        BackgroundSizeComputeMode::Auto => (width, height),
                        BackgroundSizeComputeMode::Size(_, _) => (width, height),
                        BackgroundSizeComputeMode::Intrinsic { .. } => (width, height),
                    }
                }
                (Lpa::LengthPercentage(width), Lpa::Auto) => {
                    let width = width.px();
                    let height = match mode {
                        BackgroundSizeComputeMode::Auto => container_h,
                        BackgroundSizeComputeMode::Size(bg_w, bg_h) => bg_h / bg_w * width,
                        BackgroundSizeComputeMode::Intrinsic { height, ratio, .. } => ratio
                            .map(|ratio| width / ratio)
                            .or(height)
                            .unwrap_or(container_h),
                    };
                    (width, height)
                }
                (Lpa::Auto, Lpa::LengthPercentage(height)) => {
                    let height = height.px();
                    let width = match mode {
                        BackgroundSizeComputeMode::Auto => container_w,
                        BackgroundSizeComputeMode::Size(bg_w, bg_h) => bg_w / bg_h * height,
                        BackgroundSizeComputeMode::Intrinsic { width, ratio, .. } => ratio
                            .map(|ratio| height * ratio)
                            .or(width)
                            .unwrap_or(container_w),
                    };
                    (width, height)
                }
                (Lpa::Auto, Lpa::Auto) => match mode {
                    BackgroundSizeComputeMode::Auto => (container_w, container_h),
                    BackgroundSizeComputeMode::Size(bg_w, bg_h) => (bg_w, bg_h),
                    BackgroundSizeComputeMode::Intrinsic {
                        width,
                        height,
                        ratio,
                    } => default_sizing(width, height, ratio, container_w, container_h),
                },
            }
        }
        BackgroundSize::Cover => match mode {
            BackgroundSizeComputeMode::Auto => (container_w, container_h),
            BackgroundSizeComputeMode::Size(bg_w, bg_h) => {
                // Scale to the smallest size that covers both axes
                let ratio = (container_w / bg_w).max(container_h / bg_h);
                (bg_w * ratio, bg_h * ratio)
            }
            BackgroundSizeComputeMode::Intrinsic { ratio, .. } => match ratio {
                // Scale the aspect ratio to the smallest size that covers both axes
                Some(ratio) => {
                    let scale = (container_w / ratio).max(container_h);
                    (scale * ratio, scale)
                }
                // No intrinsic aspect ratio: fill the positioning area
                None => (container_w, container_h),
            },
        },
        BackgroundSize::Contain => match mode {
            BackgroundSizeComputeMode::Auto => (container_w, container_h),
            BackgroundSizeComputeMode::Size(bg_w, bg_h) => {
                // Scale to the largest size contained by both axes
                let ratio = (container_w / bg_w).min(container_h / bg_h);
                (bg_w * ratio, bg_h * ratio)
            }
            BackgroundSizeComputeMode::Intrinsic { ratio, .. } => match ratio {
                // Scale the aspect ratio to the largest size contained by both axes
                Some(ratio) => {
                    let scale = (container_w / ratio).min(container_h);
                    (scale * ratio, scale)
                }
                // No intrinsic aspect ratio: fill the positioning area
                None => (container_w, container_h),
            },
        },
    };

    kurbo::Size {
        width: width as f64,
        height: height as f64,
    }
}

enum BackgroundSizeComputeMode {
    Auto,
    Size(f32, f32),
    /// Intrinsic dimensions of an image which may lack an intrinsic width,
    /// height, and/or aspect ratio (e.g. SVG), sized per the CSS default
    /// sizing algorithm (https://drafts.csswg.org/css-images/#default-sizing)
    Intrinsic {
        width: Option<f32>,
        height: Option<f32>,
        ratio: Option<f32>,
    },
}

/// The CSS default sizing algorithm for the unconstrained (`auto auto`) case:
/// resolve the concrete object size from whichever intrinsic dimensions exist,
/// falling back to the default object size (the background positioning area).
fn default_sizing(
    width: Option<f32>,
    height: Option<f32>,
    ratio: Option<f32>,
    container_w: f32,
    container_h: f32,
) -> (f32, f32) {
    match (width, height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, ratio.map(|r| w / r).unwrap_or(container_h)),
        (None, Some(h)) => (ratio.map(|r| h * r).unwrap_or(container_w), h),
        (None, None) => match ratio {
            // Intrinsic aspect ratio only: size as if `contain` were specified
            Some(ratio) => {
                let scale = (container_w / ratio).min(container_h);
                (scale * ratio, scale)
            }
            None => (container_w, container_h),
        },
    }
}

/// The placement and tiling of a background layer along one axis: a
/// translation applied to the layer as a whole, the length of each filled
/// rect, and the number of explicit tiles with the stride between them.
struct AxisTiling {
    /// Translation (in device pixels) positioning the first tile
    translate: f64,
    /// Length of each filled rect, in the coordinate space of the fill's transform
    rect_len: f64,
    /// Number of explicitly drawn tiles
    count: u32,
    /// Stride (in device pixels) between the starts of consecutive tiles
    stride: f64,
}

/// Per-axis placement and tiling for a raster image layer. `Repeat`/`Round`
/// produce a single fill covering the whole positioning area (relying on the
/// image brush repeating), while `Space` produces `count` explicit tiles
/// spaced `stride` apart.
///
/// The fill rect is in image pixel coordinates (the drawing transform is
/// pre-scaled by `ratio`), while translations are in device pixels.
fn raster_axis_tiling(
    repeat: BackgroundRepeatKeyword,
    origin_start: f64,
    origin_len: f64,
    bg_pos: f64,
    tile_len: f64,
    image_len: f64,
    ratio: f64,
) -> AxisTiling {
    use BackgroundRepeatKeyword::*;

    match repeat {
        Repeat | Round => {
            let extend_len = extend(bg_pos, tile_len);
            AxisTiling {
                translate: origin_start - extend_len,
                rect_len: (origin_len + extend_len) / ratio,
                count: 1,
                stride: 0.0,
            }
        }
        Space => {
            let (count, stride) = compute_space_count_and_stride(origin_len, tile_len);
            AxisTiling {
                translate: origin_start + if count == 1 { bg_pos } else { 0.0 },
                rect_len: image_len,
                count,
                stride,
            }
        }
        NoRepeat => AxisTiling {
            translate: origin_start + bg_pos,
            rect_len: image_len,
            count: 1,
            stride: 0.0,
        },
    }
}

/// Narrow one axis of a tiling to the tiles that can reach the render surface.
///
/// `tile_lo`/`tile_hi` are the axis extent of one tile and `surface_lo`/`surface_hi` the
/// surface, both in the space the per-tile offset is applied in, so tile `i` is visible iff
/// `tile_lo + translate + i * stride < surface_hi` and
/// `tile_hi + translate + i * stride > surface_lo`. Solving for `i` and keeping one extra
/// tile at each end makes this conservative by construction: it can only ever drop a tile
/// whose own bounding box misses the surface entirely, so it cannot change a rendered pixel.
///
/// Anything non-finite disables culling rather than risking a saturating cast to zero.
fn cull_axis_to_surface(
    tiling: &mut AxisTiling,
    tile_lo: f64,
    tile_hi: f64,
    surface_lo: f64,
    surface_hi: f64,
) {
    if tiling.count <= 1 || tiling.stride <= 0.0 {
        return;
    }
    let bounds = [
        tile_lo,
        tile_hi,
        surface_lo,
        surface_hi,
        tiling.translate,
        tiling.stride,
    ];
    if bounds.iter().any(|value| !value.is_finite()) {
        return;
    }

    let first = ((surface_lo - tile_hi - tiling.translate) / tiling.stride).floor() - 1.0;
    let last = ((surface_hi - tile_lo - tiling.translate) / tiling.stride).ceil() + 1.0;
    let max_index = f64::from(tiling.count - 1);
    if last < 0.0 || first > max_index {
        tiling.count = 0;
        return;
    }

    let first = first.clamp(0.0, max_index);
    let last = last.clamp(first, max_index);
    tiling.translate += first * tiling.stride;
    tiling.count = (last - first) as u32 + 1;
}

/// Per-axis placement and tiling for a gradient layer. Unlike raster images,
/// gradients cannot rely on brush repetition, so `Repeat`/`Round` also produce
/// explicit tiles. When the clip box extends beyond the origin box, tiling
/// starts from the clip box edge so the pattern covers the whole clipped area.
#[allow(clippy::too_many_arguments)]
fn gradient_axis_tiling(
    repeat: BackgroundRepeatKeyword,
    origin_start: f64,
    origin_len: f64,
    clip_start: f64,
    clip_len: f64,
    bg_pos: f64,
    tile_len: f64,
) -> AxisTiling {
    use BackgroundRepeatKeyword::*;

    // A repeated tile narrower than one device pixel resolves none of its own detail, so
    // tiling it at its exact size only multiplies the fill count without changing a pixel:
    // `background-size: 0.2px` over a 100px axis asks for 500 fills, and a near-zero size
    // asks for an unbounded number. Widening such a tile to a single device pixel is
    // visually equivalent, and together with `cull_axis_to_surface` it bounds the tile
    // count at one fill per device pixel of the render surface. `NoRepeat` is a single fill
    // whatever its size, so it keeps its exact geometry.
    let repeated_tile_len = tile_len.max(1.0);

    match repeat {
        Repeat | Round => {
            let tile_len = repeated_tile_len;
            // The clip and origin boxes are nested, so the clip box extends
            // beyond the origin box iff it does so at either end
            let clip_is_outer =
                clip_start < origin_start || clip_start + clip_len > origin_start + origin_len;
            let (area_start, area_len) = if clip_is_outer {
                (clip_start, clip_len)
            } else {
                (origin_start, origin_len)
            };
            let extend_len = extend((origin_start - area_start) + bg_pos, tile_len);
            let count = ((area_len + extend_len) / tile_len).ceil() as u32;
            AxisTiling {
                translate: area_start - extend_len,
                rect_len: tile_len,
                count,
                stride: tile_len,
            }
        }
        Space => {
            let tile_len = repeated_tile_len;
            let (count, stride) = compute_space_count_and_stride(origin_len, tile_len);
            AxisTiling {
                translate: origin_start + if count == 1 { bg_pos } else { 0.0 },
                rect_len: tile_len,
                count,
                stride,
            }
        }
        NoRepeat => AxisTiling {
            translate: origin_start + bg_pos,
            rect_len: tile_len,
            count: 1,
            stride: 0.0,
        },
    }
}

fn compute_space_count_and_stride(bg_size: f64, size: f64) -> (u32, f64) {
    let modulo = bg_size % size;
    let count = (((bg_size - modulo) / size) as u32).max(1);
    let stride = if count > 1 {
        modulo / (count - 1) as f64
    } else {
        0.0
    } + size;

    (count, stride)
}

/// Whether drawing the layer is guaranteed to paint nothing, making its clip
/// layer unnecessary
fn layer_paints_nothing(layer: &ImageLayerStyles) -> bool {
    match layer.stylo_image {
        GenericImage::None => true,
        GenericImage::Url(_) => layer.image_data.is_none(),
        _ => false,
    }
}

#[inline]
pub(super) fn get_cyclic<T>(values: &[T], layer_index: usize) -> &T {
    &values[layer_index % values.len()]
}

fn extend(offset: f64, length: f64) -> f64 {
    let extend_length = offset % length;
    if extend_length > 0.0 {
        length - extend_length
    } else {
        -extend_length
    }
}
