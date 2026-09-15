use std::sync::{Arc, Mutex};

use crate::stylo_to_parley;
use app_units::Au;
use parley::{FontContext, FontVariation};
use skrifa::MetadataProvider as _;
use skrifa::charmap::Charmap;
use style::properties::ComputedValues;
use style::properties::style_structs::Font as FontStyles;
use style::{
    device::servo::FontMetricsProvider,
    font_metrics::FontMetrics,
    values::computed::{CSSPixelLength, font::LineHeight, font::QueryFontMetricsFlags},
};

#[derive(Clone)]
pub(crate) struct BlitzFontMetricsProvider {
    pub(crate) font_ctx: Arc<Mutex<FontContext>>,
}

impl core::fmt::Debug for BlitzFontMetricsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlitzFontMetricsProvider")
    }
}

/// Resolve `line-height: normal` for `font_styles`, the way Blink does.
///
/// Blink takes the *primary* font's ascent and descent, rounds each to a whole CSS pixel
/// (`FontMetrics::AscentDescentWithHacks` -> `SkScalarRoundToScalar`), and uses their sum
/// as the strut height (`FontMetrics::FloatHeight()`; the line gap is deliberately
/// excluded). Gecko and WebKit resolve `normal` from font metrics too.
///
/// Approximating `normal` as a fixed multiple of the font size is off by up to a pixel per
/// line box for fonts whose ascent + descent happens to be near that multiple, and off
/// outright for fonts where it is not (DotGothic16, for instance, is 1.448em). The error
/// accumulates down a block layout.
///
/// Returns `None` when no font in the family list resolves to a real face, in which case
/// the caller keeps its previous approximation.
fn normal_line_height(
    font_ctx: &mut FontContext,
    font_styles: &FontStyles,
    font_size: f32,
) -> Option<f32> {
    use parley::fontique::{Attributes, QueryFont, QueryStatus};
    use skrifa::instance::{LocationRef, Size};
    use skrifa::metrics::Metrics;

    // Explicit reborrow required for the borrow checker, as in `query_font_metrics`.
    let font_ctx = &mut *font_ctx;
    let mut query = font_ctx.collection.query(&mut font_ctx.source_cache);
    query.set_families(
        font_styles
            .font_family
            .families
            .iter()
            .map(stylo_to_parley::query_font_family),
    );
    query.set_attributes(Attributes {
        width: stylo_to_parley::font_width(font_styles.font_stretch),
        weight: stylo_to_parley::font_weight(font_styles.font_weight),
        style: stylo_to_parley::font_style(font_styles.font_style),
    });

    // The strut comes from the primary font: the first family that resolves to a real
    // face, independent of which face ends up covering any particular glyph.
    let mut primary: Option<QueryFont> = None;
    query.matches_with(|font| {
        primary = Some(font.clone());
        QueryStatus::Stop
    });
    let primary = primary?;

    let font_ref = skrifa::FontRef::from_index(primary.blob.as_ref(), primary.index).ok()?;
    let variations = stylo_to_parley::font_variations(&font_styles.font_variation_settings);
    let location = font_ref.axes().location(
        variations
            .iter()
            .map(|v| (skrifa::Tag::from_be_bytes(v.tag.to_bytes()), v.value)),
    );
    let metrics = Metrics::new(
        &font_ref,
        Size::new(font_size),
        LocationRef::from(&location),
    );

    // `Metrics::descent` points down the y axis, so it is negative.
    Some(metrics.ascent.round() + (-metrics.descent).round())
}

/// Resolve `line-height: normal` for `style`, if that is what `line-height` computes to.
///
/// Returns `None` for every other `line-height` value: those need no font metrics.
pub(crate) fn resolve_normal_line_height(
    font_ctx: &mut FontContext,
    style: &ComputedValues,
) -> Option<f32> {
    let font_styles = style.get_font();
    if !matches!(font_styles.line_height, LineHeight::Normal) {
        return None;
    }
    let font_size = font_styles.font_size.used_size.0.px();
    normal_line_height(font_ctx, font_styles, font_size)
}

impl FontMetricsProvider for BlitzFontMetricsProvider {
    fn query_font_metrics(
        &self,
        _vertical: bool,
        font_styles: &FontStyles,
        font_size: CSSPixelLength,
        _flags: QueryFontMetricsFlags,
    ) -> FontMetrics {
        use parley::fontique::{Attributes, Query, QueryFont, QueryStatus};
        use skrifa::instance::{LocationRef, Size};
        use skrifa::metrics::{GlyphMetrics, Metrics};

        // Lock font_ctx. Explicit reborrow required for borrow checker.
        let mut font_ctx = self.font_ctx.lock().unwrap();
        let font_ctx = &mut *font_ctx;

        // Query fontique for the font that matches the font styles
        let mut query = font_ctx.collection.query(&mut font_ctx.source_cache);
        let families = font_styles
            .font_family
            .families
            .iter()
            .map(stylo_to_parley::query_font_family);
        query.set_families(families);
        query.set_attributes(Attributes {
            width: stylo_to_parley::font_width(font_styles.font_stretch),
            weight: stylo_to_parley::font_weight(font_styles.font_weight),
            style: stylo_to_parley::font_style(font_styles.font_style),
        });
        // let fb_script = crate::swash_convert::script_to_fontique(script);
        // let fb_language = locale.and_then(crate::swash_convert::locale_to_fontique);
        // query.set_fallbacks(fontique::FallbackKey::new(fb_script, fb_language.as_ref()));

        let variations = stylo_to_parley::font_variations(&font_styles.font_variation_settings);
        // let features = self.rcx.features(style.font_features).unwrap_or(&[]);

        // fn name_of(font_ref: &skrifa::FontRef) -> String {
        //     use skrifa::string::StringId;
        //     font_ref
        //         .localized_strings(StringId::POSTSCRIPT_NAME)
        //         .english_or_first()
        //         .unwrap()
        //         .chars()
        //         .collect()
        // }

        fn find_font_for(query: &mut Query, ch: char) -> Option<QueryFont> {
            let mut font = None;
            query.matches_with(|q_font: &QueryFont| {
                use skrifa::MetadataProvider;

                let Ok(font_ref) = skrifa::FontRef::from_index(q_font.blob.as_ref(), q_font.index)
                else {
                    return QueryStatus::Continue;
                };

                let charmap = font_ref.charmap();
                if charmap.map(ch).is_some() {
                    font = Some(q_font.clone());
                    QueryStatus::Stop
                } else {
                    QueryStatus::Continue
                }
            });
            font
        }

        fn advance_of(
            query: &mut Query,
            ch: char,
            font_size: Size,
            variations: &[FontVariation],
        ) -> Option<f32> {
            let font = find_font_for(query, ch)?;
            let font_ref = skrifa::FontRef::from_index(font.blob.as_ref(), font.index).ok()?;
            let location = font_ref.axes().location(
                variations
                    .iter()
                    .map(|v| (skrifa::Tag::from_be_bytes(v.tag.to_bytes()), v.value)),
            );
            let location_ref = LocationRef::from(&location);
            let glyph_metrics = GlyphMetrics::new(&font_ref, font_size, location_ref);
            let char_map = Charmap::new(&font_ref);
            let glyph_id = char_map.map(ch)?;
            glyph_metrics.advance_width(glyph_id)
        }

        fn metrics_of(
            query: &mut Query,
            ch: char,
            font_size: Size,
            variations: &[FontVariation],
        ) -> Option<(f32, Option<f32>, Option<f32>)> {
            let font = find_font_for(query, ch)?;
            let font_ref = skrifa::FontRef::from_index(font.blob.as_ref(), font.index).ok()?;
            let location = font_ref.axes().location(
                variations
                    .iter()
                    .map(|v| (skrifa::Tag::from_be_bytes(v.tag.to_bytes()), v.value)),
            );
            let location_ref = LocationRef::from(&location);
            let metrics = Metrics::new(&font_ref, font_size, location_ref);
            Some((metrics.ascent, metrics.x_height, metrics.cap_height))
        }

        let font_size = Size::new(font_size.px());
        let zero_advance = advance_of(&mut query, '0', font_size, &variations);
        let ic_advance = advance_of(&mut query, '\u{6C34}', font_size, &variations);
        let (ascent, x_height, cap_height) =
            metrics_of(&mut query, ' ', font_size, &variations).unwrap_or((0.0, None, None));

        FontMetrics {
            ascent: CSSPixelLength::new(ascent),
            x_height: x_height.filter(|xh| *xh != 0.0).map(CSSPixelLength::new),
            cap_height: cap_height.map(CSSPixelLength::new),
            zero_advance_measure: zero_advance.map(CSSPixelLength::new),
            ic_width: ic_advance.map(CSSPixelLength::new),
            script_percent_scale_down: None,
            script_script_percent_scale_down: None,
        }
    }

    fn base_size_for_generic(
        &self,
        generic: style::values::computed::font::GenericFontFamily,
    ) -> style::values::computed::Length {
        let size = match generic {
            style::values::computed::font::GenericFontFamily::Monospace => 13.0,
            _ => 16.0,
        };
        style::values::computed::Length::from(Au::from_f32_px(size))
    }
}
