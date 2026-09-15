//! An inline `<svg>` flex item shrinks, because the UA style sheet makes it a
//! scroll container.
//!
//! SVG 2 § 3.11 puts a non-root `svg` element's `overflow` in the User Agent
//! style sheet:
//!
//! > In the User Agent style sheet, overflow is overridden for the ‘svg’
//! > element when it is not the root element of a stand-alone document, the
//! > ‘pattern’ element, and the ‘marker’ element to be hidden by default.
//!
//! That computed `overflow: hidden` is load-bearing for flex layout, because
//! css-flexbox-1 § 4.5 keys the automatic minimum size off it:
//!
//! > To provide a more reasonable default minimum size for flex items, the used
//! > value of a main axis automatic minimum size on a flex item whose computed
//! > overflow value is non-scrollable is its content-based minimum size; for
//! > scroll containers the automatic minimum size is zero, as usual.
//!
//! Blitz's UA style sheet had no `svg` rule, so an inline `<svg>` computed
//! `overflow: visible`, took the content-based minimum size branch, and could
//! never shrink below its own `width` — pushing its flex siblings out of the
//! container instead of giving up space to them.
//!
//! <https://svgwg.org/svg2-draft/render.html#OverflowAndClipProperties>
//! <https://drafts.csswg.org/css-flexbox-1/#min-size-auto>

use blitz_dom::DocumentConfig;
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;

fn layout_doc(html: &str) -> HtmlDocument {
    let mut doc = HtmlDocument::from_html(
        html,
        DocumentConfig {
            viewport: Some(Viewport::new(800, 600, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    doc
}

/// A flex row `container_px` wide holding a 22x20 `<svg>` icon and an
/// unshrinkable 100px sibling, so the icon must absorb the whole deficit.
/// `svg_style` is applied to the icon.
fn row(container_px: u32, svg_style: &str) -> String {
    format!(
        r#"<html><body style="margin:0">
          <div style="display:flex; width:{container_px}px">
            <svg id="icon" width="22" height="20" viewBox="0 0 22 20" style="{svg_style}"></svg>
            <div id="sibling" style="width:100px; flex-shrink:0"></div>
          </div>
        </body></html>"#
    )
}

fn icon_width(html: &str) -> f32 {
    let doc = layout_doc(html);
    let id = doc.query_selector("#icon").unwrap().expect("#icon");
    doc.get_node(id).unwrap().final_layout().size.width
}

fn sibling_x(html: &str) -> f32 {
    let doc = layout_doc(html);
    let id = doc.query_selector("#sibling").unwrap().expect("#sibling");
    doc.get_node(id).unwrap().final_layout().location.x
}

#[test]
fn an_inline_svg_computes_overflow_hidden() {
    let doc = layout_doc(
        r#"<html><body><svg id="icon" width="22" height="20" viewBox="0 0 22 20"></svg></body></html>"#,
    );
    let id = doc.query_selector("#icon").unwrap().expect("#icon");
    let node = doc.get_node(id).unwrap();
    let styles = node.primary_styles().unwrap();
    let box_styles = styles.get_box();
    assert_eq!(
        (box_styles.overflow_x, box_styles.overflow_y),
        (
            style::values::computed::Overflow::Hidden,
            style::values::computed::Overflow::Hidden
        ),
        "SVG 2 § 3.11: the UA style sheet sets `overflow: hidden` on a non-root `svg`"
    );
}

#[test]
fn an_svg_flex_item_shrinks_below_its_specified_width() {
    // 122px fits both items exactly; every pixel below that must come out of
    // the icon, down to a zero automatic minimum size.
    assert_eq!(icon_width(&row(122, "")), 22.0, "no deficit, no shrink");
    assert_eq!(icon_width(&row(115, "")), 15.0);
    assert_eq!(icon_width(&row(107, "")), 7.0);
    assert_eq!(icon_width(&row(100, "")), 0.0, "clamped at zero, not at 22");
    assert_eq!(icon_width(&row(80, "")), 0.0);
}

#[test]
fn a_shrunk_svg_flex_item_does_not_push_its_sibling_out() {
    for container_px in [100, 90, 80] {
        assert_eq!(
            sibling_x(&row(container_px, "")),
            0.0,
            "at {container_px}px the icon must yield all its space so the \
             sibling still starts at the container's content edge"
        );
    }
}

#[test]
fn an_svg_flex_item_with_visible_overflow_keeps_its_minimum_size() {
    // The carve-out is specifically for scroll containers: an author who opts
    // back into `overflow: visible` gets the content-based minimum size again,
    // and the icon holds its 22px. This is what Blitz did for every inline
    // `<svg>` before the UA rule existed.
    for container_px in [115, 107, 100, 80] {
        assert_eq!(
            icon_width(&row(container_px, "overflow:visible")),
            22.0,
            "`overflow: visible` restores the content-based automatic minimum size"
        );
    }
}
