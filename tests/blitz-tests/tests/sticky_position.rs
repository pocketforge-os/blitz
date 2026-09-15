//! A `position: sticky` box's insets are a constraint, not a translation.
//!
//! CSS Positioned Layout 3 § 3.4 lays a sticky box out exactly where relative
//! positioning with no offset would put it, and only shifts it when its
//! scrollport would otherwise push its border edge outside the sticky view
//! rectangle its insets describe. In particular:
//!
//! > Note: A sticky positioned element with a non-`auto` `top` value and an
//! > `auto` `bottom` value will only ever be pushed down by sticky
//! > positioning; it will never be offset upwards.
//!
//! Stylo's `position: sticky` was converted to Taffy as `position: relative`
//! *with the computed insets attached*, so `top: -8px` translated the box 8px
//! up — the one direction that note forbids — and a positive `top` translated
//! it down whether or not any scrollport was constraining it.
//!
//! <https://drafts.csswg.org/css-position-3/#sticky-pos>

use blitz_dom::DocumentConfig;
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;

/// Lay `html` out at 800x600 and return the `y` of `#probe`'s border box
/// relative to its parent.
fn probe_y(html: &str) -> f32 {
    let mut doc = HtmlDocument::from_html(
        html,
        DocumentConfig {
            viewport: Some(Viewport::new(800, 600, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);

    let id = doc.query_selector("#probe").unwrap().expect("#probe");
    doc.get_node(id).unwrap().final_layout().location.y
}

/// Lay `html` out at 800x600 and return the `x` of `#probe`'s border box
/// relative to its parent.
fn probe_x(html: &str) -> f32 {
    let mut doc = HtmlDocument::from_html(
        html,
        DocumentConfig {
            viewport: Some(Viewport::new(800, 600, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);

    let id = doc.query_selector("#probe").unwrap().expect("#probe");
    doc.get_node(id).unwrap().final_layout().location.x
}

/// A scroll container holding a probe styled by `probe_style`, preceded by
/// `lead_px` of content and followed by enough content to make it scrollable.
fn scroller(probe_style: &str, lead_px: u32) -> String {
    format!(
        r#"<html><body style="margin:0">
          <div id="scroller" style="height:200px; overflow:scroll">
            <div style="height:{lead_px}px"></div>
            <div id="probe" style="height:30px; {probe_style}"></div>
            <div style="height:1000px"></div>
          </div>
        </body></html>"#
    )
}

#[test]
fn a_negative_top_never_lifts_a_sticky_box() {
    let flow = probe_y(&scroller("", 40));
    let sticky = probe_y(&scroller("position:sticky; top:-8px", 40));
    assert_eq!(
        sticky, flow,
        "an unconstrained sticky box must stay at its in-flow position; \
         `top: -8px` may never lift it (flow y={flow}, sticky y={sticky})"
    );
}

#[test]
fn a_sticky_box_is_not_translated_by_its_insets() {
    let flow = probe_y(&scroller("", 40));
    for inset in ["top:12px", "bottom:12px", "top:12px; bottom:4px"] {
        let sticky = probe_y(&scroller(&format!("position:sticky; {inset}"), 40));
        assert_eq!(
            sticky, flow,
            "`position:sticky; {inset}` must not translate the box during layout \
             (flow y={flow}, sticky y={sticky})"
        );
    }
}

#[test]
fn a_sticky_box_is_not_translated_in_the_inline_axis_either() {
    let flow = probe_x(&scroller("", 40));
    let sticky = probe_x(&scroller("position:sticky; left:25px", 40));
    assert_eq!(
        sticky, flow,
        "`position:sticky; left:25px` must not translate the box during layout \
         (flow x={flow}, sticky x={sticky})"
    );
}

#[test]
fn a_sticky_box_still_takes_part_in_normal_flow() {
    // Sticky is in-flow: the box keeps its slot, so the content after it sits
    // below it just as it would for a static box (CSS Positioned Layout 3
    // § 3.4 — "similar to relative positioning").
    let html = r#"<html><body style="margin:0">
      <div id="scroller" style="height:200px; overflow:scroll">
        <div id="probe" style="height:30px; position:sticky; top:-8px"></div>
        <div id="after" style="height:10px"></div>
        <div style="height:1000px"></div>
      </div>
    </body></html>"#;
    let mut doc = HtmlDocument::from_html(
        html,
        DocumentConfig {
            viewport: Some(Viewport::new(800, 600, 1.0, ColorScheme::Light)),
            html_parser_provider: Some(Arc::new(HtmlProvider) as _),
            ..Default::default()
        },
    );
    doc.resolve(0.0);
    let after = doc.query_selector("#after").unwrap().expect("#after");
    assert_eq!(
        doc.get_node(after).unwrap().final_layout().location.y,
        30.0,
        "a sticky box must still occupy its slot in normal flow"
    );
}

#[test]
fn relative_positioning_still_applies_its_insets() {
    // The guard on the fix: only `sticky` drops its insets. `relative` is the
    // translation `sticky` is not (CSS Positioned Layout 3 § 3.3).
    let flow = probe_y(&scroller("", 40));
    let relative = probe_y(&scroller("position:relative; top:-8px", 40));
    assert_eq!(
        relative,
        flow - 8.0,
        "`position:relative; top:-8px` must still lift the box by 8px \
         (flow y={flow}, relative y={relative})"
    );
}
