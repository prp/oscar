//! The menu bar's icon: the handwritten O as path data (a perfect oval, a hand-drawn flick), idle as a template image,
//! the prefill lines and decode dots in the accent colour, the O in the label colour, drawn under the bar's appearance.

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::Bool;
use objc2_app_kit::{NSBezierPath, NSColor, NSImage, NSImageSymbolConfiguration, NSLineCapStyle, NSLineJoinStyle};
use objc2_foundation::{NSArray, NSPoint, NSRect, NSSize, NSString};

/// The 20-point box; the oval's radii and its centre with the tail in the box; the cusp at one o'clock; the pen.
const BOX: f64 = 20.0;
const RX: f64 = 6.0;
const RY: f64 = 7.0;
const CENTRE: (f64, f64) = (8.61, 10.0);
const CUSP_DEG: f64 = -68.0;
const LINE: f64 = 2.4;
/// The flick, traced from a hand-drawn sketch at an oval of radius 5, as offsets from the stroke's end; scaled with
/// the oval, as the marks are.
const SCALE: f64 = RX / 5.0;
const TAIL: [(f64, f64); 49] = [
    (0.00, 0.00),
    (0.04, 0.05),
    (0.09, 0.10),
    (0.17, 0.16),
    (0.28, 0.24),
    (0.39, 0.30),
    (0.50, 0.34),
    (0.62, 0.36),
    (0.74, 0.39),
    (0.85, 0.39),
    (0.97, 0.40),
    (1.09, 0.41),
    (1.21, 0.41),
    (1.32, 0.41),
    (1.44, 0.42),
    (1.56, 0.43),
    (1.68, 0.44),
    (1.79, 0.45),
    (1.91, 0.46),
    (2.03, 0.47),
    (2.15, 0.47),
    (2.26, 0.47),
    (2.38, 0.47),
    (2.50, 0.47),
    (2.62, 0.47),
    (2.74, 0.47),
    (2.85, 0.47),
    (2.97, 0.47),
    (3.09, 0.47),
    (3.21, 0.47),
    (3.32, 0.47),
    (3.44, 0.47),
    (3.56, 0.47),
    (3.68, 0.47),
    (3.79, 0.47),
    (3.91, 0.45),
    (4.03, 0.44),
    (4.15, 0.43),
    (4.26, 0.41),
    (4.38, 0.39),
    (4.50, 0.36),
    (4.62, 0.34),
    (4.74, 0.30),
    (4.85, 0.27),
    (4.97, 0.24),
    (5.09, 0.19),
    (5.21, 0.14),
    (5.32, 0.10),
    (5.44, 0.04),
];

/// Prefill: three lines scrolling up through the counter, a 1.1 s cycle; decode: three dots appearing, 1.4 s.
pub const PREFILL_FRAMES: usize = 12;
pub const PREFILL_CYCLE_S: f64 = 1.1;
pub const DECODE_FRAMES: usize = 4;
pub const DECODE_CYCLE_S: f64 = 1.4;

fn point(x: f64, y: f64) -> NSPoint {
    NSPoint::new(x, y)
}

fn oval(cx: f64, cy: f64, rx: f64, ry: f64) -> Retained<NSBezierPath> {
    NSBezierPath::bezierPathWithOvalInRect(NSRect::new(point(cx - rx, cy - ry), NSSize::new(2.0 * rx, 2.0 * ry)))
}

/// From the cusp a full turn counter-clockwise plus 3°, then the flick; the box's y runs down, as the glyph's.
fn o_path() -> Retained<NSBezierPath> {
    let (cx, cy) = CENTRE;
    let at = |deg: f64| point(cx + RX * deg.to_radians().cos(), cy + RY * deg.to_radians().sin());
    let path = NSBezierPath::bezierPath();
    path.moveToPoint(at(CUSP_DEG));
    for i in 1..=150 {
        path.lineToPoint(at(CUSP_DEG - 363.0 * i as f64 / 150.0));
    }
    let end = at(CUSP_DEG - 363.0);
    for (dx, dy) in TAIL {
        path.lineToPoint(point(end.x + dx * SCALE, end.y + dy * SCALE));
    }
    path.setLineWidth(LINE);
    path.setLineCapStyle(NSLineCapStyle::Round);
    path.setLineJoinStyle(NSLineJoinStyle::Round);
    path
}

/// The O in the text colour (black or white outright, the label colour being translucent), then `marks` in the
/// accent colour; the handler runs under the appearance the image is drawn in.
fn image(template: bool, marks: impl Fn() + 'static) -> Retained<NSImage> {
    let handler = RcBlock::new(move |_: NSRect| -> Bool {
        NSColor::textColor().setStroke();
        o_path().stroke();
        let accent = NSColor::controlAccentColor();
        accent.setStroke();
        accent.setFill();
        marks();
        Bool::YES
    });
    let image = NSImage::imageWithSize_flipped_drawingHandler(NSSize::new(BOX, BOX), true, &handler);
    image.setTemplate(template);
    image
}

/// The trouble line's triangle: the system's symbol, its mark black on yellow for a warning, white on red for an
/// error.
pub fn warning(warn: bool) -> Retained<NSImage> {
    let (mark, tint) = if warn {
        (NSColor::blackColor(), NSColor::systemYellowColor())
    } else {
        (NSColor::whiteColor(), NSColor::systemRedColor())
    };
    let palette = NSArray::from_retained_slice(&[mark, tint]);
    let symbol = NSImage::imageWithSystemSymbolName_accessibilityDescription(
        &NSString::from_str("exclamationmark.triangle.fill"),
        None,
    )
    .expect("a system symbol");
    symbol
        .imageWithSymbolConfiguration(&NSImageSymbolConfiguration::configurationWithPaletteColors(&palette))
        .expect("a palette")
}

/// The template image: the system tints it for the bar.
pub fn idle() -> Retained<NSImage> {
    image(true, || {})
}

/// Four lines a pitch apart, moved up one pitch over the cycle, clipped to the counter (the oval less the pen).
pub fn prefill(frame: usize) -> Retained<NSImage> {
    image(false, move || {
        let (cx, cy) = CENTRE;
        let (half, pitch) = (2.2 * SCALE, 2.3 * SCALE);
        oval(cx, cy, RX - LINE, RY - LINE).addClip();
        let off = pitch * frame as f64 / PREFILL_FRAMES as f64;
        let lines = NSBezierPath::bezierPath();
        for k in -1..=2 {
            let y = cy + k as f64 * pitch - off;
            lines.moveToPoint(point(cx - half, y));
            lines.lineToPoint(point(cx + half, y));
        }
        lines.setLineWidth(2.0);
        lines.setLineCapStyle(NSLineCapStyle::Round);
        lines.stroke();
    })
}

/// Left to right, one dot a frame, none on the first.
pub fn decode(frame: usize) -> Retained<NSImage> {
    image(false, move || {
        let (cx, cy) = CENTRE;
        for dx in [-2.5 * SCALE, 0.0, 2.5 * SCALE].iter().take(frame % DECODE_FRAMES) {
            oval(cx + dx, cy, 1.3, 1.3).fill();
        }
    })
}
