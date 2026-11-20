//! The memory bar under the status line: Oscar's share in the accent colour, every other process and the kernel in
//! orange, free in grey, over the memory the kernel pages; the three figures as a legend under it.

use std::cell::Cell;

use objc2::rc::Retained;
use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSBezierPath, NSColor, NSView};
use objc2_foundation::{NSPoint, NSRect, NSSize};

use super::rows::{draw_text, WIDTH};

const HEIGHT: f64 = 30.0;

pub struct Figures {
    /// (own, other, total) in GB.
    pub gb: Cell<(f64, f64, f64)>,
}

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "OscarMemoryBar"]
    #[ivars = Figures]
    pub struct MemoryBar;

    impl MemoryBar {
        #[unsafe(method(isFlipped))]
        fn is_flipped(&self) -> bool {
            true
        }

        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty: NSRect) {
            let (own, other, total) = self.ivars().gb.get();
            let free = (total - own - other).max(0.0);
            let (left, right) = (30.0, WIDTH - 18.0);
            let mut x = left;
            let mut legend = Vec::new();
            for (gb, colour, name) in [
                (own, NSColor::controlAccentColor(), "Oscar"),
                (other, NSColor::systemOrangeColor(), "Other"),
                (free, NSColor::systemGrayColor(), "Free"),
            ] {
                let width = if total > 0.0 { (right - left) * gb / total } else { 0.0 };
                if name == "Free" { colour.colorWithAlphaComponent(0.4).setFill() } else { colour.setFill() };
                NSBezierPath::fillRect(NSRect::new(NSPoint::new(x, 4.0), NSSize::new(width, 8.0)));
                x += width;
                legend.push((colour, format!("{name} {gb:.0} GB")));
            }
            let mut x = left;
            for (colour, text) in legend {
                draw_text("■", NSPoint::new(x, 16.0), &colour, false, 11.0);
                let grey = NSColor::secondaryLabelColor();
                x += 13.0 + draw_text(&text, NSPoint::new(x + 13.0, 16.0), &grey, false, 11.0) + 10.0;
            }
        }
    }
);

impl MemoryBar {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(Figures { gb: Cell::new((0.0, 0.0, 0.0)) });
        unsafe {
            msg_send![super(this), initWithFrame: NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(WIDTH, HEIGHT))]
        }
    }

    pub fn refresh(&self, own: f64, other: f64, total: f64) {
        if self.ivars().gb.replace((own, other, total)) != (own, other, total) {
            self.setNeedsDisplay(true);
        }
    }
}
