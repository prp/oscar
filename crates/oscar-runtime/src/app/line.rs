//! The menu's first line: the phase and up to three figures, each right-aligned at its column with its unit after
//! it, so a label stands still as its number changes; drawn in the label colour, the rows' width. The trouble line
//! is one too, its triangle drawn here, as a menu shows no image of its own on macOS 26.

use std::cell::RefCell;

use objc2::rc::Retained;
use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSColor, NSCompositingOperation, NSView};
use objc2_foundation::{NSPoint, NSRect, NSSize};

use super::rows::{draw_text, WIDTH};

const HEIGHT: f64 = 22.0;
/// The right edge of each column's number.
const COLUMNS: [f64; 3] = [150.0, 260.0, 356.0];

/// The phase word, then (number, unit) cells in column order; a trouble line's mark, true for a warning.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Line {
    pub phase: String,
    pub cells: Vec<(String, &'static str)>,
    pub mark: Option<bool>,
}

impl Line {
    pub fn new(phase: &str, cells: &[(String, &'static str)]) -> Self {
        Self { phase: phase.to_string(), cells: cells.to_vec(), mark: None }
    }

    pub fn trouble(text: &str, warn: bool) -> Self {
        Self { mark: Some(warn), ..Self::new(text, &[]) }
    }
}

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "OscarStatusLine"]
    #[ivars = RefCell<Line>]
    pub struct StatusLine;

    impl StatusLine {
        #[unsafe(method(isFlipped))]
        fn is_flipped(&self) -> bool {
            true
        }

        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty: NSRect) {
            let line = self.ivars().borrow();
            let ink = NSColor::labelColor();
            if let Some(warn) = line.mark {
                let at = NSRect::new(NSPoint::new(9.0, 3.0), NSSize::new(16.0, 16.0));
                unsafe {
                    super::icon::warning(warn).drawInRect_fromRect_operation_fraction_respectFlipped_hints(
                        at,
                        NSRect::ZERO,
                        NSCompositingOperation::SourceOver,
                        1.0,
                        true,
                        None,
                    )
                };
            }
            draw_text(&line.phase, NSPoint::new(30.0, 3.0), &ink, false, 0.0);
            for ((number, unit), right) in line.cells.iter().zip(COLUMNS) {
                draw_text(number, NSPoint::new(right, 3.0), &ink, true, 0.0);
                draw_text(unit, NSPoint::new(right + 4.0, 3.0), &ink, false, 0.0);
            }
        }
    }
);

impl StatusLine {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(RefCell::new(Line::default()));
        unsafe {
            msg_send![super(this), initWithFrame: NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(WIDTH, HEIGHT))]
        }
    }

    pub fn refresh(&self, line: Line) {
        if *self.ivars().borrow() != line {
            *self.ivars().borrow_mut() = line;
            self.setNeedsDisplay(true);
        }
    }
}
