//! The model rows, custom item views as Control Center's: the check on the served row, the name, the size, and at the
//! right edge the row's mark (an X on a downloaded row, the download arrow alone in ink on one not on disk, the X on a
//! running download cancelling it); a click on the name switches or downloads.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{
    NSBezierPath, NSColor, NSEvent, NSFont, NSFontAttributeName, NSForegroundColorAttributeName, NSLineCapStyle,
    NSStringDrawing, NSView,
};
use objc2_foundation::{NSDictionary, NSPoint, NSRect, NSSize, NSString};
use oscar_models::rows::ModelPreset;

use crate::model_store::download::{self, Event};
use crate::model_store::hub;

pub const WIDTH: f64 = 400.0;

/// The row's name: the family, with its total parameters in brackets where the name does not carry them.
pub fn name(preset: ModelPreset) -> String {
    let def = preset.def();
    let family = def.family_dir.trim_end_matches("-GGUF");
    let total = if def.parameters >= 1000 {
        format!("{:.1}T", def.parameters as f64 / 1000.0)
    } else {
        format!("{}B", def.parameters)
    };
    if family.contains(&total) {
        family.to_string()
    } else {
        format!("{family} ({total})")
    }
}

#[cfg(test)]
#[test]
fn a_name_carries_its_total_once() {
    let names: Vec<String> = <ModelPreset as clap::ValueEnum>::value_variants().iter().map(|&p| name(p)).collect();
    assert!(names.contains(&"Qwen3.5-397B-A17B".to_string()), "{names:?}");
    assert!(names.contains(&"Kimi-K3 (2.8T)".to_string()), "{names:?}");
    assert!(names.contains(&"GLM-5.2 (754B)".to_string()), "{names:?}");
}
const HEIGHT: f64 = 22.0;
/// A row with a bar under its text.
const TALL: f64 = 30.0;
/// The mark's column at the right edge.
const MARK: f64 = 30.0;

/// What the row shows; a download's bar fill in 0..1, its text at the right; the RAM in GB a machine too small
/// would need.
#[derive(Clone, PartialEq)]
pub enum Kind {
    Absent,
    TooLarge(u64),
    OnDisk,
    Served,
    Loading(String),
    Downloading(f64, String),
}

pub struct RowState {
    pub preset: ModelPreset,
    pub name: String,
    pub kind: RefCell<Kind>,
    pub size: RefCell<String>,
    pub app: Retained<super::App>,
}

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "OscarRow"]
    #[ivars = RowState]
    pub struct Row;

    impl Row {
        /// Drawn top-down, as the glyphs are.
        #[unsafe(method(isFlipped))]
        fn is_flipped(&self) -> bool {
            true
        }

        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty: NSRect) {
            self.draw();
        }

        /// The menu closes first; the mark's column acts on the mark, the rest on the name.
        #[unsafe(method(mouseUp:))]
        fn mouse_up(&self, event: &NSEvent) {
            let at = self.convertPoint_fromView(event.locationInWindow(), None);
            if let Some(menu) = self.enclosingMenuItem().and_then(|item| unsafe { item.menu() }) {
                menu.cancelTracking();
            }
            let s = self.ivars();
            let kind = s.kind.borrow().clone();
            match (kind, at.x >= WIDTH - MARK - 4.0) {
                (Kind::Absent, _) => s.app.download(s.preset),
                (Kind::OnDisk | Kind::Served | Kind::Loading(..), true) => {
                    s.app.remove(s.preset, &s.name, &s.size.borrow())
                }
                (Kind::OnDisk | Kind::Served, false) => s.app.switch(s.preset),
                (Kind::Downloading(..), true) => s.app.cancel(s.preset),
                _ => {}
            }
        }
    }
);

impl Row {
    pub fn new(mtm: MainThreadMarker, state: RowState) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(state);
        unsafe {
            msg_send![super(this), initWithFrame: NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(WIDTH, HEIGHT))]
        }
    }

    /// Redrawn only on a change; a bar makes the row taller, read when the menu next opens.
    pub fn refresh(&self, kind: Kind, size: String) {
        let s = self.ivars();
        if *s.kind.borrow() == kind && *s.size.borrow() == size {
            return;
        }
        let tall = matches!(kind, Kind::Downloading(..));
        *s.kind.borrow_mut() = kind;
        *s.size.borrow_mut() = size;
        self.setFrameSize(NSSize::new(WIDTH, if tall { TALL } else { HEIGHT }));
        self.setNeedsDisplay(true);
    }

    fn draw(&self) {
        let s = self.ivars();
        let kind = s.kind.borrow().clone();
        let bounds = self.bounds();
        let highlighted = self.enclosingMenuItem().is_some_and(|item| item.isHighlighted());
        if highlighted {
            NSColor::selectedContentBackgroundColor().setFill();
            let inset = NSRect::new(NSPoint::new(5.0, 0.0), NSSize::new(bounds.size.width - 10.0, bounds.size.height));
            NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(inset, 5.0, 5.0).fill();
        }
        let ink = match (highlighted, &kind) {
            (true, _) => NSColor::selectedMenuItemTextColor(),
            (false, Kind::Absent | Kind::TooLarge(_)) => NSColor::tertiaryLabelColor(),
            _ => NSColor::labelColor(),
        };
        let grey = if highlighted { NSColor::selectedMenuItemTextColor() } else { NSColor::secondaryLabelColor() };
        let accent = if highlighted { NSColor::selectedMenuItemTextColor() } else { NSColor::controlAccentColor() };
        let right = WIDTH - MARK - 6.0;
        let size = s.size.borrow().clone();
        let (text, bar) = match &kind {
            Kind::Downloading(fill, text) => (text.as_str(), Some(*fill)),
            Kind::Loading(text) => (text.as_str(), None),
            _ => (size.as_str(), None),
        };
        let name_end = 30.0 + draw_text(&s.name, NSPoint::new(30.0, 3.0), &ink, false, 0.0);
        if let Kind::TooLarge(gb) = kind {
            draw_text(&format!("[needs >{gb} GB of RAM]"), NSPoint::new(name_end + 6.0, 3.0), &grey, false, 0.0);
        }
        draw_text(text, NSPoint::new(right, 3.0), &grey, true, 0.0);
        if let Some(fill) = bar {
            let track = NSRect::new(NSPoint::new(30.0, TALL - 7.0), NSSize::new(right - 30.0, 4.0));
            NSColor::quaternarySystemFillColor().setFill();
            NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(track, 2.0, 2.0).fill();
            let done = NSRect::new(track.origin, NSSize::new(track.size.width * fill, 4.0));
            accent.setFill();
            NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(done, 2.0, 2.0).fill();
        }
        let glyph = |segments: &[&[(f64, f64)]], at: (f64, f64), width: f64, colour: &NSColor| {
            let path = NSBezierPath::bezierPath();
            for seg in segments {
                path.moveToPoint(NSPoint::new(at.0 + seg[0].0, at.1 + seg[0].1));
                for &(x, y) in &seg[1..] {
                    path.lineToPoint(NSPoint::new(at.0 + x, at.1 + y));
                }
            }
            path.setLineWidth(width);
            path.setLineCapStyle(NSLineCapStyle::Round);
            colour.setStroke();
            path.stroke();
        };
        let mark = (WIDTH - MARK + 6.0, 2.0);
        if matches!(kind, Kind::Served | Kind::Loading(..)) {
            glyph(&[&[(3.0, 11.5), (7.0, 15.5), (15.0, 6.5)]], (10.0, 1.0), 2.2, &ink);
        }
        match kind {
            Kind::TooLarge(_) => {}
            Kind::Absent => glyph(
                &[&[(9.0, 3.5), (9.0, 11.5)], &[(5.5, 8.5), (9.0, 12.0), (12.5, 8.5)], &[(4.5, 15.0), (13.5, 15.0)]],
                mark,
                1.7,
                &NSColor::labelColor(),
            ),
            _ => glyph(&[&[(5.5, 5.5), (12.5, 12.5)], &[(12.5, 5.5), (5.5, 12.5)]], mark, 1.8, &grey),
        }
    }
}

/// The menu's font at `size` (0 its own) in `colour`.
fn attrs(colour: &NSColor, size: f64) -> Retained<NSDictionary<NSString, AnyObject>> {
    let font = NSFont::menuFontOfSize(size);
    let keys = unsafe { [NSFontAttributeName, NSForegroundColorAttributeName] };
    let values: [&AnyObject; 2] = [&font, colour];
    NSDictionary::from_slices(&keys, &values)
}

/// The width of `text` in the menu's font.
pub fn width(text: &str, size: f64) -> f64 {
    unsafe { NSString::from_str(text).sizeWithAttributes(Some(&attrs(&NSColor::labelColor(), size))) }.width
}

/// `right_aligned` ends the text at `at`; the width drawn.
pub fn draw_text(text: &str, at: NSPoint, colour: &NSColor, right_aligned: bool, size: f64) -> f64 {
    let attrs = attrs(colour, size);
    let text = NSString::from_str(text);
    let width = unsafe { text.sizeWithAttributes(Some(&attrs)) }.width;
    let at = if right_aligned { NSPoint::new(at.x - width, at.y) } else { at };
    unsafe { text.drawAtPoint_withAttributes(at, Some(&attrs)) };
    width
}

/// A download on a thread of its own with its own runtime; the counts the row's bar; cancelled, the staged part stays
/// under `.partial` for the next download to resume.
pub struct Download {
    pub progress: Arc<Mutex<Progress>>,
    cancel: Option<tokio::sync::oneshot::Sender<()>>,
}

#[derive(Default)]
pub struct Progress {
    pub total: u64,
    pub fetched: u64,
    pub kept: u64,
    /// Set when the thread ends: the model stored, or why not.
    pub done: Option<Result<(), String>>,
}

impl Download {
    pub fn start(preset: ModelPreset) -> Self {
        let progress = Arc::new(Mutex::new(Progress::default()));
        let (cancel, cancelled) = tokio::sync::oneshot::channel::<()>();
        let shared = progress.clone();
        std::thread::Builder::new()
            .name("oscar-download".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build();
                let done = runtime.map_err(|e| e.to_string()).and_then(|runtime| {
                    runtime.block_on(async {
                        let fetch = async {
                            let client = download::client()?;
                            let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
                            let mut on = |ev: Event| {
                                let mut p = shared.lock().unwrap_or_else(|e| e.into_inner());
                                match ev {
                                    Event::FetchTotal(total) => p.total = total,
                                    Event::Fetched(delta) => p.fetched += delta,
                                    Event::Kept(bytes) => p.kept += bytes,
                                    _ => {}
                                }
                            };
                            download::download_variant(&client, hub::HF_BASE, preset, threads, &mut on).await?;
                            crate::cli::download::prefetch_tokenizer(preset.def().tok_repo).await;
                            anyhow::Ok(())
                        };
                        tokio::select! {
                            r = fetch => r.map_err(|e| format!("{e:#}")),
                            _ = cancelled => Err("cancelled".to_string()),
                        }
                    })
                });
                shared.lock().unwrap_or_else(|e| e.into_inner()).done = Some(done);
            })
            .expect("the download's thread spawns");
        Self { progress, cancel: Some(cancel) }
    }

    pub fn cancel(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}
