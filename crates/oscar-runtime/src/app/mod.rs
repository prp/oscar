//! Oscar.app: the menu bar over `serve` as a child of this binary. The app runs only the CLI's own paths; nothing in
//! the engine changes for it.

mod child;
mod icon;
mod line;
mod login;
mod memory;
mod notify;
mod opencode;
mod rows;
mod setup;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::ValueEnum;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{define_class, msg_send, sel, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{
    NSAlert, NSAlertFirstButtonReturn, NSApplication, NSApplicationActivationPolicy, NSControlStateValueOff,
    NSControlStateValueOn, NSImage, NSMenu, NSMenuDelegate, NSMenuItem, NSModalResponseOK, NSOpenPanel, NSStatusBar,
    NSStatusItem, NSVariableStatusItemLength, NSWorkspace,
};
use objc2_foundation::{NSObject, NSObjectProtocol, NSRunLoop, NSRunLoopCommonModes, NSString, NSTimer, NSURL};
use oscar_models::rows::ModelPreset;

use crate::cli::settings::{self, Settings};
use crate::cli::ui::Status;
use crate::machine::GB;
use crate::model_store;
use line::Line;
use rows::{Download, Kind, Row};

/// The app, not the CLI: no terminal on stdin and no ssh session (a double-click, launchd and the login item).
pub fn wanted() -> bool {
    let tty = unsafe { libc::isatty(0) } == 1;
    !tty && std::env::var_os("SSH_CONNECTION").is_none() && std::env::var_os("SSH_TTY").is_none()
}

/// What macOS and the other applications keep beside oscar after the squeeze: 8-12 GB on the machines measured.
/// With a model's least budget, the installed RAM below which it cannot load.
const SYSTEM_GB: f64 = 10.0;

/// Cut to the rows' width with an ellipsis, so the menu keeps its width; the whole text goes to the tooltip.
fn fit(text: &str) -> String {
    let room = rows::WIDTH - 60.0;
    if rows::width(text, 0.0) <= room {
        return text.to_string();
    }
    let mut chars: Vec<char> = text.chars().collect();
    while chars.pop().is_some() && rows::width(&format!("{}…", chars.iter().collect::<String>()), 0.0) > room {}
    format!("{}…", chars.iter().collect::<String>().trim_end())
}

/// The timer's period, a prefill frame; the child's status read every third tick.
const TICK_S: f64 = icon::PREFILL_CYCLE_S / icon::PREFILL_FRAMES as f64;

struct State {
    log: PathBuf,
    child: RefCell<Option<child::Child>>,
    /// The model the child serves, and the one to serve once the child has ended (a switch, or the first start).
    served: Cell<Option<ModelPreset>>,
    pending: RefCell<Option<(ModelPreset, Option<String>)>>,
    /// When the child was last restarted by the app after dying on its own.
    restarted: Cell<Option<Instant>>,
    /// The child's port once its ready line names it; OpenCode's file is written for it then.
    port: Cell<Option<u16>>,
    rows: RefCell<Vec<Retained<Row>>>,
    downloads: RefCell<HashMap<usize, Download>>,
    item: Retained<NSStatusItem>,
    /// The status line, the memory bar and, shown only with one, the trouble line in its item.
    line: Retained<line::StatusLine>,
    memory: Retained<memory::MemoryBar>,
    trouble: Retained<NSMenuItem>,
    trouble_line: Retained<line::StatusLine>,
    /// The app's own trouble (a download, a removal, a launch), shown over the child's until the next action.
    own_trouble: RefCell<Option<String>>,
    /// Start at Login, checked while the login item is registered.
    login: Retained<NSMenuItem>,
    idle: Retained<NSImage>,
    prefill: Vec<Retained<NSImage>>,
    decode: Vec<Retained<NSImage>>,
    tick: Cell<u64>,
    shown: Cell<*const NSImage>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "OscarApp"]
    #[ivars = State]
    struct App;

    impl App {
        #[unsafe(method(tick:))]
        fn tick(&self, _timer: &NSTimer) {
            let s = self.ivars();
            let n = s.tick.get();
            s.tick.set(n + 1);
            if n % 3 == 0 {
                self.read_status();
                self.refresh_rows();
                self.refresh_memory();
            }
            let frame = |cycle: f64, frames: usize| ((n as f64 * TICK_S / cycle * frames as f64) as usize) % frames;
            let image = match self.status().as_ref().map(|st| st.phase.as_str()) {
                Some("prefill") => &s.prefill[frame(icon::PREFILL_CYCLE_S, icon::PREFILL_FRAMES)],
                Some("decode") => &s.decode[frame(icon::DECODE_CYCLE_S, icon::DECODE_FRAMES)],
                _ => &s.idle,
            };
            if s.shown.get() != Retained::as_ptr(image) {
                s.shown.set(Retained::as_ptr(image));
                if let Some(button) = s.item.button(self.mtm()) {
                    button.setImage(Some(image));
                }
            }
        }

        #[unsafe(method(openLog:))]
        fn open_log(&self, _sender: Option<&AnyObject>) {
            let url = NSURL::fileURLWithPath(&NSString::from_str(&self.ivars().log.to_string_lossy()));
            NSWorkspace::sharedWorkspace().openURL(&url);
        }

        #[unsafe(method(toggleLogin:))]
        fn toggle_login(&self, _sender: Option<&AnyObject>) {
            if std::env::current_exe().is_ok_and(|e| setup::translocated(&e)) {
                return self.show_trouble("Move Oscar to the Applications folder first");
            }
            if let Err(e) = login::set(!login::enabled()) {
                self.show_trouble(&format!("{e:#}"));
            }
            self.show_login();
        }

        /// A Terminal window, white on black, running `oscar chat` through the served model; a clean exit (Ctrl-D)
        /// ends the shell and the app closes the window, an error leaves it with the text and the prompt. With no
        /// server running, chat would load a model of its own: refused.
        #[unsafe(method(launchChat:))]
        fn launch_chat(&self, _sender: Option<&AnyObject>) {
            self.ivars().own_trouble.take();
            if !self.ivars().child.borrow_mut().as_mut().is_some_and(|c| c.exited().is_none()) {
                return self.show_trouble("No server is running - pick a model first");
            }
            let exe = std::env::current_exe().map(|e| e.display().to_string()).unwrap_or_default();
            match opencode::terminal(&format!("clear; '{exe}' chat && exit"), Some("Pro")) {
                Ok(tty) => opencode::close_when_done(&tty),
                Err(e) => self.show_trouble(&format!("{e:#}")),
            }
        }

        /// The folder OpenCode works in, asked each time, the last one preselected and remembered.
        #[unsafe(method(launchOpencode:))]
        fn launch_opencode(&self, _sender: Option<&AnyObject>) {
            self.ivars().own_trouble.take();
            let Some(bin) = opencode::find() else { return self.show_trouble("OpenCode not found") };
            let remembered = Settings::load().unwrap_or_default();
            let panel = NSOpenPanel::openPanel(self.mtm());
            panel.setCanChooseDirectories(true);
            panel.setCanChooseFiles(false);
            panel.setMessage(Some(&NSString::from_str("The folder OpenCode works in")));
            panel.setPrompt(Some(&NSString::from_str("Launch")));
            if let Some(dir) = remembered.opencode_dir.as_deref() {
                panel.setDirectoryURL(Some(&NSURL::fileURLWithPath(&NSString::from_str(dir))));
            }
            activate(self.mtm());
            if panel.runModal() != NSModalResponseOK {
                return;
            }
            let Some(dir) = panel.URL().and_then(|url| url.path()).map(|p| p.to_string()) else { return };
            let settings = Settings { opencode_dir: Some(dir.clone()), ..remembered };
            if let Err(e) = settings.save().and_then(|()| opencode::launch(&bin, Path::new(&dir))) {
                self.show_trouble(&format!("{e:#}"));
            }
        }

        /// The child ended first: the store flushes, up to 20 s.
        #[unsafe(method(quit:))]
        fn quit(&self, _sender: Option<&AnyObject>) {
            if let Some(child) = self.ivars().child.borrow_mut().take() {
                child.stop();
            }
            NSApplication::sharedApplication(self.mtm()).terminate(None);
        }
    }

    unsafe impl NSObjectProtocol for App {}

    unsafe impl NSMenuDelegate for App {
        #[unsafe(method(menuNeedsUpdate:))]
        fn menu_needs_update(&self, menu: &NSMenu) {
            self.show_opencode(menu);
        }
    }
);

impl App {
    fn new(mtm: MainThreadMarker, state: State) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(state);
        unsafe { msg_send![super(this), init] }
    }

    fn item(&self, title: &str, action: Option<objc2::runtime::Sel>, key: &str) -> Retained<NSMenuItem> {
        let item = unsafe {
            NSMenuItem::initWithTitle_action_keyEquivalent(
                NSMenuItem::alloc(self.mtm()),
                &NSString::from_str(title),
                action,
                &NSString::from_str(key),
            )
        };
        unsafe { item.setTarget(Some(self)) };
        item
    }

    /// Top to bottom: the status line, the trouble line, the model rows, Launch Chat, Launch OpenCode, Open Log, Start
    /// at Login, Quit.
    fn menu(&self) -> Retained<NSMenu> {
        let s = self.ivars();
        let menu = NSMenu::new(self.mtm());
        menu.setAutoenablesItems(false);
        let status = self.item("", None, "");
        status.setView(Some(&s.line));
        menu.addItem(&status);
        let bar = self.item("", None, "");
        bar.setView(Some(&s.memory));
        menu.addItem(&bar);
        menu.addItem(&s.trouble);
        menu.addItem(&NSMenuItem::separatorItem(self.mtm()));
        let header = self.item("MODELS", None, "");
        header.setEnabled(false);
        menu.addItem(&header);
        for row in s.rows.borrow().iter() {
            let item = self.item("", None, "");
            item.setView(Some(row));
            menu.addItem(&item);
        }
        menu.addItem(&NSMenuItem::separatorItem(self.mtm()));
        menu.addItem(&self.item("Launch Chat", Some(sel!(launchChat:)), ""));
        menu.addItem(&self.item("", Some(sel!(launchOpencode:)), ""));
        self.show_opencode(&menu);
        menu.addItem(&NSMenuItem::separatorItem(self.mtm()));
        menu.addItem(&self.item("Open Log", Some(sel!(openLog:)), ""));
        unsafe { s.login.setTarget(Some(self)) };
        self.show_login();
        menu.addItem(&s.login);
        menu.addItem(&NSMenuItem::separatorItem(self.mtm()));
        menu.addItem(&self.item("Quit Oscar", Some(sel!(quit:)), "q"));
        menu
    }

    /// The remembered model to serve; none remembered is a pick from the rows, a settings file that does not read
    /// the trouble line.
    fn start(&self) {
        match Settings::load().map(|settings| settings.model.as_deref().map(|_| settings.preset(None, None))) {
            Ok(Some(Ok(next))) => *self.ivars().pending.borrow_mut() = Some(next),
            Ok(_) => self.show(Line::new("Pick a model", &[]), None),
            Err(e) => self.show(Line::new("Pick a model", &[]), Some(&format!("{e:#}"))),
        }
    }

    /// A click on a downloaded row's name: remembered, then served once the running child has ended; the ticked
    /// row's click restarts a child that died.
    pub fn switch(&self, preset: ModelPreset) {
        let s = self.ivars();
        s.own_trouble.take();
        if s.served.get() == Some(preset) {
            if s.child.borrow_mut().as_mut().is_none_or(|c| c.exited().is_some()) {
                *s.pending.borrow_mut() = Some((preset, Settings::load().unwrap_or_default().quant));
                s.restarted.set(None);
            }
            return;
        }
        let remembered = Settings::load().unwrap_or_default();
        let settings = Settings { model: Some(preset.label()), quant: None, ..remembered };
        if let Err(e) = settings.save() {
            return self.show(Line::new("Pick a model", &[]), Some(&format!("{e:#}")));
        }
        *s.pending.borrow_mut() = Some((preset, None));
        if let Some(child) = s.child.borrow().as_ref() {
            child.terminate();
        }
        self.show(Line::new("Stopping", &[]), None);
    }

    pub fn download(&self, preset: ModelPreset) {
        self.ivars().own_trouble.take();
        self.ivars().downloads.borrow_mut().entry(preset as usize).or_insert_with(|| Download::start(preset));
    }

    pub fn cancel(&self, preset: ModelPreset) {
        if let Some(download) = self.ivars().downloads.borrow_mut().get_mut(&(preset as usize)) {
            download.cancel();
        }
    }

    /// The row's X: the alert, then the removal; the model can be downloaded again. The served model's server
    /// stops and nothing is remembered to serve; it holds its files until it ends.
    pub fn remove(&self, preset: ModelPreset, name: &str, size: &str) {
        let s = self.ivars();
        let served = s.served.get() == Some(preset);
        let alert = NSAlert::new(self.mtm());
        alert.setMessageText(&NSString::from_str(&format!("Remove “{name}”?")));
        alert.setInformativeText(&NSString::from_str(&format!(
            "{size} on disk will be freed.{}\nThe model can be downloaded again.",
            if served { " The server stops." } else { "" }
        )));
        alert.addButtonWithTitle(&NSString::from_str("Remove"));
        alert.addButtonWithTitle(&NSString::from_str("Cancel"));
        activate(self.mtm());
        if alert.runModal() != NSAlertFirstButtonReturn {
            return;
        }
        s.own_trouble.take();
        if served {
            s.served.set(None);
            s.pending.borrow_mut().take_if(|(next, _)| *next == preset);
            if let Some(child) = s.child.borrow().as_ref() {
                child.terminate();
            }
            let settings = Settings { model: None, quant: None, ..Settings::load().unwrap_or_default() };
            if let Err(e) = settings.save() {
                self.show_trouble(&format!("{e:#}"));
            }
        }
        if let Err(e) = model_store::remove_variant(preset, preset.def().production_variant) {
            self.show_trouble(&format!("removing {name}: {e:#}"));
        }
    }

    /// Launch OpenCode, greyed out while OpenCode is not installed: the menu then asks at each opening, until found.
    fn show_opencode(&self, menu: &NSMenu) {
        let found = opencode::find().is_some();
        let index = unsafe { menu.indexOfItemWithTarget_andAction(Some(self), Some(sel!(launchOpencode:))) };
        let Some(item) = menu.itemAtIndex(index) else { return };
        item.setTitle(&NSString::from_str(if found { "Launch OpenCode" } else { "OpenCode not found" }));
        item.setEnabled(found);
        menu.setDelegate((!found).then(|| ProtocolObject::from_ref(self)));
    }

    fn show_login(&self) {
        self.ivars().login.setState(if login::enabled() { NSControlStateValueOn } else { NSControlStateValueOff });
    }

    fn show_trouble(&self, trouble: &str) {
        *self.ivars().own_trouble.borrow_mut() = Some(trouble.to_string());
        self.set_trouble(Some(trouble));
    }

    /// The child's figures while it runs, the machine's own reading otherwise (Oscar then holding nothing).
    fn refresh_memory(&self) {
        let s = self.ivars();
        let gb = |b: u64| b as f64 / GB;
        let figures = match self.status() {
            Some(st) => (st.own_gb, st.other_gb, st.total_gb),
            None => match oscar_offload::machine::MemorySnapshot::read() {
                Ok(m) => (Some(0.0), Some(gb(m.foreign() + m.own_footprint)), Some(gb(m.physical))),
                Err(_) => (None, None, None),
            },
        };
        if let (Some(own), Some(other), Some(total)) = figures {
            s.memory.refresh(own, other, total);
        }
    }

    /// Each row from the disk, the downloads and the served child; a finished download leaves the map, a failed one
    /// the trouble line.
    fn refresh_rows(&self) {
        let s = self.ivars();
        let mut downloads = s.downloads.borrow_mut();
        let loading = self.status().filter(|st| st.phase == "load").map(|st| st.elapsed_s.unwrap_or(0.0));
        let physical_gb = oscar_offload::machine::MemorySnapshot::read().map_or(f64::MAX, |m| m.physical as f64 / GB);
        for row in s.rows.borrow().iter() {
            let (preset, name) = (row.ivars().preset, row.ivars().name.clone());
            let def = preset.def();
            let on_disk = model_store::variants_in_root(&model_store::managed_root(), def.family_dir)
                .into_iter()
                .find_map(|(variant, bytes)| (variant == def.production_variant).then_some(bytes));
            let size = on_disk.map_or(preset.help_size_gb(), |b| (b as f64 / GB).round() as u64);
            let download = downloads.get(&(preset as usize)).map(|d| {
                let p = d.progress.lock().unwrap_or_else(|e| e.into_inner());
                (p.done.clone(), p.total, p.fetched + p.kept)
            });
            let kind = match download {
                Some((Some(done), ..)) => {
                    downloads.remove(&(preset as usize));
                    if let Err(e) = done {
                        self.show_trouble(&format!("downloading {name}: {e}"));
                    }
                    continue;
                }
                Some((None, total, at)) => Kind::Downloading(
                    if total > 0 { at as f64 / total as f64 } else { 0.0 },
                    format!("{:.1} of {} GB", at as f64 / GB, (total as f64 / GB).round()),
                ),
                None if s.served.get() == Some(preset) => match loading {
                    Some(secs) => Kind::Loading(format!("loading {}:{:02}", secs as u64 / 60, secs as u64 % 60)),
                    None => Kind::Served,
                },
                // The whole GB shown is the largest that is not enough, so the figure and the rule agree.
                None if physical_gb < preset.def().least_gb + SYSTEM_GB => {
                    Kind::TooLarge((preset.def().least_gb + SYSTEM_GB).ceil() as u64 - 1)
                }
                None if on_disk.is_some() => Kind::OnDisk,
                None => Kind::Absent,
            };
            row.refresh(kind, format!("{size} GB"));
        }
        // No model served: a download said, and every row greyed, as "Pick a model" would then offer nothing.
        if s.served.get().is_none() && s.pending.borrow().is_none() {
            let unfit = s.rows.borrow().iter().all(|r| matches!(*r.ivars().kind.borrow(), Kind::TooLarge(_)));
            let line = match (unfit, downloads.is_empty()) {
                (true, _) => format!("No model fits in {} GB of RAM", physical_gb.round()),
                (false, false) => "Downloading".to_string(),
                (false, true) => "Pick a model".to_string(),
            };
            s.line.refresh(Line::new(&line, &[]));
        }
    }

    fn show(&self, line: Line, trouble: Option<&str>) {
        self.ivars().line.refresh(line);
        let own = self.ivars().own_trouble.borrow();
        self.set_trouble(own.as_deref().or(trouble));
    }

    /// Shown only with one, in its headline's words, the log's whole entry its tooltip; a new one is notified too.
    fn set_trouble(&self, trouble: Option<&str>) {
        let s = self.ivars();
        let entry = trouble.unwrap_or_default();
        let (warn, head) = child::headline(entry);
        if trouble.is_some() && s.trouble_line.toolTip().is_none_or(|t| t.to_string() != entry) {
            notify::post(head);
        }
        s.trouble.setHidden(trouble.is_none());
        s.trouble_line.refresh(Line::trouble(&fit(head), warn));
        s.trouble_line.setToolTip(Some(&NSString::from_str(entry)));
    }

    /// The child's last status line, none without a running child.
    fn status(&self) -> Option<Status> {
        self.ivars().child.borrow().as_ref().and_then(|c| c.status())
    }

    /// The pending model started once no child runs; the child's status line while one runs, its exit, with the
    /// last trouble line, after. A child that died on its own is restarted once; dying again within a minute it
    /// stays stopped with its reason, for the ticked row's click.
    fn read_status(&self) {
        let s = self.ivars();
        let mut child = s.child.borrow_mut();
        let exited = child.as_mut().and_then(|c| c.exited());
        if let Some((preset, quant)) = s.pending.borrow_mut().take_if(|_| child.is_none() || exited.is_some()) {
            match child::Child::spawn(&preset.label(), quant.as_deref(), &s.log) {
                Ok(spawned) => {
                    *child = Some(spawned);
                    s.served.set(Some(preset));
                    s.port.set(None);
                    self.show(Line::new("Starting", &[]), None);
                }
                Err(e) => self.show(Line::new("Pick a model", &[]), Some(&format!("{e:#}"))),
            }
            return;
        }
        let Some(running) = child.as_mut() else { return };
        let trouble = running.trouble();
        if let Some(exit) = exited {
            let how = exit.code().map_or("stopped".to_string(), |c| format!("exited {c}"));
            // No model served: its removal stopped the server on purpose.
            let Some(preset) = s.served.get() else {
                return self.show(Line::new("Pick a model", &[]), trouble.as_deref());
            };
            if let Some(t) = &trouble {
                self.show_trouble(t);
            }
            if s.restarted.get().is_none_or(|t| t.elapsed() > Duration::from_secs(60)) {
                s.restarted.set(Some(Instant::now()));
                *s.pending.borrow_mut() = Some((preset, Settings::load().unwrap_or_default().quant));
                return self.show(Line::new(&format!("Restarting: oscar serve {how}"), &[]), None);
            }
            // The dead child dropped: only the ticked row's click starts another.
            *child = None;
            return self.show(Line::new(&format!("Stopped: oscar serve {how}"), &[]), None);
        }
        let mut line = running.status().as_ref().map_or_else(|| Line::new("Starting", &[]), status_line);
        if let Some(port) = running.port() {
            let settings = Settings::load().unwrap_or_default();
            if s.port.replace(Some(port)) != Some(port) {
                let written =
                    s.served.get().map_or(Ok(()), |p| opencode::write(p, settings.quant.as_deref(), port).map(drop));
                if let Err(e) = written {
                    self.show_trouble(&format!("{e:#}"));
                }
            }
            let usual = settings.port(None);
            if port != usual {
                line.cells.push((port.to_string(), "port"));
                line.cells.push((usual.to_string(), "taken"));
            }
        }
        self.show(line, trouble.as_deref());
    }
}

/// In front, for an alert; its replacement, `activate`, needs macOS 14.
#[allow(deprecated)]
fn activate(mtm: MainThreadMarker) {
    NSApplication::sharedApplication(mtm).activateIgnoringOtherApps(true);
}

/// The phase and its figures by column: tokens, the rate, the drive.
fn status_line(st: &Status) -> Line {
    let rate = |r: Option<f64>| r.map_or("-".to_string(), |r| format!("{r:.1}"));
    let count = |n: Option<u64>| thousands(n.unwrap_or(0));
    let drive = |r: Option<f64>| r.map_or("-".to_string(), |r| format!("{r:.1}"));
    match st.phase.as_str() {
        "load" => match st.elapsed_s {
            Some(s) => Line::new("Loading", &[(format!("{}:{:02}", s as u64 / 60, s as u64 % 60), "")]),
            None => Line::new("Loading", &[]),
        },
        "prefill" => Line::new(
            "Prefill",
            &[(count(st.tokens), "tokens"), (rate(st.tok_per_s), "tok/s"), (drive(st.nvme_gb_s), "GB/s")],
        ),
        "decode" => Line::new(
            "Decoding",
            &[(rate(st.tok_per_s), "tok/s"), (count(st.tokens), "tokens"), (drive(st.nvme_gb_s), "GB/s")],
        ),
        _ => match st.tok_per_s {
            Some(r) => Line::new("Idle", &[(format!("{r:.1}"), "tok/s last decode")]),
            None => Line::new("Idle", &[]),
        },
    }
}

fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// One session, one log; the status item with its menu and the timer, for the app's life.
pub fn main() -> anyhow::Result<()> {
    let mtm = MainThreadMarker::new().context("the app starts on the main thread")?;
    let dir = settings::dir();
    std::fs::create_dir_all(&dir)?;
    // The last launch's log stays as oscar.log.1.
    let log = dir.join("oscar.log");
    let _ = std::fs::rename(&log, dir.join("oscar.log.1"));
    std::fs::write(&log, b"")?;
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
    if setup::relocate(mtm)? {
        return Ok(());
    }
    let item = NSStatusBar::systemStatusBar().statusItemWithLength(NSVariableStatusItemLength);
    let plain = |title: &str| {
        let item = NSMenuItem::new(mtm);
        item.setTitle(&NSString::from_str(title));
        item.setEnabled(false);
        item
    };
    let trouble_line = line::StatusLine::new(mtm);
    let trouble = plain("");
    trouble.setView(Some(&trouble_line));
    trouble.setHidden(true);
    trouble.setEnabled(true);
    let login = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str("Start at Login"),
            Some(sel!(toggleLogin:)),
            &NSString::from_str(""),
        )
    };
    let this = App::new(
        mtm,
        State {
            log,
            child: RefCell::new(None),
            served: Cell::new(None),
            pending: RefCell::new(None),
            restarted: Cell::new(None),
            port: Cell::new(None),
            rows: RefCell::new(Vec::new()),
            downloads: RefCell::new(HashMap::new()),
            item,
            line: line::StatusLine::new(mtm),
            memory: memory::MemoryBar::new(mtm),
            trouble,
            trouble_line,
            own_trouble: RefCell::new(None),
            login,
            idle: icon::idle(),
            prefill: (0..icon::PREFILL_FRAMES).map(icon::prefill).collect(),
            decode: (0..icon::DECODE_FRAMES).map(icon::decode).collect(),
            tick: Cell::new(0),
            shown: Cell::new(std::ptr::null()),
        },
    );
    // The rows keep one order, smallest model first.
    let mut presets = ModelPreset::value_variants().to_vec();
    presets.sort_by_key(|p| p.help_size_gb());
    let rows = presets.iter().map(|&preset| {
        Row::new(
            mtm,
            rows::RowState {
                preset,
                name: rows::name(preset),
                kind: RefCell::new(Kind::Absent),
                size: RefCell::new(String::new()),
                app: this.clone(),
            },
        )
    });
    this.ivars().rows.borrow_mut().extend(rows);
    this.ivars().item.setMenu(Some(&this.menu()));
    this.start();
    this.refresh_rows();
    this.refresh_memory();
    notify::request_leave();
    if let Err(e) = setup::offer(this.mtm()) {
        this.show_trouble(&format!("{e:#}"));
    }
    // In the common modes, so the ticks run while the menu is open.
    let timer = unsafe {
        NSTimer::timerWithTimeInterval_target_selector_userInfo_repeats(TICK_S, &this, sel!(tick:), None, true)
    };
    unsafe { NSRunLoop::mainRunLoop().addTimer_forMode(&timer, NSRunLoopCommonModes) };
    app.run();
    Ok(())
}

#[cfg(test)]
#[test]
fn the_status_line_takes_its_three_forms_and_the_loads() {
    let st = |phase: &str, elapsed_s: Option<f64>, tokens: Option<u64>, tok_per_s: Option<f64>| Status {
        phase: phase.to_string(),
        elapsed_s,
        tokens,
        tok_per_s,
        nvme_gb_s: None,
        own_gb: None,
        other_gb: None,
        total_gb: None,
        level: None,
    };
    assert_eq!(status_line(&st("load", None, None, None)), Line::new("Loading", &[]));
    assert_eq!(status_line(&st("load", Some(130.0), None, None)), Line::new("Loading", &[("2:10".into(), "")]));
    assert_eq!(
        status_line(&st("prefill", Some(1.0), Some(1204), Some(661.2))).cells,
        [("1,204".to_string(), "tokens"), ("661.2".into(), "tok/s"), ("-".into(), "GB/s")]
    );
    assert_eq!(
        status_line(&st("decode", Some(1.0), Some(40), Some(128.04))).cells,
        [("128.0".to_string(), "tok/s"), ("40".into(), "tokens"), ("-".into(), "GB/s")]
    );
    assert_eq!(status_line(&st("idle", None, Some(40), Some(24.8))).cells, [("24.8".to_string(), "tok/s last decode")]);
    assert_eq!(status_line(&st("idle", None, None, None)), Line::new("Idle", &[]));
    assert_eq!((thousands(0), thousands(999), thousands(1_000_000)), ("0".into(), "999".into(), "1,000,000".into()));
}
