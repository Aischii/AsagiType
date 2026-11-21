//! Main window: title bar, menus, toolbar, project/page lists, canvas, balloon panel, dialogs.

mod editor;
mod history;
mod pipeline;

use crate::vision_worker::VisionWorker;
use crate::{
    credits,
    settings::{Settings, ROLES},
    theme,
};
use at2k::Manifest;
use editor::Drag;
use history::{History, PaintLayers};
use eframe::egui::{
    self, Color32, ColorImage, Key, Rect, Sense, TextureHandle, TextureOptions, Vec2,
    ViewportCommand,
};
use pipeline::{AutoRun, LiveClean};
use provider::{BalloonIn, BalloonOut, PageRequest, PRESETS};
use render::Renderer;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::mpsc::{channel, Receiver},
};

#[derive(PartialEq, Clone, Copy)]
enum Tool {
    Select,
    Draw,
}

struct TranslateJob {
    page: usize,
    only: Option<u32>,
    rx: Receiver<anyhow::Result<Vec<BalloonOut>>>,
}

#[derive(Default)]
struct SettingsForm {
    open: bool,
    draft: Option<Settings>,
    key: String,
}

#[derive(Default)]
struct ImportForm {
    open: bool,
    filter: String,
    chapters: Vec<at2k::WorkspaceChapter>,
    selected: Option<usize>,
    lang: String,
}

pub struct AsagiApp {
    settings: Settings,
    renderer: Renderer,
    projects: Vec<at2k::ProjectInfo>,
    path: Option<PathBuf>,
    manifest: Option<Manifest>,
    page: usize,
    page_tex: Option<(String, TextureHandle)>,
    letter_cache: HashMap<u64, Option<(TextureHandle, bool)>>,
    selected: Option<u32>,
    dirty: bool,
    zoom: f32,
    tool: Tool,
    drag: Option<Drag>,
    side_by_side: bool,
    show_text: bool,
    show_boxes: bool,
    status: String,
    job: Option<TranslateJob>,
    vision: Option<VisionWorker>,
    /// Clean layers produced since the last save, PNG by page index.
    /// Clean layer of the current page, recomputed from the boxes after edits.
    live: Option<LiveClean>,
    clean_inflight: Option<u64>,
    clean_wanted: Option<(u64, std::time::Instant)>,
    show_mask: bool,
    auto: Option<AutoRun>,
    auto_all: bool,
    settings_form: SettingsForm,
    import_form: ImportForm,
    about_open: bool,
    message: Option<(String, bool)>,
    history: History,
    /// Healing-brush layers loaded or painted this session, by page index.
    paint: PaintLayers,
    /// Pages whose paint layer changed since the last save.
    paint_dirty: HashSet<usize>,
}

impl AsagiApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::apply(&cc.egui_ctx);
        let settings = Settings::load();
        let mut renderer = Renderer::new();
        let mut status = "Ready".to_string();
        if Path::new(&settings.fonts_dir).is_dir() {
            let n = renderer.load_fonts_dir(Path::new(&settings.fonts_dir));
            status = format!("Ready. Loaded {n} fonts from {}", settings.fonts_dir);
        }
        let mut app = Self {
            projects: vec![],
            settings,
            renderer,
            path: None,
            manifest: None,
            page: 0,
            page_tex: None,
            letter_cache: HashMap::new(),
            selected: None,
            dirty: false,
            zoom: 0.5,
            tool: Tool::Select,
            drag: None,
            side_by_side: true,
            show_text: true,
            show_boxes: true,
            status,
            job: None,
            vision: None,
            live: None,
            clean_inflight: None,
            clean_wanted: None,
            show_mask: false,
            auto: None,
            auto_all: false,
            settings_form: SettingsForm::default(),
            import_form: ImportForm::default(),
            about_open: false,
            message: None,
            history: History::default(),
            paint: PaintLayers::new(),
            paint_dirty: HashSet::new(),
        };
        app.refresh_projects();
        app
    }

    fn error(&mut self, e: impl std::fmt::Display) {
        self.status = "Error".into();
        self.message = Some((e.to_string(), true));
    }

    fn refresh_projects(&mut self) {
        self.projects = at2k::list_projects(Path::new(&self.settings.projects_dir));
    }

    fn open_project(&mut self, path: PathBuf) {
        match at2k::open(&path) {
            Ok(mut m) => {
                self.status = format!("Opened {}", m.title);
                // Clean layers are regenerated from the boxes, so stored 0.4 layers are dropped.
                for p in &mut m.pages {
                    p.clean = None;
                }
                self.history.reset(&m.pages);
                self.paint.clear();
                self.paint_dirty.clear();
                self.manifest = Some(m);
                self.path = Some(path);
                self.page = 0;
                self.page_tex = None;
                self.live = None;
                self.clean_wanted = None;
                self.auto = None;
                self.selected = None;
                self.dirty = false;
            }
            Err(e) => self.error(format!("Could not open project:\n{e:#}")),
        }
    }

    fn save(&mut self) {
        let (Some(path), Some(m)) = (&self.path, &self.manifest) else {
            return;
        };
        let mut m = m.clone();
        let mut files = Vec::new();
        for &i in &self.paint_dirty {
            let (Some(page), Some(layer)) = (m.pages.get_mut(i), self.paint.get(&i)) else {
                continue;
            };
            if layer.pixels().any(|p| p.0[3] > 0) {
                let mut png = Vec::new();
                if let Err(e) =
                    layer.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                {
                    let msg = format!("Could not encode paint layer:\n{e:#}");
                    self.error(msg);
                    return;
                }
                let name = page.paint_entry_name();
                page.paint = Some(name.clone());
                files.push((name, png));
            } else {
                page.paint = None;
            }
        }
        let path = path.clone();
        match at2k::save_with_files(&path, &m, &files) {
            Ok(()) => {
                if let Some(cur) = &mut self.manifest {
                    for (cur, saved) in cur.pages.iter_mut().zip(&m.pages) {
                        cur.paint = saved.paint.clone();
                    }
                }
                self.paint_dirty.clear();
                self.dirty = false;
                self.status = format!("Saved {}", path.display());
                self.refresh_projects();
            }
            Err(e) => self.error(format!("Save failed:\n{e:#}")),
        }
    }

    fn import_folder(&mut self) {
        let Some(dir) = rfd::FileDialog::new()
            .set_title("Folder with page images")
            .pick_folder()
        else {
            return;
        };
        let images = at2k::image_files(&dir);
        let chapter = dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let series = dir
            .parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".into());
        self.create_project(
            series,
            chapter,
            String::new(),
            &images,
            self.settings.target_lang.clone(),
        );
    }

    fn create_project(
        &mut self,
        series: String,
        chapter: String,
        glossary: String,
        images: &[PathBuf],
        lang: String,
    ) {
        let out = Path::new(&self.settings.projects_dir)
            .join(at2k::project_file_name(&series, &chapter, &lang));
        let meta = at2k::NewProject {
            series,
            chapter,
            source_lang: "ja".into(),
            target_lang: lang,
            glossary,
            presets: self.settings.presets.clone(),
        };
        match at2k::create(&out, meta, images) {
            Ok(_) => {
                self.refresh_projects();
                self.open_project(out);
            }
            Err(e) => self.error(format!("Import failed:\n{e:#}")),
        }
    }

    /// Load a page's saved paint layer, or start an empty one. Returns false on read errors.
    #[allow(dead_code)] // used by the healing brush (0.6 part 4)
    fn ensure_paint(&mut self, index: usize) -> bool {
        if self.paint.contains_key(&index) {
            return true;
        }
        let page = self.manifest.as_ref().and_then(|m| m.pages.get(index)).cloned();
        let (Some(path), Some(page)) = (self.path.clone(), page) else {
            return false;
        };
        let layer = match &page.paint {
            Some(entry) => match at2k::read_entry(&path, entry)
                .and_then(|b| Ok(image::load_from_memory(&b)?.to_rgba8()))
            {
                Ok(img) => img,
                Err(e) => {
                    self.error(format!("Could not read paint layer of {}:\n{e:#}", page.id));
                    return false;
                }
            },
            None => image::RgbaImage::new(page.width, page.height),
        };
        self.paint.insert(index, layer);
        true
    }

    /// True while a drag, a held mouse button or a focused text field may still change the page.
    fn gesture_active(&self, ctx: &egui::Context) -> bool {
        self.drag.is_some() || ctx.egui_wants_keyboard_input() || ctx.input(|i| i.pointer.any_down())
    }

    fn record_history(&mut self, ctx: &egui::Context) {
        let hold = self.gesture_active(ctx).then_some(self.page);
        if let Some(m) = &self.manifest {
            self.history.observe(&m.pages, hold);
        }
    }

    fn undo_redo(&mut self, redo: bool) {
        let Some(m) = &mut self.manifest else { return };
        let changed = if redo {
            self.history.redo(&mut m.pages, &mut self.paint)
        } else {
            self.history.undo(&mut m.pages, &mut self.paint)
        };
        let Some(page) = changed else { return };
        self.paint_dirty.insert(page);
        self.page = page;
        let ids: Vec<u32> = m.pages[page].balloons.iter().map(|b| b.id).collect();
        if self.selected.is_some_and(|id| !ids.contains(&id)) {
            self.selected = None;
        }
        self.drag = None;
        self.dirty = true;
        self.status = if redo { "Redo" } else { "Undo" }.into();
    }

    fn cur_page(&self) -> Option<&at2k::Page> {
        self.manifest.as_ref()?.pages.get(self.page)
    }

    fn go_page(&mut self, i: isize) {
        let Some(m) = &self.manifest else { return };
        let n = m.pages.len() as isize;
        self.page = i.clamp(0, n - 1) as usize;
        self.selected = None;
    }

    fn ensure_page_texture(&mut self, ctx: &egui::Context) {
        let (Some(path), Some(page)) = (self.path.clone(), self.cur_page().cloned()) else {
            return;
        };
        let key = format!("{}#{}", path.display(), page.image);
        if self.page_tex.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let loaded = at2k::read_entry(&path, &page.image)
            .and_then(|bytes| Ok(image::load_from_memory(&bytes)?.to_rgba8()));
        match loaded {
            Ok(img) => {
                let size = [img.width() as usize, img.height() as usize];
                let ci = ColorImage::from_rgba_unmultiplied(size, img.as_raw());
                self.page_tex = Some((key, ctx.load_texture("page", ci, TextureOptions::LINEAR)));
            }
            Err(e) => {
                self.page_tex = None;
                self.error(format!("Could not read page {}:\n{e:#}", page.id));
            }
        }
    }

    fn start_translate(&mut self, only: Option<u32>) {
        self.start_translate_page(self.page, only);
    }

    fn start_translate_page(&mut self, page_idx: usize, only: Option<u32>) {
        if self.job.is_some() {
            return;
        }
        let Some(m) = &self.manifest else { return };
        let Some(page) = m.pages.get(page_idx) else {
            return;
        };
        if page.balloons.is_empty() {
            self.message = Some((
                "Draw balloons on this page first (Tool: Draw), then type the source text.".into(),
                false,
            ));
            return;
        }
        let previous_lines = if page_idx > 0 {
            let prev = &m.pages[page_idx - 1].balloons;
            {
                let lines: Vec<String> = prev
                    .iter()
                    .map(|b| b.text.clone())
                    .filter(|t| !t.is_empty())
                    .collect();
                lines[lines.len().saturating_sub(4)..].to_vec()
            }
        } else {
            vec![]
        };
        let req = PageRequest {
            series: m.series.clone(),
            page_id: page.id.clone(),
            source_lang: m.source_lang.clone(),
            target_lang: m.target_lang.clone(),
            glossary: m.glossary.clone(),
            rules: m.rules.clone(),
            balloons: page
                .balloons
                .iter()
                .filter(|b| !b.ignore)
                .map(|b| BalloonIn {
                    id: b.id,
                    kind: b.kind.clone(),
                    source_text: b.source_text.clone(),
                    text_box: b.text_box,
                })
                .collect(),
            previous_lines,
        };
        let cfg = self.settings.provider.clone();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(provider::translate_page(&cfg, &req));
        });
        self.status = format!(
            "Translating page {} with {} ({})...",
            page.id,
            cfg_label(&self.settings),
            self.settings.provider.model
        );
        self.job = Some(TranslateJob {
            page: page_idx,
            only,
            rx,
        });
    }

    fn poll_job(&mut self) {
        let Some(job) = &self.job else { return };
        let Ok(result) = job.rx.try_recv() else {
            return;
        };
        let (page_idx, only) = (job.page, job.only);
        self.job = None;
        match result {
            Ok(out) => {
                let Some(m) = &mut self.manifest else { return };
                let page = &mut m.pages[page_idx];
                let mut n = 0;
                for o in out {
                    if only.is_some_and(|id| id != o.id) {
                        continue;
                    }
                    if let Some(b) = page.balloons.iter_mut().find(|b| b.id == o.id) {
                        b.text = o.text;
                        n += 1;
                    }
                }
                self.dirty = true;
                self.status = format!("Translated {n} balloon(s) on page {}", page.id);
            }
            Err(e) => self.error(format!("Translation failed:\n{e:#}")),
        }
    }

    fn delete_selected(&mut self) {
        let Some(id) = self.selected else { return };
        let page = self.page;
        if let Some(m) = &mut self.manifest {
            if m.pages[page]
                .balloons
                .iter()
                .any(|b| b.id == id && b.locked)
            {
                self.status = "That box is locked.".into();
                return;
            }
            m.pages[page].balloons.retain(|b| b.id != id);
            self.selected = None;
            self.dirty = true;
        }
    }

    // ---------- UI pieces ----------

    fn title_bar(&mut self, root: &mut egui::Ui) {
        let ctx = &root.ctx().clone();
        egui::Panel::top("title")
            .exact_size(22.0)
            .frame(egui::Frame::new().fill(theme::FACE))
            .show(root, |ui| {
                let rect = ui.max_rect().shrink2(egui::vec2(2.0, 2.0));
                let focused = ctx.input(|i| i.viewport().focused.unwrap_or(true));
                let (l, r) = if focused {
                    (theme::ASAGI, theme::ASAGI_LIGHT)
                } else {
                    (theme::INACTIVE, theme::INACTIVE_LIGHT)
                };
                theme::gradient(ui.painter(), rect, l, r);
                let drag = ui.interact(rect, ui.scope_id().with("drag"), Sense::click_and_drag());
                if drag.drag_started() {
                    ctx.send_viewport_cmd(ViewportCommand::StartDrag);
                }
                if drag.double_clicked() {
                    let max = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                    ctx.send_viewport_cmd(ViewportCommand::Maximized(!max));
                }
                let mut title = "AsagiType 2000".to_string();
                if let Some(m) = &self.manifest {
                    title = format!(
                        "AsagiType 2000 - {}{}",
                        m.title,
                        if self.dirty { " *" } else { "" }
                    );
                }
                ui.painter().text(
                    rect.left_center() + egui::vec2(6.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    title,
                    egui::FontId::proportional(12.0),
                    Color32::WHITE,
                );
                let size = egui::vec2(16.0, 14.0);
                let mut x = rect.right() - size.x - 2.0;
                for (label, cmd) in [("x", 0), ("[]", 1), ("_", 2)] {
                    let b =
                        Rect::from_min_size(egui::pos2(x, rect.center().y - size.y / 2.0), size);
                    let resp = ui.interact(b, ui.scope_id().with(label), Sense::click());
                    ui.painter().rect_filled(b, 0.0, theme::FACE);
                    theme::bevel(ui.painter(), b, !resp.is_pointer_button_down_on());
                    ui.painter().text(
                        b.center(),
                        egui::Align2::CENTER_CENTER,
                        label,
                        egui::FontId::monospace(9.0),
                        Color32::BLACK,
                    );
                    if resp.clicked() {
                        match cmd {
                            0 => ctx.send_viewport_cmd(ViewportCommand::Close),
                            1 => {
                                let max = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                                ctx.send_viewport_cmd(ViewportCommand::Maximized(!max));
                            }
                            _ => ctx.send_viewport_cmd(ViewportCommand::Minimized(true)),
                        }
                    }
                    x -= size.x + if cmd == 0 { 4.0 } else { 2.0 };
                }
            });
    }

    fn menu_bar(&mut self, root: &mut egui::Ui) {
        let ctx = &root.ctx().clone();
        egui::Panel::top("menu").show(root, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Import chapter from workspace...").clicked() {
                        self.open_import();
                        ui.close();
                    }
                    if ui.button("Import folder of pages...").clicked() {
                        self.import_folder();
                        ui.close();
                    }
                    if ui.button("Open .at2k...").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("AsagiType project", &["at2k"])
                            .pick_file()
                        {
                            self.open_project(p);
                        }
                        ui.close();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(self.manifest.is_some(), egui::Button::new("Save  (Ctrl+S)"))
                        .clicked()
                    {
                        self.save();
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(ViewportCommand::Close);
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui
                        .add_enabled(self.history.can_undo(), egui::Button::new("Undo  (Ctrl+Z)"))
                        .clicked()
                    {
                        self.undo_redo(false);
                        ui.close();
                    }
                    if ui
                        .add_enabled(self.history.can_redo(), egui::Button::new("Redo  (Ctrl+Y)"))
                        .clicked()
                    {
                        self.undo_redo(true);
                        ui.close();
                    }
                });
                ui.menu_button("Page", |ui| {
                    let busy = self.vision.as_ref().is_some_and(|v| v.pending > 0);
                    if ui
                        .add_enabled(!busy, egui::Button::new("Detect + OCR page  (F5)"))
                        .clicked()
                    {
                        self.detect_pages(ctx, vec![self.page], false);
                        ui.close();
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("Detect + OCR whole chapter"))
                        .clicked()
                    {
                        let n = self.manifest.as_ref().map(|m| m.pages.len()).unwrap_or(0);
                        self.detect_pages(ctx, (0..n).collect(), false);
                        ui.close();
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("Redetect page (replace balloons)"))
                        .clicked()
                    {
                        self.detect_pages(ctx, vec![self.page], true);
                        ui.close();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(self.auto.is_none(), egui::Button::new("Auto: this page"))
                        .clicked()
                    {
                        self.start_auto(vec![self.page]);
                        ui.close();
                    }
                    if ui
                        .add_enabled(self.auto.is_none(), egui::Button::new("Auto: all pages"))
                        .clicked()
                    {
                        let n = self.manifest.as_ref().map(|m| m.pages.len()).unwrap_or(0);
                        self.start_auto((0..n).collect());
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Translate page  (F6)").clicked() {
                        self.start_translate(None);
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            self.selected.is_some(),
                            egui::Button::new("Retranslate balloon"),
                        )
                        .clicked()
                    {
                        self.start_translate(self.selected);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Previous page  (PgUp)").clicked() {
                        self.go_page(self.page as isize - 1);
                        ui.close();
                    }
                    if ui.button("Next page  (PgDn)").clicked() {
                        self.go_page(self.page as isize + 1);
                        ui.close();
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.side_by_side, "Original | Translated side by side");
                    ui.checkbox(&mut self.show_text, "Show translation (single pane)");
                    ui.checkbox(&mut self.show_boxes, "Show boxes");
                    ui.checkbox(&mut self.show_mask, "Show clean mask (red)");
                });
                ui.menu_button("Tools", |ui| {
                    if ui.button("Settings...").clicked() {
                        self.settings_form = SettingsForm {
                            open: true,
                            draft: Some(self.settings.clone()),
                            key: String::new(),
                        };
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("About AsagiType 2000").clicked() {
                        self.about_open = true;
                        ui.close();
                    }
                });
            });
        });
    }

    fn toolbar(&mut self, root: &mut egui::Ui) {
        egui::Panel::top("toolbar").show(root, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Import").clicked() {
                    self.open_import();
                }
                if ui
                    .add_enabled(self.dirty, egui::Button::new("Save"))
                    .clicked()
                {
                    self.save();
                }
                ui.separator();
                if ui.button("< Prev").clicked() {
                    self.go_page(self.page as isize - 1);
                }
                let label = match &self.manifest {
                    Some(m) if !m.pages.is_empty() => format!(
                        "{}  ({} / {})",
                        m.pages[self.page].id,
                        self.page + 1,
                        m.pages.len()
                    ),
                    _ => "-- / --".into(),
                };
                ui.add(egui::Label::new(label).extend());
                if ui.button("Next >").clicked() {
                    self.go_page(self.page as isize + 1);
                }
                ui.separator();
                ui.selectable_value(&mut self.tool, Tool::Select, "Select")
                    .on_hover_text("V");
                ui.selectable_value(&mut self.tool, Tool::Draw, "Draw balloon")
                    .on_hover_text("D");
                ui.separator();
                let has_project = self.manifest.is_some();
                if ui
                    .add_enabled(
                        has_project && self.auto.is_none(),
                        egui::Button::new(egui::RichText::new("Auto").strong()),
                    )
                    .on_hover_text("Run the missing steps: detect + OCR, translate, clean")
                    .clicked()
                {
                    let pages = if self.auto_all {
                        (0..self.manifest.as_ref().map(|m| m.pages.len()).unwrap_or(0)).collect()
                    } else {
                        vec![self.page]
                    };
                    self.start_auto(pages);
                }
                egui::ComboBox::from_id_salt("auto_scope")
                    .selected_text(if self.auto_all {
                        "All pages"
                    } else {
                        "This page"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.auto_all, false, "This page");
                        ui.selectable_value(&mut self.auto_all, true, "All pages");
                    });
                ui.separator();
                let detecting = self.vision.as_ref().map(|v| v.pending).unwrap_or(0);
                let label = if detecting > 0 {
                    format!("Detecting ({detecting})...")
                } else {
                    "Detect + OCR".to_string()
                };
                if ui
                    .add_enabled(
                        detecting == 0 && self.manifest.is_some(),
                        egui::Button::new(label),
                    )
                    .clicked()
                {
                    let ctx = ui.ctx().clone();
                    self.detect_pages(&ctx, vec![self.page], false);
                }
                let busy = self.job.is_some();
                if ui
                    .add_enabled(
                        !busy,
                        egui::Button::new(if busy {
                            "Translating..."
                        } else {
                            "Translate page"
                        }),
                    )
                    .clicked()
                {
                    self.start_translate(None);
                }
                ui.separator();
                ui.checkbox(&mut self.show_text, "Translation");
                ui.checkbox(&mut self.show_boxes, "Boxes");
                ui.separator();
                ui.label("Zoom");
                egui::ComboBox::from_id_salt("zoom")
                    .selected_text(format!("{:.0}%", self.zoom * 100.0))
                    .show_ui(ui, |ui| {
                        for z in [0.25, 0.35, 0.5, 0.75, 1.0] {
                            ui.selectable_value(&mut self.zoom, z, format!("{:.0}%", z * 100.0));
                        }
                    });
            });
        });
    }

    fn status_bar(&mut self, root: &mut egui::Ui) {
        egui::Panel::bottom("status")
            .exact_size(22.0)
            .show(root, |ui| {
                ui.horizontal_centered(|ui| {
                    let w = ui.available_width();
                    let (r1, _) =
                        ui.allocate_exact_size(egui::vec2(w - 360.0, 18.0), Sense::hover());
                    let (r2, _) = ui.allocate_exact_size(egui::vec2(220.0, 18.0), Sense::hover());
                    let (r3, _) = ui.allocate_exact_size(egui::vec2(120.0, 18.0), Sense::hover());
                    let font = egui::FontId::proportional(12.0);
                    let prov = format!(
                        "{} / {}",
                        cfg_label(&self.settings),
                        self.settings.provider.model
                    );
                    let saved = if self.dirty { "Modified" } else { "Saved" };
                    for (r, t) in [(r1, self.status.as_str()), (r2, prov.as_str()), (r3, saved)] {
                        theme::bevel(ui.painter(), r, false);
                        ui.painter().with_clip_rect(r.shrink(2.0)).text(
                            r.left_center() + egui::vec2(4.0, 0.0),
                            egui::Align2::LEFT_CENTER,
                            t,
                            font.clone(),
                            Color32::BLACK,
                        );
                    }
                });
            });
    }

    fn left_panel(&mut self, root: &mut egui::Ui) {
        egui::Panel::left("left")
            .resizable(true)
            .default_size(220.0)
            .show(root, |ui| {
                theme::caption(ui, "Projects");
                let mut open = None;
                egui::ScrollArea::vertical()
                    .id_salt("projects")
                    .max_height(ui.available_height() * 0.4)
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        if self.projects.is_empty() {
                            ui.weak("(no projects yet: File > Import)");
                        }
                        for p in &self.projects {
                            let sel = self.path.as_deref() == Some(p.path.as_path());
                            if ui
                                .selectable_label(sel, &p.name)
                                .on_hover_text(p.path.display().to_string())
                                .clicked()
                            {
                                open = Some(p.path.clone());
                            }
                        }
                    });
                if let Some(p) = open {
                    if !self.dirty || self.path.as_ref() == Some(&p) {
                        self.open_project(p);
                    } else {
                        self.message = Some((
                            "Save the current project before opening another.".into(),
                            false,
                        ));
                    }
                }
                theme::caption(ui, "Pages");
                let mut go = None;
                egui::ScrollArea::vertical()
                    .id_salt("pages")
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        if let Some(m) = &self.manifest {
                            for (i, p) in m.pages.iter().enumerate() {
                                let done = !p.balloons.is_empty()
                                    && p.balloons.iter().all(|b| !b.text.trim().is_empty());
                                let meta = if p.balloons.is_empty() {
                                    "empty".to_string()
                                } else {
                                    format!(
                                        "{} balloons{}",
                                        p.balloons.len(),
                                        if done { ", done" } else { "" }
                                    )
                                };
                                if ui
                                    .selectable_label(
                                        i == self.page,
                                        format!("{}   {}", p.id, meta),
                                    )
                                    .clicked()
                                {
                                    go = Some(i);
                                }
                            }
                        }
                    });
                if let Some(i) = go {
                    self.go_page(i as isize);
                }
            });
    }

    fn right_panel(&mut self, root: &mut egui::Ui) {
        egui::Panel::right("right")
            .resizable(true)
            .default_size(300.0)
            .show(root, |ui| {
                theme::caption(ui, "Balloons");
                let page_idx = self.page;
                let mut select = None;
                egui::ScrollArea::vertical()
                    .id_salt("balloons")
                    .max_height(ui.available_height() * 0.4)
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        if let Some(p) = self.manifest.as_ref().and_then(|m| m.pages.get(page_idx))
                        {
                            for b in &p.balloons {
                                let shown = if !b.text.is_empty() {
                                    &b.text
                                } else if !b.source_text.is_empty() {
                                    &b.source_text
                                } else {
                                    "(empty)"
                                };
                                let line: String = shown
                                    .lines()
                                    .next()
                                    .unwrap_or("")
                                    .chars()
                                    .take(48)
                                    .collect();
                                if ui
                                    .selectable_label(
                                        self.selected == Some(b.id),
                                        format!("{}. {}", b.id, line),
                                    )
                                    .clicked()
                                {
                                    select = Some(b.id);
                                }
                            }
                        }
                    });
                if select.is_some() {
                    self.selected = select;
                }
                theme::caption(ui, "Properties");
                egui::ScrollArea::vertical()
                    .id_salt("props")
                    .show(ui, |ui| self.properties(ui));
            });
    }

    // ---------- dialogs ----------

    fn open_import(&mut self) {
        let ws = PathBuf::from(&self.settings.workspace_root);
        self.import_form = ImportForm {
            open: true,
            filter: String::new(),
            chapters: at2k::workspace_chapters(&ws),
            selected: None,
            lang: self.settings.target_lang.clone(),
        };
        self.status = format!("{} chapters in workspace", self.import_form.chapters.len());
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        // Import
        if self.import_form.open {
            let mut open = true;
            let mut do_import = None;
            egui::Window::new("Import chapter from workspace").open(&mut open).default_size([560.0, 420.0]).collapsible(false).show(ctx, |ui| {
                let f = &mut self.import_form;
                ui.horizontal(|ui| {
                    ui.label("Filter");
                    ui.text_edit_singleline(&mut f.filter);
                    ui.label("Target");
                    egui::ComboBox::from_id_salt("ilang").selected_text(f.lang.clone()).show_ui(ui, |ui| {
                        ui.selectable_value(&mut f.lang, "en".to_string(), "English");
                        ui.selectable_value(&mut f.lang, "id".to_string(), "Indonesian");
                    });
                });
                let q = f.filter.to_lowercase();
                egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                    ui.set_min_width(ui.available_width());
                    let mut last = String::new();
                    for (i, c) in f.chapters.iter().enumerate() {
                        if !q.is_empty() && !format!("{} {}", c.series, c.chapter).to_lowercase().contains(&q) {
                            continue;
                        }
                        if c.series != last {
                            ui.strong(&c.series);
                            last = c.series.clone();
                        }
                        let r = ui.selectable_label(f.selected == Some(i), format!("    {}   ({} pages)", c.chapter, c.pages));
                        if r.clicked() {
                            f.selected = Some(i);
                        }
                        if r.double_clicked() {
                            do_import = Some(i);
                        }
                    }
                });
                ui.weak("Source scans and glossaries are copied into a new .at2k project. The workspace is not changed.");
                ui.horizontal(|ui| {
                    if ui.add_enabled(f.selected.is_some(), egui::Button::new("Import")).clicked() {
                        do_import = f.selected;
                    }
                });
            });
            self.import_form.open = open;
            if let Some(i) = do_import {
                let c = &self.import_form.chapters[i];
                let ws = PathBuf::from(&self.settings.workspace_root);
                let glossary = at2k::workspace_glossary(&ws, &c.series_dir);
                let images = at2k::image_files(&c.source_dir);
                let (series, chapter, lang) = (
                    c.series.clone(),
                    c.chapter.clone(),
                    self.import_form.lang.clone(),
                );
                self.import_form.open = false;
                self.create_project(series, chapter, glossary, &images, lang);
            }
        }

        // Settings
        if self.settings_form.open {
            let mut open = true;
            let mut apply = false;
            let families = self.renderer.families();
            egui::Window::new("Settings")
                .open(&mut open)
                .default_width(480.0)
                .collapsible(false)
                .show(ctx, |ui| {
                    let form = &mut self.settings_form;
                    let Some(d) = form.draft.as_mut() else { return };
                    egui::Grid::new("settings")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Workspace");
                            ui.add(
                                egui::TextEdit::singleline(&mut d.workspace_root)
                                    .desired_width(320.0),
                            );
                            ui.end_row();
                            ui.label("Projects");
                            ui.add(
                                egui::TextEdit::singleline(&mut d.projects_dir)
                                    .desired_width(320.0),
                            );
                            ui.end_row();
                            ui.label("Models folder");
                            ui.add(
                                egui::TextEdit::singleline(&mut d.models_dir).desired_width(320.0),
                            )
                            .on_hover_text("Holds bubble-detector/ and manga-ocr/");
                            ui.end_row();
                            ui.label("Fonts folder");
                            ui.add(
                                egui::TextEdit::singleline(&mut d.fonts_dir).desired_width(320.0),
                            );
                            ui.end_row();
                            ui.label("Provider");
                            let current = PRESETS
                                .iter()
                                .find(|p| p.id == d.provider.name)
                                .map(|p| p.label)
                                .unwrap_or("Custom");
                            egui::ComboBox::from_id_salt("preset")
                                .selected_text(current)
                                .show_ui(ui, |ui| {
                                    for p in PRESETS {
                                        if ui
                                            .selectable_label(d.provider.name == p.id, p.label)
                                            .clicked()
                                        {
                                            d.provider.name = p.id.into();
                                            if !p.base_url.is_empty() {
                                                d.provider.base_url = p.base_url.into();
                                                d.provider.model = p.model.into();
                                            }
                                        }
                                    }
                                });
                            ui.end_row();
                            ui.label("Base URL");
                            ui.add(
                                egui::TextEdit::singleline(&mut d.provider.base_url)
                                    .desired_width(320.0),
                            );
                            ui.end_row();
                            ui.label("Model");
                            ui.add(
                                egui::TextEdit::singleline(&mut d.provider.model)
                                    .desired_width(320.0),
                            );
                            ui.end_row();
                            ui.label("API key");
                            ui.add(
                                egui::TextEdit::singleline(&mut form.key)
                                    .password(true)
                                    .hint_text("leave blank to keep")
                                    .desired_width(320.0),
                            );
                            ui.end_row();
                            ui.label("");
                            ui.weak(if provider::api_key(&d.provider.name).is_some() {
                                "A key is stored for this provider."
                            } else {
                                "No key stored for this provider."
                            });
                            ui.end_row();
                            ui.label("Target");
                            egui::ComboBox::from_id_salt("slang")
                                .selected_text(d.target_lang.clone())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut d.target_lang,
                                        "en".to_string(),
                                        "English",
                                    );
                                    ui.selectable_value(
                                        &mut d.target_lang,
                                        "id".to_string(),
                                        "Indonesian",
                                    );
                                });
                            ui.end_row();
                            for role in ROLES {
                                ui.label(*role);
                                let cur = d.role_fonts.entry(role.to_string()).or_default();
                                egui::ComboBox::from_id_salt(*role)
                                    .width(320.0)
                                    .selected_text(cur.clone())
                                    .show_ui(ui, |ui| {
                                        for f in &families {
                                            ui.selectable_value(cur, f.clone(), f);
                                        }
                                    });
                                ui.end_row();
                            }
                        });
                    ui.weak("API keys are stored in Windows Credential Manager, never in files.");
                    ui.horizontal(|ui| {
                        apply = ui.button("OK").clicked();
                    });
                });
            self.settings_form.open = open && !apply;
            if apply {
                if let Some(d) = self.settings_form.draft.take() {
                    if !self.settings_form.key.trim().is_empty() {
                        if let Err(e) =
                            provider::set_api_key(&d.provider.name, &self.settings_form.key)
                        {
                            self.error(format!("Could not store key: {e:#}"));
                        }
                    }
                    self.settings_form.key.clear();
                    if d.fonts_dir != self.settings.fonts_dir && Path::new(&d.fonts_dir).is_dir() {
                        self.renderer.load_fonts_dir(Path::new(&d.fonts_dir));
                    }
                    self.settings = d;
                    self.letter_cache.clear();
                    match self.settings.save() {
                        Ok(()) => self.status = "Settings saved".into(),
                        Err(e) => self.error(e),
                    }
                    self.refresh_projects();
                }
            }
        }

        // About
        if self.about_open {
            let mut open = true;
            egui::Window::new("About AsagiType 2000")
                .open(&mut open)
                .default_size([560.0, 480.0])
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        let (r, _) = ui.allocate_exact_size(egui::vec2(40.0, 40.0), Sense::hover());
                        ui.painter().rect_filled(r, 0.0, theme::ASAGI);
                        theme::bevel(ui.painter(), r, true);
                        ui.painter().text(
                            r.center(),
                            egui::Align2::CENTER_CENTER,
                            "AT",
                            egui::FontId::proportional(18.0),
                            Color32::WHITE,
                        );
                        ui.vertical(|ui| {
                            ui.strong(format!("AsagiType 2000  v{}", env!("CARGO_PKG_VERSION")));
                            ui.label("Pure-Rust manga scanlation editor for AsagiScans.");
                            ui.hyperlink_to(
                                "github.com/Aischii/AsagiType",
                                "https://github.com/Aischii/AsagiType",
                            );
                            ui.label("Licence: MIT OR Apache-2.0");
                        });
                    });
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for s in credits::SECTIONS {
                            theme::caption(ui, s.title);
                            ui.weak(s.note);
                            egui::Grid::new(s.title)
                                .striped(true)
                                .num_columns(3)
                                .spacing([10.0, 2.0])
                                .show(ui, |ui| {
                                    for c in s.items {
                                        ui.hyperlink_to(c.name, c.url);
                                        ui.label(c.author);
                                        ui.label(format!("{}  ({})", c.licence, c.role));
                                        ui.end_row();
                                    }
                                });
                            ui.add_space(6.0);
                        }
                    });
                });
            self.about_open = open;
        }

        // Message box
        if let Some((text, is_err)) = self.message.clone() {
            let mut close = false;
            egui::Window::new(if is_err { "Error" } else { "AsagiType 2000" })
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        let (r, _) = ui.allocate_exact_size(egui::vec2(32.0, 32.0), Sense::hover());
                        let col = if is_err {
                            Color32::from_rgb(0xc0, 0, 0)
                        } else {
                            Color32::from_rgb(0x10, 0x60, 0xc0)
                        };
                        ui.painter().circle_filled(r.center(), 15.0, col);
                        ui.painter().text(
                            r.center(),
                            egui::Align2::CENTER_CENTER,
                            if is_err { "X" } else { "i" },
                            egui::FontId::proportional(16.0),
                            Color32::WHITE,
                        );
                        ui.label(&text);
                    });
                    ui.vertical_centered(|ui| {
                        close = ui.button("   OK   ").clicked();
                    });
                });
            if close {
                self.message = None;
            }
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        let typing = ctx.egui_wants_keyboard_input();
        if !typing {
            let (undo, redo, select, draw) = ctx.input_mut(|i| {
                let redo = i.consume_key(egui::Modifiers::COMMAND, Key::Y)
                    || i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, Key::Z);
                (
                    i.consume_key(egui::Modifiers::COMMAND, Key::Z),
                    redo,
                    i.consume_key(egui::Modifiers::NONE, Key::V),
                    i.consume_key(egui::Modifiers::NONE, Key::D),
                )
            });
            if undo {
                self.undo_redo(false);
            }
            if redo {
                self.undo_redo(true);
            }
            if select {
                self.tool = Tool::Select;
            }
            if draw {
                self.tool = Tool::Draw;
            }
        }
        let (save, f5, f6, pgdn, pgup, del, esc) = ctx.input(|i| {
            (
                i.modifiers.ctrl && i.key_pressed(Key::S),
                i.key_pressed(Key::F5),
                i.key_pressed(Key::F6),
                i.key_pressed(Key::PageDown),
                i.key_pressed(Key::PageUp),
                i.key_pressed(Key::Delete),
                i.key_pressed(Key::Escape),
            )
        });
        if save {
            self.save();
        }
        if f5 {
            self.detect_pages(ctx, vec![self.page], false);
        }
        if f6 {
            self.start_translate(None);
        }
        if !typing {
            if pgdn {
                self.go_page(self.page as isize + 1);
            }
            if pgup {
                self.go_page(self.page as isize - 1);
            }
            if del {
                self.delete_selected();
            }
        }
        if esc {
            self.message = None;
            self.tool = Tool::Select;
        }
    }
}

fn cfg_label(s: &Settings) -> &'static str {
    PRESETS
        .iter()
        .find(|p| p.id == s.provider.name)
        .map(|p| p.label)
        .unwrap_or("Custom")
}

impl eframe::App for AsagiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_job();
        self.poll_vision(&ctx);
        self.step_auto(&ctx);
        self.update_live_clean(&ctx);
        if self.job.is_some() || self.vision_busy() || self.auto.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
        self.shortcuts(&ctx);
        self.nudge(&ctx);
        self.title_bar(ui);
        self.menu_bar(ui);
        self.toolbar(ui);
        self.status_bar(ui);
        self.left_panel(ui);
        self.right_panel(ui);
        self.canvas(ui);
        self.dialogs(&ctx);
        self.record_history(&ctx);
        self.auto_dialog(&ctx);
    }
}
