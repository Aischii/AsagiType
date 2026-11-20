//! Detection + OCR, live cleaning, and the Auto pipeline (detect -> translate)
//! with a progress dialog and Cancel.
//!
//! Cleaning is not stored: the clean layer of the current page is recomputed from the
//! current boxes shortly after any edit, so deleting or moving a box restores the original.

use super::AsagiApp;
use crate::vision_worker::{Outcome, Request, VisionWorker};
use at2k::{Balloon, BubbleShape};
use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};
use std::{
    collections::VecDeque,
    hash::{Hash, Hasher},
    path::PathBuf,
    time::{Duration, Instant},
};
use vision::clean::{CleanBox, Shape, ShapeStyle};

/// Wait this long after the last edit before recleaning.
const CLEAN_DEBOUNCE: Duration = Duration::from_millis(150);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Stage {
    Detect,
    Translate,
}

impl Stage {
    fn label(self) -> &'static str {
        match self {
            Stage::Detect => "detecting text and running OCR",
            Stage::Translate => "translating",
        }
    }
}

pub(super) struct AutoRun {
    queue: VecDeque<usize>,
    current: Option<usize>,
    /// Stages already started for the current page, so a stage that finds nothing is not repeated.
    tried: Vec<Stage>,
    stage: Option<Stage>,
    total: usize,
    done: usize,
    cancel: bool,
}

/// The clean layer shown for one page and the box state it was computed from.
pub(super) struct LiveClean {
    pub page: usize,
    pub sig: u64,
    pub tex: TextureHandle,
    pub mask_tex: TextureHandle,
    /// Balloon ids whose text touches an outline and needs a manual touch-up.
    pub touch_up: Vec<u32>,
}

/// Stages a page still needs, in order. Cleaning is live, so it is never a stage.
pub(super) fn missing_stages(page: &at2k::Page) -> Vec<Stage> {
    if page.raw_only {
        return vec![];
    }
    let mut out = vec![];
    if page.balloons.is_empty() {
        out.push(Stage::Detect);
    }
    let pending = page
        .balloons
        .iter()
        .filter(|b| !b.ignore)
        .any(|b| !b.source_text.trim().is_empty() && b.text.trim().is_empty());
    if page.balloons.is_empty() || pending {
        out.push(Stage::Translate);
    }
    out
}

/// Clean instructions for a balloon, or None when it is skipped (ignored or hidden).
pub(super) fn clean_box_for(b: &Balloon) -> Option<CleanBox> {
    if b.ignore || b.hidden {
        return None;
    }
    let mut c = CleanBox::auto(b.text_box, b.bubble);
    let shape = match b.shape {
        BubbleShape::Auto => None,
        BubbleShape::Rect => Some(Shape::Rect),
        BubbleShape::Rounded => Some(Shape::Rounded),
        BubbleShape::Ellipse => Some(Shape::Ellipse),
    };
    match shape {
        None => c.fill = b.fill.map(|f| [f[0], f[1], f[2]]),
        Some(shape) => {
            c.shape = Some(ShapeStyle {
                shape,
                fill: b.fill.unwrap_or([255, 255, 255, 255]),
                outline: b.outline.map(|o| (o.color, o.width)),
            })
        }
    }
    Some(c)
}

/// Fingerprint of everything that affects a page's clean layer.
fn clean_signature(page_idx: usize, page: &at2k::Page) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    page_idx.hash(&mut h);
    page.image.hash(&mut h);
    for b in &page.balloons {
        (
            b.id,
            b.text_box,
            b.bubble,
            b.ignore,
            b.hidden,
            format!("{:?}{:?}{:?}", b.shape, b.fill, b.outline),
        )
            .hash(&mut h);
    }
    h.finish()
}

fn rgba_texture(ctx: &egui::Context, name: &str, img: &image::RgbaImage) -> TextureHandle {
    let ci = ColorImage::from_rgba_unmultiplied(
        [img.width() as usize, img.height() as usize],
        img.as_raw(),
    );
    ctx.load_texture(name, ci, TextureOptions::LINEAR)
}

impl AsagiApp {
    fn ensure_worker(&mut self, ctx: &egui::Context) -> bool {
        let dir = PathBuf::from(&self.settings.models_dir);
        if self.vision.as_ref().is_none_or(|v| v.models_dir != dir) {
            if self.vision.as_ref().is_some_and(|v| v.pending > 0) {
                return false;
            }
            self.vision = Some(VisionWorker::spawn(dir, ctx.clone()));
            self.clean_inflight = None;
        }
        true
    }

    /// Detection or translation is running (live cleaning does not count).
    pub(super) fn vision_busy(&self) -> bool {
        let cleaning = self.clean_inflight.is_some() as usize;
        self.vision.as_ref().is_some_and(|v| v.pending > cleaning)
    }

    fn page_bytes(&mut self, page: usize) -> Option<Vec<u8>> {
        let path = self.path.clone()?;
        let image = self.manifest.as_ref()?.pages.get(page)?.image.clone();
        match at2k::read_entry(&path, &image) {
            Ok(b) => Some(b),
            Err(e) => {
                self.error(format!("Could not read {image}:\n{e:#}"));
                None
            }
        }
    }

    /// Queue detection + OCR. Pages that already have balloons are skipped unless `replace`.
    pub(super) fn detect_pages(&mut self, ctx: &egui::Context, pages: Vec<usize>, replace: bool) {
        let Some(m) = &self.manifest else { return };
        let todo: Vec<usize> = pages
            .into_iter()
            .filter(|&i| {
                m.pages
                    .get(i)
                    .is_some_and(|p| !p.raw_only && (replace || p.balloons.is_empty()))
            })
            .collect();
        if todo.is_empty() {
            self.status =
                "Nothing to detect: these pages already have balloons (use Redetect to replace)."
                    .into();
            return;
        }
        if !self.ensure_worker(ctx) {
            self.message = Some(("Wait for the current detection to finish.".into(), false));
            return;
        }
        let mut queued = 0;
        for page in todo {
            let Some(bytes) = self.page_bytes(page) else {
                return;
            };
            self.vision.as_mut().unwrap().submit(Request::Analyze {
                page,
                bytes,
                replace,
            });
            queued += 1;
        }
        self.status = format!("Detecting text and running OCR on {queued} page(s)... (the first run loads the models)");
    }

    /// Keep the current page's clean layer in step with its boxes.
    pub(super) fn update_live_clean(&mut self, ctx: &egui::Context) {
        let page_idx = self.page;
        let Some(page) = self.manifest.as_ref().and_then(|m| m.pages.get(page_idx)) else {
            return;
        };
        let sig = clean_signature(page_idx, page);
        if self
            .live
            .as_ref()
            .is_some_and(|l| l.page == page_idx && l.sig == sig)
        {
            self.clean_wanted = None;
            return;
        }
        let now = Instant::now();
        match self.clean_wanted {
            Some((s, _)) if s == sig => {}
            _ => self.clean_wanted = Some((sig, now)),
        }
        let (_, since) = self.clean_wanted.unwrap();
        if now.duration_since(since) < CLEAN_DEBOUNCE {
            ctx.request_repaint_after(CLEAN_DEBOUNCE);
            return;
        }
        if self.clean_inflight.is_some() {
            return; // the result comes back first, then the newest state is cleaned
        }
        let (ids, boxes): (Vec<u32>, Vec<CleanBox>) = page
            .balloons
            .iter()
            .filter_map(|b| clean_box_for(b).map(|c| (b.id, c)))
            .unzip();
        if !self.ensure_worker(ctx) {
            return;
        }
        let Some(bytes) = self.page_bytes(page_idx) else {
            return;
        };
        self.vision.as_mut().unwrap().submit(Request::Clean {
            page: page_idx,
            sig,
            bytes,
            boxes,
            ids,
        });
        self.clean_inflight = Some(sig);
    }

    pub(super) fn poll_vision(&mut self, ctx: &egui::Context) {
        let Some(worker) = &mut self.vision else {
            return;
        };
        let results = worker.poll();
        let mut errors = vec![];
        for r in results {
            match r.result {
                Ok(Outcome::Cleaned {
                    sig,
                    layer,
                    mask,
                    touch_up,
                }) => {
                    self.clean_inflight = None;
                    if r.page != self.page {
                        continue;
                    }
                    let red = image::RgbaImage::from_fn(mask.width(), mask.height(), |x, y| {
                        if mask.get_pixel(x, y)[0] > 0 {
                            image::Rgba([255, 0, 0, 150])
                        } else {
                            image::Rgba([0, 0, 0, 0])
                        }
                    });
                    self.live = Some(LiveClean {
                        page: r.page,
                        sig,
                        tex: rgba_texture(ctx, "clean", &layer),
                        mask_tex: rgba_texture(ctx, "clean-mask", &red),
                        touch_up,
                    });
                }
                Ok(Outcome::Analyzed { replace, regions }) => {
                    let Some(page) = self.manifest.as_mut().and_then(|m| m.pages.get_mut(r.page))
                    else {
                        continue;
                    };
                    if !replace && !page.balloons.is_empty() {
                        continue;
                    }
                    page.balloons = regions
                        .into_iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let mut b = Balloon::new(i as u32 + 1, t.kind, t.text_box);
                            b.bubble = t.bubble;
                            b.source_text = t.source_text;
                            b.rotation = t.rotation;
                            b.frame = t.frame;
                            b
                        })
                        .collect();
                    let msg = format!("Page {}: {} text areas.", page.id, page.balloons.len());
                    if r.page == self.page {
                        self.selected = None;
                    }
                    self.dirty = true;
                    let left = self.vision.as_ref().map(|v| v.pending).unwrap_or(0);
                    self.status = if left > 0 {
                        format!("{msg} {left} left...")
                    } else {
                        msg
                    };
                }
                Err(e) => {
                    if self.clean_inflight.is_some() && r.page == self.page {
                        self.clean_inflight = None;
                    }
                    let id = self
                        .manifest
                        .as_ref()
                        .and_then(|m| m.pages.get(r.page))
                        .map(|p| p.id.clone())
                        .unwrap_or_default();
                    errors.push(format!("Page {id} failed:\n{e}"));
                }
            }
        }
        if !errors.is_empty() {
            self.error(errors.join("\n\n"));
        }
    }

    /// Clean layer and mask for the current page, if computed.
    pub(super) fn live_clean(&self) -> Option<&LiveClean> {
        self.live.as_ref().filter(|l| l.page == self.page)
    }

    // ---------- Auto ----------

    pub(super) fn start_auto(&mut self, pages: Vec<usize>) {
        if self.auto.is_some() || self.manifest.is_none() {
            return;
        }
        if self.message.as_ref().is_some_and(|(_, err)| *err) {
            self.message = None;
        }
        let total = pages.len();
        self.auto = Some(AutoRun {
            queue: pages.into(),
            current: None,
            tried: vec![],
            stage: None,
            total,
            done: 0,
            cancel: false,
        });
    }

    /// Advance the Auto run by one step when nothing is in flight.
    pub(super) fn step_auto(&mut self, ctx: &egui::Context) {
        if self.auto.is_none() || self.vision_busy() || self.job.is_some() {
            return;
        }
        // Any error stops the run; the message box shows it.
        if self.message.as_ref().is_some_and(|(_, err)| *err) {
            let run = self.auto.take().unwrap();
            self.status = format!("Auto stopped after {} of {} page(s).", run.done, run.total);
            return;
        }
        loop {
            let run = self.auto.as_mut().unwrap();
            if run.cancel {
                let (done, total) = (run.done, run.total);
                self.auto = None;
                self.status = format!("Auto cancelled after {done} of {total} page(s).");
                return;
            }
            let Some(page) = run.current.or_else(|| run.queue.pop_front()) else {
                let total = run.total;
                self.auto = None;
                self.status = format!("Auto finished {total} page(s). Save to keep the results.");
                return;
            };
            if run.current != Some(page) {
                run.current = Some(page);
                run.tried.clear();
            }
            let Some(p) = self.manifest.as_ref().and_then(|m| m.pages.get(page)) else {
                return;
            };
            let run = self.auto.as_mut().unwrap();
            match missing_stages(p)
                .into_iter()
                .find(|s| !run.tried.contains(s))
            {
                None => {
                    run.done += 1;
                    run.current = None;
                    run.stage = None;
                }
                Some(stage) => {
                    run.tried.push(stage);
                    run.stage = Some(stage);
                    // Nothing to translate when detection found nothing.
                    if stage == Stage::Translate && p.balloons.iter().all(|b| b.ignore) {
                        continue;
                    }
                    match stage {
                        Stage::Detect => self.detect_pages(ctx, vec![page], false),
                        Stage::Translate => self.start_translate_page(page, None),
                    }
                    return;
                }
            }
        }
    }

    pub(super) fn auto_dialog(&mut self, ctx: &egui::Context) {
        let Some(run) = &self.auto else { return };
        let page_id = run
            .current
            .and_then(|p| self.manifest.as_ref()?.pages.get(p))
            .map(|p| p.id.clone())
            .unwrap_or_default();
        let stage = run.stage.map(Stage::label).unwrap_or("starting");
        let (done, total, cancel) = (run.done, run.total, run.cancel);
        let mut cancel_clicked = false;
        egui::Window::new("Auto")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                ui.label(format!(
                    "Page {} of {total}  ({page_id}): {stage}...",
                    (done + 1).min(total)
                ));
                ui.add(egui::ProgressBar::new(done as f32 / total.max(1) as f32).show_percentage());
                ui.add_space(4.0);
                ui.vertical_centered(|ui| {
                    if cancel {
                        ui.weak("Cancelling after the current step...");
                    } else if ui.button("   Cancel   ").clicked() {
                        cancel_clicked = true;
                    }
                });
            });
        if cancel_clicked {
            if let Some(run) = &mut self.auto {
                run.cancel = true;
            }
        }
        ctx.request_repaint_after(Duration::from_millis(200));
    }
}

/// Amber marks boxes whose text touches an outline.
pub(super) const TOUCH_UP: Color32 = Color32::from_rgb(0xe0, 0xa0, 0x00);

#[cfg(test)]
mod tests {
    use super::*;

    fn page() -> at2k::Page {
        at2k::Page {
            id: "001".into(),
            image: "pages/001.png".into(),
            width: 10,
            height: 10,
            balloons: vec![],
            sfx: vec![],
            raw_only: false,
            clean: None,
        }
    }

    #[test]
    fn stages_follow_page_state() {
        let mut p = page();
        assert_eq!(missing_stages(&p), vec![Stage::Detect, Stage::Translate]);
        let mut b = Balloon::new(1, "dialogue", [0, 0, 5, 5]);
        b.source_text = "あ".into();
        p.balloons.push(b);
        assert_eq!(missing_stages(&p), vec![Stage::Translate]);
        p.balloons[0].text = "AH".into();
        assert!(missing_stages(&p).is_empty());
        p.balloons[0].text.clear();
        p.balloons[0].ignore = true;
        assert!(missing_stages(&p).is_empty());
        p.raw_only = true;
        assert!(missing_stages(&p).is_empty());
    }

    #[test]
    fn signature_changes_with_boxes_not_text() {
        let mut p = page();
        p.balloons.push(Balloon::new(1, "dialogue", [0, 0, 5, 5]));
        let a = clean_signature(0, &p);
        p.balloons[0].text = "HELLO".into();
        assert_eq!(
            a,
            clean_signature(0, &p),
            "translation edits do not reclean"
        );
        p.balloons[0].text_box = [1, 0, 5, 5];
        assert_ne!(a, clean_signature(0, &p));
        p.balloons.clear();
        assert_ne!(a, clean_signature(0, &p), "deleting a box recleans");
    }

    #[test]
    fn ignored_and_hidden_boxes_are_not_cleaned() {
        let mut b = Balloon::new(1, "dialogue", [0, 0, 5, 5]);
        assert!(clean_box_for(&b).is_some());
        b.ignore = true;
        assert!(clean_box_for(&b).is_none());
        b.ignore = false;
        b.hidden = true;
        assert!(clean_box_for(&b).is_none());
        b.hidden = false;
        b.shape = BubbleShape::Ellipse;
        assert!(clean_box_for(&b).unwrap().shape.is_some());
    }
}
