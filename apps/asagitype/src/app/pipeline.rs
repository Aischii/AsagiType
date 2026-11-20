//! Detection + OCR, cleaning, and the Auto pipeline (detect -> translate -> clean)
//! with a progress dialog and Cancel.

use super::AsagiApp;
use crate::vision_worker::{Outcome, Request, VisionWorker};
use at2k::Balloon;
use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use std::{collections::VecDeque, path::PathBuf};
use vision::clean::CleanBox;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Stage {
    Detect,
    Translate,
    Clean,
}

impl Stage {
    fn label(self) -> &'static str {
        match self {
            Stage::Detect => "detecting text and running OCR",
            Stage::Translate => "translating",
            Stage::Clean => "cleaning",
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

/// Stages a page still needs, in order. `has_clean` covers saved and unsaved clean layers.
pub(super) fn missing_stages(page: &at2k::Page, has_clean: bool) -> Vec<Stage> {
    if page.raw_only {
        return vec![];
    }
    let mut out = vec![];
    if page.balloons.is_empty() {
        out.push(Stage::Detect);
    }
    let active = |b: &&Balloon| !b.ignore;
    if page.balloons.is_empty()
        || page
            .balloons
            .iter()
            .filter(active)
            .any(|b| !b.source_text.trim().is_empty() && b.text.trim().is_empty())
    {
        out.push(Stage::Translate);
    }
    if !has_clean {
        out.push(Stage::Clean);
    }
    out
}

impl AsagiApp {
    fn ensure_worker(&mut self, ctx: &egui::Context) -> bool {
        let dir = PathBuf::from(&self.settings.models_dir);
        if self.vision.as_ref().is_none_or(|v| v.models_dir != dir) {
            if self.vision.as_ref().is_some_and(|v| v.pending > 0) {
                self.message = Some((
                    "Wait for the current detection or cleaning to finish.".into(),
                    false,
                ));
                return false;
            }
            self.vision = Some(VisionWorker::spawn(dir, ctx.clone()));
        }
        true
    }

    pub(super) fn vision_busy(&self) -> bool {
        self.vision.as_ref().is_some_and(|v| v.pending > 0)
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

    /// Queue cleaning of every non-ignored balloon on these pages.
    pub(super) fn clean_pages(&mut self, ctx: &egui::Context, pages: Vec<usize>) {
        let Some(m) = &self.manifest else { return };
        let todo: Vec<(usize, Vec<CleanBox>)> = pages
            .into_iter()
            .filter_map(|i| m.pages.get(i).map(|p| (i, p)))
            .filter(|(_, p)| !p.raw_only)
            .map(|(i, p)| {
                let boxes = p
                    .balloons
                    .iter()
                    .filter(|b| !b.ignore)
                    .map(|b| CleanBox {
                        text_box: b.text_box,
                        bubble: b.bubble,
                    })
                    .collect();
                (i, boxes)
            })
            .filter(|(_, b): &(usize, Vec<CleanBox>)| !b.is_empty())
            .collect();
        if todo.is_empty() {
            self.status = "Nothing to clean: detect or draw balloons first.".into();
            return;
        }
        if !self.ensure_worker(ctx) {
            return;
        }
        let n = todo.len();
        for (page, boxes) in todo {
            let Some(bytes) = self.page_bytes(page) else {
                return;
            };
            self.vision
                .as_mut()
                .unwrap()
                .submit(Request::Clean { page, bytes, boxes });
        }
        self.status = format!("Cleaning {n} page(s)...");
    }

    pub(super) fn poll_vision(&mut self) {
        let Some(worker) = &mut self.vision else {
            return;
        };
        let results = worker.poll();
        let pending = worker.pending;
        let mut errors = vec![];
        for r in results {
            let Some(page) = self.manifest.as_mut().and_then(|m| m.pages.get_mut(r.page)) else {
                continue;
            };
            let msg = match r.result {
                Ok(Outcome::Analyzed { replace, regions }) => {
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
                            b
                        })
                        .collect();
                    if r.page == self.page {
                        self.selected = None;
                    }
                    format!("Page {}: {} text areas.", page.id, page.balloons.len())
                }
                Ok(Outcome::Cleaned { png }) => {
                    page.clean = Some(page.clean_entry_name());
                    let id = page.id.clone();
                    self.pending_clean.insert(r.page, png);
                    self.clean_tex = None;
                    format!("Page {id}: cleaned.")
                }
                Err(e) => {
                    errors.push(format!("Page {} failed:\n{e}", page.id));
                    continue;
                }
            };
            self.dirty = true;
            self.status = if pending > 0 {
                format!("{msg} {pending} left...")
            } else {
                msg
            };
        }
        if !errors.is_empty() {
            self.error(errors.join("\n\n"));
        }
    }

    /// Texture of the current page's clean layer: unsaved result first, then the saved one.
    pub(super) fn clean_texture(&mut self, ctx: &egui::Context) -> Option<TextureHandle> {
        let page = self.manifest.as_ref()?.pages.get(self.page)?.clone();
        let key = format!(
            "{}#{}",
            self.page,
            self.pending_clean.contains_key(&self.page)
        );
        if let Some((k, t)) = &self.clean_tex {
            if *k == key {
                return Some(t.clone());
            }
        }
        let bytes = match self.pending_clean.get(&self.page) {
            Some(b) => b.clone(),
            None => at2k::read_entry(self.path.as_ref()?, page.clean.as_ref()?).ok()?,
        };
        let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
        let ci = ColorImage::from_rgba_unmultiplied(
            [img.width() as usize, img.height() as usize],
            img.as_raw(),
        );
        let tex = ctx.load_texture("clean", ci, TextureOptions::LINEAR);
        self.clean_tex = Some((key, tex.clone()));
        Some(tex)
    }

    pub(super) fn has_clean(&self, page: usize) -> bool {
        self.pending_clean.contains_key(&page)
            || self
                .manifest
                .as_ref()
                .and_then(|m| m.pages.get(page))
                .is_some_and(|p| p.clean.is_some())
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
            let has_clean = self.has_clean(page);
            let Some(p) = self.manifest.as_ref().and_then(|m| m.pages.get(page)) else {
                return;
            };
            let run = self.auto.as_mut().unwrap();
            let next = missing_stages(p, has_clean)
                .into_iter()
                .find(|s| !run.tried.contains(s));
            match next {
                None => {
                    run.done += 1;
                    run.current = None;
                    run.stage = None;
                    continue;
                }
                Some(stage) => {
                    run.tried.push(stage);
                    run.stage = Some(stage);
                    // Skip translate and clean when detection found nothing.
                    if stage != Stage::Detect && p.balloons.iter().all(|b| b.ignore) {
                        continue;
                    }
                    match stage {
                        Stage::Detect => self.detect_pages(ctx, vec![page], false),
                        Stage::Translate => self.start_translate_page(page, None),
                        Stage::Clean => self.clean_pages(ctx, vec![page]),
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
        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
}

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
        assert_eq!(
            missing_stages(&p, false),
            vec![Stage::Detect, Stage::Translate, Stage::Clean]
        );
        let mut b = Balloon::new(1, "dialogue", [0, 0, 5, 5]);
        b.source_text = "あ".into();
        p.balloons.push(b);
        assert_eq!(
            missing_stages(&p, false),
            vec![Stage::Translate, Stage::Clean]
        );
        p.balloons[0].text = "AH".into();
        assert_eq!(missing_stages(&p, true), vec![]);
        // Ignored balloons never need translating.
        p.balloons[0].text.clear();
        p.balloons[0].ignore = true;
        assert_eq!(missing_stages(&p, true), vec![]);
        p.raw_only = true;
        assert!(missing_stages(&p, false).is_empty());
    }
}
