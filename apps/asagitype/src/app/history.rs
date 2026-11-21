//! Project-wide undo and redo.
//!
//! Page edits are not recorded by each widget. Instead the history keeps a baseline copy of
//! every page and, once no gesture is in progress, compares the live pages with it. Each page
//! that changed becomes one step holding its previous state. A gesture is a drag or typing in a
//! field, so a whole drag or field edit is one step. Paint strokes are recorded explicitly as the
//! pixels they replaced. Applying a step swaps its stored state with the live one, so the same
//! step moves between the undo and redo stacks.

use at2k::Page;
use image::RgbaImage;
use std::collections::HashMap;

pub const LIMIT: usize = 100;

/// Healing-brush layers by page index, page-sized RGBA.
pub type PaintLayers = HashMap<usize, RgbaImage>;

enum Step {
    Page { index: usize, state: Page },
    Paint { index: usize, x: u32, y: u32, pixels: RgbaImage },
}

impl Step {
    fn page(&self) -> usize {
        match self {
            Step::Page { index, .. } | Step::Paint { index, .. } => *index,
        }
    }
}

#[derive(Default)]
pub struct History {
    undo: Vec<Step>,
    redo: Vec<Step>,
    baseline: Vec<Page>,
}

impl History {
    /// Forget everything; call when a project is opened.
    pub fn reset(&mut self, pages: &[Page]) {
        self.undo.clear();
        self.redo.clear();
        self.baseline = pages.to_vec();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    fn push(&mut self, step: Step) {
        self.redo.clear();
        self.undo.push(step);
        if self.undo.len() > LIMIT {
            self.undo.remove(0);
        }
    }

    /// Record page changes since the last call. `hold` is a page with a gesture still in
    /// progress; its changes wait for the next call. Returns true when a step was added.
    pub fn observe(&mut self, pages: &[Page], hold: Option<usize>) -> bool {
        if self.baseline.len() != pages.len() {
            self.baseline = pages.to_vec();
            return false;
        }
        let mut added = false;
        for (i, page) in pages.iter().enumerate() {
            if Some(i) == hold || self.baseline[i] == *page {
                continue;
            }
            let state = std::mem::replace(&mut self.baseline[i], page.clone());
            if !same_content(&state, page) {
                self.push(Step::Page { index: i, state });
                added = true;
            }
        }
        added
    }

    /// Record a paint stroke: `before` holds the pixels at (x, y) as they were before it.
    pub fn record_paint(&mut self, index: usize, x: u32, y: u32, before: RgbaImage) {
        self.push(Step::Paint {
            index,
            x,
            y,
            pixels: before,
        });
    }

    /// Undo one step. Returns the page it changed.
    pub fn undo(&mut self, pages: &mut [Page], paint: &mut PaintLayers) -> Option<usize> {
        let mut step = self.undo.pop()?;
        self.apply(&mut step, pages, paint);
        let page = step.page();
        self.redo.push(step);
        Some(page)
    }

    /// Redo one step. Returns the page it changed.
    pub fn redo(&mut self, pages: &mut [Page], paint: &mut PaintLayers) -> Option<usize> {
        let mut step = self.redo.pop()?;
        self.apply(&mut step, pages, paint);
        let page = step.page();
        self.undo.push(step);
        Some(page)
    }

    fn apply(&mut self, step: &mut Step, pages: &mut [Page], paint: &mut PaintLayers) {
        match step {
            Step::Page { index, state } => {
                if let Some(live) = pages.get_mut(*index) {
                    std::mem::swap(live, state);
                    // Archive entry names are bookkeeping, not edits: keep the live ones.
                    std::mem::swap(&mut live.paint, &mut state.paint);
                    std::mem::swap(&mut live.clean, &mut state.clean);
                    if let Some(b) = self.baseline.get_mut(*index) {
                        *b = live.clone();
                    }
                }
            }
            Step::Paint {
                index,
                x,
                y,
                pixels,
            } => {
                if let Some(layer) = paint.get_mut(index) {
                    swap_region(layer, *x, *y, pixels);
                }
            }
        }
    }
}

/// Equal apart from archive entry names, which saving changes.
fn same_content(a: &Page, b: &Page) -> bool {
    a.balloons == b.balloons && a.sfx == b.sfx && a.raw_only == b.raw_only
}

/// Exchange the pixels of `patch` with the same-sized region of `layer` at (x, y).
pub fn swap_region(layer: &mut RgbaImage, x: u32, y: u32, patch: &mut RgbaImage) {
    for py in 0..patch.height() {
        for px in 0..patch.width() {
            let (lx, ly) = (x + px, y + py);
            if lx < layer.width() && ly < layer.height() {
                let a = *layer.get_pixel(lx, ly);
                let b = *patch.get_pixel(px, py);
                layer.put_pixel(lx, ly, b);
                patch.put_pixel(px, py, a);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use at2k::Balloon;
    use image::Rgba;

    fn page(id: &str) -> Page {
        Page {
            id: id.into(),
            image: format!("pages/{id}.png"),
            width: 8,
            height: 8,
            balloons: vec![],
            sfx: vec![],
            raw_only: false,
            clean: None,
            paint: None,
        }
    }

    #[test]
    fn edits_undo_and_redo_across_pages() {
        let mut pages = vec![page("001"), page("002")];
        let mut paint = PaintLayers::new();
        let mut h = History::default();
        h.reset(&pages);
        assert!(!h.observe(&pages, None));

        pages[0].balloons.push(Balloon::new(1, "dialogue", [0, 0, 4, 4]));
        assert!(h.observe(&pages, None));
        pages[1].balloons.push(Balloon::new(1, "dialogue", [1, 1, 2, 2]));
        pages[1].balloons[0].text = "HI.".into();
        assert!(h.observe(&pages, None));

        assert_eq!(h.undo(&mut pages, &mut paint), Some(1));
        assert!(pages[1].balloons.is_empty());
        assert_eq!(pages[0].balloons.len(), 1);
        assert_eq!(h.undo(&mut pages, &mut paint), Some(0));
        assert!(pages[0].balloons.is_empty());
        assert_eq!(h.undo(&mut pages, &mut paint), None);
        // Undo itself is not recorded as a new edit.
        assert!(!h.observe(&pages, None));

        assert_eq!(h.redo(&mut pages, &mut paint), Some(0));
        assert_eq!(h.redo(&mut pages, &mut paint), Some(1));
        assert_eq!(pages[1].balloons[0].text, "HI.");
        assert!(!h.can_redo());
    }

    #[test]
    fn a_gesture_is_one_step_and_new_edits_clear_redo() {
        let mut pages = vec![page("001")];
        let mut paint = PaintLayers::new();
        let mut h = History::default();
        h.reset(&pages);
        pages[0].balloons.push(Balloon::new(1, "dialogue", [0, 0, 4, 4]));
        h.observe(&pages, None);
        // Typing H, E, Y while the field has focus, then leaving it.
        for t in ["H", "HE", "HEY"] {
            pages[0].balloons[0].text = t.into();
            assert!(!h.observe(&pages, Some(0)));
        }
        assert!(h.observe(&pages, None));
        h.undo(&mut pages, &mut paint);
        assert_eq!(pages[0].balloons[0].text, "");
        assert!(h.can_redo());
        pages[0].balloons[0].locked = true;
        h.observe(&pages, None);
        assert!(!h.can_redo());
    }

    #[test]
    fn saving_entry_names_is_not_an_edit() {
        let mut pages = vec![page("001")];
        let mut paint = PaintLayers::new();
        let mut h = History::default();
        h.reset(&pages);
        pages[0].raw_only = true;
        h.observe(&pages, None);
        pages[0].paint = Some("paint/001.png".into());
        assert!(!h.observe(&pages, None));
        h.undo(&mut pages, &mut paint);
        assert!(!pages[0].raw_only);
        assert_eq!(pages[0].paint.as_deref(), Some("paint/001.png"));
    }

    #[test]
    fn history_is_capped() {
        let mut pages = vec![page("001")];
        let mut h = History::default();
        h.reset(&pages);
        for i in 0..(LIMIT + 20) {
            pages[0].sfx.push([i as i32, 0, 1, 1]);
            h.observe(&pages, None);
        }
        assert_eq!(h.undo.len(), LIMIT);
    }

    #[test]
    fn paint_strokes_swap_pixels() {
        let mut pages = vec![page("001")];
        let mut paint = PaintLayers::new();
        paint.insert(0, RgbaImage::new(8, 8));
        let mut h = History::default();
        h.reset(&pages);
        // A stroke over (2..4, 3..5): save the old pixels, then paint.
        let before = image::imageops::crop_imm(&paint[&0], 2, 3, 2, 2).to_image();
        for y in 3..5 {
            for x in 2..4 {
                paint.get_mut(&0).unwrap().put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        h.record_paint(0, 2, 3, before);
        assert_eq!(h.undo(&mut pages, &mut paint), Some(0));
        assert!(paint[&0].pixels().all(|p| p.0[3] == 0));
        assert_eq!(h.redo(&mut pages, &mut paint), Some(0));
        assert_eq!(paint[&0].get_pixel(3, 4).0, [255, 0, 0, 255]);
        assert_eq!(paint[&0].get_pixel(4, 4).0[3], 0);
    }
}
