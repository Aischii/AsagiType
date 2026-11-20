//! Page editor: synced Original | Translated panes, selection, move/resize handles,
//! drawing new balloons, nudging, and the balloon properties / typography panel.

use super::{AsagiApp, Tool};
use crate::{settings::ROLES, theme};
use at2k::{Align, Balloon, Preset, TextCase, PRESET_COUNT};
use eframe::egui::{
    self, Color32, ColorImage, CursorIcon, Key, Pos2, Rect, Sense, Stroke, StrokeKind,
    TextureHandle, TextureOptions, Vec2,
};
use std::hash::{Hash, Hasher};

/// Smallest box side, in page pixels.
const MIN_SIDE: i32 = 8;
/// Handle square size on screen.
const HANDLE: f32 = 7.0;
const GAP: f32 = 24.0;
const HEADER: f32 = 18.0;
const SELECT: Color32 = Color32::from_rgb(0xff, 0x00, 0xff);
const IGNORED: Color32 = Color32::from_rgb(0x90, 0x90, 0x90);

#[derive(Clone, Copy)]
pub(super) enum Drag {
    Move {
        id: u32,
        start: [i32; 4],
        from: Pos2,
    },
    /// `edges` = (left, top, right, bottom) sides that follow the pointer.
    Resize {
        id: u32,
        start: [i32; 4],
        from: Pos2,
        edges: [bool; 4],
    },
    Draw {
        from: Pos2,
    },
}

/// Handle positions on a box: corners and edge midpoints, with the edges each one moves.
fn handles(r: Rect) -> [(Pos2, [bool; 4], CursorIcon); 8] {
    let (l, t, rr, b) = (r.left(), r.top(), r.right(), r.bottom());
    let (cx, cy) = (r.center().x, r.center().y);
    [
        (
            Pos2::new(l, t),
            [true, true, false, false],
            CursorIcon::ResizeNwSe,
        ),
        (
            Pos2::new(cx, t),
            [false, true, false, false],
            CursorIcon::ResizeVertical,
        ),
        (
            Pos2::new(rr, t),
            [false, true, true, false],
            CursorIcon::ResizeNeSw,
        ),
        (
            Pos2::new(rr, cy),
            [false, false, true, false],
            CursorIcon::ResizeHorizontal,
        ),
        (
            Pos2::new(rr, b),
            [false, false, true, true],
            CursorIcon::ResizeNwSe,
        ),
        (
            Pos2::new(cx, b),
            [false, false, false, true],
            CursorIcon::ResizeVertical,
        ),
        (
            Pos2::new(l, b),
            [true, false, false, true],
            CursorIcon::ResizeNeSw,
        ),
        (
            Pos2::new(l, cy),
            [true, false, false, false],
            CursorIcon::ResizeHorizontal,
        ),
    ]
}

/// Apply a drag delta (page px) to a box; keeps it inside the page and at least MIN_SIDE.
fn resized(start: [i32; 4], edges: [bool; 4], d: Vec2, w: i32, h: i32) -> [i32; 4] {
    let (dx, dy) = (d.x.round() as i32, d.y.round() as i32);
    let mut b = start;
    if edges[0] {
        b[0] = (start[0] + dx).clamp(0, start[2] - MIN_SIDE);
    }
    if edges[1] {
        b[1] = (start[1] + dy).clamp(0, start[3] - MIN_SIDE);
    }
    if edges[2] {
        b[2] = (start[2] + dx).clamp(start[0] + MIN_SIDE, w);
    }
    if edges[3] {
        b[3] = (start[3] + dy).clamp(start[1] + MIN_SIDE, h);
    }
    b
}

fn moved(start: [i32; 4], dx: i32, dy: i32, w: i32, h: i32) -> [i32; 4] {
    let dx = dx.clamp(-start[0], w - start[2]);
    let dy = dy.clamp(-start[1], h - start[3]);
    [start[0] + dx, start[1] + dy, start[2] + dx, start[3] + dy]
}

fn render_align(a: Align) -> render::Align {
    match a {
        Align::Left => render::Align::Left,
        Align::Center => render::Align::Center,
        Align::Right => render::Align::Right,
    }
}

impl AsagiApp {
    fn balloon_mut(&mut self, id: u32) -> Option<&mut Balloon> {
        let page = self.page;
        self.manifest
            .as_mut()?
            .pages
            .get_mut(page)?
            .balloons
            .iter_mut()
            .find(|b| b.id == id)
    }

    pub(super) fn lettering(
        &mut self,
        ctx: &egui::Context,
        b: &Balloon,
    ) -> Option<(TextureHandle, bool)> {
        let family = b
            .font
            .clone()
            .unwrap_or_else(|| self.settings.font_for(&b.role));
        let text = b.display_text();
        let w = (b.text_box[2] - b.text_box[0]).max(1) as u32;
        let h = (b.text_box[3] - b.text_box[1]).max(1) as u32;
        let style = render::Style {
            family: &family,
            bold: b.bold,
            italic: b.italic,
            align: render_align(b.align),
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (
            &text,
            &family,
            w,
            h,
            b.size.to_bits(),
            b.bold,
            b.italic,
            style.align,
        )
            .hash(&mut hasher);
        let key = hasher.finish();
        if let Some(hit) = self.letter_cache.get(&key) {
            return hit.clone();
        }
        let made = self.renderer.letter(&text, &style, w, h, b.size).map(|l| {
            let ci = ColorImage::from_rgba_premultiplied([w as usize, h as usize], l.pixmap.data());
            (
                ctx.load_texture(format!("letter-{key}"), ci, TextureOptions::LINEAR),
                l.fits,
            )
        });
        if self.letter_cache.len() > 400 {
            self.letter_cache.clear();
        }
        self.letter_cache.insert(key, made.clone());
        made
    }

    /// Arrow keys move the selected box 1 px, Shift+arrow 10 px.
    pub(super) fn nudge(&mut self, ctx: &egui::Context) {
        let Some(id) = self.selected else { return };
        if ctx.memory(|m| m.focused().is_some()) {
            return;
        }
        let (dx, dy) = ctx.input(|i| {
            let step = if i.modifiers.shift { 10 } else { 1 };
            let mut d = (0, 0);
            if i.key_pressed(Key::ArrowLeft) {
                d.0 -= step;
            }
            if i.key_pressed(Key::ArrowRight) {
                d.0 += step;
            }
            if i.key_pressed(Key::ArrowUp) {
                d.1 -= step;
            }
            if i.key_pressed(Key::ArrowDown) {
                d.1 += step;
            }
            d
        });
        if (dx, dy) == (0, 0) {
            return;
        }
        let Some((w, h)) = self.cur_page().map(|p| (p.width as i32, p.height as i32)) else {
            return;
        };
        if let Some(b) = self.balloon_mut(id) {
            b.text_box = moved(b.text_box, dx, dy, w, h);
            self.dirty = true;
        }
    }

    pub(super) fn canvas(&mut self, root: &mut egui::Ui) {
        let ctx = &root.ctx().clone();
        self.ensure_page_texture(ctx);
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::CANVAS_BG))
            .show(root, |ui| {
                let Some((_, tex)) = self.page_tex.clone() else {
                    ui.centered_and_justified(|ui| {
                        ui.colored_label(
                            Color32::WHITE,
                            "Open a project, or File > Import chapter.",
                        );
                    });
                    return;
                };
                let Some(page) = self.cur_page().cloned() else {
                    return;
                };
                let z = self.zoom;
                // One scroll area holds both panes, so zoom and scroll are always in sync.
                egui::ScrollArea::both().id_salt("canvas").show(ui, |ui| {
                    let size = Vec2::new(page.width as f32 * z, page.height as f32 * z);
                    let panes = if self.side_by_side { 2 } else { 1 };
                    let total = Vec2::new(
                        size.x * panes as f32 + GAP * (panes as f32 + 1.0),
                        size.y + HEADER + GAP * 2.0,
                    );
                    let (outer, _) = ui.allocate_exact_size(total, Sense::hover());
                    let rects: Vec<Rect> = (0..panes)
                        .map(|i| {
                            Rect::from_min_size(
                                outer.min
                                    + Vec2::new(GAP + i as f32 * (size.x + GAP), GAP + HEADER),
                                size,
                            )
                        })
                        .collect();
                    // (pane rect, shows translation)
                    let layout: Vec<(Rect, bool)> = if self.side_by_side {
                        vec![(rects[0], false), (rects[1], true)]
                    } else {
                        vec![(rects[0], self.show_text)]
                    };
                    let p = ui.painter_at(outer);
                    for &(rect, translated) in &layout {
                        let title = if !self.side_by_side {
                            "PAGE"
                        } else if translated {
                            "TRANSLATED"
                        } else {
                            "ORIGINAL"
                        };
                        let head = Rect::from_min_size(
                            rect.min - Vec2::new(0.0, HEADER),
                            Vec2::new(rect.width(), HEADER - 2.0),
                        );
                        theme::gradient(&p, head, theme::ASAGI, theme::ASAGI_LIGHT);
                        p.text(
                            head.left_center() + Vec2::new(4.0, 0.0),
                            egui::Align2::LEFT_CENTER,
                            title,
                            egui::FontId::proportional(11.0),
                            Color32::WHITE,
                        );
                        p.rect_filled(rect.translate(Vec2::splat(3.0)), 0.0, theme::DARK);
                        p.image(
                            tex.id(),
                            rect,
                            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    }

                    let to_screen = |rect: Rect, b: [i32; 4]| {
                        Rect::from_min_max(
                            rect.min + Vec2::new(b[0] as f32, b[1] as f32) * z,
                            rect.min + Vec2::new(b[2] as f32, b[3] as f32) * z,
                        )
                    };

                    // Lettering on the translated pane: white fill over the source text, then the English.
                    let clean = self.clean_texture(ctx);
                    for &(rect, translated) in &layout {
                        if !translated {
                            continue;
                        }
                        if let Some(c) = &clean {
                            p.image(
                                c.id(),
                                rect,
                                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                Color32::WHITE,
                            );
                        }
                        for b in &page.balloons {
                            if b.ignore || b.text.trim().is_empty() {
                                continue;
                            }
                            let r = to_screen(rect, b.text_box);
                            // Without a clean layer, a white box stands in as a preview.
                            if clean.is_none() {
                                p.rect_filled(r, 0.0, Color32::WHITE);
                            }
                            if let Some((t, fits)) = self.lettering(ctx, b) {
                                p.image(
                                    t.id(),
                                    r,
                                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                    Color32::WHITE,
                                );
                                if !fits {
                                    p.rect_stroke(
                                        r,
                                        0.0,
                                        Stroke::new(2.0, Color32::RED),
                                        StrokeKind::Outside,
                                    );
                                }
                            }
                        }
                    }

                    // Boxes: numbered on the original pane; only the selection on the translated pane.
                    for &(rect, translated) in &layout {
                        for b in &page.balloons {
                            let sel = self.selected == Some(b.id);
                            if !(self.show_boxes || sel)
                                || (translated && self.side_by_side && !sel)
                            {
                                continue;
                            }
                            let r = to_screen(rect, b.text_box);
                            let col = if sel {
                                SELECT
                            } else if b.ignore {
                                IGNORED
                            } else if b.kind == "aside" {
                                Color32::from_rgb(0xd0, 0x70, 0x00)
                            } else {
                                theme::ASAGI
                            };
                            p.rect_stroke(
                                r,
                                0.0,
                                Stroke::new(if sel { 1.5 } else { 2.0 }, col),
                                StrokeKind::Outside,
                            );
                            if !translated || !self.side_by_side {
                                let tag = Rect::from_min_size(
                                    r.left_top() - Vec2::new(2.0, 16.0),
                                    Vec2::new(22.0, 15.0),
                                );
                                p.rect_filled(tag, 0.0, col);
                                p.text(
                                    tag.center(),
                                    egui::Align2::CENTER_CENTER,
                                    b.id.to_string(),
                                    egui::FontId::proportional(11.0),
                                    Color32::WHITE,
                                );
                            }
                            if sel {
                                for (h, _, _) in handles(r) {
                                    let hr = Rect::from_center_size(h, Vec2::splat(HANDLE));
                                    p.rect_filled(hr, 0.0, Color32::WHITE);
                                    p.rect_stroke(
                                        hr,
                                        0.0,
                                        Stroke::new(1.0, SELECT),
                                        StrokeKind::Inside,
                                    );
                                }
                            }
                        }
                    }

                    // Interaction: either pane edits the same balloons.
                    let (pw, ph) = (page.width as i32, page.height as i32);
                    for (pane, &(rect, _)) in layout.iter().enumerate() {
                        let resp = ui.interact(
                            rect.expand(HANDLE),
                            ui.scope_id().with(("pane", pane)),
                            Sense::click_and_drag(),
                        );
                        let hover = resp.hover_pos();
                        let to_page = |pos: Pos2| (pos - rect.min) / z;
                        let selected_rect = self
                            .selected
                            .and_then(|id| page.balloons.iter().find(|b| b.id == id))
                            .map(|b| to_screen(rect, b.text_box));
                        let handle_at = |pos: Pos2| {
                            selected_rect.and_then(|r| {
                                handles(r).into_iter().find(|(h, _, _)| {
                                    Rect::from_center_size(*h, Vec2::splat(HANDLE + 4.0))
                                        .contains(pos)
                                })
                            })
                        };
                        let box_at = |pos: Pos2| {
                            let pp = to_page(pos);
                            page.balloons
                                .iter()
                                .rev()
                                .find(|b| {
                                    pp.x >= b.text_box[0] as f32
                                        && pp.x <= b.text_box[2] as f32
                                        && pp.y >= b.text_box[1] as f32
                                        && pp.y <= b.text_box[3] as f32
                                })
                                .map(|b| (b.id, b.text_box))
                        };
                        if let Some(pos) = hover {
                            if self.tool == Tool::Draw {
                                ctx.set_cursor_icon(CursorIcon::Crosshair);
                            } else if let Some((_, _, icon)) = handle_at(pos) {
                                ctx.set_cursor_icon(icon);
                            } else if box_at(pos).is_some() {
                                ctx.set_cursor_icon(CursorIcon::Move);
                            }
                        }
                        if resp.clicked() && self.tool == Tool::Select {
                            if let Some(pos) = resp.interact_pointer_pos() {
                                self.selected = box_at(pos).map(|(id, _)| id);
                            }
                        }
                        if resp.drag_started() {
                            if let Some(pos) = resp.interact_pointer_pos() {
                                self.drag = if self.tool == Tool::Draw {
                                    Some(Drag::Draw { from: pos })
                                } else if let (Some(id), Some((_, edges, _))) =
                                    (self.selected, handle_at(pos))
                                {
                                    page.balloons.iter().find(|b| b.id == id).map(|b| {
                                        Drag::Resize {
                                            id,
                                            start: b.text_box,
                                            from: pos,
                                            edges,
                                        }
                                    })
                                } else if let Some((id, start)) = box_at(pos) {
                                    self.selected = Some(id);
                                    Some(Drag::Move {
                                        id,
                                        start,
                                        from: pos,
                                    })
                                } else {
                                    None
                                };
                            }
                        }
                        if !resp.dragged() && !resp.drag_stopped() {
                            continue;
                        }
                        let Some(now) = ctx.pointer_latest_pos() else {
                            continue;
                        };
                        match self.drag {
                            Some(Drag::Move { id, start, from }) => {
                                let d = (now - from) / z;
                                let nb =
                                    moved(start, d.x.round() as i32, d.y.round() as i32, pw, ph);
                                if let Some(b) = self.balloon_mut(id) {
                                    if b.text_box != nb {
                                        b.text_box = nb;
                                        self.dirty = true;
                                    }
                                }
                            }
                            Some(Drag::Resize {
                                id,
                                start,
                                from,
                                edges,
                            }) => {
                                let nb = resized(start, edges, (now - from) / z, pw, ph);
                                if let Some(b) = self.balloon_mut(id) {
                                    if b.text_box != nb {
                                        b.text_box = nb;
                                        self.dirty = true;
                                    }
                                }
                            }
                            Some(Drag::Draw { from }) => {
                                let r = Rect::from_two_pos(from, now).intersect(rect);
                                p.rect_stroke(
                                    r,
                                    0.0,
                                    Stroke::new(1.0, Color32::BLACK),
                                    StrokeKind::Inside,
                                );
                                if resp.drag_stopped() && r.width() > 6.0 && r.height() > 6.0 {
                                    let (a, b) = (to_page(r.min).round(), to_page(r.max).round());
                                    let idx = self.page;
                                    if let Some(pg) =
                                        self.manifest.as_mut().and_then(|m| m.pages.get_mut(idx))
                                    {
                                        let id = pg.next_balloon_id();
                                        pg.balloons.push(Balloon::new(
                                            id,
                                            "dialogue",
                                            [a.x as i32, a.y as i32, b.x as i32, b.y as i32],
                                        ));
                                        self.selected = Some(id);
                                        self.dirty = true;
                                        self.status = format!(
                                            "Added balloon {id}; type its source text on the right"
                                        );
                                    }
                                }
                            }
                            None => {}
                        }
                        if resp.drag_stopped() {
                            self.drag = None;
                        }
                    }
                });
            });
    }

    /// Properties of the selected balloon: text, role, typography and presets.
    pub(super) fn properties(&mut self, ui: &mut egui::Ui) {
        let page_idx = self.page;
        let sel = self.selected;
        let families = self.renderer.families();
        let role_font = |role: &str| self.settings.font_for(role);
        let Some(m) = self.manifest.as_mut() else {
            ui.weak("Open a project first.");
            return;
        };
        if m.presets.len() < PRESET_COUNT {
            let defaults = self.settings.presets.clone();
            m.presets.extend(
                defaults
                    .into_iter()
                    .skip(m.presets.len())
                    .take(PRESET_COUNT - m.presets.len()),
            );
        }
        let presets = m.presets.clone();
        let Some(b) = m
            .pages
            .get_mut(page_idx)
            .and_then(|p| p.balloons.iter_mut().find(|b| Some(b.id) == sel))
        else {
            ui.weak("Select a balloon, or use Draw balloon to add one.");
            return;
        };
        let before = b.clone();
        let mut delete = false;
        let mut retranslate = false;
        let mut save_preset: Option<usize> = None;
        let default_font = role_font(&b.role);

        egui::Grid::new("props").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("Id");
            ui.horizontal(|ui| {
                ui.label(b.id.to_string());
                ui.checkbox(&mut b.ignore, "Ignore").on_hover_text("Keep the box but skip it when translating and lettering (credits, false hits)");
            });
            ui.end_row();
            ui.label("Kind");
            egui::ComboBox::from_id_salt("kind").selected_text(b.kind.clone()).show_ui(ui, |ui| {
                for k in ["dialogue", "aside", "caption"] {
                    ui.selectable_value(&mut b.kind, k.to_string(), k);
                }
            });
            ui.end_row();
            ui.label("Role");
            egui::ComboBox::from_id_salt("role").selected_text(b.role.clone()).show_ui(ui, |ui| {
                for r in ROLES {
                    ui.selectable_value(&mut b.role, r.to_string(), *r);
                }
            });
            ui.end_row();
            ui.label("Box");
            ui.label(format!("{} x {} at {}, {}", b.text_box[2] - b.text_box[0], b.text_box[3] - b.text_box[1], b.text_box[0], b.text_box[1]));
            ui.end_row();
        });

        theme::caption(ui, "Typography");
        egui::Grid::new("type")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Presets");
                ui.horizontal(|ui| {
                    for (i, p) in presets.iter().enumerate().take(PRESET_COUNT) {
                        let r = ui
                            .selectable_label(b.preset == Some(i as u8 + 1), format!(" {} ", i + 1))
                            .on_hover_text(format!(
                                "{}: click to apply, right-click to save this balloon's style here",
                                p.name
                            ));
                        if r.clicked() {
                            b.apply_preset(i as u8 + 1, p);
                        }
                        r.context_menu(|ui| {
                            if ui
                                .button(format!(
                                    "Save current style as preset {} ({})",
                                    i + 1,
                                    p.name
                                ))
                                .clicked()
                            {
                                save_preset = Some(i);
                                ui.close();
                            }
                        });
                    }
                });
                ui.end_row();
                ui.label("Font");
                let shown = b
                    .font
                    .clone()
                    .unwrap_or_else(|| format!("{default_font} (role)"));
                egui::ComboBox::from_id_salt("font")
                    .width(180.0)
                    .selected_text(shown)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut b.font,
                            None,
                            format!("{default_font} (role default)"),
                        );
                        for f in &families {
                            ui.selectable_value(&mut b.font, Some(f.clone()), f);
                        }
                    });
                ui.end_row();
                ui.label("Size");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut b.size)
                            .range(0.0..=72.0)
                            .speed(0.25)
                            .suffix(" pt"),
                    );
                    if ui
                        .selectable_label(b.size == 0.0, "Auto")
                        .on_hover_text("Fit the box automatically")
                        .clicked()
                    {
                        b.size = 0.0;
                    }
                });
                ui.end_row();
                ui.label("Style");
                ui.horizontal(|ui| {
                    ui.toggle_value(&mut b.bold, egui::RichText::new(" B ").strong());
                    ui.toggle_value(&mut b.italic, egui::RichText::new(" I ").italics());
                    egui::ComboBox::from_id_salt("case")
                        .selected_text(b.case.label())
                        .show_ui(ui, |ui| {
                            for c in TextCase::ALL {
                                ui.selectable_value(&mut b.case, c, c.label());
                            }
                        });
                });
                ui.end_row();
                ui.label("Align");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut b.align, Align::Left, "Left");
                    ui.selectable_value(&mut b.align, Align::Center, "Centre");
                    ui.selectable_value(&mut b.align, Align::Right, "Right");
                });
                ui.end_row();
            });

        ui.label("Source");
        ui.add(
            egui::TextEdit::multiline(&mut b.source_text)
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        ui.label("Translation");
        ui.add(
            egui::TextEdit::multiline(&mut b.text)
                .desired_rows(4)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal(|ui| {
            retranslate = ui.button("Retranslate").clicked();
            delete = ui.button("Delete").clicked();
        });

        // Picking a style by hand clears the preset marker.
        let typography_changed = (&b.font, b.size, b.bold, b.italic, b.case, b.align)
            != (
                &before.font,
                before.size,
                before.bold,
                before.italic,
                before.case,
                before.align,
            );
        if typography_changed && b.preset == before.preset {
            b.preset = None;
        }
        let changed = *b != before;
        if let Some(i) = save_preset {
            let name = m.presets[i].name.clone();
            let style = m.pages[page_idx]
                .balloons
                .iter()
                .find(|x| Some(x.id) == sel)
                .map(|x| Preset::from_balloon(&name, x));
            if let Some(p) = style {
                m.presets[i] = p;
                self.dirty = true;
                self.status = format!("Saved preset {} ({name}) for this project", i + 1);
            }
        }
        if changed {
            self.dirty = true;
        }
        if delete {
            self.delete_selected();
        }
        if retranslate {
            self.start_translate(sel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_keeps_minimum_and_page_bounds() {
        let start = [100, 100, 200, 200];
        // Drag the right edge far left: stops at MIN_SIDE.
        assert_eq!(
            resized(
                start,
                [false, false, true, false],
                Vec2::new(-500.0, 0.0),
                1000,
                1000
            )[2],
            100 + MIN_SIDE
        );
        // Drag the top-left corner past the page origin.
        assert_eq!(
            resized(
                start,
                [true, true, false, false],
                Vec2::new(-500.0, -500.0),
                1000,
                1000
            ),
            [0, 0, 200, 200]
        );
    }

    #[test]
    fn move_stays_on_page() {
        assert_eq!(moved([10, 10, 50, 50], -40, 5, 100, 100), [0, 15, 40, 55]);
        assert_eq!(moved([10, 10, 50, 50], 90, 0, 100, 100), [60, 10, 100, 50]);
    }
}
