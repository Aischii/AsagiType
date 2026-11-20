//! Windows 2000 classic look with asagi teal title bars.

use eframe::egui::{
    self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, Painter, Rect, Shadow,
    Stroke,
};
use std::sync::Arc;

pub const FACE: Color32 = Color32::from_rgb(0xd4, 0xd0, 0xc8);
pub const LIGHT: Color32 = Color32::WHITE;
pub const SHADOW: Color32 = Color32::from_rgb(0x80, 0x80, 0x80);
pub const DARK: Color32 = Color32::from_rgb(0x40, 0x40, 0x40);
pub const ASAGI: Color32 = Color32::from_rgb(0x00, 0xa3, 0xaf);
pub const ASAGI_LIGHT: Color32 = Color32::from_rgb(0x7f, 0xd1, 0xd8);
pub const INACTIVE: Color32 = Color32::from_rgb(0x80, 0x80, 0x80);
pub const INACTIVE_LIGHT: Color32 = Color32::from_rgb(0xc0, 0xc0, 0xc0);
pub const SELECT: Color32 = Color32::from_rgb(0x00, 0x82, 0x8c);
pub const CANVAS_BG: Color32 = Color32::from_rgb(0x80, 0x80, 0x80);

pub fn apply(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    // Tahoma is the Windows 2000 UI font; use it when the system has it.
    for (name, file) in [("tahoma", "tahoma.ttf"), ("tahomabd", "tahomabd.ttf")] {
        if let Ok(bytes) = std::fs::read(format!("C:/Windows/Fonts/{file}")) {
            fonts
                .font_data
                .insert(name.into(), Arc::new(FontData::from_owned(bytes)));
        }
    }
    if fonts.font_data.contains_key("tahoma") {
        fonts
            .families
            .entry(FontFamily::Proportional)
            .or_default()
            .insert(0, "tahoma".into());
    }
    // Japanese fallback so OCR text and glossaries render instead of empty boxes.
    // The first Windows font found wins; .ttc collections use face 0.
    for file in ["YuGothM.ttc", "meiryo.ttc", "BIZ-UDGothicR.ttc", "NotoSansJP-VF.ttf", "msgothic.ttc"] {
        if let Ok(bytes) = std::fs::read(format!("C:/Windows/Fonts/{file}")) {
            fonts
                .font_data
                .insert("cjk".into(), Arc::new(FontData::from_owned(bytes)));
            for family in [FontFamily::Proportional, FontFamily::Monospace] {
                fonts.families.entry(family).or_default().push("cjk".into());
            }
            break;
        }
    }
    ctx.set_fonts(fonts);

    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(4.0, 3.0);
    style.spacing.button_padding = egui::vec2(8.0, 3.0);
    style.spacing.interact_size.y = 20.0;
    for (_, font) in style.text_styles.iter_mut() {
        font.size = font.size.min(13.0).max(11.0);
    }
    let v = &mut style.visuals;
    *v = egui::Visuals::light();
    v.panel_fill = FACE;
    v.window_fill = FACE;
    v.faint_bg_color = Color32::from_rgb(0xec, 0xe9, 0xe2);
    v.extreme_bg_color = Color32::WHITE;
    v.override_text_color = Some(Color32::BLACK);
    v.window_corner_radius = CornerRadius::ZERO;
    v.menu_corner_radius = CornerRadius::ZERO;
    v.window_shadow = Shadow {
        offset: [2, 2],
        blur: 0,
        spread: 0,
        color: DARK,
    };
    v.popup_shadow = Shadow {
        offset: [2, 2],
        blur: 0,
        spread: 0,
        color: DARK,
    };
    v.window_stroke = Stroke::new(1.0, DARK);
    v.selection.bg_fill = SELECT;
    v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    v.hyperlink_color = Color32::from_rgb(0x00, 0x00, 0xcc);
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::ZERO;
        w.bg_fill = FACE;
        w.weak_bg_fill = FACE;
        w.bg_stroke = Stroke::new(1.0, SHADOW);
        w.fg_stroke = Stroke::new(1.0, Color32::BLACK);
        w.expansion = 0.0;
    }
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, SHADOW);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, DARK);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(0xc4, 0xc0, 0xb8);
    v.widgets.active.bg_stroke = Stroke::new(1.0, Color32::BLACK);
    v.widgets.open.weak_bg_fill = SELECT;
    v.widgets.open.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    ctx.set_global_style(style);
}

/// Classic 3D bevel: raised for buttons and windows, sunken for fields and lists.
pub fn bevel(p: &Painter, r: Rect, raised: bool) {
    let (tl, br, tl2, br2) = if raised {
        (LIGHT, DARK, FACE, SHADOW)
    } else {
        (SHADOW, LIGHT, DARK, FACE)
    };
    let s = |c| Stroke::new(1.0, c);
    p.line_segment([r.left_top(), r.right_top()], s(tl));
    p.line_segment([r.left_top(), r.left_bottom()], s(tl));
    p.line_segment([r.left_bottom(), r.right_bottom()], s(br));
    p.line_segment([r.right_top(), r.right_bottom()], s(br));
    let r2 = r.shrink(1.0);
    p.line_segment([r2.left_top(), r2.right_top()], s(tl2));
    p.line_segment([r2.left_top(), r2.left_bottom()], s(tl2));
    p.line_segment([r2.left_bottom(), r2.right_bottom()], s(br2));
    p.line_segment([r2.right_top(), r2.right_bottom()], s(br2));
}

/// Horizontal two-colour gradient, as on Windows 2000 title bars.
pub fn gradient(p: &Painter, r: Rect, left: Color32, right: Color32) {
    let steps = 48;
    let w = r.width() / steps as f32;
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        let c = Color32::from_rgb(
            (left.r() as f32 + (right.r() as f32 - left.r() as f32) * t) as u8,
            (left.g() as f32 + (right.g() as f32 - left.g() as f32) * t) as u8,
            (left.b() as f32 + (right.b() as f32 - left.b() as f32) * t) as u8,
        );
        let x = r.left() + i as f32 * w;
        p.rect_filled(
            Rect::from_min_max(egui::pos2(x, r.top()), egui::pos2(x + w + 1.0, r.bottom())),
            0.0,
            c,
        );
    }
}

/// A teal caption strip used for panel headings.
pub fn caption(ui: &mut egui::Ui, text: &str) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 18.0), egui::Sense::hover());
    gradient(ui.painter(), rect, ASAGI, ASAGI_LIGHT);
    ui.painter().text(
        rect.left_center() + egui::vec2(4.0, 0.0),
        egui::Align2::LEFT_CENTER,
        text,
        egui::FontId::proportional(12.0),
        Color32::WHITE,
    );
}
