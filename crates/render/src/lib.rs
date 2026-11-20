//! Lettering: shape text with cosmic-text and rasterise with tiny-skia.
//! The same output is used on the canvas and in exports.

use cosmic_text::{
    Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, Style as FontStyle, SwashCache,
    Weight, Wrap,
};
use std::path::Path;
use tiny_skia::{Paint, Pixmap, Rect, Transform};

/// Scans are 168 DPI: 1 pt = 168/72 px.
pub const PX_PER_PT: f32 = 168.0 / 72.0;
/// Smallest readable line height on a 1600 px tall page.
pub const MIN_LINE_PX: f32 = 16.0;
pub const LINE_HEIGHT: f32 = 1.15;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Align {
    Left,
    #[default]
    Center,
    Right,
}

/// How a block of lettering looks. Case changes are applied by the caller.
#[derive(Clone, Copy, Debug)]
pub struct Style<'a> {
    pub family: &'a str,
    pub bold: bool,
    pub italic: bool,
    pub align: Align,
}

impl<'a> Style<'a> {
    pub fn plain(family: &'a str) -> Self {
        Self {
            family,
            bold: false,
            italic: false,
            align: Align::Center,
        }
    }
}

pub struct Lettering {
    pub pixmap: Pixmap,
    /// Font size actually used, in px.
    pub font_px: f32,
    /// False when the text overflows even at the minimum size.
    pub fits: bool,
}

pub struct Renderer {
    fonts: FontSystem,
    cache: SwashCache,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    /// Starts with the system fonts.
    pub fn new() -> Self {
        Self {
            fonts: FontSystem::new(),
            cache: SwashCache::new(),
        }
    }

    /// Add every .ttf/.otf in a folder (for example the user's comic fonts).
    pub fn load_fonts_dir(&mut self, dir: &Path) -> usize {
        let before = self.fonts.db().len();
        self.fonts.db_mut().load_fonts_dir(dir);
        self.fonts.db().len() - before
    }

    pub fn has_family(&self, family: &str) -> bool {
        self.fonts.db().faces().any(|f| {
            f.families
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case(family))
        })
    }

    pub fn families(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .fonts
            .db()
            .faces()
            .filter_map(|f| f.families.first().map(|(n, _)| n.clone()))
            .collect();
        v.sort();
        v.dedup();
        v
    }

    fn layout(
        &mut self,
        text: &str,
        style: &Style,
        font_px: f32,
        width: f32,
    ) -> (Buffer, f32, f32) {
        let mut buffer = Buffer::new(
            &mut self.fonts,
            Metrics::new(font_px, font_px * LINE_HEIGHT),
        );
        buffer.set_wrap(Wrap::Word);
        buffer.set_size(Some(width), None);
        let attrs = Attrs::new()
            .family(Family::Name(style.family))
            .weight(if style.bold {
                Weight::BOLD
            } else {
                Weight::NORMAL
            })
            .style(if style.italic {
                FontStyle::Italic
            } else {
                FontStyle::Normal
            });
        let align = match style.align {
            Align::Left => cosmic_text::Align::Left,
            Align::Center => cosmic_text::Align::Center,
            Align::Right => cosmic_text::Align::Right,
        };
        buffer.set_text(text, &attrs, Shaping::Advanced, Some(align));
        buffer.shape_until_scroll(&mut self.fonts, false);
        let mut height: f32 = 0.0;
        let mut widest: f32 = 0.0;
        for run in buffer.layout_runs() {
            height = height.max(run.line_top + run.line_height);
            widest = widest.max(run.line_w);
        }
        (buffer, widest, height)
    }

    /// Render `text` in a `width` x `height` px box, vertically centred.
    /// `size_pt` > 0 fixes the size; otherwise the largest size that fits is used.
    pub fn letter(
        &mut self,
        text: &str,
        style: &Style,
        width: u32,
        height: u32,
        size_pt: f32,
    ) -> Option<Lettering> {
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        let min_px = MIN_LINE_PX / LINE_HEIGHT;
        let mut font_px = if size_pt > 0.0 {
            size_pt * PX_PER_PT
        } else {
            (h / 2.5).min(80.0).max(min_px)
        };
        let (mut buffer, mut widest, mut used_h) = self.layout(text, style, font_px, w);
        if size_pt <= 0.0 {
            while (used_h > h || widest > w) && font_px > min_px {
                font_px = (font_px * 0.94).max(min_px);
                (buffer, widest, used_h) = self.layout(text, style, font_px, w);
            }
        }
        let fits = used_h <= h && widest <= w;
        let mut pixmap = Pixmap::new(width.max(1), height.max(1))?;
        let offset_y = ((h - used_h) / 2.0).max(0.0);
        let mut paint = Paint::default();
        paint.anti_alias = false;
        buffer.draw(
            &mut self.fonts,
            &mut self.cache,
            Color::rgb(0, 0, 0),
            |x, y, gw, gh, color| {
                let a = color.a();
                if a == 0 {
                    return;
                }
                if let Some(rect) =
                    Rect::from_xywh(x as f32, y as f32 + offset_y, gw as f32, gh as f32)
                {
                    paint.set_color_rgba8(color.r(), color.g(), color.b(), a);
                    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
                }
            },
        );
        Some(Lettering {
            pixmap,
            font_px,
            fits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_text_and_draws_ink() {
        let mut r = Renderer::new();
        let family = r.families().into_iter().next().expect("a system font");
        let l = r
            .letter(
                "SO THIS IS WHERE YOU ALWAYS SHOP.",
                &Style::plain(&family),
                180,
                260,
                0.0,
            )
            .unwrap();
        assert!(l.fits);
        assert!(l.font_px * LINE_HEIGHT >= MIN_LINE_PX - 0.01);
        assert!(l.pixmap.pixels().iter().any(|p| p.alpha() > 0));
    }

    /// Ink bounding box columns: (first, last) column with any ink.
    fn ink_columns(l: &Lettering) -> (u32, u32) {
        let w = l.pixmap.width();
        let cols: Vec<u32> = l
            .pixmap
            .pixels()
            .iter()
            .enumerate()
            .filter(|(_, p)| p.alpha() > 0)
            .map(|(i, _)| i as u32 % w)
            .collect();
        (*cols.iter().min().unwrap(), *cols.iter().max().unwrap())
    }

    #[test]
    fn alignment_moves_short_lines() {
        let mut r = Renderer::new();
        let family = r.families().into_iter().next().unwrap();
        let mut style = Style::plain(&family);
        let mut at = |align| {
            style.align = align;
            ink_columns(&r.letter("HI", &style, 300, 60, 8.0).unwrap())
        };
        let (left, center, right) = (at(Align::Left), at(Align::Center), at(Align::Right));
        assert!(
            left.0 < center.0 && center.0 < right.0,
            "{left:?} {center:?} {right:?}"
        );
    }
}
