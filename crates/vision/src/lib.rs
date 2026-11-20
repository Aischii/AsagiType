//! Pure-Rust (candle) vision models for AsagiType 2000.

pub mod clean;
pub mod detect;
pub mod ocr;
pub mod rotation;

pub use detect::{Class, ComicDetector, Detection};
pub use ocr::MangaOcr;

use anyhow::{Context, Result};
use candle_core::Device;
use image::DynamicImage;
use std::path::Path;

/// A text area found on a page.
#[derive(Clone, Debug)]
pub struct TextRegion {
    /// "dialogue" when inside a detected bubble, otherwise "aside".
    pub kind: &'static str,
    pub text_box: [i32; 4],
    pub bubble: Option<[i32; 4]>,
    pub source_text: String,
    pub score: f32,
    /// Text angle in degrees (clockwise); 0 for level text.
    pub rotation: f32,
    /// Lettering frame for rotated text, as a fraction of the box size [w, h].
    pub frame: Option<[f32; 2]>,
}

pub struct Models {
    pub detector: ComicDetector,
    pub ocr: MangaOcr,
}

impl Models {
    /// `models_dir` holds `bubble-detector/` and `manga-ocr/`.
    pub fn load(models_dir: &Path) -> Result<Self> {
        let dev = Device::Cpu;
        Ok(Self {
            detector: ComicDetector::load(&models_dir.join("bubble-detector"), &dev)
                .context("loading the detector")?,
            ocr: MangaOcr::load(&models_dir.join("manga-ocr"), &dev)
                .context("loading manga-ocr")?,
        })
    }

    /// Detect text, pair it with bubbles, OCR it and sort in manga reading order.
    pub fn analyze(&self, img: &DynamicImage, threshold: f32) -> Result<Vec<TextRegion>> {
        let dets = dedupe(self.detector.detect(img, threshold)?);
        let bubbles: Vec<[f32; 4]> = dets
            .iter()
            .filter(|d| d.class == Class::Bubble)
            .map(|d| d.bbox)
            .collect();
        let (w, h) = (img.width() as i32, img.height() as i32);
        let clamp = |b: [f32; 4]| -> [i32; 4] {
            [
                b[0].round().max(0.0) as i32,
                b[1].round().max(0.0) as i32,
                (b[2].round() as i32).min(w),
                (b[3].round() as i32).min(h),
            ]
        };
        let mut regions = vec![];
        for d in dets.iter().filter(|d| d.class != Class::Bubble) {
            let tb = clamp(d.bbox);
            if tb[2] - tb[0] < 4 || tb[3] - tb[1] < 4 {
                continue;
            }
            let bubble = bubbles
                .iter()
                .filter(|b| overlap_of(&d.bbox, b) > 0.6)
                .min_by(|a, b| area(a).total_cmp(&area(b)))
                .map(|b| clamp(*b));
            let crop = img.crop_imm(
                tb[0] as u32,
                tb[1] as u32,
                (tb[2] - tb[0]) as u32,
                (tb[3] - tb[1]) as u32,
            );
            // Tilted text: OCR the straightened crop, letter in a rotated frame.
            let angle = rotation::estimate(&crop.to_rgb8());
            let (ocr_crop, frame) = if angle.degrees != 0.0 {
                let straight = rotation::straighten(&crop.to_rgb8(), angle.degrees);
                let (bw, bh) = ((tb[2] - tb[0]) as f32, (tb[3] - tb[1]) as f32);
                let [along, across] = angle.extent;
                // Frame is along x across for lines, across x along for columns.
                let (fw, fh) = if angle.vertical {
                    (across, along)
                } else {
                    (along, across)
                };
                (
                    image::DynamicImage::ImageRgb8(straight),
                    Some([(fw / bw).min(1.0), (fh / bh).min(1.0)]),
                )
            } else {
                (crop.clone(), None)
            };
            regions.push(TextRegion {
                kind: if bubble.is_some() {
                    "dialogue"
                } else {
                    "aside"
                },
                text_box: tb,
                bubble,
                source_text: self.ocr.recognize(&ocr_crop)?,
                score: d.score,
                rotation: angle.degrees,
                frame,
            });
        }
        reading_order(&mut regions, h);
        Ok(regions)
    }
}

fn area(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

/// Fraction of `a` covered by `b`.
fn overlap_of(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let i = [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ];
    area(&i) / area(a).max(1.0)
}

/// Drop lower-scoring boxes of the same class that mostly cover a kept one.
fn dedupe(mut dets: Vec<Detection>) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = vec![];
    for d in dets {
        let dup = kept.iter().any(|k| {
            k.class == d.class && {
                let i = area(&[
                    d.bbox[0].max(k.bbox[0]),
                    d.bbox[1].max(k.bbox[1]),
                    d.bbox[2].min(k.bbox[2]),
                    d.bbox[3].min(k.bbox[3]),
                ]);
                i / (area(&d.bbox) + area(&k.bbox) - i).max(1.0) > 0.7
            }
        });
        if !dup {
            kept.push(d);
        }
    }
    kept
}

/// Manga order: top-to-bottom bands, right-to-left inside a band.
pub fn reading_order(regions: &mut [TextRegion], page_height: i32) {
    let band = (page_height / 12).max(1);
    regions.sort_by(|a, b| {
        let (ay, by) = (a.text_box[1] / band, b.text_box[1] / band);
        ay.cmp(&by).then(b.text_box[2].cmp(&a.text_box[2]))
    });
}
