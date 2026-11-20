//! Text angle estimation by projection search, and straightening for OCR.
//!
//! Ink pixels (the minority tone after Otsu, so white text on a dark strip works too) are
//! projected across the text direction for every angle from -45 to 45 degrees in 1 degree
//! steps, for both horizontal lines and vertical columns. Aligned text gives sharp peaks, so
//! the angle with the most concentrated projection wins. Angles under 3 degrees count as level.

use image::{Rgb, RgbImage};

/// Angles smaller than this are treated as level text.
pub const MIN_ANGLE: f32 = 3.0;
const MAX_ANGLE: i32 = 45;
/// A tilted angle must beat level by this factor, so noise does not tilt level text.
const MARGIN: f32 = 1.15;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextAngle {
    /// Degrees, clockwise on screen (image y points down).
    pub degrees: f32,
    /// Size of the text block measured along and across the lines at that angle, in px.
    pub extent: [f32; 2],
    /// True for vertical columns (Japanese tategaki), false for horizontal lines.
    pub vertical: bool,
}

fn lum(p: &Rgb<u8>) -> f32 {
    0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
}

fn otsu(values: &[f32]) -> f32 {
    let mut hist = [0usize; 256];
    for &v in values {
        hist[v.clamp(0.0, 255.0) as usize] += 1;
    }
    let total = values.len() as f64;
    let sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let (mut w0, mut sum0, mut best, mut best_t) = (0.0f64, 0.0f64, -1.0f64, 128usize);
    for (t, &c) in hist.iter().enumerate() {
        w0 += c as f64;
        if w0 == 0.0 || w0 == total {
            continue;
        }
        sum0 += t as f64 * c as f64;
        let (m0, m1) = (sum0 / w0, (sum - sum0) / (total - w0));
        let between = w0 * (total - w0) * (m0 - m1).powi(2);
        if between > best {
            best = between;
            best_t = t;
        }
    }
    best_t as f32 + 0.5
}

/// Ink pixel coordinates (relative to the crop centre).
fn ink_points(crop: &RgbImage) -> Vec<(f32, f32)> {
    let lums: Vec<f32> = crop.pixels().map(lum).collect();
    if lums.is_empty() {
        return vec![];
    }
    let thr = otsu(&lums);
    let dark = lums.iter().filter(|&&v| v < thr).count();
    let ink_dark = dark * 2 <= lums.len();
    let (cx, cy) = (crop.width() as f32 / 2.0, crop.height() as f32 / 2.0);
    let w = crop.width() as usize;
    lums.iter()
        .enumerate()
        .filter(|(_, &l)| if ink_dark { l < thr } else { l > thr })
        .map(|(i, _)| ((i % w) as f32 - cx, (i / w) as f32 - cy))
        .collect()
}

/// Sharpness of a projection: sum of squared bin counts, normalised.
fn sharpness(values: impl Iterator<Item = f32>, n: usize) -> f32 {
    let vals: Vec<f32> = values.collect();
    let lo = vals.iter().cloned().fold(f32::MAX, f32::min);
    let hi = vals.iter().cloned().fold(f32::MIN, f32::max);
    let bins = (hi - lo).ceil() as usize + 1;
    let mut hist = vec![0u32; bins.max(1)];
    for v in &vals {
        hist[((v - lo) as usize).min(bins - 1)] += 1;
    }
    hist.iter().map(|&c| (c as f32).powi(2)).sum::<f32>() / (n as f32).powi(2)
}

/// Estimate the angle of the text in `crop`. Returns level (0 degrees) when unsure.
pub fn estimate(crop: &RgbImage) -> TextAngle {
    let pts = ink_points(crop);
    let level = TextAngle {
        degrees: 0.0,
        extent: [crop.width() as f32, crop.height() as f32],
        vertical: crop.height() > crop.width(),
    };
    if pts.len() < 30 {
        return level;
    }
    let n = pts.len();
    // (score, degrees, vertical)
    let mut best = (0f32, 0i32, false);
    let mut level_score = 0f32;
    for deg in -MAX_ANGLE..=MAX_ANGLE {
        let (s, c) = (deg as f32).to_radians().sin_cos();
        // Horizontal lines: distance across lines is v = -x sin + y cos.
        let h = sharpness(pts.iter().map(|&(x, y)| -x * s + y * c), n);
        // Vertical columns: distance across columns is u = x cos + y sin.
        let v = sharpness(pts.iter().map(|&(x, y)| x * c + y * s), n);
        let (score, vertical) = if v > h { (v, true) } else { (h, false) };
        if deg == 0 {
            level_score = score;
        }
        if score > best.0 {
            best = (score, deg, vertical);
        }
    }
    let (score, deg, vertical) = best;
    if (deg.abs() as f32) < MIN_ANGLE || score < level_score * MARGIN {
        return level;
    }
    let (s, c) = (deg as f32).to_radians().sin_cos();
    let along: Vec<f32> = pts.iter().map(|&(x, y)| x * c + y * s).collect();
    let across: Vec<f32> = pts.iter().map(|&(x, y)| -x * s + y * c).collect();
    let span = |v: &[f32]| {
        v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min)
    };
    TextAngle {
        degrees: deg as f32,
        extent: [span(&along) + 8.0, span(&across) + 8.0],
        vertical,
    }
}

/// Rotate `crop` by `-degrees` around its centre so the text is level, on a canvas large
/// enough to hold it, padded with the border colour. Bilinear sampling.
pub fn straighten(crop: &RgbImage, degrees: f32) -> RgbImage {
    if degrees.abs() < f32::EPSILON {
        return crop.clone();
    }
    let (w, h) = (crop.width() as f32, crop.height() as f32);
    let (s, c) = degrees.to_radians().sin_cos();
    let (nw, nh) = (
        (w * c.abs() + h * s.abs()).ceil() as u32,
        (w * s.abs() + h * c.abs()).ceil() as u32,
    );
    let bg = border_colour(crop);
    let (cx, cy, ncx, ncy) = (w / 2.0, h / 2.0, nw as f32 / 2.0, nh as f32 / 2.0);
    RgbImage::from_fn(nw, nh, |x, y| {
        // Output pixel -> source: rotate by +degrees.
        let (dx, dy) = (x as f32 + 0.5 - ncx, y as f32 + 0.5 - ncy);
        let sx = dx * c - dy * s + cx - 0.5;
        let sy = dx * s + dy * c + cy - 0.5;
        sample(crop, sx, sy, bg)
    })
}

fn border_colour(img: &RgbImage) -> Rgb<u8> {
    let (w, h) = (img.width(), img.height());
    let mut ch: [Vec<u8>; 3] = Default::default();
    for x in 0..w {
        for y in [0, h.saturating_sub(1)] {
            let p = img.get_pixel(x, y);
            for k in 0..3 {
                ch[k].push(p[k]);
            }
        }
    }
    Rgb(std::array::from_fn(|k| {
        ch[k].sort_unstable();
        ch[k].get(ch[k].len() / 2).copied().unwrap_or(255)
    }))
}

fn sample(img: &RgbImage, x: f32, y: f32, bg: Rgb<u8>) -> Rgb<u8> {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let get = |xi: i64, yi: i64| -> [f32; 3] {
        if xi < 0 || yi < 0 || xi >= img.width() as i64 || yi >= img.height() as i64 {
            bg.0.map(|v| v as f32)
        } else {
            img.get_pixel(xi as u32, yi as u32).0.map(|v| v as f32)
        }
    };
    let (xi, yi) = (x0 as i64, y0 as i64);
    let (a, b, c, d) = (
        get(xi, yi),
        get(xi + 1, yi),
        get(xi, yi + 1),
        get(xi + 1, yi + 1),
    );
    Rgb(std::array::from_fn(|k| {
        let top = a[k] * (1.0 - fx) + b[k] * fx;
        let bot = c[k] * (1.0 - fx) + d[k] * fx;
        (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White crop with three dark text lines tilted by `deg` (clockwise).
    fn lines(deg: f32, vertical: bool, dark_on_light: bool) -> RgbImage {
        let (bg, ink) = if dark_on_light {
            ([255, 255, 255], [0, 0, 0])
        } else {
            ([20, 20, 20], [250, 250, 250])
        };
        let mut img = RgbImage::from_pixel(240, 240, Rgb(bg));
        let (s, c) = deg.to_radians().sin_cos();
        for k in -1..=1 {
            for t in -70..=70 {
                for w in -2..=2 {
                    // Dashes so it looks like characters, not solid bars.
                    if (t + 70) % 14 > 10 {
                        continue;
                    }
                    let (a, b) = (t as f32, (k * 30 + w) as f32);
                    let (lx, ly) = if vertical { (b, a) } else { (a, b) };
                    let (x, y) = (lx * c - ly * s + 120.0, lx * s + ly * c + 120.0);
                    img.put_pixel(x as u32, y as u32, Rgb(ink));
                }
            }
        }
        img
    }

    #[test]
    fn finds_tilt_of_horizontal_lines() {
        for deg in [-20.0f32, -8.0, 12.0, 30.0] {
            let a = estimate(&lines(deg, false, true));
            assert!((a.degrees - deg).abs() <= 1.0, "{deg}: got {a:?}");
            assert!(!a.vertical);
        }
    }

    #[test]
    fn finds_tilt_of_vertical_columns_and_light_text() {
        let a = estimate(&lines(15.0, true, true));
        assert!((a.degrees - 15.0).abs() <= 1.0 && a.vertical, "{a:?}");
        let b = estimate(&lines(-10.0, false, false));
        assert!((b.degrees + 10.0).abs() <= 1.0, "{b:?}");
    }

    #[test]
    fn level_and_small_tilts_stay_level() {
        assert_eq!(estimate(&lines(0.0, true, true)).degrees, 0.0);
        assert_eq!(estimate(&lines(2.0, false, true)).degrees, 0.0);
    }

    #[test]
    fn straighten_levels_the_text() {
        let tilted = lines(20.0, false, true);
        let level = straighten(&tilted, 20.0);
        let a = estimate(&level);
        assert_eq!(a.degrees, 0.0, "{a:?}");
        assert!(level.width() > tilted.width());
    }
}
