//! Cleaning: erase source text and keep everything else.
//!
//! The output is a page-size RGBA layer that is transparent everywhere except the
//! repainted pixels, plus the erase mask and a per-box "needs touch-up" flag.
//!
//! Per text box (Auto):
//! 1. Ink polarity and threshold come from Otsu on the text box: the smaller class is ink
//!    (dark letters on white, or white letters on a dark strip).
//! 2. Ink pixels near the box are split into 8-connected strokes. A stroke is erased only if
//!    it lies wholly inside the text box plus [`STROKE_SLACK`] px. Strokes crossing that edge
//!    (bubble outlines, panel borders, art) are kept. A kept stroke that also reaches deep into
//!    the box is probably a letter merged with an outline, so the box is flagged for touch-up.
//! 3. The erase mask is the erasable strokes grown by [`HALO`] px (anti-aliasing).
//! 4. The mask is filled according to what surrounds it: a solid colour on plain backgrounds
//!    (bubble interiors), smooth diffusion (Telea-style onion peel) on gradients such as sky,
//!    or exemplar patch synthesis on texture such as screentone.
//!
//! Shaped boxes instead paint a new bubble (rectangle, rounded or ellipse) with fill and outline.

use image::{GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub struct CleanBox {
    pub text_box: [i32; 4],
    pub bubble: Option<[i32; 4]>,
    /// Fill colour instead of the sampled one (Auto, plain backgrounds).
    pub fill: Option<[u8; 3]>,
    /// Paint a new bubble instead of erasing.
    pub shape: Option<ShapeStyle>,
    /// Text angle in degrees, clockwise. Tilted text is cleaned in its own straightened frame.
    pub rotation: f32,
    /// Text frame for tilted text as a fraction of the box [w, h]; None uses the box.
    pub frame: Option<[f32; 2]>,
}

impl CleanBox {
    pub fn auto(text_box: [i32; 4], bubble: Option<[i32; 4]>) -> Self {
        Self {
            text_box,
            bubble,
            fill: None,
            shape: None,
            rotation: 0.0,
            frame: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Rect,
    Rounded,
    Ellipse,
}

#[derive(Clone, Copy, Debug)]
pub struct ShapeStyle {
    pub shape: Shape,
    pub fill: [u8; 4],
    /// Colour and width in page pixels.
    pub outline: Option<([u8; 4], f32)>,
}

/// How a box's mask was filled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillMethod {
    Solid,
    Smooth,
    Texture,
    Shape,
    Nothing,
}

pub struct CleanResult {
    pub layer: RgbaImage,
    /// 255 where pixels were erased (Auto boxes) or painted (shapes).
    pub mask: GrayImage,
    /// Per box: a stroke that reaches into the text was kept because it also crosses the box edge.
    pub touch_up: Vec<bool>,
    pub methods: Vec<FillMethod>,
}

/// Strokes may extend this far past the text box and still be erased.
pub const STROKE_SLACK: i32 = 3;
/// Erase mask growth around strokes, for anti-aliased edges.
const HALO: i32 = 2;
/// Context around the box used to sample the background.
const CONTEXT: i32 = 16;
/// A kept stroke with this many pixels inside the box core flags the box.
const TOUCH_PIXELS: usize = 15;
const PLAIN_STD: f32 = 4.0;
const TEXTURE_LAPLACIAN: f32 = 7.0;

fn lum(p: &Rgb<u8>) -> f32 {
    0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
}

fn clamp_box(b: [i32; 4], w: i32, h: i32) -> [i32; 4] {
    [
        b[0].clamp(0, w),
        b[1].clamp(0, h),
        b[2].clamp(0, w),
        b[3].clamp(0, h),
    ]
}

fn grow(b: [i32; 4], by: i32) -> [i32; 4] {
    [b[0] - by, b[1] - by, b[2] + by, b[3] + by]
}

fn inside(b: &[i32; 4], x: i32, y: i32) -> bool {
    x >= b[0] && y >= b[1] && x < b[2] && y < b[3]
}

/// Clean every box onto one transparent layer the size of `img`.
pub fn clean_page(img: &RgbImage, boxes: &[CleanBox]) -> RgbaImage {
    clean_page_full(img, boxes).layer
}

pub fn clean_page_full(img: &RgbImage, boxes: &[CleanBox]) -> CleanResult {
    clean_page_with_balloons(img, boxes, &[])
}

/// Like [`clean_page_full`], with balloon masks from the segmentation model: text inside a
/// segmented balloon is erased only within the balloon interior, so outlines stay.
pub fn clean_page_with_balloons(
    img: &RgbImage,
    boxes: &[CleanBox],
    balloons: &[crate::bubbles::Balloon],
) -> CleanResult {
    let mut res = CleanResult {
        layer: RgbaImage::new(img.width(), img.height()),
        mask: GrayImage::new(img.width(), img.height()),
        touch_up: vec![],
        methods: vec![],
    };
    for b in boxes {
        let (touch, method) = match b.shape {
            Some(style) => {
                paint_shape(b.text_box, &style, &mut res.layer, &mut res.mask);
                (false, FillMethod::Shape)
            }
            None if b.rotation.abs() >= MIN_TILT => {
                clean_rotated(img, b, &mut res.layer, &mut res.mask)
            }
            None => match balloon_for(&b.text_box, balloons) {
                Some(bal) => clean_in_balloon(img, b, bal, &mut res.layer, &mut res.mask)
                    .unwrap_or_else(|| clean_auto(img, b, &mut res.layer, &mut res.mask)),
                _ => clean_auto(img, b, &mut res.layer, &mut res.mask),
            },
        };
        res.touch_up.push(touch);
        res.methods.push(method);
    }
    res
}

/// Tilts below this are cleaned upright.
const MIN_TILT: f32 = 1.0;
/// Margin around the straightened text frame, so the paper around the letters is sampled.
const FRAME_PAD: f32 = 4.0;

/// Clean tilted text in its own frame: straighten the area around the box, clean the level
/// text frame there, then map the erased pixels back. Cleaning the upright box instead mixes
/// the tilted strip with the art around it, which picks the wrong ink polarity.
fn clean_rotated(
    img: &RgbImage,
    cb: &CleanBox,
    layer: &mut RgbaImage,
    mask_out: &mut GrayImage,
) -> (bool, FillMethod) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let t = clamp_box(cb.text_box, w, h);
    let region = clamp_box(grow(t, CONTEXT), w, h);
    let (rw, rh) = (region[2] - region[0], region[3] - region[1]);
    if t[2] - t[0] < 3 || t[3] - t[1] < 3 || rw < 3 || rh < 3 {
        return (false, FillMethod::Nothing);
    }
    let crop =
        image::imageops::crop_imm(img, region[0] as u32, region[1] as u32, rw as u32, rh as u32)
            .to_image();
    let straight = crate::rotation::straighten(&crop, cb.rotation);
    let (s, c) = cb.rotation.to_radians().sin_cos();
    let (cx, cy) = (rw as f32 / 2.0, rh as f32 / 2.0);
    let (ncx, ncy) = (straight.width() as f32 / 2.0, straight.height() as f32 / 2.0);
    // Crop pixel centre -> straightened coordinates (inverse of `straighten`'s mapping).
    let fwd = |x: f32, y: f32| {
        let (ux, uy) = (x - cx, y - cy);
        (c * ux + s * uy + ncx, -s * ux + c * uy + ncy)
    };
    let (bw, bh) = ((t[2] - t[0]) as f32, (t[3] - t[1]) as f32);
    let [fx, fy] = cb.frame.unwrap_or([1.0, 1.0]);
    let (fw, fh) = (fx * bw + 2.0 * FRAME_PAD, fy * bh + 2.0 * FRAME_PAD);
    let (mx, my) = fwd(
        (t[0] + t[2]) as f32 / 2.0 - region[0] as f32,
        (t[1] + t[3]) as f32 / 2.0 - region[1] as f32,
    );
    let approx = [
        (mx - fw / 2.0).floor() as i32,
        (my - fh / 2.0).floor() as i32,
        (mx + fw / 2.0).ceil() as i32,
        (my + fh / 2.0).ceil() as i32,
    ];
    // Tilted captions usually sit on a strip of paper over the art: clean just that strip.
    let band = paper_band(&straight, approx);
    let frame = band.map(|(r, _)| r).unwrap_or(approx);
    let inner = CleanBox {
        text_box: frame,
        bubble: None,
        // A strip of paper is flat: fill it with its own tone, never with the art around it.
        fill: cb.fill.or(band.map(|(_, paper)| paper)),
        shape: None,
        rotation: 0.0,
        frame: None,
    };
    let res = clean_page_full(&straight, &[inner]);
    let (sw, sh) = (straight.width() as i32, straight.height() as i32);
    for y in region[1]..region[3] {
        for x in region[0]..region[2] {
            let (sx, sy) = fwd(
                (x - region[0]) as f32 + 0.5,
                (y - region[1]) as f32 + 0.5,
            );
            let (sx, sy) = (sx.floor() as i32, sy.floor() as i32);
            if sx < 0 || sy < 0 || sx >= sw || sy >= sh {
                continue;
            }
            if res.mask.get_pixel(sx as u32, sy as u32).0[0] == 0 {
                continue;
            }
            let p = res.layer.get_pixel(sx as u32, sy as u32);
            if p.0[3] == 0 {
                continue;
            }
            layer.put_pixel(x as u32, y as u32, Rgba([p.0[0], p.0[1], p.0[2], 255]));
            mask_out.put_pixel(x as u32, y as u32, Luma([255]));
        }
    }
    (res.touch_up[0], res.methods[0])
}

/// In a straightened crop, the strip of paper that carries the text line through the middle
/// of `approx`: rows and columns, grown out from the centre, in which at least a third of the
/// pixels have the paper's tone. Once a row of plain paper (the strip's margin) is reached, growth
/// continues only through plain paper, so it stops at the strip's edge rather than running into
/// light art beside it. Returns the strip and the paper colour; None when no clear strip is found.
fn paper_band(img: &RgbImage, approx: [i32; 4]) -> Option<([i32; 4], [u8; 3])> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let a = clamp_box(approx, w, h);
    if a[2] - a[0] < 8 || a[3] - a[1] < 8 {
        return None;
    }
    let cy = (a[1] + a[3]) / 2;
    let (iw0, iw1) = (a[0] + (a[2] - a[0]) / 10, a[2] - (a[2] - a[0]) / 10);
    // Candidate paper tones: the most common tones along the text line. Dense lettering can
    // outnumber the paper there, so each is tried and only one with plain margins is accepted.
    let mut bins = [0u32; 16];
    let mut sums = [0f32; 16];
    for y in (cy - 2).max(0)..=(cy + 2).min(h - 1) {
        for x in iw0..iw1 {
            let v = lum(img.get_pixel(x as u32, y as u32));
            let k = ((v / 16.0) as usize).min(15);
            bins[k] += 1;
            sums[k] += v;
        }
    }
    let mut order: Vec<usize> = (0..16).filter(|&k| bins[k] > 0).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(bins[k]));
    order
        .into_iter()
        .take(3)
        .find_map(|k| band_for_tone(img, a, cy, sums[k] / bins[k] as f32))
}

fn band_for_tone(img: &RgbImage, a: [i32; 4], cy: i32, paper: f32) -> Option<([i32; 4], [u8; 3])> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let l = |x: i32, y: i32| lum(img.get_pixel(x as u32, y as u32));
    let is_paper = |x: i32, y: i32| (l(x, y) - paper).abs() < 48.0;
    const MIN_FRAC: f32 = 0.35;
    const MARGIN_FRAC: f32 = 0.9;
    let (iw0, iw1) = (a[0] + (a[2] - a[0]) / 10, a[2] - (a[2] - a[0]) / 10);
    let row_frac = |y: i32| {
        let n = (iw0..iw1).filter(|&x| is_paper(x, y)).count();
        n as f32 / (iw1 - iw0).max(1) as f32
    };
    if row_frac(cy) < MIN_FRAC {
        return None;
    }
    let limit = CONTEXT * 2;
    let grow_rows = |step: i32, stop: i32| {
        let (mut y, mut in_margin) = (cy, false);
        while y != stop {
            let f = row_frac(y + step);
            if f < if in_margin { MARGIN_FRAC } else { MIN_FRAC } {
                break;
            }
            in_margin |= f >= MARGIN_FRAC;
            y += step;
        }
        (y, in_margin)
    };
    let (top, m_top) = grow_rows(-1, (a[1] - limit).max(0));
    let (bot, m_bot) = grow_rows(1, (a[3] + limit).min(h - 1));
    // Paper has plain margins around the text; a run of ink or art does not.
    if !m_top && !m_bot {
        return None;
    }
    let col_ok = |x: i32| {
        let n = (top..=bot).filter(|&y| is_paper(x, y)).count();
        n as f32 >= MIN_FRAC * (bot - top + 1) as f32
    };
    let cx = (a[0] + a[2]) / 2;
    let (mut left, mut right) = (cx, cx);
    while left > (a[0] - limit).max(0) && col_ok(left - 1) {
        left -= 1;
    }
    while right < (a[2] + limit).min(w - 1) && col_ok(right + 1) {
        right += 1;
    }
    // A strip covering the whole search area is not a strip: let the caller decide.
    let tall = bot - top + 1;
    if tall < 8 || tall >= (a[3] - a[1]) + 2 * limit - 2 || right - left < 8 {
        return None;
    }
    let mut ch: [Vec<u8>; 3] = Default::default();
    for y in top..=bot {
        for x in left..=right {
            if is_paper(x, y) {
                let p = img.get_pixel(x as u32, y as u32);
                for k in 0..3 {
                    ch[k].push(p[k]);
                }
            }
        }
    }
    let colour = std::array::from_fn(|k| {
        ch[k].sort_unstable();
        ch[k].get(ch[k].len() / 2).copied().unwrap_or(255)
    });
    Some(([left, top, right + 1, bot + 1], colour))
}

/// Pixels shaved off the balloon mask so the outline is never painted.
const OUTLINE_GUARD: i32 = 3;

/// The segmented balloon covering most of the text box (at least 60%).
fn balloon_for<'a>(
    t: &[i32; 4],
    balloons: &'a [crate::bubbles::Balloon],
) -> Option<&'a crate::bubbles::Balloon> {
    let area = ((t[2] - t[0]).max(1) * (t[3] - t[1]).max(1)) as f32;
    balloons
        .iter()
        .map(|b| {
            let mut n = 0usize;
            for y in (t[1]..t[3]).step_by(2) {
                for x in (t[0]..t[2]).step_by(2) {
                    if b.contains(x, y) {
                        n += 1;
                    }
                }
            }
            (n as f32 * 4.0 / area, b)
        })
        .filter(|(f, _)| *f >= 0.6)
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, b)| b)
}

/// Fill the text box (plus margin) inside the balloon interior with the balloon colour.
/// None when the interior is not plain (screentone, gradients): the general method runs.
fn clean_in_balloon(
    img: &RgbImage,
    cb: &CleanBox,
    bal: &crate::bubbles::Balloon,
    layer: &mut RgbaImage,
    mask_out: &mut GrayImage,
) -> Option<(bool, FillMethod)> {
    const MARGIN: i32 = 6;
    let (w, h) = (img.width() as i32, img.height() as i32);
    let t = clamp_box(cb.text_box, w, h);
    let fill = clamp_box(grow(t, MARGIN), w, h);
    // Interior = mask eroded by OUTLINE_GUARD.
    let interior = |x: i32, y: i32| {
        (-OUTLINE_GUARD..=OUTLINE_GUARD)
            .all(|dy| (-OUTLINE_GUARD..=OUTLINE_GUARD).all(|dx| bal.contains(x + dx, y + dy)))
    };
    // Background colour: the light tone of the interior inside the box.
    let mut lums = vec![];
    for y in t[1]..t[3] {
        for x in t[0]..t[2] {
            if interior(x, y) {
                lums.push(lum(img.get_pixel(x as u32, y as u32)));
            }
        }
    }
    if lums.len() < 20 {
        return None;
    }
    lums.sort_by(f32::total_cmp);
    let bg = lums[lums.len() * 9 / 10];
    let light: Vec<f32> = lums.iter().copied().filter(|&l| l >= bg - 40.0).collect();
    let mean = light.iter().sum::<f32>() / light.len() as f32;
    let std = (light.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / light.len() as f32).sqrt();
    if bg < 120.0 || std > 12.0 {
        return None;
    }
    let colour = cb.fill.unwrap_or_else(|| {
        median_colour(img, &t, |x, y| {
            interior(x, y) && lum(img.get_pixel(x as u32, y as u32)) >= bg - 40.0
        })
    });
    for y in fill[1]..fill[3] {
        for x in fill[0]..fill[2] {
            if interior(x, y) {
                layer.put_pixel(
                    x as u32,
                    y as u32,
                    Rgba([colour[0], colour[1], colour[2], 255]),
                );
                mask_out.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }
    }
    Some((false, FillMethod::Solid))
}

/// Otsu threshold over luminance values.
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

fn clean_auto(
    img: &RgbImage,
    cb: &CleanBox,
    layer: &mut RgbaImage,
    mask_out: &mut GrayImage,
) -> (bool, FillMethod) {
    if cb.shape.is_none() {
        if let Some(done) = clean_closed_bubble(img, cb, layer, mask_out) {
            return done;
        }
    }
    let (w, h) = (img.width() as i32, img.height() as i32);
    let t = clamp_box(cb.text_box, w, h);
    if t[2] - t[0] < 3 || t[3] - t[1] < 3 {
        return (false, FillMethod::Nothing);
    }
    let erase_limit = clamp_box(grow(t, STROKE_SLACK), w, h);
    let region = clamp_box(grow(t, CONTEXT), w, h);
    let (rx, ry) = (region[0], region[1]);
    let (rw, rh) = ((region[2] - rx) as usize, (region[3] - ry) as usize);
    let idx = |x: i32, y: i32| (y - ry) as usize * rw + (x - rx) as usize;

    // 1. Ink polarity and threshold.
    let box_lums: Vec<f32> = (t[1]..t[3])
        .flat_map(|y| (t[0]..t[2]).map(move |x| (x, y)))
        .map(|(x, y)| lum(img.get_pixel(x as u32, y as u32)))
        .collect();
    let lo = box_lums.iter().cloned().fold(f32::MAX, f32::min);
    let hi = box_lums.iter().cloned().fold(f32::MIN, f32::max);
    if hi - lo < 40.0 {
        return (false, FillMethod::Nothing); // no visible text
    }
    let thr = otsu(&box_lums);
    let region_lums: Vec<f32> = (ry..region[3])
        .flat_map(|y| (rx..region[2]).map(move |x| (x, y)))
        .map(|(x, y)| lum(img.get_pixel(x as u32, y as u32)))
        .collect();
    // Ink is usually the minority tone in the box. When dark is the majority (dark strips,
    // dense screentone), compare with the ring around the box: text adds its tone to the box.
    let box_dark = box_lums.iter().filter(|&&v| v < thr).count() as f32 / box_lums.len() as f32;
    let ink_is_dark = if box_dark <= 0.5 {
        true
    } else {
        let ring: Vec<f32> = (ry..region[3])
            .flat_map(|y| (rx..region[2]).map(move |x| (x, y)))
            .filter(|&(x, y)| !inside(&t, x, y))
            .map(|(x, y)| region_lums[idx(x, y)])
            .collect();
        let ring_dark = ring.iter().filter(|&&v| v < thr).count() as f32 / ring.len().max(1) as f32;
        box_dark > ring_dark
    };
    let mut ink: Vec<bool> = region_lums
        .iter()
        .map(|&l| if ink_is_dark { l < thr } else { l > thr })
        .collect();

    // 2. Strokes; if one stroke swallows the area (screentone counted as ink), split the ink
    //    tones again so only the extreme tone (the lettering) counts as ink.
    let mut thick = stroke_thickness(&ink, &region);
    let mut strokes = classify_strokes(&ink, &thick, &region, &erase_limit, &t);
    for _ in 0..2 {
        let protected_n = strokes.protected.iter().filter(|&&p| p).count();
        if protected_n * 10 < ink.len() * 4 {
            break;
        }
        let ink_lums: Vec<f32> = (t[1]..t[3])
            .flat_map(|y| (t[0]..t[2]).map(move |x| (x, y)))
            .filter(|&(x, y)| ink[idx(x, y)])
            .map(|(x, y)| region_lums[idx(x, y)])
            .collect();
        if ink_lums.len() < 8 {
            break;
        }
        let t2 = otsu(&ink_lums);
        ink = region_lums
            .iter()
            .zip(&ink)
            .map(|(&l, &k)| k && if ink_is_dark { l < t2 } else { l > t2 })
            .collect();
        thick = stroke_thickness(&ink, &region);
        strokes = classify_strokes(&ink, &thick, &region, &erase_limit, &t);
    }
    let Strokes {
        erasable,
        protected,
        touch_up,
    } = strokes;

    // 3. Erase mask: strokes plus halo, never over protected strokes, within the erase limit.
    let mut mask = vec![false; rw * rh];
    for y in ry..region[3] {
        for x in rx..region[2] {
            if !erasable[idx(x, y)] {
                continue;
            }
            for dy in -HALO..=HALO {
                for dx in -HALO..=HALO {
                    let (nx, ny) = (x + dx, y + dy);
                    if inside(&region, nx, ny) && !protected[idx(nx, ny)] {
                        mask[idx(nx, ny)] = true;
                    }
                }
            }
        }
    }
    if !mask.iter().any(|&m| m) {
        return (touch_up, FillMethod::Nothing);
    }

    // 3b. Grow the mask into pixels that still differ from the nearby background: text glow
    //     or outline strokes, and JPEG fringes around letters.
    for _ in 0..3 {
        if !grow_deviating(img, &region, &erase_limit, &ink, &protected, &mut mask) {
            break;
        }
    }

    // 4. Pick a fill method from the known pixels inside the box (not the art around it).
    let known = |x: i32, y: i32| !mask[idx(x, y)] && !ink[idx(x, y)];
    let mut known_lums = vec![];
    let (mut lap_sum, mut lap_n) = (0.0f32, 0usize);
    const N4: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let measure = clamp_box(erase_limit, region[2] - 1, region[3] - 1);
    for y in measure[1].max(ry + 1)..measure[3] {
        for x in measure[0].max(rx + 1)..measure[2] {
            if !known(x, y) {
                continue;
            }
            let l = lum(img.get_pixel(x as u32, y as u32));
            known_lums.push(l);
            if N4.iter().all(|&(dx, dy)| known(x + dx, y + dy)) {
                let n: f32 = N4
                    .iter()
                    .map(|&(dx, dy)| lum(img.get_pixel((x + dx) as u32, (y + dy) as u32)))
                    .sum();
                lap_sum += (n - 4.0 * l).abs();
                lap_n += 1;
            }
        }
    }
    let count = known_lums.len().max(1) as f32;
    let mean = known_lums.iter().sum::<f32>() / count;
    let std = (known_lums.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / count).sqrt();
    let lap = lap_sum / lap_n.max(1) as f32;
    let method = if cb.fill.is_some() || std < PLAIN_STD {
        FillMethod::Solid
    } else if lap > TEXTURE_LAPLACIAN {
        FillMethod::Texture
    } else {
        FillMethod::Smooth
    };

    // Smooth fills sample the ring around the hole; widen the hole so faint glow is not sampled.
    let mut fill_mask = mask.clone();
    if method == FillMethod::Smooth {
        for y in ry..region[3] {
            for x in rx..region[2] {
                if !mask[idx(x, y)] {
                    continue;
                }
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        let (nx, ny) = (x + dx, y + dy);
                        if inside(&region, nx, ny) && !protected[idx(nx, ny)] {
                            fill_mask[idx(nx, ny)] = true;
                        }
                    }
                }
            }
        }
    }
    let mut patch = Patch {
        img,
        region,
        rw,
        mask: &fill_mask,
        filled: vec![None; rw * rh],
    };
    match method {
        FillMethod::Solid => {
            let colour = cb
                .fill
                .unwrap_or_else(|| median_colour(img, &measure, known));
            // Plain background: also repaint near-background pixels in the box, which removes
            // JPEG ringing that the mask misses.
            for y in erase_limit[1]..erase_limit[3] {
                for x in erase_limit[0]..erase_limit[2] {
                    let i = idx(x, y);
                    if protected[i] {
                        continue;
                    }
                    let p = img.get_pixel(x as u32, y as u32);
                    let close = (0..3).all(|k| (p[k] as i32 - colour[k] as i32).abs() <= 48);
                    if close && !mask[i] {
                        patch.filled[i] = Some(colour);
                    }
                }
            }
            for (i, m) in mask.iter().enumerate() {
                if *m {
                    patch.filled[i] = Some(colour);
                }
            }
        }
        FillMethod::Smooth => patch.smooth(),
        FillMethod::Texture => patch.texture(),
        _ => {}
    }
    redraw_lines(
        img,
        &region,
        &fill_mask,
        &protected,
        &thick,
        &mut patch.filled,
    );
    for y in ry..region[3] {
        for x in rx..region[2] {
            if let Some(c) = patch.filled[idx(x, y)] {
                layer.put_pixel(x as u32, y as u32, Rgba([c[0], c[1], c[2], 255]));
                mask_out.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }
    }
    (touch_up, method)
}

/// Band width grown around the mask, in px.
const GROW: i32 = 5;
/// Colour distance (max channel) that marks a pixel as not background.
const DEVIATION: i32 = 18;

/// Add pixels near the mask whose colour differs from the local background, where the
/// background is the mean of non-ink pixels further than `GROW` px from the mask.
fn grow_deviating(
    img: &RgbImage,
    region: &[i32; 4],
    erase_limit: &[i32; 4],
    ink: &[bool],
    protected: &[bool],
    mask: &mut [bool],
) -> bool {
    let (rx, ry) = (region[0], region[1]);
    let rw = (region[2] - rx) as usize;
    let idx = |x: i32, y: i32| (y - ry) as usize * rw + (x - rx) as usize;
    // Chebyshev distance to the mask, capped.
    let cap = GROW + 1;
    let mut dist = vec![cap; mask.len()];
    let mut queue = VecDeque::new();
    for y in ry..region[3] {
        for x in rx..region[2] {
            if mask[idx(x, y)] {
                dist[idx(x, y)] = 0;
                queue.push_back((x, y));
            }
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        let d = dist[idx(x, y)];
        if d >= GROW {
            continue;
        }
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if inside(region, nx, ny) && dist[idx(nx, ny)] > d + 1 {
                    dist[idx(nx, ny)] = d + 1;
                    queue.push_back((nx, ny));
                }
            }
        }
    }
    let _ = erase_limit;
    let limit = *region;
    const R: i32 = 10;
    let mut add = vec![];
    for y in ry..region[3] {
        for x in rx..region[2] {
            let i = idx(x, y);
            let d = dist[i];
            if d == 0 || d > GROW || protected[i] || !inside(&limit, x, y) {
                continue;
            }
            let mut samples: Vec<[u8; 3]> = vec![];
            for wy in (y - R).max(ry)..(y + R + 1).min(region[3]) {
                for wx in (x - R).max(rx)..(x + R + 1).min(region[2]) {
                    let j = idx(wx, wy);
                    if dist[j] >= cap && !ink[j] {
                        samples.push(img.get_pixel(wx as u32, wy as u32).0);
                    }
                }
            }
            if samples.len() < 6 {
                continue;
            }
            let n = samples.len() as f32;
            let mean: [f32; 3] =
                std::array::from_fn(|k| samples.iter().map(|s| s[k] as f32).sum::<f32>() / n);
            // Textured backgrounds (screentone) vary a lot by themselves: only pixels standing
            // out more than the background's own spread are added.
            let spread = samples
                .iter()
                .map(|s| {
                    (0..3)
                        .map(|k| (s[k] as f32 - mean[k]).abs())
                        .fold(0f32, f32::max)
                })
                .sum::<f32>()
                / n;
            let p = img.get_pixel(x as u32, y as u32);
            let dev = (0..3)
                .map(|k| (p[k] as f32 - mean[k]).abs())
                .fold(0f32, f32::max);
            if dev > DEVIATION as f32 + 2.0 * spread {
                add.push(i);
            }
        }
    }
    let grew = !add.is_empty();
    for i in add {
        mask[i] = true;
    }
    grew
}

/// Text inside a detected, closed bubble (the 0.4 method): light pixels connected to the
/// text box form the interior L; everything reachable from the bubble box edge without
/// crossing L is outline or art and stays. L and the enclosed letters are repainted.
/// Returns None when there is no bubble or it is open (L leaks to the bubble box edge).
fn clean_closed_bubble(
    img: &RgbImage,
    cb: &CleanBox,
    layer: &mut RgbaImage,
    mask_out: &mut GrayImage,
) -> Option<(bool, FillMethod)> {
    const TOLERANCE: f32 = 60.0;
    const MARGIN: i32 = 6;
    let bb = cb.bubble?;
    let (w, h) = (img.width() as i32, img.height() as i32);
    let t = clamp_box(cb.text_box, w, h);
    if t[2] - t[0] < 3 || t[3] - t[1] < 3 {
        return None;
    }
    let union = [
        bb[0].min(t[0]),
        bb[1].min(t[1]),
        bb[2].max(t[2]),
        bb[3].max(t[3]),
    ];
    let region = clamp_box(grow(union, 2), w, h);
    let (rx, ry) = (region[0], region[1]);
    let rw = (region[2] - rx) as usize;
    let rh = (region[3] - ry) as usize;
    let idx = |x: i32, y: i32| (y - ry) as usize * rw + (x - rx) as usize;

    let mut lums: Vec<f32> = (t[1]..t[3])
        .flat_map(|y| (t[0]..t[2]).map(move |x| (x, y)))
        .map(|(x, y)| lum(img.get_pixel(x as u32, y as u32)))
        .collect();
    lums.sort_by(f32::total_cmp);
    let bg_lum = lums[(lums.len() * 9 / 10).min(lums.len() - 1)];
    if bg_lum < 120.0 {
        return None; // dark bubble: leave it to the general method
    }
    let threshold = bg_lum - TOLERANCE;
    let light: Vec<bool> = (ry..region[3])
        .flat_map(|y| (rx..region[2]).map(move |x| (x, y)))
        .map(|(x, y)| lum(img.get_pixel(x as u32, y as u32)) >= threshold)
        .collect();

    // L: light pixels connected (4-way) to light pixels in the text box.
    let mut in_l = vec![false; rw * rh];
    let mut queue = VecDeque::new();
    for y in t[1]..t[3] {
        for x in t[0]..t[2] {
            let i = idx(x, y);
            if light[i] && !in_l[i] {
                in_l[i] = true;
                queue.push_back((x, y));
            }
        }
    }
    let mut leaks = false;
    while let Some((x, y)) = queue.pop_front() {
        if x == rx || y == ry || x == region[2] - 1 || y == region[3] - 1 {
            leaks = true;
        }
        for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
            if inside(&region, nx, ny) {
                let i = idx(nx, ny);
                if light[i] && !in_l[i] {
                    in_l[i] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
    }
    let _ = leaks; // open bubbles are handled by the protected strokes below
                   // Outside: reachable from the region edge without crossing L.
    let mut outside = vec![false; rw * rh];
    for y in ry..region[3] {
        for x in rx..region[2] {
            let edge = x == rx || y == ry || x == region[2] - 1 || y == region[3] - 1;
            if edge && !in_l[idx(x, y)] {
                outside[idx(x, y)] = true;
                queue.push_back((x, y));
            }
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
            if inside(&region, nx, ny) {
                let i = idx(nx, ny);
                if !in_l[i] && !outside[i] {
                    outside[i] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
    }
    // Dark strokes reaching the bubble box edge are outline (also in open bubbles, where the
    // light interior leaks round the outline): never paint over them.
    let mut keep = vec![false; rw * rh];
    for y in ry..region[3] {
        for x in rx..region[2] {
            let edge = x == rx || y == ry || x == region[2] - 1 || y == region[3] - 1;
            if edge && !light[idx(x, y)] && !keep[idx(x, y)] {
                keep[idx(x, y)] = true;
                queue.push_back((x, y));
            }
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if inside(&region, nx, ny) {
                    let i = idx(nx, ny);
                    if !light[i] && !keep[i] {
                        keep[i] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
    }
    // Outlines often pass through a text box, so kept strokes do not flag bubble text.
    let touch_up = false;
    let colour = cb
        .fill
        .unwrap_or_else(|| median_colour(img, &t, |x, y| in_l[idx(x, y)]));
    let fill = clamp_box(grow(t, MARGIN), w, h);
    for y in fill[1].max(ry)..fill[3].min(region[3]) {
        for x in fill[0].max(rx)..fill[2].min(region[2]) {
            let i = idx(x, y);
            if (in_l[i] || !outside[i]) && !keep[i] {
                layer.put_pixel(
                    x as u32,
                    y as u32,
                    Rgba([colour[0], colour[1], colour[2], 255]),
                );
                mask_out.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }
    }
    Some((touch_up, FillMethod::Solid))
}

struct Strokes {
    erasable: Vec<bool>,
    protected: Vec<bool>,
    touch_up: bool,
}

/// Split ink into 8-connected strokes seeded inside `erase_limit`. A stroke that leaves the
/// erase limit (or reaches the region edge) is protected; others are erasable.
fn classify_strokes(
    ink: &[bool],
    thickness: &[u8],
    region: &[i32; 4],
    erase_limit: &[i32; 4],
    t: &[i32; 4],
) -> Strokes {
    let (rx, ry) = (region[0], region[1]);
    let rw = (region[2] - rx) as usize;
    let idx = |x: i32, y: i32| (y - ry) as usize * rw + (x - rx) as usize;
    let core = grow(*t, -STROKE_SLACK);
    let mut seen = vec![false; ink.len()];
    let mut erasable = vec![false; ink.len()];
    let mut protected = vec![false; ink.len()];
    let mut touch_up = false;
    let mut queue = VecDeque::new();
    let mut inner: Vec<Vec<(i32, i32)>> = vec![];
    for sy in erase_limit[1]..erase_limit[3] {
        for sx in erase_limit[0]..erase_limit[2] {
            let si = idx(sx, sy);
            if !ink[si] || seen[si] {
                continue;
            }
            seen[si] = true;
            queue.push_back((sx, sy));
            let mut pixels = vec![];
            let mut crosses = false;
            while let Some((x, y)) = queue.pop_front() {
                pixels.push((x, y));
                if x == rx || y == ry || x == region[2] - 1 || y == region[3] - 1 {
                    crosses = true;
                }
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (x + dx, y + dy);
                        if (dx, dy) == (0, 0) || !inside(region, nx, ny) {
                            continue;
                        }
                        let ni = idx(nx, ny);
                        if ink[ni] && !seen[ni] {
                            seen[ni] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }
            if crosses {
                let deep =
                    pixels.iter().filter(|&&(x, y)| inside(&core, x, y)).count() >= TOUCH_PIXELS;
                for &(x, y) in &pixels {
                    protected[idx(x, y)] = true;
                }
                if deep {
                    // Letters merged with thin lines: thick parts that stay inside are letters.
                    let letters = thick_letters(&pixels, thickness, t, region, &core);
                    if letters.is_empty() {
                        touch_up = true;
                    }
                    for (x, y) in letters {
                        protected[idx(x, y)] = false;
                        erasable[idx(x, y)] = true;
                    }
                }
            } else {
                inner.push(pixels);
            }
        }
    }
    // Strokes wholly inside: letters, unless they are long thin curves compared with the
    // characters in this box (an outline shared by two touching bubbles, a panel line).
    let shape = |px: &Vec<(i32, i32)>| {
        let (x0, x1) = (
            px.iter().map(|p| p.0).min().unwrap(),
            px.iter().map(|p| p.0).max().unwrap(),
        );
        let (y0, y1) = (
            px.iter().map(|p| p.1).min().unwrap(),
            px.iter().map(|p| p.1).max().unwrap(),
        );
        let (bw, bh) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
        (bw.max(bh), bw.min(bh))
    };
    let mut char_sizes: Vec<f32> = inner
        .iter()
        .filter(|px| px.len() >= 12)
        .map(shape)
        .filter(|(long, short)| long / short <= 2.5)
        .map(|(long, _)| long)
        .collect();
    char_sizes.sort_by(f32::total_cmp);
    let char_size = char_sizes
        .get(char_sizes.len() / 2)
        .copied()
        .unwrap_or(30.0)
        .max(12.0);
    for px in inner {
        let (long, short) = shape(&px);
        let outline_like = long / short > 4.0 && long > 2.6 * char_size;
        for (x, y) in px {
            if outline_like {
                protected[idx(x, y)] = true;
            } else {
                erasable[idx(x, y)] = true;
            }
        }
    }
    Strokes {
        erasable,
        protected,
        touch_up,
    }
}

/// Chebyshev distance from each ink pixel to the nearest non-ink pixel, capped at 4:
/// 1 for hairlines, 2+ for strokes at least 3 px wide.
fn stroke_thickness(ink: &[bool], region: &[i32; 4]) -> Vec<u8> {
    let (rx, ry) = (region[0], region[1]);
    let rw = (region[2] - rx) as usize;
    let idx = |x: i32, y: i32| (y - ry) as usize * rw + (x - rx) as usize;
    let mut d = vec![0u8; ink.len()];
    let mut queue = VecDeque::new();
    for y in ry..region[3] {
        for x in rx..region[2] {
            let i = idx(x, y);
            if !ink[i] {
                continue;
            }
            let border = (-1..=1).any(|dy: i32| {
                (-1..=1).any(|dx: i32| {
                    let (nx, ny) = (x + dx, y + dy);
                    inside(region, nx, ny) && !ink[idx(nx, ny)]
                })
            });
            if border {
                d[i] = 1;
                queue.push_back((x, y));
            }
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        let v = d[idx(x, y)];
        if v >= 4 {
            continue;
        }
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if inside(region, nx, ny) {
                    let ni = idx(nx, ny);
                    if ink[ni] && d[ni] == 0 {
                        d[ni] = v + 1;
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
    }
    d
}

/// Within one stroke that crosses the box edge, find letter pixels: parts at least 3 px
/// thick that stay inside the box, grown back by their thickness. Thin lines and thick
/// strokes leaving the box core (outlines) are left out.
fn thick_letters(
    pixels: &[(i32, i32)],
    thickness: &[u8],
    text_box: &[i32; 4],
    region: &[i32; 4],
    core: &[i32; 4],
) -> Vec<(i32, i32)> {
    use std::collections::HashSet;
    let (rx, ry) = (region[0], region[1]);
    let rw = (region[2] - rx) as usize;
    let idx = |x: i32, y: i32| (y - ry) as usize * rw + (x - rx) as usize;
    let in_stroke: HashSet<(i32, i32)> = pixels.iter().copied().collect();
    let thick: Vec<(i32, i32)> = pixels
        .iter()
        .copied()
        .filter(|&(x, y)| thickness[idx(x, y)] >= 2)
        .collect();
    let thick_set: HashSet<(i32, i32)> = thick.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut out = vec![];
    for &start in &thick {
        if !seen.insert(start) {
            continue;
        }
        let mut part = vec![start];
        let mut queue = VecDeque::from([start]);
        let mut escapes = false;
        while let Some((x, y)) = queue.pop_front() {
            if !inside(core, x, y) {
                escapes = true;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let n = (x + dx, y + dy);
                    if thick_set.contains(&n) && seen.insert(n) {
                        part.push(n);
                        queue.push_back(n);
                    }
                }
            }
        }
        // Letters are compact; outline fragments are long thin curves.
        let xs = part.iter().map(|p| p.0);
        let ys = part.iter().map(|p| p.1);
        let (bw, bh) = (
            xs.clone().max().unwrap() - xs.min().unwrap() + 1,
            ys.clone().max().unwrap() - ys.min().unwrap() + 1,
        );
        let (long, short) = (bw.max(bh) as f32, bw.min(bh).max(1) as f32);
        let letter_max =
            48f32.max(0.6 * (text_box[2] - text_box[0]).min(text_box[3] - text_box[1]) as f32);
        if escapes || part.len() < 6 || long / short > 2.5 || long > letter_max {
            continue;
        }
        // Grow the letter core back to its full width, staying on the stroke.
        for &(x, y) in &part {
            let r = thickness[idx(x, y)] as i32;
            for dy in -r..=r {
                for dx in -r..=r {
                    if in_stroke.contains(&(x + dx, y + dy)) {
                        out.push((x + dx, y + dy));
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Continue thin lines (focus lines, hatching) straight through the erased area: each line
/// end touching the mask is fitted with a direction and extended until it leaves the mask.
fn redraw_lines(
    img: &RgbImage,
    region: &[i32; 4],
    mask: &[bool],
    protected: &[bool],
    thickness: &[u8],
    filled: &mut [Option<[u8; 3]>],
) {
    let (rx, ry) = (region[0], region[1]);
    let rw = (region[2] - rx) as usize;
    let idx = |x: i32, y: i32| (y - ry) as usize * rw + (x - rx) as usize;
    // Mask dilated by 6 px: line ends must start near the hole.
    let mut near = vec![false; mask.len()];
    for y in ry..region[3] {
        for x in rx..region[2] {
            if !mask[idx(x, y)] {
                continue;
            }
            for dy in -6..=6 {
                for dx in -6..=6 {
                    if inside(region, x + dx, y + dy) {
                        near[idx(x + dx, y + dy)] = true;
                    }
                }
            }
        }
    }
    let line_px = |i: usize| protected[i] && thickness[i] <= 1 && !mask[i] && near[i];
    let mut seen = vec![false; mask.len()];
    for sy in ry..region[3] {
        for sx in rx..region[2] {
            let si = idx(sx, sy);
            if seen[si] || !line_px(si) {
                continue;
            }
            seen[si] = true;
            let mut group = vec![(sx, sy)];
            let mut queue = VecDeque::from([(sx, sy)]);
            while let Some((x, y)) = queue.pop_front() {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (x + dx, y + dy);
                        if !inside(region, nx, ny) {
                            continue;
                        }
                        let ni = idx(nx, ny);
                        if !seen[ni] && line_px(ni) {
                            seen[ni] = true;
                            group.push((nx, ny));
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }
            if group.len() < 5 {
                continue;
            }
            // Principal direction of the group.
            let n = group.len() as f32;
            let mx = group.iter().map(|p| p.0 as f32).sum::<f32>() / n;
            let my = group.iter().map(|p| p.1 as f32).sum::<f32>() / n;
            let (mut sxx, mut syy, mut sxy) = (0f32, 0f32, 0f32);
            for &(x, y) in &group {
                let (dx, dy) = (x as f32 - mx, y as f32 - my);
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
            }
            let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
            let (ux, uy) = (angle.cos(), angle.sin());
            // Elongated groups only: lines, not blobs.
            let along: Vec<f32> = group
                .iter()
                .map(|&(x, y)| (x as f32 - mx) * ux + (y as f32 - my) * uy)
                .collect();
            let length = along.iter().cloned().fold(f32::MIN, f32::max)
                - along.iter().cloned().fold(f32::MAX, f32::min);
            if length < 4.0 || length * length < 4.0 * n {
                continue;
            }
            let mut ch: [Vec<u8>; 3] = Default::default();
            for &(x, y) in &group {
                let p = img.get_pixel(x as u32, y as u32);
                for k in 0..3 {
                    ch[k].push(p[k]);
                }
            }
            let colour: [u8; 3] = std::array::from_fn(|k| {
                ch[k].sort_unstable();
                ch[k][ch[k].len() / 2]
            });
            let half_width = ((n / length).max(1.0) / 2.0).max(0.6);
            // March both ways from the group's centre through masked pixels.
            for sign in [1.0f32, -1.0] {
                let mut t = 0.0f32;
                let mut in_hole = false;
                while t < 200.0 {
                    let (px, py) = (mx + sign * ux * t, my + sign * uy * t);
                    let (ix, iy) = (px.round() as i32, py.round() as i32);
                    if !inside(region, ix, iy) {
                        break;
                    }
                    let masked = mask[idx(ix, iy)];
                    if in_hole && !masked {
                        break; // left the hole on the far side
                    }
                    in_hole |= masked;
                    if masked {
                        let r = half_width.ceil() as i32;
                        for oy in -r..=r {
                            for ox in -r..=r {
                                let (qx, qy) = (ix + ox, iy + oy);
                                if !inside(region, qx, qy) || !mask[idx(qx, qy)] {
                                    continue;
                                }
                                let d = ((qx as f32 - mx) * -uy + (qy as f32 - my) * ux).abs();
                                if d <= half_width {
                                    filled[idx(qx, qy)] = Some(colour);
                                }
                            }
                        }
                    }
                    t += 0.5;
                }
            }
        }
    }
}

fn median_colour(img: &RgbImage, region: &[i32; 4], ok: impl Fn(i32, i32) -> bool) -> [u8; 3] {
    let mut ch: [Vec<u8>; 3] = Default::default();
    for y in region[1]..region[3] {
        for x in region[0]..region[2] {
            if ok(x, y) {
                let p = img.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    ch[c].push(p[c]);
                }
            }
        }
    }
    if ch[0].is_empty() {
        return [255, 255, 255];
    }
    std::array::from_fn(|c| {
        ch[c].sort_unstable();
        ch[c][ch[c].len() / 2]
    })
}

/// Fills masked pixels of a region from the unmasked ones.
struct Patch<'a> {
    img: &'a RgbImage,
    region: [i32; 4],
    rw: usize,
    mask: &'a [bool],
    filled: Vec<Option<[u8; 3]>>,
}

impl Patch<'_> {
    fn idx(&self, x: i32, y: i32) -> usize {
        (y - self.region[1]) as usize * self.rw + (x - self.region[0]) as usize
    }

    /// Colour of a pixel if known: original when unmasked, or already filled.
    fn value(&self, x: i32, y: i32) -> Option<[u8; 3]> {
        if !inside(&self.region, x, y) {
            return None;
        }
        let i = self.idx(x, y);
        if self.mask[i] {
            self.filled[i]
        } else {
            Some(self.img.get_pixel(x as u32, y as u32).0)
        }
    }

    /// Masked pixels ordered from the mask edge inward (onion peel).
    fn peel_order(&self) -> Vec<(i32, i32)> {
        let r = self.region;
        let mut seen = vec![false; self.mask.len()];
        let mut queue = VecDeque::new();
        for y in r[1]..r[3] {
            for x in r[0]..r[2] {
                let i = self.idx(x, y);
                let edge = [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| {
                    inside(&r, x + dx, y + dy) && !self.mask[self.idx(x + dx, y + dy)]
                });
                if self.mask[i] && edge {
                    seen[i] = true;
                    queue.push_back((x, y));
                }
            }
        }
        let mut order = vec![];
        while let Some((x, y)) = queue.pop_front() {
            order.push((x, y));
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x + dx, y + dy);
                if inside(&r, nx, ny) {
                    let ni = self.idx(nx, ny);
                    if self.mask[ni] && !seen[ni] {
                        seen[ni] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
        order
    }

    /// Telea-style: each pixel becomes a distance-weighted average of known neighbours.
    fn smooth(&mut self) {
        const R: i32 = 4;
        for (x, y) in self.peel_order() {
            let (mut acc, mut wsum) = ([0f32; 3], 0f32);
            for dy in -R..=R {
                for dx in -R..=R {
                    let d2 = (dx * dx + dy * dy) as f32;
                    if d2 == 0.0 || d2 > (R * R) as f32 {
                        continue;
                    }
                    if let Some(c) = self.value(x + dx, y + dy) {
                        let original = !self.mask[self.idx(x + dx, y + dy)];
                        let wgt = (if original { 1.0 } else { 0.5 }) / d2;
                        for k in 0..3 {
                            acc[k] += c[k] as f32 * wgt;
                        }
                        wsum += wgt;
                    }
                }
            }
            if wsum > 0.0 {
                let i = self.idx(x, y);
                self.filled[i] = Some(std::array::from_fn(|k| (acc[k] / wsum).round() as u8));
            }
        }
    }

    /// Exemplar synthesis: copy the pixel whose neighbourhood best matches, searching random
    /// candidates nearby and the offsets that worked for already-filled neighbours.
    fn texture(&mut self) {
        const P: i32 = 3; // 7x7 patch
        const SEARCH: i32 = 40;
        const CANDIDATES: usize = 80;
        let r = self.region;
        let mut source: Vec<Option<(i32, i32)>> = vec![None; self.mask.len()];
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = move |n: i32| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % n.max(1) as u64) as i32
        };
        for (x, y) in self.peel_order() {
            let mut cands = vec![];
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                if inside(&r, x + dx, y + dy) {
                    if let Some((sx, sy)) = source[self.idx(x + dx, y + dy)] {
                        cands.push((sx - dx, sy - dy));
                    }
                }
            }
            for _ in 0..CANDIDATES {
                cands.push((
                    x + rand(2 * SEARCH + 1) - SEARCH,
                    y + rand(2 * SEARCH + 1) - SEARCH,
                ));
            }
            let mut best: Option<(f32, (i32, i32))> = None;
            for (cx, cy) in cands {
                let usable = cx - P >= r[0]
                    && cy - P >= r[1]
                    && cx + P < r[2]
                    && cy + P < r[3]
                    && !self.mask[self.idx(cx, cy)];
                if !usable {
                    continue;
                }
                let (mut ssd, mut n) = (0f32, 0usize);
                for dy in -P..=P {
                    for dx in -P..=P {
                        if self.mask[self.idx(cx + dx, cy + dy)] {
                            continue;
                        }
                        let (Some(a), Some(b)) =
                            (self.value(x + dx, y + dy), self.value(cx + dx, cy + dy))
                        else {
                            continue;
                        };
                        ssd += (0..3)
                            .map(|k| (a[k] as f32 - b[k] as f32).powi(2))
                            .sum::<f32>();
                        n += 1;
                    }
                }
                if n < 4 {
                    continue;
                }
                let score = ssd / n as f32;
                if best.is_none_or(|(s, _)| score < s) {
                    best = Some((score, (cx, cy)));
                }
            }
            let i = self.idx(x, y);
            if let Some((_, (sx, sy))) = best {
                self.filled[i] = Some(self.img.get_pixel(sx as u32, sy as u32).0);
                source[i] = Some((sx, sy));
            }
        }
        // Anything the search could not reach falls back to smoothing.
        if self
            .filled
            .iter()
            .zip(self.mask)
            .any(|(f, &m)| m && f.is_none())
        {
            self.smooth();
        }
    }
}

/// Signed distance to the shape edge (negative inside), for a shape centred at the origin
/// with half-size (`hx`, `hy`).
fn shape_distance(shape: Shape, x: f32, y: f32, hx: f32, hy: f32) -> f32 {
    match shape {
        Shape::Ellipse => {
            let k = ((x / hx).powi(2) + (y / hy).powi(2)).sqrt();
            (k - 1.0) * hx.min(hy)
        }
        Shape::Rect | Shape::Rounded => {
            let r = if shape == Shape::Rounded {
                0.3 * hx.min(hy)
            } else {
                0.0
            };
            let qx = x.abs() - (hx - r);
            let qy = y.abs() - (hy - r);
            let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
            outside + qx.max(qy).min(0.0) - r
        }
    }
}

fn blend(dst: Rgba<u8>, src: [u8; 4], coverage: f32) -> Rgba<u8> {
    let a = src[3] as f32 / 255.0 * coverage.clamp(0.0, 1.0);
    let da = dst[3] as f32 / 255.0;
    let out_a = a + da * (1.0 - a);
    if out_a <= 0.0 {
        return Rgba([0, 0, 0, 0]);
    }
    let ch =
        |i: usize| ((src[i] as f32 * a + dst[i] as f32 * da * (1.0 - a)) / out_a).round() as u8;
    Rgba([ch(0), ch(1), ch(2), (out_a * 255.0).round() as u8])
}

/// Paint a new bubble around `text_box`, grown by 12% so the lettering does not touch the edge.
fn paint_shape(
    text_box: [i32; 4],
    style: &ShapeStyle,
    layer: &mut RgbaImage,
    mask: &mut GrayImage,
) {
    let (w, h) = (layer.width() as i32, layer.height() as i32);
    let [x1, y1, x2, y2] = text_box;
    let pad = ((x2 - x1).min(y2 - y1) as f32 * 0.12).max(4.0);
    let (cx, cy) = ((x1 + x2) as f32 / 2.0, (y1 + y2) as f32 / 2.0);
    let (hx, hy) = ((x2 - x1) as f32 / 2.0 + pad, (y2 - y1) as f32 / 2.0 + pad);
    let (ocolor, ow) = style.outline.unwrap_or((style.fill, 0.0));
    let ext = (pad + 2.0).ceil() as i32;
    for y in (y1 - ext).max(0)..(y2 + ext).min(h) {
        for x in (x1 - ext).max(0)..(x2 + ext).min(w) {
            let d = shape_distance(
                style.shape,
                x as f32 + 0.5 - cx,
                y as f32 + 0.5 - cy,
                hx,
                hy,
            );
            let outer = 0.5 - d;
            if outer <= 0.0 {
                continue;
            }
            let inner = (0.5 - (d + ow)).clamp(0.0, 1.0);
            let col: [u8; 4] = std::array::from_fn(|i| {
                (ocolor[i] as f32 * (1.0 - inner) + style.fill[i] as f32 * inner).round() as u8
            });
            let px = *layer.get_pixel(x as u32, y as u32);
            layer.put_pixel(x as u32, y as u32, blend(px, col, outer));
            mask.put_pixel(x as u32, y as u32, Luma([255]));
        }
    }
}

/// Composite a clean layer over the page (for export and tests).
pub fn composite(img: &RgbImage, layer: &RgbaImage) -> RgbImage {
    let mut out = img.clone();
    for (x, y, p) in layer.enumerate_pixels() {
        if p[3] > 0 {
            let a = p[3] as f32 / 255.0;
            let d = out.get_pixel(x, y);
            let mix = |i: usize| (p[i] as f32 * a + d[i] as f32 * (1.0 - a)).round() as u8;
            out.put_pixel(x, y, Rgb([mix(0), mix(1), mix(2)]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(img: &mut RgbImage, rows: &[(u32, u32)], len: u32, colour: [u8; 3]) {
        for &(x0, y0) in rows {
            for y in y0..y0 + 4 {
                for x in x0..x0 + len {
                    img.put_pixel(x, y, Rgb(colour));
                }
            }
        }
    }

    /// Grey page, white ellipse bubble with a 3 px black outline, black "text" bars inside.
    fn bubble_page() -> RgbImage {
        let mut img = RgbImage::from_pixel(200, 200, Rgb([150, 150, 150]));
        for y in 0..200 {
            for x in 0..200 {
                let d = ((x as f32 - 100.0) / 70.0).powi(2) + ((y as f32 - 100.0) / 50.0).powi(2);
                if d <= 1.0 {
                    img.put_pixel(x, y, Rgb([255, 255, 255]));
                }
                if (0.88..=1.0).contains(&d) {
                    img.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        bars(
            &mut img,
            &[(80, 85), (80, 100), (80, 115)],
            40,
            [10, 10, 10],
        );
        img
    }

    #[test]
    fn cleans_text_in_bubble_and_keeps_outline_and_art() {
        let img = bubble_page();
        let r = clean_page_full(
            &img,
            &[CleanBox::auto([76, 80, 124, 122], Some([30, 50, 170, 150]))],
        );
        let out = composite(&img, &r.layer);
        for (x, y) in [(90, 86), (100, 101), (110, 116)] {
            assert_eq!(out.get_pixel(x, y).0, [255, 255, 255], "text at {x},{y}");
        }
        assert_eq!(out.get_pixel(31, 100).0, [0, 0, 0]);
        assert_eq!(out.get_pixel(5, 5).0, [150, 150, 150]);
        assert_eq!(r.methods, vec![FillMethod::Solid]);
        assert_eq!(r.touch_up, vec![false]);
    }

    #[test]
    fn outline_crossing_the_box_is_kept() {
        // The text box overlaps a bubble outline that runs on past it (open or tight bubble).
        let mut img = RgbImage::from_pixel(160, 120, Rgb([255, 255, 255]));
        for y in 0..120 {
            for x in 118..121 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        bars(&mut img, &[(60, 40), (60, 60)], 40, [0, 0, 0]);
        let r = clean_page_full(&img, &[CleanBox::auto([55, 35, 125, 70], None)]);
        let out = composite(&img, &r.layer);
        assert_eq!(out.get_pixel(80, 42).0, [255, 255, 255], "text erased");
        for y in [36, 50, 68] {
            assert_eq!(out.get_pixel(119, y).0, [0, 0, 0], "outline kept at y={y}");
        }
    }

    #[test]
    fn letter_touching_outline_is_flagged() {
        let mut img = RgbImage::from_pixel(160, 120, Rgb([255, 255, 255]));
        // A 4 px outline: as thick as the letter, so they cannot be told apart.
        for y in 0..120 {
            for x in 118..122 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        bars(&mut img, &[(70, 50)], 50, [0, 0, 0]); // this bar runs into the outline
        let r = clean_page_full(&img, &[CleanBox::auto([60, 40, 125, 64], None)]);
        assert_eq!(r.touch_up, vec![true]);
        assert_eq!(
            composite(&img, &r.layer).get_pixel(120, 30).0,
            [0, 0, 0],
            "outline kept"
        );
    }

    #[test]
    fn glow_around_text_on_sky_is_removed() {
        let mut img = RgbImage::from_fn(140, 100, |_, y| Rgb([150 + y as u8 / 2, 190, 235]));
        // White 2 px glow around black bars.
        for &(x0, y0) in &[(40u32, 40u32), (40, 54)] {
            for y in y0 - 2..y0 + 6 {
                for x in x0 - 2..x0 + 62 {
                    img.put_pixel(x, y, Rgb([255, 255, 255]));
                }
            }
        }
        bars(&mut img, &[(40, 40), (40, 54)], 60, [0, 0, 0]);
        let out = composite(
            &img,
            &clean_page(&img, &[CleanBox::auto([35, 35, 105, 62], None)]),
        );
        for (x, y) in [(70, 38), (70, 45), (39, 42)] {
            let p = out.get_pixel(x, y);
            assert!(p[0] < 230 && p[2] > 220, "glow left at {x},{y}: {:?}", p.0);
        }
    }

    #[test]
    fn letters_on_focus_lines_are_erased_and_lines_continue() {
        let mut img = RgbImage::from_pixel(200, 120, Rgb([255, 255, 255]));
        // Thin diagonal lines crossing the whole page.
        for k in 0..6u32 {
            for x in 0..200u32 {
                let y = x / 2 + k * 18;
                if y < 120 {
                    img.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        // Letter-shaped 14 px blocks overlapping the lines.
        for y0 in [45u32, 65] {
            for x0 in [70u32, 90, 110] {
                for y in y0..y0 + 14 {
                    for x in x0..x0 + 14 {
                        img.put_pixel(x, y, Rgb([0, 0, 0]));
                    }
                }
            }
        }
        let r = clean_page_full(&img, &[CleanBox::auto([65, 40, 129, 84], None)]);
        let out = composite(&img, &r.layer);
        assert_eq!(r.touch_up, vec![false]);
        let white = (70..124)
            .filter(|&x| out.get_pixel(x, 52).0 == [255, 255, 255])
            .count();
        assert!(white > 30, "only {white} white pixels in the erased row");
        // Line k=1 (y = x/2 + 18) crosses the block at x 90..104, y 65..79; it is redrawn.
        let dark = (94..104u32)
            .filter(|&x| out.get_pixel(x, x / 2 + 18)[0] < 128)
            .count();
        assert!(dark >= 6, "line not redrawn ({dark} of 10 pixels)");
    }

    /// Two bubbles share a thick outline that thins out inside the text box. The outline
    /// pieces must not be mistaken for letters (regression from Kozinsa v01-001 p004).
    #[test]
    fn shared_outline_between_bubbles_is_kept() {
        let mut img = RgbImage::from_pixel(200, 200, Rgb([255, 255, 255]));
        for y in 10..190u32 {
            let width = if (90..96).contains(&y) || (130..134).contains(&y) {
                1
            } else {
                4
            };
            for x in 100..100 + width {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        // Letters close to the shared outline, in the left bubble.
        for y0 in [50u32, 110] {
            for x0 in [70u32, 84] {
                for y in y0..y0 + 12 {
                    for x in x0..x0 + 12 {
                        img.put_pixel(x, y, Rgb([0, 0, 0]));
                    }
                }
            }
        }
        let r = clean_page_full(&img, &[CleanBox::auto([60, 40, 110, 160], None)]);
        let out = composite(&img, &r.layer);
        // A short divider wholly inside the box plus 16 px is kept too.
        let mut img2 = RgbImage::from_pixel(200, 200, Rgb([255, 255, 255]));
        for y in 50..150u32 {
            for x in 100..104 {
                img2.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        for y in 60..72u32 {
            for x in 70..82 {
                img2.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        let out2 = composite(
            &img2,
            &clean_page(&img2, &[CleanBox::auto([60, 40, 110, 160], None)]),
        );
        assert_eq!(out2.get_pixel(101, 100)[0], 0, "short divider erased");
        assert_eq!(
            out2.get_pixel(75, 65).0,
            [255, 255, 255],
            "letter left in img2"
        );
        for y in [45u32, 70, 100, 120, 150] {
            assert_eq!(out.get_pixel(101, y)[0], 0, "outline erased at y={y}");
        }
        assert_eq!(out.get_pixel(75, 55).0, [255, 255, 255], "letter left");
    }

    #[test]
    fn white_text_on_dark_strip() {
        let mut img = RgbImage::from_pixel(120, 50, Rgb([20, 20, 20]));
        bars(&mut img, &[(30, 20)], 60, [250, 250, 250]);
        let out = composite(
            &img,
            &clean_page(&img, &[CleanBox::auto([25, 15, 95, 30], None)]),
        );
        assert_eq!(out.get_pixel(60, 21).0, [20, 20, 20]);
    }

    #[test]
    fn text_on_gradient_is_filled_smoothly() {
        let mut img = RgbImage::from_fn(140, 100, |_, y| {
            Rgb([120 + y as u8, 160 + y as u8 / 2, 230])
        });
        bars(&mut img, &[(40, 40), (40, 52)], 60, [0, 0, 0]);
        let r = clean_page_full(&img, &[CleanBox::auto([35, 35, 105, 60], None)]);
        assert_eq!(r.methods, vec![FillMethod::Smooth]);
        let out = composite(&img, &r.layer);
        let p = out.get_pixel(70, 42);
        let expect = [120 + 42, 160 + 21, 230];
        for k in 0..3 {
            assert!(
                (p[k] as i32 - expect[k]).abs() <= 10,
                "{:?} vs {:?}",
                p.0,
                expect
            );
        }
    }

    #[test]
    fn text_on_screentone_keeps_texture() {
        let mut img = RgbImage::from_fn(140, 100, |x, y| {
            if (x / 2 + y / 2) % 2 == 0 {
                Rgb([60, 60, 60])
            } else {
                Rgb([220, 220, 220])
            }
        });
        bars(&mut img, &[(40, 40), (40, 52)], 60, [0, 0, 0]);
        let r = clean_page_full(&img, &[CleanBox::auto([35, 35, 105, 60], None)]);
        assert_eq!(r.methods, vec![FillMethod::Texture]);
        let out = composite(&img, &r.layer);
        let row: Vec<u8> = (45..95).map(|x| out.get_pixel(x, 42)[0]).collect();
        assert!(
            row.iter().any(|&v| v < 110) && row.iter().any(|&v| v > 170),
            "{row:?}"
        );
        assert!(row.iter().all(|&v| v > 30), "{row:?}");
    }

    #[test]
    fn fill_override_and_shapes() {
        let mut img = RgbImage::from_pixel(60, 40, Rgb([250, 250, 250]));
        bars(&mut img, &[(20, 18)], 20, [0, 0, 0]);
        let mut b = CleanBox::auto([15, 14, 45, 26], None);
        b.fill = Some([200, 10, 10]);
        assert_eq!(
            clean_page(&img, &[b]).get_pixel(30, 19).0,
            [200, 10, 10, 255]
        );

        let style = ShapeStyle {
            shape: Shape::Ellipse,
            fill: [255, 255, 255, 255],
            outline: Some(([0, 0, 0, 255], 3.0)),
        };
        let boxes = [CleanBox {
            rotation: 0.0,
            frame: None,
            text_box: [30, 20, 90, 60],
            bubble: None,
            fill: None,
            shape: Some(style),
        }];
        let layer = clean_page(&RgbImage::new(120, 80), &boxes);
        assert_eq!(layer.get_pixel(60, 40).0, [255, 255, 255, 255]);
        assert_eq!(layer.get_pixel(0, 0)[3], 0);
        assert!((0..60).any(|x| layer.get_pixel(x, 40).0 == [0, 0, 0, 255]));
    }
}

#[cfg(test)]
mod balloon_tests {
    use super::*;
    use crate::bubbles::Balloon;

    #[test]
    fn letter_touching_outline_is_erased_inside_the_balloon_only() {
        // White disc bubble with a 4 px outline; a letter bar fused to the right outline.
        let mut img = RgbImage::from_pixel(200, 200, Rgb([150, 150, 150]));
        let inside = |x: i32, y: i32| ((x - 100).pow(2) + (y - 100).pow(2)) as f32;
        for y in 0..200 {
            for x in 0..200 {
                let d = inside(x, y).sqrt();
                if d <= 70.0 {
                    img.put_pixel(x as u32, y as u32, Rgb([255, 255, 255]));
                }
                if (66.0..=70.0).contains(&d) {
                    img.put_pixel(x as u32, y as u32, Rgb([0, 0, 0]));
                }
            }
        }
        for y in 96..104 {
            for x in 120..168 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        let mask = GrayImage::from_fn(140, 140, |x, y| {
            let d = inside(x as i32 + 30, y as i32 + 30).sqrt();
            image::Luma([if d <= 70.0 { 255 } else { 0 }])
        });
        let bal = Balloon {
            bbox: [30, 30, 170, 170],
            score: 0.9,
            mask,
        };
        let r =
            clean_page_with_balloons(&img, &[CleanBox::auto([110, 90, 172, 110], None)], &[bal]);
        let out = composite(&img, &r.layer);
        assert_eq!(out.get_pixel(140, 100).0, [255, 255, 255], "letter erased");
        assert_eq!(out.get_pixel(168, 100).0, [0, 0, 0], "outline kept");
        assert_eq!(
            out.get_pixel(185, 100).0,
            [150, 150, 150],
            "outside untouched"
        );
    }

    #[test]
    fn tilted_caption_strip_over_dark_art_is_cleaned_in_its_frame() {
        // A white strip tilted up to the right on black, with dark letters along it and a light
        // shoe sole just above it. Its upright box is mostly black art.
        let (a, c) = (11f32.to_radians(), (120.0f32, 80.0f32));
        let u = (a.cos(), -a.sin());
        let n = (a.sin(), a.cos());
        let along = |x: u32, y: u32| (x as f32 - c.0) * u.0 + (y as f32 - c.1) * u.1;
        let across = |x: u32, y: u32| (x as f32 - c.0) * n.0 + (y as f32 - c.1) * n.1;
        let in_strip = |x: u32, y: u32| along(x, y).abs() <= 100.0 && across(x, y).abs() <= 15.0;
        let in_letter = |x: u32, y: u32| {
            let t = along(x, y) + 90.0;
            across(x, y).abs() <= 6.0 && t >= 0.0 && t <= 180.0 && t % 20.0 < 12.0
        };
        let sole = |x: u32, y: u32| (60..90).contains(&x) && (78..84).contains(&y) && across(x, y) < -19.0;
        let img = RgbImage::from_fn(240, 160, |x, y| {
            if in_letter(x, y) {
                Rgb([20, 20, 20])
            } else if in_strip(x, y) {
                Rgb([240, 240, 240])
            } else if sole(x, y) {
                Rgb([225, 225, 225])
            } else {
                Rgb([0, 0, 0])
            }
        });
        let pts: Vec<(u32, u32)> = (0..240)
            .flat_map(|x| (0..160).map(move |y| (x, y)))
            .filter(|&(x, y)| in_strip(x, y))
            .collect();
        let bx = |f: fn(&(u32, u32)) -> u32| pts.iter().map(f);
        let tb = [
            bx(|p| p.0).min().unwrap() as i32,
            bx(|p| p.1).min().unwrap() as i32,
            bx(|p| p.0).max().unwrap() as i32 + 1,
            bx(|p| p.1).max().unwrap() as i32 + 1,
        ];
        let crop = image::imageops::crop_imm(&img, tb[0] as u32, tb[1] as u32, (tb[2] - tb[0]) as u32, (tb[3] - tb[1]) as u32).to_image();
        let angle = crate::rotation::estimate(&crop);
        assert!((angle.degrees.abs() - 11.0).abs() <= 2.0, "angle {}", angle.degrees);
        let mut cb = CleanBox::auto(tb, None);
        cb.rotation = angle.degrees;
        let r = clean_page_full(&img, &[cb]);
        let out = composite(&img, &r.layer);
        let mut left = 0;
        for (x, y) in pts.iter().copied().filter(|&(x, y)| in_letter(x, y)) {
            if lum(out.get_pixel(x, y)) < 160.0 {
                left += 1;
            }
        }
        assert_eq!(left, 0, "letter pixels left");
        for y in 0..160 {
            for x in 0..240 {
                if sole(x, y) {
                    assert_eq!(out.get_pixel(x, y), img.get_pixel(x, y), "sole at {x},{y}");
                }
                if !in_strip(x, y) && across(x, y).abs() > 18.0 {
                    assert_eq!(out.get_pixel(x, y), img.get_pixel(x, y), "art at {x},{y}");
                }
            }
        }
    }
}
