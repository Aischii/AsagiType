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
}

impl CleanBox {
    pub fn auto(text_box: [i32; 4], bubble: Option<[i32; 4]>) -> Self {
        Self {
            text_box,
            bubble,
            fill: None,
            shape: None,
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
            None => clean_auto(img, b, &mut res.layer, &mut res.mask),
        };
        res.touch_up.push(touch);
        res.methods.push(method);
    }
    res
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
                    let letters = thick_letters(&pixels, thickness, region, &core);
                    if letters.is_empty() {
                        touch_up = true;
                    }
                    for (x, y) in letters {
                        protected[idx(x, y)] = false;
                        erasable[idx(x, y)] = true;
                    }
                }
            } else {
                for (x, y) in pixels {
                    erasable[idx(x, y)] = true;
                }
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
        if escapes || part.len() < 6 {
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
        // Thick "letters" (5 px bars) overlapping the lines.
        for &(x0, y0) in &[(70u32, 45u32), (70, 65)] {
            for y in y0..y0 + 5 {
                for x in x0..x0 + 50 {
                    img.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        let r = clean_page_full(&img, &[CleanBox::auto([65, 40, 125, 75], None)]);
        let out = composite(&img, &r.layer);
        assert_eq!(r.touch_up, vec![false]);
        let white = (70..120)
            .filter(|&x| out.get_pixel(x, 47).0 == [255, 255, 255])
            .count();
        assert!(white > 30, "only {white} white pixels in the erased row");
        // Lines crossing the erased bars come back: line k=1 (y = x/2 + 18) passes y 45..50 at x 54..63
        // and k=2 (y = x/2 + 36) passes y 45..50 at x 18..27; check k=0 through bar 2: y = x/2 at x 130.. no.
        // Use line k=1 through bar 2 (y 65..70 at x 94..103).
        let dark = (94..104u32)
            .filter(|&x| out.get_pixel(x, x / 2 + 18)[0] < 128)
            .count();
        assert!(dark >= 6, "line not redrawn ({dark} of 10 pixels)");
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
