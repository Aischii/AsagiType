//! Solid-fill cleaning: erase source text by filling it with the balloon's own colour,
//! clipped to the bubble so outlines and art survive. Produces a page-size RGBA layer
//! that is transparent everywhere except the filled pixels.
//!
//! Per text box:
//! 1. Work inside a region B: the bubble box (plus 2 px) or, for free text, the text box plus 12 px.
//! 2. Pixels at least as light as the background threshold are "light". The threshold comes
//!    from the text box itself: background is its 90th-percentile luminance.
//! 3. L = light pixels connected to light pixels inside the text box (4-connected, within B).
//! 4. "Outside" = everything reachable from B's border without crossing L. Bubble outlines and
//!    art touch the border, so they are outside; glyphs enclosed by L are not.
//! 5. Fill = (L or not outside) inside the text box plus a small margin, painted with L's median colour.

use image::{Rgb, RgbImage, Rgba, RgbaImage};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub struct CleanBox {
    pub text_box: [i32; 4],
    pub bubble: Option<[i32; 4]>,
}

/// Margin around the text box that may be filled.
const MARGIN: i32 = 6;
const FREE_TEXT_PAD: i32 = 12;
/// Light pixels are at most this much darker than the background.
const TOLERANCE: f32 = 60.0;

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

/// Clean every box onto one transparent layer the size of `img`.
pub fn clean_page(img: &RgbImage, boxes: &[CleanBox]) -> RgbaImage {
    let mut layer = RgbaImage::new(img.width(), img.height());
    for b in boxes {
        clean_box(img, b, &mut layer);
    }
    layer
}

fn clean_box(img: &RgbImage, cb: &CleanBox, layer: &mut RgbaImage) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let t = clamp_box(cb.text_box, w, h);
    if t[2] - t[0] < 2 || t[3] - t[1] < 2 {
        return;
    }
    let region = clamp_box(
        match cb.bubble {
            Some(bb) => {
                // The bubble must contain the text box.
                let u = [
                    bb[0].min(t[0]),
                    bb[1].min(t[1]),
                    bb[2].max(t[2]),
                    bb[3].max(t[3]),
                ];
                grow(u, 2)
            }
            None => grow(t, FREE_TEXT_PAD),
        },
        w,
        h,
    );
    let (rx, ry) = (region[0], region[1]);
    let (rw, rh) = ((region[2] - rx) as usize, (region[3] - ry) as usize);
    let idx = |x: i32, y: i32| (y - ry) as usize * rw + (x - rx) as usize;

    // Background threshold from the text box's own pixels.
    let mut lums: Vec<f32> = (t[1]..t[3])
        .flat_map(|y| (t[0]..t[2]).map(move |x| (x, y)))
        .map(|(x, y)| lum(img.get_pixel(x as u32, y as u32)))
        .collect();
    lums.sort_by(f32::total_cmp);
    let bg_lum = lums[(lums.len() * 9 / 10).min(lums.len() - 1)];
    let threshold = bg_lum - TOLERANCE;

    let mut light = vec![false; rw * rh];
    for y in ry..region[3] {
        for x in rx..region[2] {
            light[idx(x, y)] = lum(img.get_pixel(x as u32, y as u32)) >= threshold;
        }
    }

    // L: light pixels connected to light seeds inside the text box.
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
    flood(
        &mut queue,
        &mut in_l,
        |x, y| x >= rx && y >= ry && x < region[2] && y < region[3] && light[idx(x, y)],
        &idx,
    );

    // Outside: reachable from the region border without crossing L.
    let mut outside = vec![false; rw * rh];
    for y in ry..region[3] {
        for x in rx..region[2] {
            let border = x == rx || y == ry || x == region[2] - 1 || y == region[3] - 1;
            let i = idx(x, y);
            if border && !in_l[i] {
                outside[i] = true;
                queue.push_back((x, y));
            }
        }
    }
    flood(
        &mut queue,
        &mut outside,
        |x, y| x >= rx && y >= ry && x < region[2] && y < region[3] && !in_l[idx(x, y)],
        &idx,
    );

    // Fill colour: median of L inside the text box.
    let mut chans: [Vec<u8>; 3] = Default::default();
    for y in t[1]..t[3] {
        for x in t[0]..t[2] {
            if in_l[idx(x, y)] {
                let p = img.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    chans[c].push(p[c]);
                }
            }
        }
    }
    if chans[0].is_empty() {
        return;
    }
    let colour: [u8; 3] = std::array::from_fn(|c| {
        chans[c].sort_unstable();
        chans[c][chans[c].len() / 2]
    });

    let fill = clamp_box(grow(t, MARGIN), w, h);
    for y in fill[1].max(ry)..fill[3].min(region[3]) {
        for x in fill[0].max(rx)..fill[2].min(region[2]) {
            let i = idx(x, y);
            if in_l[i] || !outside[i] {
                layer.put_pixel(
                    x as u32,
                    y as u32,
                    Rgba([colour[0], colour[1], colour[2], 255]),
                );
            }
        }
    }
}

fn flood(
    queue: &mut VecDeque<(i32, i32)>,
    mark: &mut [bool],
    ok: impl Fn(i32, i32) -> bool,
    idx: &impl Fn(i32, i32) -> usize,
) {
    while let Some((x, y)) = queue.pop_front() {
        for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
            if ok(nx, ny) {
                let i = idx(nx, ny);
                if !mark[i] {
                    mark[i] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
    }
}

/// Composite a clean layer over the page (for export and tests).
pub fn composite(img: &RgbImage, layer: &RgbaImage) -> RgbImage {
    let mut out = img.clone();
    for (x, y, p) in layer.enumerate_pixels() {
        if p[3] > 0 {
            out.put_pixel(x, y, Rgb([p[0], p[1], p[2]]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grey page, white ellipse bubble with a 3 px black outline, black "text" bars inside.
    fn page() -> RgbImage {
        let mut img = RgbImage::from_pixel(200, 200, Rgb([150, 150, 150]));
        let (cx, cy, a, b) = (100.0f32, 100.0f32, 70.0f32, 50.0f32);
        for y in 0..200 {
            for x in 0..200 {
                let d = ((x as f32 - cx) / a).powi(2) + ((y as f32 - cy) / b).powi(2);
                if d <= 1.0 {
                    img.put_pixel(x, y, Rgb([255, 255, 255]));
                }
                if (0.88..=1.0).contains(&d) {
                    img.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        for (x0, y0) in [(80, 85), (80, 100), (80, 115)] {
            for y in y0..y0 + 4 {
                for x in x0..x0 + 40 {
                    img.put_pixel(x, y, Rgb([10, 10, 10]));
                }
            }
        }
        img
    }

    #[test]
    fn cleans_text_and_keeps_outline_and_art() {
        let img = page();
        let layer = clean_page(
            &img,
            &[CleanBox {
                text_box: [76, 80, 124, 122],
                bubble: Some([30, 50, 170, 150]),
            }],
        );
        let out = composite(&img, &layer);
        // Text gone.
        for (x, y) in [(90, 86), (100, 101), (110, 116)] {
            assert_eq!(out.get_pixel(x, y).0, [255, 255, 255], "text at {x},{y}");
        }
        // Outline (left edge at y=100) and grey art outside untouched.
        assert_eq!(out.get_pixel(31, 100).0, [0, 0, 0]);
        assert_eq!(out.get_pixel(5, 5).0, [150, 150, 150]);
        // Nothing painted outside the text box margin.
        assert_eq!(layer.get_pixel(40, 100)[3], 0);
    }

    #[test]
    fn free_text_on_flat_background() {
        let mut img = RgbImage::from_pixel(100, 60, Rgb([240, 230, 200]));
        for x in 20..80 {
            for y in 25..29 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        let out = composite(
            &img,
            &clean_page(
                &img,
                &[CleanBox {
                    text_box: [18, 22, 82, 32],
                    bubble: None,
                }],
            ),
        );
        assert_eq!(out.get_pixel(50, 27).0, [240, 230, 200]);
    }
}
