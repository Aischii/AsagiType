//! Show the erase mask for one box. Usage: mask_check <models dir> <image> <box index 1-based> <out.png>
use std::path::PathBuf;
use vision::clean::{clean_page_full, CleanBox};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let models = vision::Models::load(&PathBuf::from(&a[1]))?;
    let img = image::open(&a[2])?.to_rgb8();
    let regions = models.analyze(&image::DynamicImage::ImageRgb8(img.clone()), 0.5)?;
    let r = &regions[a[3].parse::<usize>()? - 1];
    println!("box {:?} bubble {:?}", r.text_box, r.bubble);
    let res = clean_page_full(&img, &[CleanBox::auto(r.text_box, r.bubble)]);
    let t = r.text_box;
    let (x0, y0, x1, y1) = (
        (t[0] - 30).max(0) as u32,
        (t[1] - 30).max(0) as u32,
        (t[2] + 30) as u32,
        (t[3] + 30) as u32,
    );
    let (w, h) = (x1 - x0, y1 - y0);
    let mut out = image::RgbImage::new(w * 3, h);
    for y in 0..h {
        for x in 0..w {
            let p = *img.get_pixel(x0 + x, y0 + y);
            out.put_pixel(x, y, p);
            let m = res.mask.get_pixel(x0 + x, y0 + y)[0] > 0;
            out.put_pixel(w + x, y, if m { image::Rgb([255, 0, 0]) } else { p });
            let l = res.layer.get_pixel(x0 + x, y0 + y);
            out.put_pixel(
                2 * w + x,
                y,
                if l[3] > 0 {
                    image::Rgb([l[0], l[1], l[2]])
                } else {
                    p
                },
            );
        }
    }
    out.save(&a[4])?;
    Ok(())
}
