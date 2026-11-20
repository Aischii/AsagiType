//! Detect, then clean a page and write the result. Usage: clean_check <models dir> <image> <out.png>
use std::path::PathBuf;
use vision::clean::{clean_page, composite, CleanBox};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let models = vision::Models::load(&PathBuf::from(&a[1]))?;
    let img = image::open(&a[2])?;
    let boxes: Vec<CleanBox> = models
        .analyze(&img, 0.5)?
        .iter()
        .map(|r| CleanBox {
            text_box: r.text_box,
            bubble: r.bubble,
        })
        .collect();
    let t = std::time::Instant::now();
    let layer = clean_page(&img.to_rgb8(), &boxes);
    println!(
        "cleaned {} boxes in {} ms",
        boxes.len(),
        t.elapsed().as_millis()
    );
    composite(&img.to_rgb8(), &layer).save(&a[3])?;
    Ok(())
}
