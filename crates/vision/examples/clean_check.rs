//! Detect, then clean a page and write the result.
//! Usage: clean_check <models dir> <image> <out.png>
use std::path::PathBuf;
use vision::clean::{clean_page_with_balloons, composite, CleanBox};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let models = vision::Models::load(&PathBuf::from(&a[1]))?;
    let img = image::open(&a[2])?;
    let regions = models.analyze(&img, 0.5)?;
    let boxes: Vec<CleanBox> = regions
        .iter()
        .map(|r| CleanBox::auto(r.text_box, r.bubble))
        .collect();
    let seg = vision::load_segmenter(&PathBuf::from(&a[1]));
    let ts = std::time::Instant::now();
    let balloons = seg
        .map(|s| s.segment(&img.to_rgb8()))
        .transpose()?
        .unwrap_or_default();
    println!(
        "segmented {} balloons in {} ms",
        balloons.len(),
        ts.elapsed().as_millis()
    );
    let t = std::time::Instant::now();
    let res = clean_page_with_balloons(&img.to_rgb8(), &boxes, &balloons);
    println!(
        "cleaned {} boxes in {} ms",
        boxes.len(),
        t.elapsed().as_millis()
    );
    for (i, r) in regions.iter().enumerate() {
        let tb = r.text_box;
        let masked = (tb[1]..tb[3])
            .flat_map(|y| (tb[0]..tb[2]).map(move |x| (x, y)))
            .filter(|&(x, y)| res.mask.get_pixel(x as u32, y as u32)[0] > 0)
            .count();
        println!(
            "{:>2} {:?} touch_up={} box={:?} masked={} {}",
            i + 1,
            res.methods[i],
            res.touch_up[i],
            tb,
            masked,
            r.source_text
        );
    }
    composite(&img.to_rgb8(), &res.layer).save(&a[3])?;
    Ok(())
}
