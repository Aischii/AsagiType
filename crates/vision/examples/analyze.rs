//! Run detection + OCR on one page. Usage: analyze <models dir> <image>
use std::{path::PathBuf, time::Instant};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let t = Instant::now();
    let models = vision::Models::load(&PathBuf::from(&args[1]))?;
    println!("models loaded in {:.1}s", t.elapsed().as_secs_f32());
    let img = image::open(&args[2])?;
    let t = Instant::now();
    let regions = models.analyze(&img, 0.5)?;
    println!(
        "{} text regions in {:.1}s",
        regions.len(),
        t.elapsed().as_secs_f32()
    );
    for r in regions {
        println!(
            "{:<8} {:.2} {:?} rot={} frame={:?} {}",
            r.kind, r.score, r.text_box, r.rotation, r.frame, r.source_text
        );
    }
    Ok(())
}
