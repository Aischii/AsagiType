//! Compare candle manga-ocr token ids with reference ids from the Python model.
//! Usage: ocr_check <manga-ocr dir> <folder with refs.json and crops>
use candle_core::Device;
use std::{path::PathBuf, time::Instant};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (model, refs) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]));
    let t = Instant::now();
    let ocr = vision::MangaOcr::load(&model, &Device::Cpu)?;
    println!("loaded in {:.1}s", t.elapsed().as_secs_f32());
    let list: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refs.join("refs.json"))?)?;
    let (mut same, mut total) = (0, 0);
    for r in list.as_array().unwrap() {
        let img = image::open(refs.join(r["file"].as_str().unwrap()))?;
        let want: Vec<u32> = r["greedy_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let t = Instant::now();
        let got = ocr.token_ids(&img)?;
        let text = ocr.recognize(&img)?;
        total += 1;
        let ok = got == want;
        same += ok as usize;
        println!(
            "{} {:>5}ms {} | {}",
            if ok { "OK " } else { "DIFF" },
            t.elapsed().as_millis(),
            text,
            r["greedy"].as_str().unwrap()
        );
    }
    println!("{same}/{total} identical token sequences");
    Ok(())
}
