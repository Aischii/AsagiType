//! Compare the candle detector with reference outputs from transformers.
//! Usage: detect_check <detector dir> <folder with input.f32, logits.f32, boxes.f32, dets.json>
use candle_core::{Device, Tensor};
use std::{path::PathBuf, time::Instant};

fn read_f32(p: PathBuf) -> anyhow::Result<Vec<f32>> {
    Ok(std::fs::read(p)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (model, refs) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]));
    let det = vision::ComicDetector::load(&model, &Device::Cpu)?;
    let input = Tensor::from_vec(
        read_f32(refs.join("input.f32"))?,
        (1, 3, 640, 640),
        &Device::Cpu,
    )?;
    let t = Instant::now();
    let (logits, boxes) = det.forward(&input)?;
    println!("forward {:.2}s", t.elapsed().as_secs_f32());
    let want_l = read_f32(refs.join("logits.f32"))?;
    let want_b = read_f32(refs.join("boxes.f32"))?;
    let got_l: Vec<f32> = logits.iter().flatten().copied().collect();
    let got_b: Vec<f32> = boxes.iter().flatten().copied().collect();
    let max_diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    println!(
        "max |logit diff| {:.5}, max |box diff| {:.5}",
        max_diff(&got_l, &want_l),
        max_diff(&got_b, &want_b)
    );
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refs.join("dets.json"))?)?;
    let (w, h) = (
        meta["size"][0].as_f64().unwrap() as f32,
        meta["size"][1].as_f64().unwrap() as f32,
    );
    let dets = vision::detect::postprocess(&logits, &boxes, w, h, 0.5);
    println!(
        "{} detections (reference {})",
        dets.len(),
        meta["dets"].as_array().unwrap().len()
    );
    for (d, r) in dets.iter().zip(meta["dets"].as_array().unwrap()).take(6) {
        println!(
            "{:?} {:.3} {:?} | ref {} {} {}",
            d.class,
            d.score,
            d.bbox.map(|v| v.round() as i32),
            r["label"],
            r["score"],
            r["box"]
        );
    }
    Ok(())
}
