//! Compare the candle balloon segmenter with Ultralytics reference outputs.
//! Usage: seg_check <model dir> <folder with input.f32, pred.f32, proto.f32, dets.json>
use candle_core::{Device, Tensor};
use std::{path::PathBuf, time::Instant};

fn read_f32(p: PathBuf) -> anyhow::Result<Vec<f32>> {
    Ok(std::fs::read(p)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (model, refs) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let seg = vision::bubbles::BubbleSegmenter::load(&model, &Device::Cpu)?;
    let n = vision::bubbles::INPUT;
    let x = Tensor::from_vec(
        read_f32(refs.join("input.f32"))?,
        (1, 3, n, n),
        &Device::Cpu,
    )?;
    let t = Instant::now();
    let (pred, proto) = seg.forward(&x)?;
    println!(
        "forward {:.2}s pred {:?} proto {:?}",
        t.elapsed().as_secs_f32(),
        pred.dims(),
        proto.dims()
    );
    let want_p = read_f32(refs.join("pred.f32"))?;
    let want_q = read_f32(refs.join("proto.f32"))?;
    let got_p: Vec<f32> = pred.flatten_all()?.to_vec1()?;
    let got_q: Vec<f32> = proto.flatten_all()?.to_vec1()?;
    let maxd = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let rows = pred.dim(1)?;
    let row = |v: &[f32], r: usize| v[r * rows..(r + 1) * rows].to_vec();
    println!(
        "max diff: boxes {:.4}  score {:.5}  coef {:.4}  proto {:.4}",
        maxd(&got_p[..4 * rows], &want_p[..4 * rows]),
        maxd(&row(&got_p, 4), &row(&want_p, 4)),
        maxd(&got_p[5 * rows..], &want_p[5 * rows..]),
        maxd(&got_q, &want_q)
    );
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refs.join("dets.json"))?)?;
    let pad = (
        meta["pad"][0].as_u64().unwrap() as u32,
        meta["pad"][1].as_u64().unwrap() as u32,
    );
    let dets = vision::bubbles::postprocess(&pred, &proto, 1.0, pad, (n as u32, n as u32))?;
    println!(
        "{} balloons (reference {})",
        dets.len(),
        meta["boxes"].as_array().unwrap().len()
    );
    for (d, (b, s)) in dets
        .iter()
        .zip(
            meta["boxes"]
                .as_array()
                .unwrap()
                .iter()
                .zip(meta["scores"].as_array().unwrap()),
        )
        .take(4)
    {
        let area = d.mask.pixels().filter(|p| p[0] > 0).count();
        println!(
            "{:?} {:.4} area {} | ref {} {}",
            d.bbox.map(|v| v + 0),
            d.score,
            area,
            b,
            s
        );
    }
    Ok(())
}
