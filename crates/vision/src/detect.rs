//! Comic balloon and text detector: RT-DETR v2 with a ResNet-50-vd backbone
//! (ogkalu/comic-text-and-bubble-detector, Apache-2.0), written in candle.
//! Classes: 0 = bubble, 1 = text inside a bubble, 2 = free text.
//! Mirrors `transformers` RTDetrV2ForObjectDetection in eval mode.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{
    conv2d_no_bias, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder,
};
use image::{imageops::FilterType, DynamicImage};
use std::path::Path;

type R<T> = candle_core::Result<T>;

const INPUT: usize = 640;
const D_MODEL: usize = 256;
const HEADS: usize = 8;
const LEVELS: usize = 3;
const POINTS: usize = 4;
const QUERIES: usize = 300;
const CLASSES: usize = 3;
const DEC_LAYERS: usize = 6;
const OFFSET_SCALE: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Bubble,
    TextBubble,
    TextFree,
}

impl Class {
    fn from_index(i: usize) -> Self {
        match i {
            0 => Class::Bubble,
            1 => Class::TextBubble,
            _ => Class::TextFree,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub class: Class,
    pub score: f32,
    /// [x1, y1, x2, y2] in original image pixels.
    pub bbox: [f32; 4],
}

// ---------- building blocks ----------

/// Frozen BatchNorm folded into scale and shift.
struct BatchNorm {
    scale: Tensor,
    shift: Tensor,
}

impl BatchNorm {
    fn new(c: usize, vb: VarBuilder) -> R<Self> {
        let w = vb.get(c, "weight")?;
        let b = vb.get(c, "bias")?;
        let mean = vb.get(c, "running_mean")?;
        let var = vb.get(c, "running_var")?;
        let scale = (w / (var + 1e-5)?.sqrt()?)?;
        let shift = (b - (&mean * &scale)?)?;
        Ok(Self {
            scale: scale.reshape((1, c, 1, 1))?,
            shift: shift.reshape((1, c, 1, 1))?,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)
    }
}

#[derive(Clone, Copy)]
enum Act {
    None,
    Relu,
    Silu,
}

fn act(x: Tensor, a: Act) -> R<Tensor> {
    match a {
        Act::None => Ok(x),
        Act::Relu => x.relu(),
        Act::Silu => candle_nn::ops::silu(&x),
    }
}

struct ConvBn {
    conv: Conv2d,
    bn: BatchNorm,
    act: Act,
}

impl ConvBn {
    /// `names` = (conv, norm) sub-module names, which differ between backbone and encoder.
    #[allow(clippy::too_many_arguments)]
    fn new(
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        a: Act,
        vb: VarBuilder,
        names: (&str, &str),
    ) -> R<Self> {
        let cfg = Conv2dConfig {
            padding: pad,
            stride,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d_no_bias(cin, cout, k, cfg, vb.pp(names.0))?,
            bn: BatchNorm::new(cout, vb.pp(names.1))?,
            act: a,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        act(self.bn.forward(&self.conv.forward(x)?)?, self.act)
    }
}

fn backbone_conv(
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    a: Act,
    vb: VarBuilder,
) -> R<ConvBn> {
    ConvBn::new(
        cin,
        cout,
        k,
        stride,
        k / 2,
        a,
        vb,
        ("convolution", "normalization"),
    )
}

fn encoder_conv(
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    a: Act,
    vb: VarBuilder,
) -> R<ConvBn> {
    ConvBn::new(cin, cout, k, stride, (k - 1) / 2, a, vb, ("conv", "norm"))
}

struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn new(dims: &[usize], vb: VarBuilder) -> R<Self> {
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, w)| linear(w[0], w[1], vb.pp(format!("layers.{i}"))))
            .collect::<R<_>>()?;
        Ok(Self { layers })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut x = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(&x)?;
            if i + 1 < self.layers.len() {
                x = x.relu()?;
            }
        }
        Ok(x)
    }
}

// ---------- ResNet-50-vd backbone ----------

struct Bottleneck {
    convs: [ConvBn; 3],
    shortcut: Option<ConvBn>,
    avg_pool: bool,
}

impl Bottleneck {
    fn new(cin: usize, cout: usize, stride: usize, vb: VarBuilder) -> R<Self> {
        let mid = cout / 4;
        let l = vb.pp("layer");
        let convs = [
            backbone_conv(cin, mid, 1, 1, Act::Relu, l.pp("0"))?,
            backbone_conv(mid, mid, 3, stride, Act::Relu, l.pp("1"))?,
            backbone_conv(mid, cout, 1, 1, Act::None, l.pp("2"))?,
        ];
        let needs = cin != cout || stride != 1;
        let (shortcut, avg_pool) = if stride == 2 {
            (
                Some(backbone_conv(
                    cin,
                    cout,
                    1,
                    1,
                    Act::None,
                    vb.pp("shortcut.1"),
                )?),
                true,
            )
        } else if needs {
            (
                Some(backbone_conv(
                    cin,
                    cout,
                    1,
                    stride,
                    Act::None,
                    vb.pp("shortcut"),
                )?),
                false,
            )
        } else {
            (None, false)
        };
        Ok(Self {
            convs,
            shortcut,
            avg_pool,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = x.clone();
        for c in &self.convs {
            h = c.forward(&h)?;
        }
        let mut res = x.clone();
        if self.avg_pool {
            res = res.avg_pool2d(2)?;
        }
        if let Some(s) = &self.shortcut {
            res = s.forward(&res)?;
        }
        (h + res)?.relu()
    }
}

struct Backbone {
    stem: [ConvBn; 3],
    stages: Vec<Vec<Bottleneck>>,
}

impl Backbone {
    fn new(vb: VarBuilder) -> R<Self> {
        let e = vb.pp("embedder.embedder");
        let stem = [
            backbone_conv(3, 32, 3, 2, Act::Relu, e.pp("0"))?,
            backbone_conv(32, 32, 3, 1, Act::Relu, e.pp("1"))?,
            backbone_conv(32, 64, 3, 1, Act::Relu, e.pp("2"))?,
        ];
        let depths = [3, 4, 6, 3];
        let sizes = [256, 512, 1024, 2048];
        let mut stages = vec![];
        let mut cin = 64;
        for (s, (&depth, &cout)) in depths.iter().zip(&sizes).enumerate() {
            let sv = vb.pp(format!("encoder.stages.{s}.layers"));
            let stride = if s == 0 { 1 } else { 2 };
            let mut layers = vec![Bottleneck::new(cin, cout, stride, sv.pp("0"))?];
            for i in 1..depth {
                layers.push(Bottleneck::new(cout, cout, 1, sv.pp(i.to_string()))?);
            }
            stages.push(layers);
            cin = cout;
        }
        Ok(Self { stem, stages })
    }
    /// Feature maps of stages 2, 3 and 4 (strides 8, 16, 32).
    fn forward(&self, x: &Tensor) -> R<Vec<Tensor>> {
        let mut h = x.clone();
        for c in &self.stem {
            h = c.forward(&h)?;
        }
        // MaxPool 3x3 / 2, padding 1. Inputs are post-ReLU (>= 0), so zero padding equals -inf padding.
        h = h
            .pad_with_zeros(2, 1, 1)?
            .pad_with_zeros(3, 1, 1)?
            .max_pool2d_with_stride(3, 2)?;
        let mut outs = vec![];
        for (i, stage) in self.stages.iter().enumerate() {
            for l in stage {
                h = l.forward(&h)?;
            }
            if i >= 1 {
                outs.push(h.clone());
            }
        }
        Ok(outs)
    }
}

// ---------- hybrid encoder ----------

fn self_attention(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> R<Tensor> {
    let (b, t, c) = q.dims3()?;
    let dh = c / heads;
    let split = |x: &Tensor| x.reshape((b, t, heads, dh))?.transpose(1, 2)?.contiguous();
    let (q, k, v) = (split(q)?, split(k)?, split(v)?);
    let w = candle_nn::ops::softmax_last_dim(&(q.matmul(&k.t()?)? * (dh as f64).powf(-0.5))?)?;
    w.matmul(&v)?.transpose(1, 2)?.reshape((b, t, c))
}

struct SelfAttn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl SelfAttn {
    fn new(vb: VarBuilder) -> R<Self> {
        Ok(Self {
            q: linear(D_MODEL, D_MODEL, vb.pp("q_proj"))?,
            k: linear(D_MODEL, D_MODEL, vb.pp("k_proj"))?,
            v: linear(D_MODEL, D_MODEL, vb.pp("v_proj"))?,
            o: linear(D_MODEL, D_MODEL, vb.pp("out_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor, pos: &Tensor) -> R<Tensor> {
        let qk = x.broadcast_add(pos)?;
        let a = self_attention(
            &self.q.forward(&qk)?,
            &self.k.forward(&qk)?,
            &self.v.forward(x)?,
            HEADS,
        )?;
        self.o.forward(&a)
    }
}

struct AifiLayer {
    attn: SelfAttn,
    ln1: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ln2: LayerNorm,
}

impl AifiLayer {
    fn new(vb: VarBuilder) -> R<Self> {
        Ok(Self {
            attn: SelfAttn::new(vb.pp("self_attn"))?,
            ln1: layer_norm(D_MODEL, 1e-5, vb.pp("self_attn_layer_norm"))?,
            fc1: linear(D_MODEL, 1024, vb.pp("fc1"))?,
            fc2: linear(1024, D_MODEL, vb.pp("fc2"))?,
            ln2: layer_norm(D_MODEL, 1e-5, vb.pp("final_layer_norm"))?,
        })
    }
    fn forward(&self, x: &Tensor, pos: &Tensor) -> R<Tensor> {
        let x = self.ln1.forward(&(x + self.attn.forward(x, pos)?)?)?;
        let h = self.fc2.forward(&self.fc1.forward(&x)?.gelu_erf()?)?;
        self.ln2.forward(&(x + h)?)
    }
}

fn sincos_2d(h: usize, w: usize, dim: usize, dev: &Device) -> R<Tensor> {
    let p = dim / 4;
    let omega: Vec<f64> = (0..p)
        .map(|i| 1.0 / 10000f64.powf(i as f64 / p as f64))
        .collect();
    let mut out = Vec::with_capacity(h * w * dim);
    for y in 0..h {
        for x in 0..w {
            for f in [f64::sin, f64::cos] {
                out.extend(omega.iter().map(|o| f(y as f64 * o) as f32));
            }
            for f in [f64::sin, f64::cos] {
                out.extend(omega.iter().map(|o| f(x as f64 * o) as f32));
            }
        }
    }
    Tensor::from_vec(out, (1, h * w, dim), dev)
}

struct RepVgg {
    a: ConvBn,
    b: ConvBn,
}

struct CspRep {
    conv1: ConvBn,
    conv2: ConvBn,
    blocks: Vec<RepVgg>,
}

impl CspRep {
    fn new(vb: VarBuilder) -> R<Self> {
        let blocks = (0..3)
            .map(|i| {
                let b = vb.pp(format!("bottlenecks.{i}"));
                Ok(RepVgg {
                    a: ConvBn::new(
                        D_MODEL,
                        D_MODEL,
                        3,
                        1,
                        1,
                        Act::None,
                        b.pp("conv1"),
                        ("conv", "norm"),
                    )?,
                    b: ConvBn::new(
                        D_MODEL,
                        D_MODEL,
                        1,
                        1,
                        0,
                        Act::None,
                        b.pp("conv2"),
                        ("conv", "norm"),
                    )?,
                })
            })
            .collect::<R<_>>()?;
        Ok(Self {
            conv1: encoder_conv(2 * D_MODEL, D_MODEL, 1, 1, Act::Silu, vb.pp("conv1"))?,
            conv2: encoder_conv(2 * D_MODEL, D_MODEL, 1, 1, Act::Silu, vb.pp("conv2"))?,
            blocks,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.conv1.forward(x)?;
        for b in &self.blocks {
            h = candle_nn::ops::silu(&(b.a.forward(&h)? + b.b.forward(&h)?)?)?;
        }
        h + self.conv2.forward(x)?
    }
}

struct HybridEncoder {
    aifi: AifiLayer,
    lateral: Vec<ConvBn>,
    fpn: Vec<CspRep>,
    down: Vec<ConvBn>,
    pan: Vec<CspRep>,
}

impl HybridEncoder {
    fn new(vb: VarBuilder) -> R<Self> {
        Ok(Self {
            aifi: AifiLayer::new(vb.pp("encoder.0.layers.0"))?,
            lateral: (0..2)
                .map(|i| {
                    encoder_conv(
                        D_MODEL,
                        D_MODEL,
                        1,
                        1,
                        Act::Silu,
                        vb.pp(format!("lateral_convs.{i}")),
                    )
                })
                .collect::<R<_>>()?,
            fpn: (0..2)
                .map(|i| CspRep::new(vb.pp(format!("fpn_blocks.{i}"))))
                .collect::<R<_>>()?,
            down: (0..2)
                .map(|i| {
                    encoder_conv(
                        D_MODEL,
                        D_MODEL,
                        3,
                        2,
                        Act::Silu,
                        vb.pp(format!("downsample_convs.{i}")),
                    )
                })
                .collect::<R<_>>()?,
            pan: (0..2)
                .map(|i| CspRep::new(vb.pp(format!("pan_blocks.{i}"))))
                .collect::<R<_>>()?,
        })
    }
    fn forward(&self, mut f: Vec<Tensor>) -> R<Vec<Tensor>> {
        // AIFI transformer on the coarsest level.
        let (b, c, h, w) = f[2].dims4()?;
        let pos = sincos_2d(h, w, c, f[2].device())?;
        let seq = f[2].flatten_from(2)?.transpose(1, 2)?;
        f[2] = self
            .aifi
            .forward(&seq, &pos)?
            .transpose(1, 2)?
            .reshape((b, c, h, w))?;
        // Top-down FPN.
        let mut fpn = vec![f[2].clone()];
        for i in 0..2 {
            let top = self.lateral[i].forward(fpn.last().unwrap())?;
            *fpn.last_mut().unwrap() = top.clone();
            let (_, _, th, tw) = top.dims4()?;
            let up = top.upsample_nearest2d(th * 2, tw * 2)?;
            let fused = Tensor::cat(&[&up, &f[1 - i]], 1)?;
            fpn.push(self.fpn[i].forward(&fused)?);
        }
        fpn.reverse();
        // Bottom-up PAN.
        let mut pan = vec![fpn[0].clone()];
        for i in 0..2 {
            let d = self.down[i].forward(pan.last().unwrap())?;
            pan.push(self.pan[i].forward(&Tensor::cat(&[&d, &fpn[i + 1]], 1)?)?);
        }
        Ok(pan)
    }
}

// ---------- decoder ----------

struct DecoderLayer {
    self_attn: SelfAttn,
    ln1: LayerNorm,
    offsets: Linear,
    weights: Linear,
    value: Linear,
    out: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ln3: LayerNorm,
}

impl DecoderLayer {
    fn new(vb: VarBuilder) -> R<Self> {
        let e = vb.pp("encoder_attn");
        Ok(Self {
            self_attn: SelfAttn::new(vb.pp("self_attn"))?,
            ln1: layer_norm(D_MODEL, 1e-5, vb.pp("self_attn_layer_norm"))?,
            offsets: linear(
                D_MODEL,
                HEADS * LEVELS * POINTS * 2,
                e.pp("sampling_offsets"),
            )?,
            weights: linear(D_MODEL, HEADS * LEVELS * POINTS, e.pp("attention_weights"))?,
            value: linear(D_MODEL, D_MODEL, e.pp("value_proj"))?,
            out: linear(D_MODEL, D_MODEL, e.pp("output_proj"))?,
            ln2: layer_norm(D_MODEL, 1e-5, vb.pp("encoder_attn_layer_norm"))?,
            fc1: linear(D_MODEL, 1024, vb.pp("fc1"))?,
            fc2: linear(1024, D_MODEL, vb.pp("fc2"))?,
            ln3: layer_norm(D_MODEL, 1e-5, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        pos: &Tensor,
        refs: &[[f32; 4]],
        memory: &Tensor,
        shapes: &[(usize, usize)],
    ) -> R<Tensor> {
        let x = self.ln1.forward(&(x + self.self_attn.forward(x, pos)?)?)?;
        let q = x.broadcast_add(pos)?;
        let offsets: Vec<f32> = self.offsets.forward(&q)?.flatten_all()?.to_vec1()?;
        let weights = self
            .weights
            .forward(&q)?
            .reshape((1, QUERIES, HEADS, LEVELS * POINTS))?;
        let weights: Vec<f32> = candle_nn::ops::softmax_last_dim(&weights)?
            .flatten_all()?
            .to_vec1()?;
        let value: Vec<f32> = self.value.forward(memory)?.flatten_all()?.to_vec1()?;
        let sampled = deformable_sample(&value, shapes, refs, &offsets, &weights);
        let sampled = Tensor::from_vec(sampled, (1, QUERIES, D_MODEL), x.device())?;
        let x = self.ln2.forward(&(x + self.out.forward(&sampled)?)?)?;
        let h = self.fc2.forward(&self.fc1.forward(&x)?.relu()?)?;
        self.ln3.forward(&(x + h)?)
    }
}

/// Multi-scale deformable attention core (RT-DETR v2, "default" method): bilinear
/// sampling with zero padding, `align_corners = false`, weighted per head.
/// `value` is (S, HEADS, DH) row-major; result is (QUERIES, HEADS * DH).
fn deformable_sample(
    value: &[f32],
    shapes: &[(usize, usize)],
    refs: &[[f32; 4]],
    offsets: &[f32],
    weights: &[f32],
) -> Vec<f32> {
    let dh = D_MODEL / HEADS;
    let mut starts = vec![0usize];
    for (h, w) in shapes {
        starts.push(starts.last().unwrap() + h * w);
    }
    let mut out = vec![0f32; QUERIES * D_MODEL];
    let scale = 1.0 / POINTS as f32;
    for q in 0..QUERIES {
        let [cx, cy, bw, bh] = refs[q];
        for head in 0..HEADS {
            let o = &mut out[q * D_MODEL + head * dh..q * D_MODEL + (head + 1) * dh];
            for lp in 0..LEVELS * POINTS {
                let level = lp / POINTS;
                let (h, w) = shapes[level];
                let base = ((q * HEADS + head) * LEVELS * POINTS + lp) * 2;
                let lx = cx + offsets[base] * scale * bw * OFFSET_SCALE;
                let ly = cy + offsets[base + 1] * scale * bh * OFFSET_SCALE;
                let aw = weights[(q * HEADS + head) * LEVELS * POINTS + lp];
                // grid_sample, align_corners = false: pixel = loc * size - 0.5
                let px = lx * w as f32 - 0.5;
                let py = ly * h as f32 - 0.5;
                let (x0, y0) = (px.floor(), py.floor());
                let (fx, fy) = (px - x0, py - y0);
                for (dy, wy) in [(0i64, 1.0 - fy), (1, fy)] {
                    for (dx, wx) in [(0i64, 1.0 - fx), (1, fx)] {
                        let (xi, yi) = (x0 as i64 + dx, y0 as i64 + dy);
                        if xi < 0 || yi < 0 || xi >= w as i64 || yi >= h as i64 {
                            continue;
                        }
                        let s = starts[level] + yi as usize * w + xi as usize;
                        let v = &value[(s * HEADS + head) * dh..(s * HEADS + head + 1) * dh];
                        let k = aw * wx * wy;
                        for d in 0..dh {
                            o[d] += k * v[d];
                        }
                    }
                }
            }
        }
    }
    out
}

fn inverse_sigmoid(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    (x.max(1e-5) / (1.0 - x).max(1e-5)).ln()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct ComicDetector {
    backbone: Backbone,
    enc_proj: Vec<ConvBn>,
    encoder: HybridEncoder,
    dec_proj: Vec<ConvBn>,
    enc_output: Linear,
    enc_output_ln: LayerNorm,
    enc_score: Linear,
    enc_bbox: Mlp,
    query_pos: Mlp,
    layers: Vec<DecoderLayer>,
    class_embed: Vec<Linear>,
    bbox_embed: Vec<Mlp>,
    device: Device,
}

fn proj(cin: usize, vb: VarBuilder) -> R<ConvBn> {
    ConvBn::new(cin, D_MODEL, 1, 1, 0, Act::None, vb, ("0", "1"))
}

impl ComicDetector {
    /// `dir` holds `model.safetensors` from ogkalu/comic-text-and-bubble-detector.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                device,
            )?
        };
        let m = vb.pp("model");
        let d = m.pp("decoder");
        Ok(Self {
            backbone: Backbone::new(m.pp("backbone.model"))?,
            enc_proj: [512, 1024, 2048]
                .iter()
                .enumerate()
                .map(|(i, &c)| proj(c, m.pp(format!("encoder_input_proj.{i}"))))
                .collect::<R<_>>()?,
            encoder: HybridEncoder::new(m.pp("encoder"))?,
            dec_proj: (0..3)
                .map(|i| proj(D_MODEL, m.pp(format!("decoder_input_proj.{i}"))))
                .collect::<R<_>>()?,
            enc_output: linear(D_MODEL, D_MODEL, m.pp("enc_output.0"))?,
            enc_output_ln: layer_norm(D_MODEL, 1e-5, m.pp("enc_output.1"))?,
            enc_score: linear(D_MODEL, CLASSES, m.pp("enc_score_head"))?,
            enc_bbox: Mlp::new(&[D_MODEL, D_MODEL, D_MODEL, 4], m.pp("enc_bbox_head"))?,
            query_pos: Mlp::new(&[4, 2 * D_MODEL, D_MODEL], d.pp("query_pos_head"))?,
            layers: (0..DEC_LAYERS)
                .map(|i| DecoderLayer::new(d.pp(format!("layers.{i}"))))
                .collect::<R<_>>()?,
            class_embed: (0..DEC_LAYERS)
                .map(|i| linear(D_MODEL, CLASSES, d.pp(format!("class_embed.{i}"))))
                .collect::<R<_>>()?,
            bbox_embed: (0..DEC_LAYERS)
                .map(|i| {
                    Mlp::new(
                        &[D_MODEL, D_MODEL, D_MODEL, 4],
                        d.pp(format!("bbox_embed.{i}")),
                    )
                })
                .collect::<R<_>>()?,
            device: device.clone(),
        })
    }

    /// 640x640 RGB in [0, 1], NCHW.
    pub fn preprocess(&self, img: &DynamicImage) -> R<Tensor> {
        let rgb = image::imageops::resize(
            &img.to_rgb8(),
            INPUT as u32,
            INPUT as u32,
            FilterType::Triangle,
        );
        let data: Vec<f32> = rgb.as_raw().iter().map(|&p| p as f32 / 255.0).collect();
        Tensor::from_vec(data, (INPUT, INPUT, 3), &self.device)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()
    }

    /// Raw model output: logits (QUERIES, CLASSES) and boxes (QUERIES, 4) as cx, cy, w, h in [0, 1].
    pub fn forward(&self, pixels: &Tensor) -> R<(Vec<[f32; CLASSES]>, Vec<[f32; 4]>)> {
        let feats = self.backbone.forward(pixels)?;
        let proj: Vec<Tensor> = feats
            .iter()
            .zip(&self.enc_proj)
            .map(|(f, p)| p.forward(f))
            .collect::<R<_>>()?;
        let enc = self.encoder.forward(proj)?;
        let mut shapes = vec![];
        let mut flat = vec![];
        for (f, p) in enc.iter().zip(&self.dec_proj) {
            let s = p.forward(f)?;
            let (_, _, h, w) = s.dims4()?;
            shapes.push((h, w));
            flat.push(s.flatten_from(2)?.transpose(1, 2)?);
        }
        let memory = Tensor::cat(&flat, 1)?; // (1, S, 256)
        let s_len = memory.dim(1)?;

        // Anchors and valid mask.
        let mut anchors = Vec::with_capacity(s_len * 4);
        let mut valid = Vec::with_capacity(s_len);
        for (lvl, &(h, w)) in shapes.iter().enumerate() {
            let wh = 0.05 * 2f32.powi(lvl as i32);
            for y in 0..h {
                for x in 0..w {
                    let a = [
                        (x as f32 + 0.5) / w as f32,
                        (y as f32 + 0.5) / h as f32,
                        wh,
                        wh,
                    ];
                    let ok = a.iter().all(|&v| v > 0.01 && v < 0.99);
                    valid.push(if ok { 1f32 } else { 0.0 });
                    for v in a {
                        anchors.push(if ok { (v / (1.0 - v)).ln() } else { f32::MAX });
                    }
                }
            }
        }
        let valid_t = Tensor::from_vec(valid, (1, s_len, 1), &self.device)?;
        let out_mem = self
            .enc_output_ln
            .forward(&self.enc_output.forward(&memory.broadcast_mul(&valid_t)?)?)?;
        let scores: Vec<f32> = self
            .enc_score
            .forward(&out_mem)?
            .max(D::Minus1)?
            .flatten_all()?
            .to_vec1()?;
        let coords: Vec<f32> = self.enc_bbox.forward(&out_mem)?.flatten_all()?.to_vec1()?;

        let mut order: Vec<usize> = (0..s_len).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        order.truncate(QUERIES);
        let idx = Tensor::new(
            order.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            &self.device,
        )?;
        let mut hidden = out_mem.squeeze(0)?.index_select(&idx, 0)?.unsqueeze(0)?;
        let mut refs: Vec<[f32; 4]> = order
            .iter()
            .map(|&i| std::array::from_fn(|k| sigmoid(coords[i * 4 + k] + anchors[i * 4 + k])))
            .collect();

        let mut logits = Tensor::zeros((1, QUERIES, CLASSES), DType::F32, &self.device)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let ref_t = Tensor::from_vec(
                refs.iter().flatten().copied().collect::<Vec<_>>(),
                (1, QUERIES, 4),
                &self.device,
            )?;
            let pos = self.query_pos.forward(&ref_t)?;
            hidden = layer.forward(&hidden, &pos, &refs, &memory, &shapes)?;
            let delta: Vec<f32> = self.bbox_embed[i]
                .forward(&hidden)?
                .flatten_all()?
                .to_vec1()?;
            for (q, r) in refs.iter_mut().enumerate() {
                for k in 0..4 {
                    r[k] = sigmoid(delta[q * 4 + k] + inverse_sigmoid(r[k]));
                }
            }
            if i == DEC_LAYERS - 1 {
                logits = self.class_embed[i].forward(&hidden)?;
            }
        }
        let l: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let logits = (0..QUERIES)
            .map(|q| std::array::from_fn(|c| l[q * CLASSES + c]))
            .collect();
        Ok((logits, refs))
    }

    /// Detections above `threshold`, boxes in original image pixels.
    pub fn detect(&self, img: &DynamicImage, threshold: f32) -> Result<Vec<Detection>> {
        let (logits, boxes) = self.forward(&self.preprocess(img)?)?;
        Ok(postprocess(
            &logits,
            &boxes,
            img.width() as f32,
            img.height() as f32,
            threshold,
        ))
    }
}

pub fn postprocess(
    logits: &[[f32; CLASSES]],
    boxes: &[[f32; 4]],
    w: f32,
    h: f32,
    threshold: f32,
) -> Vec<Detection> {
    let mut all: Vec<(f32, usize, usize)> = logits
        .iter()
        .enumerate()
        .flat_map(|(q, l)| l.iter().enumerate().map(move |(c, &v)| (sigmoid(v), q, c)))
        .collect();
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    all.into_iter()
        .take(QUERIES)
        .filter(|(s, _, _)| *s >= threshold)
        .map(|(score, q, c)| {
            let [cx, cy, bw, bh] = boxes[q];
            Detection {
                class: Class::from_index(c),
                score,
                bbox: [
                    (cx - bw / 2.0) * w,
                    (cy - bh / 2.0) * h,
                    (cx + bw / 2.0) * w,
                    (cy + bh / 2.0) * h,
                ],
            }
        })
        .collect()
}
