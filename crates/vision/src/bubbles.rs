//! Speech-balloon instance segmentation: a YOLO11n-seg network written from scratch in
//! candle, loading the Manga109 balloon weights (huyvux3005/manga109-segmentation-bubble,
//! SafeTensors conversion). The weights are not bundled with AsagiType: they were trained on
//! the academic-use Manga109 dataset, so the user supplies them in their models folder.
//!
//! Output: one mask per balloon, used to keep cleaning inside the balloon.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{
    conv2d, conv2d_no_bias, conv_transpose2d, Conv2d, Conv2dConfig, ConvTranspose2d,
    ConvTranspose2dConfig, VarBuilder,
};
use image::{imageops::FilterType, GrayImage, RgbImage};
use std::path::Path;

type R<T> = candle_core::Result<T>;

pub const INPUT: usize = 1600;
const REG_MAX: usize = 16;
const MASKS: usize = 32;
const STRIDES: [usize; 3] = [8, 16, 32];
const BN_EPS: f64 = 1e-3;
const CONF: f32 = 0.25;
const IOU: f32 = 0.7;
const LETTERBOX: u8 = 114;

// ---------- building blocks ----------

struct Conv {
    conv: Conv2d,
    scale: Tensor,
    shift: Tensor,
    act: bool,
}

impl Conv {
    fn new(
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        groups: usize,
        act: bool,
        vb: VarBuilder,
    ) -> R<Self> {
        let cfg = Conv2dConfig {
            padding: k / 2,
            stride,
            groups,
            ..Default::default()
        };
        let conv = conv2d_no_bias(cin, cout, k, cfg, vb.pp("conv"))?;
        let bn = vb.pp("bn");
        let (w, b) = (bn.get(cout, "weight")?, bn.get(cout, "bias")?);
        let (mean, var) = (bn.get(cout, "running_mean")?, bn.get(cout, "running_var")?);
        let scale = (w / (var + BN_EPS)?.sqrt()?)?;
        let shift = (b - (&mean * &scale)?)?;
        Ok(Self {
            conv,
            scale: scale.reshape((1, cout, 1, 1))?,
            shift: shift.reshape((1, cout, 1, 1))?,
            act,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let y = self
            .conv
            .forward(x)?
            .broadcast_mul(&self.scale)?
            .broadcast_add(&self.shift)?;
        if self.act {
            candle_nn::ops::silu(&y)
        } else {
            Ok(y)
        }
    }
}

fn conv(cin: usize, cout: usize, k: usize, s: usize, vb: VarBuilder) -> R<Conv> {
    Conv::new(cin, cout, k, s, 1, true, vb)
}

struct Bottleneck {
    cv1: Conv,
    cv2: Conv,
    add: bool,
}

impl Bottleneck {
    fn new(c1: usize, c2: usize, shortcut: bool, e: f32, vb: VarBuilder) -> R<Self> {
        let c = (c2 as f32 * e) as usize;
        Ok(Self {
            cv1: conv(c1, c, 3, 1, vb.pp("cv1"))?,
            cv2: conv(c, c2, 3, 1, vb.pp("cv2"))?,
            add: shortcut && c1 == c2,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let y = self.cv2.forward(&self.cv1.forward(x)?)?;
        if self.add {
            x + y
        } else {
            Ok(y)
        }
    }
}

/// CSP block with two stacked bottlenecks (C3 with 3x3 kernels).
struct C3k {
    cv1: Conv,
    cv2: Conv,
    cv3: Conv,
    m: Vec<Bottleneck>,
}

impl C3k {
    fn new(c1: usize, c2: usize, n: usize, vb: VarBuilder) -> R<Self> {
        let c = c2 / 2;
        Ok(Self {
            cv1: conv(c1, c, 1, 1, vb.pp("cv1"))?,
            cv2: conv(c1, c, 1, 1, vb.pp("cv2"))?,
            cv3: conv(2 * c, c2, 1, 1, vb.pp("cv3"))?,
            m: (0..n)
                .map(|i| Bottleneck::new(c, c, true, 1.0, vb.pp(format!("m.{i}"))))
                .collect::<R<_>>()?,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut y = self.cv1.forward(x)?;
        for b in &self.m {
            y = b.forward(&y)?;
        }
        self.cv3
            .forward(&Tensor::cat(&[&y, &self.cv2.forward(x)?], 1)?)
    }
}

enum Inner {
    B(Bottleneck),
    K(C3k),
}

/// Split-and-concatenate block (C2f layout) whose inner blocks are bottlenecks or C3k.
struct C3k2 {
    cv1: Conv,
    cv2: Conv,
    c: usize,
    m: Vec<Inner>,
}

impl C3k2 {
    fn new(c1: usize, c2: usize, n: usize, c3k: bool, e: f32, vb: VarBuilder) -> R<Self> {
        let c = (c2 as f32 * e) as usize;
        let m = (0..n)
            .map(|i| {
                let v = vb.pp(format!("m.{i}"));
                Ok(if c3k {
                    Inner::K(C3k::new(c, c, 2, v)?)
                } else {
                    Inner::B(Bottleneck::new(c, c, true, 0.5, v)?)
                })
            })
            .collect::<R<_>>()?;
        Ok(Self {
            cv1: conv(c1, 2 * c, 1, 1, vb.pp("cv1"))?,
            cv2: conv((2 + n) * c, c2, 1, 1, vb.pp("cv2"))?,
            c,
            m,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let y = self.cv1.forward(x)?;
        let mut parts = vec![y.narrow(1, 0, self.c)?, y.narrow(1, self.c, self.c)?];
        for m in &self.m {
            let last = parts.last().unwrap();
            parts.push(match m {
                Inner::B(b) => b.forward(last)?,
                Inner::K(k) => k.forward(last)?,
            });
        }
        self.cv2.forward(&Tensor::cat(&parts, 1)?)
    }
}

struct Sppf {
    cv1: Conv,
    cv2: Conv,
}

impl Sppf {
    fn new(c1: usize, c2: usize, vb: VarBuilder) -> R<Self> {
        let c = c1 / 2;
        Ok(Self {
            cv1: conv(c1, c, 1, 1, vb.pp("cv1"))?,
            cv2: conv(4 * c, c2, 1, 1, vb.pp("cv2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut ys = vec![self.cv1.forward(x)?];
        for _ in 0..3 {
            // 5x5 max pool, stride 1, padding 2 (pad with -inf).
            let last = ys.last().unwrap();
            let padded = last.pad_with_same(2, 2, 2)?.pad_with_same(3, 2, 2)?;
            ys.push(padded.max_pool2d_with_stride(5, 1)?);
        }
        self.cv2.forward(&Tensor::cat(&ys, 1)?)
    }
}

/// Multi-head self-attention over the feature map with a depthwise positional conv.
struct Attention {
    qkv: Conv,
    proj: Conv,
    pe: Conv,
    heads: usize,
    key_dim: usize,
    head_dim: usize,
}

struct PsaBlock {
    attn: Attention,
    ffn1: Conv,
    ffn2: Conv,
}

struct C2Psa {
    cv1: Conv,
    cv2: Conv,
    c: usize,
    blocks: Vec<PsaBlock>,
}

impl C2Psa {
    fn new(c1: usize, n: usize, vb: VarBuilder) -> R<Self> {
        let c = c1 / 2;
        let heads = c / 64;
        let head_dim = c / heads;
        let key_dim = head_dim / 2;
        let h = c + 2 * key_dim * heads;
        let blocks = (0..n)
            .map(|i| {
                let b = vb.pp(format!("m.{i}"));
                let a = b.pp("attn");
                Ok(PsaBlock {
                    attn: Attention {
                        qkv: Conv::new(c, h, 1, 1, 1, false, a.pp("qkv"))?,
                        proj: Conv::new(c, c, 1, 1, 1, false, a.pp("proj"))?,
                        pe: Conv::new(c, c, 3, 1, c, false, a.pp("pe"))?,
                        heads,
                        key_dim,
                        head_dim,
                    },
                    ffn1: conv(c, 2 * c, 1, 1, b.pp("ffn.0"))?,
                    ffn2: Conv::new(2 * c, c, 1, 1, 1, false, b.pp("ffn.1"))?,
                })
            })
            .collect::<R<_>>()?;
        Ok(Self {
            cv1: conv(c1, 2 * c, 1, 1, vb.pp("cv1"))?,
            cv2: conv(2 * c, c1, 1, 1, vb.pp("cv2"))?,
            c,
            blocks,
        })
    }

    fn attention(a: &Attention, x: &Tensor) -> R<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let n = h * w;
        let qkv = a
            .qkv
            .forward(x)?
            .reshape((b, a.heads, 2 * a.key_dim + a.head_dim, n))?;
        let q = qkv.narrow(2, 0, a.key_dim)?;
        let k = qkv.narrow(2, a.key_dim, a.key_dim)?;
        let v = qkv.narrow(2, 2 * a.key_dim, a.head_dim)?;
        let attn = (q.transpose(2, 3)?.contiguous()?.matmul(&k.contiguous()?)?
            * (a.key_dim as f64).powf(-0.5))?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = v
            .contiguous()?
            .matmul(&attn.transpose(2, 3)?.contiguous()?)?
            .reshape((b, c, h, w))?;
        let pe = a.pe.forward(&v.contiguous()?.reshape((b, c, h, w))?)?;
        a.proj.forward(&(out + pe)?)
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let y = self.cv1.forward(x)?;
        let a = y.narrow(1, 0, self.c)?;
        let mut bpart = y.narrow(1, self.c, self.c)?;
        for blk in &self.blocks {
            bpart = (&bpart + Self::attention(&blk.attn, &bpart)?)?;
            bpart = (&bpart + blk.ffn2.forward(&blk.ffn1.forward(&bpart)?)?)?;
        }
        self.cv2.forward(&Tensor::cat(&[&a, &bpart], 1)?)
    }
}

// ---------- segmentation head ----------

struct Branch {
    a: Conv,
    b: Conv,
    out: Conv2d,
}

impl Branch {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        self.out.forward(&self.b.forward(&self.a.forward(x)?)?)
    }
}

/// Class branch: two depthwise-separable blocks then a 1x1 conv.
struct ClsBranch {
    dw1: Conv,
    pw1: Conv,
    dw2: Conv,
    pw2: Conv,
    out: Conv2d,
}

impl ClsBranch {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let y = self.pw1.forward(&self.dw1.forward(x)?)?;
        let y = self.pw2.forward(&self.dw2.forward(&y)?)?;
        self.out.forward(&y)
    }
}

struct Proto {
    cv1: Conv,
    up: ConvTranspose2d,
    cv2: Conv,
    cv3: Conv,
}

pub struct BubbleSegmenter {
    l0: Conv,
    l1: Conv,
    l2: C3k2,
    l3: Conv,
    l4: C3k2,
    l5: Conv,
    l6: C3k2,
    l7: Conv,
    l8: C3k2,
    l9: Sppf,
    l10: C2Psa,
    l13: C3k2,
    l16: C3k2,
    l17: Conv,
    l19: C3k2,
    l20: Conv,
    l22: C3k2,
    box_branch: Vec<Branch>,
    cls_branch: Vec<ClsBranch>,
    mask_branch: Vec<Branch>,
    proto: Proto,
    device: Device,
}

/// One detected balloon, in original image pixels.
#[derive(Clone, Debug)]
pub struct Balloon {
    pub bbox: [i32; 4],
    pub score: f32,
    /// Mask covering `bbox`: 255 inside the balloon.
    pub mask: GrayImage,
}

impl Balloon {
    pub fn contains(&self, x: i32, y: i32) -> bool {
        let [x0, y0, x1, y1] = self.bbox;
        x >= x0
            && y >= y0
            && x < x1
            && y < y1
            && self.mask.get_pixel((x - x0) as u32, (y - y0) as u32)[0] > 0
    }
}

impl BubbleSegmenter {
    /// `dir` holds `model.safetensors` (Manga109 YOLO11n-seg balloon weights).
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                device,
            )?
        };
        let m = |i: usize| vb.pp(format!("model.{i}"));
        let h = m(23);
        let chs = [64usize, 128, 256];
        let box_branch = (0..3)
            .map(|i| {
                let v = h.pp(format!("cv2.{i}"));
                Ok(Branch {
                    a: conv(chs[i], 64, 3, 1, v.pp("0"))?,
                    b: conv(64, 64, 3, 1, v.pp("1"))?,
                    out: conv2d(64, 4 * REG_MAX, 1, Default::default(), v.pp("2"))?,
                })
            })
            .collect::<R<_>>()?;
        let cls_branch = (0..3)
            .map(|i| {
                let v = h.pp(format!("cv3.{i}"));
                Ok(ClsBranch {
                    dw1: Conv::new(chs[i], chs[i], 3, 1, chs[i], true, v.pp("0.0"))?,
                    pw1: conv(chs[i], 64, 1, 1, v.pp("0.1"))?,
                    dw2: Conv::new(64, 64, 3, 1, 64, true, v.pp("1.0"))?,
                    pw2: conv(64, 64, 1, 1, v.pp("1.1"))?,
                    out: conv2d(64, 1, 1, Default::default(), v.pp("2"))?,
                })
            })
            .collect::<R<_>>()?;
        let mask_branch = (0..3)
            .map(|i| {
                let v = h.pp(format!("cv4.{i}"));
                Ok(Branch {
                    a: conv(chs[i], MASKS, 3, 1, v.pp("0"))?,
                    b: conv(MASKS, MASKS, 3, 1, v.pp("1"))?,
                    out: conv2d(MASKS, MASKS, 1, Default::default(), v.pp("2"))?,
                })
            })
            .collect::<R<_>>()?;
        let p = h.pp("proto");
        let proto = Proto {
            cv1: conv(64, 64, 3, 1, p.pp("cv1"))?,
            up: conv_transpose2d(
                64,
                64,
                2,
                ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                p.pp("upsample"),
            )?,
            cv2: conv(64, 64, 3, 1, p.pp("cv2"))?,
            cv3: conv(64, MASKS, 1, 1, p.pp("cv3"))?,
        };
        Ok(Self {
            l0: conv(3, 16, 3, 2, m(0))?,
            l1: conv(16, 32, 3, 2, m(1))?,
            l2: C3k2::new(32, 64, 1, false, 0.25, m(2))?,
            l3: conv(64, 64, 3, 2, m(3))?,
            l4: C3k2::new(64, 128, 1, false, 0.25, m(4))?,
            l5: conv(128, 128, 3, 2, m(5))?,
            l6: C3k2::new(128, 128, 1, true, 0.5, m(6))?,
            l7: conv(128, 256, 3, 2, m(7))?,
            l8: C3k2::new(256, 256, 1, true, 0.5, m(8))?,
            l9: Sppf::new(256, 256, m(9))?,
            l10: C2Psa::new(256, 1, m(10))?,
            l13: C3k2::new(384, 128, 1, false, 0.5, m(13))?,
            l16: C3k2::new(256, 64, 1, false, 0.5, m(16))?,
            l17: conv(64, 64, 3, 2, m(17))?,
            l19: C3k2::new(192, 128, 1, false, 0.5, m(19))?,
            l20: conv(128, 128, 3, 2, m(20))?,
            l22: C3k2::new(384, 256, 1, true, 0.5, m(22))?,
            box_branch,
            cls_branch,
            mask_branch,
            proto,
            device: device.clone(),
        })
    }

    /// Raw output: predictions (4 + 1 + 32, anchors) with boxes as cx, cy, w, h in input
    /// pixels and sigmoid class scores; and mask prototypes (32, H/4, W/4).
    pub fn forward(&self, x: &Tensor) -> R<(Tensor, Tensor)> {
        let up = |t: &Tensor| -> R<Tensor> {
            let (_, _, h, w) = t.dims4()?;
            t.upsample_nearest2d(h * 2, w * 2)
        };
        let x = self.l1.forward(&self.l0.forward(x)?)?;
        let x = self.l2.forward(&x)?;
        let x = self.l3.forward(&x)?;
        let p4 = self.l4.forward(&x)?;
        let x = self.l5.forward(&p4)?;
        let p6 = self.l6.forward(&x)?;
        let x = self.l7.forward(&p6)?;
        let x = self.l8.forward(&x)?;
        let x = self.l9.forward(&x)?;
        let p10 = self.l10.forward(&x)?;
        let p13 = self.l13.forward(&Tensor::cat(&[&up(&p10)?, &p6], 1)?)?;
        let p16 = self.l16.forward(&Tensor::cat(&[&up(&p13)?, &p4], 1)?)?;
        let p19 = self
            .l19
            .forward(&Tensor::cat(&[&self.l17.forward(&p16)?, &p13], 1)?)?;
        let p22 = self
            .l22
            .forward(&Tensor::cat(&[&self.l20.forward(&p19)?, &p10], 1)?)?;
        let feats = [p16, p19, p22];

        let proto = {
            let p = &self.proto;
            let y = p.up.forward(&p.cv1.forward(&feats[0])?)?;
            p.cv3.forward(&p.cv2.forward(&y)?)?
        };

        let (mut boxes, mut scores, mut coefs, mut anchors) = (vec![], vec![], vec![], vec![]);
        for (i, f) in feats.iter().enumerate() {
            let (_, _, h, w) = f.dims4()?;
            boxes.push(self.box_branch[i].forward(f)?.flatten_from(2)?);
            scores.push(self.cls_branch[i].forward(f)?.flatten_from(2)?);
            coefs.push(self.mask_branch[i].forward(f)?.flatten_from(2)?);
            for y in 0..h {
                for x in 0..w {
                    anchors.push((x as f32 + 0.5, y as f32 + 0.5, STRIDES[i] as f32));
                }
            }
        }
        let raw = Tensor::cat(&boxes, 2)?.squeeze(0)?; // (64, A)
        let a = raw.dim(1)?;
        // Distribution focal loss decode: softmax over 16 bins per side, expectation.
        let dist = candle_nn::ops::softmax(&raw.reshape((4, REG_MAX, a))?, 1)?;
        let bins = Tensor::arange(0f32, REG_MAX as f32, &self.device)?.reshape((1, REG_MAX, 1))?;
        let ltrb: Vec<Vec<f32>> = dist.broadcast_mul(&bins)?.sum(1)?.to_vec2()?;
        let mut xywh = vec![0f32; 4 * a];
        for (j, &(ax, ay, s)) in anchors.iter().enumerate() {
            let (l, t, r, b) = (ltrb[0][j], ltrb[1][j], ltrb[2][j], ltrb[3][j]);
            let (x1, y1, x2, y2) = (ax - l, ay - t, ax + r, ay + b);
            xywh[j] = (x1 + x2) / 2.0 * s;
            xywh[a + j] = (y1 + y2) / 2.0 * s;
            xywh[2 * a + j] = (x2 - x1) * s;
            xywh[3 * a + j] = (y2 - y1) * s;
        }
        let xywh = Tensor::from_vec(xywh, (4, a), &self.device)?;
        let cls = candle_nn::ops::sigmoid(&Tensor::cat(&scores, 2)?.squeeze(0)?)?;
        let mc = Tensor::cat(&coefs, 2)?.squeeze(0)?;
        Ok((Tensor::cat(&[&xywh, &cls, &mc], 0)?, proto.squeeze(0)?))
    }

    /// Letterbox to INPUT x INPUT (centred, grey 114); returns tensor, scale and padding.
    fn letterbox(&self, img: &RgbImage) -> R<(Tensor, f32, (u32, u32))> {
        let (w, h) = (img.width() as f32, img.height() as f32);
        let r = (INPUT as f32 / w).min(INPUT as f32 / h);
        let (nw, nh) = ((w * r).round() as u32, (h * r).round() as u32);
        let resized = image::imageops::resize(img, nw, nh, FilterType::Triangle);
        let (px, py) = ((INPUT as u32 - nw) / 2, (INPUT as u32 - nh) / 2);
        let mut canvas =
            RgbImage::from_pixel(INPUT as u32, INPUT as u32, image::Rgb([LETTERBOX; 3]));
        image::imageops::replace(&mut canvas, &resized, px as i64, py as i64);
        let data: Vec<f32> = canvas.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
        let t = Tensor::from_vec(data, (INPUT, INPUT, 3), &self.device)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?;
        Ok((t, r, (px, py)))
    }

    /// Balloons with masks, in original image pixels.
    pub fn segment(&self, img: &RgbImage) -> Result<Vec<Balloon>> {
        let (x, r, (px, py)) = self.letterbox(img)?;
        let (pred, proto) = self.forward(&x)?;
        Ok(postprocess(
            &pred,
            &proto,
            r,
            (px, py),
            (img.width(), img.height()),
        )?)
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let i = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    i / ua.max(1e-6)
}

/// Confidence filter, NMS, then masks: sigmoid(coefficients . prototypes), cropped to the
/// box, upsampled to the input size, thresholded at 0.5 and mapped back to the image.
pub fn postprocess(
    pred: &Tensor,
    proto: &Tensor,
    r: f32,
    pad: (u32, u32),
    size: (u32, u32),
) -> R<Vec<Balloon>> {
    let p: Vec<Vec<f32>> = pred.to_vec2()?;
    let a = p[0].len();
    let mut cands: Vec<(f32, [f32; 4], usize)> = (0..a)
        .filter(|&j| p[4][j] >= CONF)
        .map(|j| {
            let (cx, cy, w, h) = (p[0][j], p[1][j], p[2][j], p[3][j]);
            (
                p[4][j],
                [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                j,
            )
        })
        .collect();
    cands.sort_by(|x, y| y.0.total_cmp(&x.0));
    let mut kept: Vec<(f32, [f32; 4], usize)> = vec![];
    for c in cands {
        if kept.iter().all(|k| iou(&k.1, &c.1) <= IOU) {
            kept.push(c);
        }
    }
    let (nm, ph, pw) = proto.dims3()?;
    let protos = proto.reshape((nm, ph * pw))?;
    let scale = INPUT as f32 / pw as f32;
    let (iw, ih) = (size.0 as f32, size.1 as f32);
    let mut out = vec![];
    for (score, b, j) in kept {
        let coef: Vec<f32> = (0..MASKS).map(|k| p[5 + k][j]).collect();
        let coef = Tensor::from_vec(coef, (1, MASKS), proto.device())?;
        let m: Vec<f32> = candle_nn::ops::sigmoid(&coef.matmul(&protos)?)?
            .flatten_all()?
            .to_vec1()?;
        // Box in original pixels.
        let to_img = |v: f32, padv: u32, lim: f32| ((v - padv as f32) / r).clamp(0.0, lim);
        let bb = [
            to_img(b[0], pad.0, iw),
            to_img(b[1], pad.1, ih),
            to_img(b[2], pad.0, iw),
            to_img(b[3], pad.1, ih),
        ];
        let bbox = [
            bb[0].floor() as i32,
            bb[1].floor() as i32,
            bb[2].ceil() as i32,
            bb[3].ceil() as i32,
        ];
        let (bw, bh) = (
            (bbox[2] - bbox[0]).max(1) as u32,
            (bbox[3] - bbox[1]).max(1) as u32,
        );
        // Sample the prototype mask bilinearly at each image pixel inside the box.
        let mask = GrayImage::from_fn(bw, bh, |x, y| {
            let ix = bbox[0] as f32 + x as f32 + 0.5;
            let iy = bbox[1] as f32 + y as f32 + 0.5;
            let (inx, iny) = (ix * r + pad.0 as f32, iy * r + pad.1 as f32);
            if inx < b[0] || inx > b[2] || iny < b[1] || iny > b[3] {
                return image::Luma([0]);
            }
            let (mx, my) = (inx / scale - 0.5, iny / scale - 0.5);
            let (x0, y0) = (
                mx.floor().clamp(0.0, (pw - 1) as f32) as usize,
                my.floor().clamp(0.0, (ph - 1) as f32) as usize,
            );
            let (x1, y1) = ((x0 + 1).min(pw - 1), (y0 + 1).min(ph - 1));
            let (fx, fy) = (
                (mx - x0 as f32).clamp(0.0, 1.0),
                (my - y0 as f32).clamp(0.0, 1.0),
            );
            let v = m[y0 * pw + x0] * (1.0 - fx) * (1.0 - fy)
                + m[y0 * pw + x1] * fx * (1.0 - fy)
                + m[y1 * pw + x0] * (1.0 - fx) * fy
                + m[y1 * pw + x1] * fx * fy;
            image::Luma([if v > 0.5 { 255 } else { 0 }])
        });
        out.push(Balloon { bbox, score, mask });
    }
    Ok(out)
}

/// Unused-dimension helper kept for readability of the decode above.
#[allow(dead_code)]
fn last_dim(t: &Tensor) -> R<usize> {
    t.dim(D::Minus1)
}
