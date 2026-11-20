//! manga-ocr (kha-white, Apache-2.0) in candle: a ViT encoder with a 2-layer BERT
//! decoder that cross-attends to it. Greedy decoding; on test crops it agrees with the
//! reference beam search in nearly all cases.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};
use image::{imageops::FilterType, DynamicImage};
use std::path::Path;

const HIDDEN: usize = 768;
const HEADS: usize = 12;
const IMAGE: usize = 224;
const PATCH: usize = 16;
const START: u32 = 2; // [CLS]
const EOS: u32 = 3; // [SEP]
const MAX_TOKENS: usize = 300;

fn multi_head(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let (b, tq, _) = q.dims3()?;
    let tk = k.dim(1)?;
    let dh = HIDDEN / HEADS;
    let split = |x: &Tensor, t| x.reshape((b, t, HEADS, dh))?.transpose(1, 2)?.contiguous();
    let (q, k, v) = (split(q, tq)?, split(k, tk)?, split(v, tk)?);
    let mut scores = (q.matmul(&k.t()?)? / (dh as f64).sqrt())?;
    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }
    let p = candle_nn::ops::softmax_last_dim(&scores)?;
    p.matmul(&v)?.transpose(1, 2)?.reshape((b, tq, HIDDEN))
}

struct Qkv {
    q: Linear,
    k: Linear,
    v: Linear,
}

impl Qkv {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            q: linear(HIDDEN, HIDDEN, vb.pp("query"))?,
            k: linear(HIDDEN, HIDDEN, vb.pp("key"))?,
            v: linear(HIDDEN, HIDDEN, vb.pp("value"))?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        kv: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        multi_head(
            &self.q.forward(x)?,
            &self.k.forward(kv)?,
            &self.v.forward(kv)?,
            mask,
        )
    }
}

// ---------- ViT encoder (pre-LayerNorm) ----------

struct VitLayer {
    ln1: LayerNorm,
    attn: Qkv,
    attn_out: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl VitLayer {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            ln1: layer_norm(HIDDEN, 1e-12, vb.pp("layernorm_before"))?,
            attn: Qkv::new(vb.pp("attention.attention"))?,
            attn_out: linear(HIDDEN, HIDDEN, vb.pp("attention.output.dense"))?,
            ln2: layer_norm(HIDDEN, 1e-12, vb.pp("layernorm_after"))?,
            fc1: linear(HIDDEN, 3072, vb.pp("intermediate.dense"))?,
            fc2: linear(3072, HIDDEN, vb.pp("output.dense"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let x = (x + self.attn_out.forward(&self.attn.forward(&h, &h, None)?)?)?;
        let h = self.fc1.forward(&self.ln2.forward(&x)?)?.gelu_erf()?;
        x + self.fc2.forward(&h)?
    }
}

struct Vit {
    patch: Conv2d,
    cls: Tensor,
    pos: Tensor,
    layers: Vec<VitLayer>,
    ln: LayerNorm,
}

impl Vit {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig {
            stride: PATCH,
            ..Default::default()
        };
        Ok(Self {
            patch: conv2d(
                3,
                HIDDEN,
                PATCH,
                cfg,
                vb.pp("embeddings.patch_embeddings.projection"),
            )?,
            cls: vb.get((1, 1, HIDDEN), "embeddings.cls_token")?,
            pos: vb.get((1, 197, HIDDEN), "embeddings.position_embeddings")?,
            layers: (0..12)
                .map(|i| VitLayer::new(vb.pp(format!("encoder.layer.{i}"))))
                .collect::<candle_core::Result<_>>()?,
            ln: layer_norm(HIDDEN, 1e-12, vb.pp("layernorm"))?,
        })
    }
    fn forward(&self, pixels: &Tensor) -> candle_core::Result<Tensor> {
        let b = pixels.dim(0)?;
        let x = self
            .patch
            .forward(pixels)?
            .flatten_from(2)?
            .transpose(1, 2)?; // (b, 196, 768)
        let mut x =
            Tensor::cat(&[&self.cls.expand((b, 1, HIDDEN))?, &x], 1)?.broadcast_add(&self.pos)?;
        for l in &self.layers {
            x = l.forward(&x)?;
        }
        self.ln.forward(&x)
    }
}

// ---------- BERT decoder (post-LayerNorm, causal self-attention + cross-attention) ----------

struct BertLayer {
    self_attn: Qkv,
    self_out: Linear,
    self_ln: LayerNorm,
    cross_attn: Qkv,
    cross_out: Linear,
    cross_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    out_ln: LayerNorm,
}

impl BertLayer {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            self_attn: Qkv::new(vb.pp("attention.self"))?,
            self_out: linear(HIDDEN, HIDDEN, vb.pp("attention.output.dense"))?,
            self_ln: layer_norm(HIDDEN, 1e-12, vb.pp("attention.output.LayerNorm"))?,
            cross_attn: Qkv::new(vb.pp("crossattention.self"))?,
            cross_out: linear(HIDDEN, HIDDEN, vb.pp("crossattention.output.dense"))?,
            cross_ln: layer_norm(HIDDEN, 1e-12, vb.pp("crossattention.output.LayerNorm"))?,
            fc1: linear(HIDDEN, 3072, vb.pp("intermediate.dense"))?,
            fc2: linear(3072, HIDDEN, vb.pp("output.dense"))?,
            out_ln: layer_norm(HIDDEN, 1e-12, vb.pp("output.LayerNorm"))?,
        })
    }
    fn forward(&self, x: &Tensor, enc: &Tensor, causal: &Tensor) -> candle_core::Result<Tensor> {
        let a = self.self_ln.forward(
            &(x + self
                .self_out
                .forward(&self.self_attn.forward(x, x, Some(causal))?)?)?,
        )?;
        let c = self.cross_ln.forward(
            &(&a + self
                .cross_out
                .forward(&self.cross_attn.forward(&a, enc, None)?)?)?,
        )?;
        let h = self.fc1.forward(&c)?.gelu_erf()?;
        self.out_ln.forward(&(c + self.fc2.forward(&h)?)?)
    }
}

struct BertDecoder {
    word: Tensor,
    pos: Tensor,
    token_type: Tensor,
    emb_ln: LayerNorm,
    layers: Vec<BertLayer>,
    head_dense: Linear,
    head_ln: LayerNorm,
    head_out: Linear,
}

impl BertDecoder {
    fn new(vb: VarBuilder, vocab: usize) -> candle_core::Result<Self> {
        let e = vb.pp("bert.embeddings");
        let p = vb.pp("cls.predictions");
        Ok(Self {
            word: e.get((vocab, HIDDEN), "word_embeddings.weight")?,
            pos: e.get((512, HIDDEN), "position_embeddings.weight")?,
            token_type: e.get((2, HIDDEN), "token_type_embeddings.weight")?.get(0)?,
            emb_ln: layer_norm(HIDDEN, 1e-12, e.pp("LayerNorm"))?,
            layers: (0..2)
                .map(|i| BertLayer::new(vb.pp(format!("bert.encoder.layer.{i}"))))
                .collect::<candle_core::Result<_>>()?,
            head_dense: linear(HIDDEN, HIDDEN, p.pp("transform.dense"))?,
            head_ln: layer_norm(HIDDEN, 1e-12, p.pp("transform.LayerNorm"))?,
            head_out: linear(HIDDEN, vocab, p.pp("decoder"))?,
        })
    }

    /// Logits for the last position.
    fn next_logits(&self, ids: &[u32], enc: &Tensor) -> candle_core::Result<Tensor> {
        let dev = enc.device();
        let t = ids.len();
        let idx = Tensor::new(ids, dev)?;
        let x = (self.word.index_select(&idx, 0)? + self.pos.narrow(0, 0, t)?)?
            .broadcast_add(&self.token_type)?;
        let mut x = self.emb_ln.forward(&x.unsqueeze(0)?)?;
        let mask: Vec<f32> = (0..t * t)
            .map(|i| {
                if i % t > i / t {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let causal = Tensor::from_vec(mask, (t, t), dev)?;
        for l in &self.layers {
            x = l.forward(&x, enc, &causal)?;
        }
        let last = x.narrow(1, t - 1, 1)?.squeeze(1)?;
        let h = self
            .head_ln
            .forward(&self.head_dense.forward(&last)?.gelu_erf()?)?;
        self.head_out.forward(&h)
    }
}

pub struct MangaOcr {
    encoder: Vit,
    decoder: BertDecoder,
    vocab: Vec<String>,
    device: Device,
}

impl MangaOcr {
    /// `dir` holds `model.safetensors` and `vocab.txt` from kha-white/manga-ocr-base.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let vocab: Vec<String> = std::fs::read_to_string(dir.join("vocab.txt"))
            .context("manga-ocr vocab.txt")?
            .lines()
            .map(str::to_string)
            .collect();
        let weights = dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
        Ok(Self {
            encoder: Vit::new(vb.pp("encoder"))?,
            decoder: BertDecoder::new(vb.pp("decoder"), vocab.len())?,
            vocab,
            device: device.clone(),
        })
    }

    /// Grayscale, 224x224 bilinear, scaled to [-1, 1]: the manga-ocr preprocessing.
    fn pixels(&self, img: &DynamicImage) -> candle_core::Result<Tensor> {
        let g = img.to_luma8();
        let g = image::imageops::resize(&g, IMAGE as u32, IMAGE as u32, FilterType::Triangle);
        let data: Vec<f32> = g.as_raw().iter().map(|&p| p as f32 / 127.5 - 1.0).collect();
        let t = Tensor::from_vec(data, (1, 1, IMAGE, IMAGE), &self.device)?;
        t.repeat((1, 3, 1, 1))
    }

    pub fn token_ids(&self, img: &DynamicImage) -> Result<Vec<u32>> {
        let enc = self.encoder.forward(&self.pixels(img)?)?;
        let mut ids = vec![START];
        while ids.len() < MAX_TOKENS {
            let next = self
                .decoder
                .next_logits(&ids, &enc)?
                .argmax(D::Minus1)?
                .squeeze(0)?
                .to_scalar::<u32>()?;
            ids.push(next);
            if next == EOS {
                break;
            }
        }
        Ok(ids)
    }

    pub fn recognize(&self, img: &DynamicImage) -> Result<String> {
        let ids = self.token_ids(img)?;
        let text: String = ids
            .iter()
            .filter_map(|&i| self.vocab.get(i as usize))
            .filter(|t| !(t.starts_with('[') && t.ends_with(']')))
            .map(|t| t.trim_start_matches("##"))
            .collect();
        Ok(post_process(&text))
    }
}

/// manga-ocr post-processing: no spaces, '…' as '...', runs of ・/. as dots,
/// half-width ASCII and digits to full width.
pub fn post_process(text: &str) -> String {
    let s: String = text
        .split_whitespace()
        .collect::<String>()
        .replace('…', "...");
    let mut out = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '・' || chars[i] == '.' {
            let start = i;
            while i < chars.len() && (chars[i] == '・' || chars[i] == '.') {
                i += 1;
            }
            if i - start >= 2 {
                out.extend(std::iter::repeat_n('.', i - start));
            } else {
                out.push(chars[start]);
            }
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out.chars()
        .map(|c| match c {
            '!'..='~' => char::from_u32(c as u32 - 0x21 + 0xFF01).unwrap_or(c),
            _ => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::post_process;

    #[test]
    fn post_processing_matches_manga_ocr() {
        assert_eq!(post_process("あ い…"), "あい．．．");
        assert_eq!(post_process("A1・x"), "Ａ１・ｘ");
        assert_eq!(post_process("ん・・・"), "ん．．．");
    }
}
