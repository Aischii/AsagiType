# AsagiType 2000

AsagiType 2000 is a manga scanlation editor written in pure Rust, with a boxy Windows 2000 look and asagi-teal title bars. It is built for AsagiScans.

It runs as one native Windows app, built on egui/eframe and wgpu, like [PhotoCraft](https://github.com/storytold/photocraft). It needs no web view and no Python.

## Status: v0.1 skeleton

**Working:**
- Import a chapter, either from a folder of pages or from a scanlation workspace (`series/<S>/chapters/<C>/source`), into a self-contained `.at2k` project.
- Page editor: draw balloon boxes, type the source text, and set the role and size.
- Translate a page (F6) with any OpenAI-compatible provider: DeepSeek (`deepseek-chat`, the default), OpenRouter, Ollama or LM Studio. API keys are kept in Windows Credential Manager.
- Lettering preview through cosmic-text and tiny-skia. Text is fitted to the box at 168 DPI with a 16 px minimum line height, and anything that overflows is outlined in red.
- Save (Ctrl+S). Help > About lists credits for every library, model, font and inspiration.

**Planned:** see [docs/design.md](docs/design.md).
- Detection and OCR with [candle](https://github.com/huggingface/candle): the ogkalu RT-DETR detector and manga-ocr.
- Solid-fill cleaning, then LaMa inpainting.
- JPG, CBZ and layered PSD export.
- Glossary editor and QC checks.

## Build

```powershell
cargo run --release -p asagitype
```

You need Rust 1.85 or newer on Windows 10 or 11. Comic fonts are loaded from the folder set in Tools > Settings.

## Layout

| Path | Contents |
| --- | --- |
| `apps/asagitype` | egui app: window, theme, canvas, dialogs, About |
| `crates/at2k` | `.at2k` project format: a zip holding `manifest.json` and the page images |
| `crates/provider` | OpenAI-compatible translation and key storage |
| `crates/render` | cosmic-text and tiny-skia lettering and fitting |

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. Third-party credits are in [CREDITS.md](CREDITS.md).
