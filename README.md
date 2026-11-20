# AsagiType 2000

AsagiType 2000 is a manga scanlation editor written in pure Rust, with a boxy Windows 2000 look and asagi-teal title bars. It is built for AsagiScans.

It runs as one native Windows app, built on egui/eframe and wgpu, like [PhotoCraft](https://github.com/storytold/photocraft). It needs no web view and no Python.

## Status: v0.5

**Working:**
- Import a chapter, either from a folder of pages or from a scanlation workspace (`series/<S>/chapters/<C>/source`), into a self-contained `.at2k` project.
- Detect + OCR (F5) in pure Rust with [candle](https://github.com/huggingface/candle): the ogkalu RT-DETR v2 detector finds balloons and text, and manga-ocr reads the Japanese. Both run per page or for a whole chapter, on CPU. On test pages the output matches the Python models: the same 18 detections, and the same OCR tokens on 14 of 15 crops. A page takes about 5 seconds.
- Page editor: Original and Translated panes side by side, sharing zoom and scroll. Drag a box to move it or use its 8 handles to resize it, and the lettering refits live. Arrow keys nudge the selected box 1 px, or 10 px with Shift. Draw new balloons and Ignore false hits.
- Cleaning: each text area is filled with its balloon's own colour, kept inside the bubble so outlines and art survive. A page takes milliseconds. The result is a separate clean layer saved in the project, shown under the lettering in the Translated pane. Run Clean again after moving boxes.
- **Auto** button for This page or All pages: runs only the missing steps (detect + OCR, translate, clean), with a progress dialog and Cancel. It stops at the first error.
- Typography per balloon: font (overrides the role), size or Auto, bold, italic, case (UPPER, Title, lower) and alignment. Five style presets are stored in each project: click one to apply it, right-click to save the current style to it.
- Translate a page (F6) with any OpenAI-compatible provider: DeepSeek (`deepseek-chat`, the default), OpenRouter, Ollama or LM Studio. API keys are kept in Windows Credential Manager.
- Lettering preview through cosmic-text and tiny-skia. Text is fitted to the box at 168 DPI with a 16 px minimum line height, and anything that overflows is outlined in red.
- Save (Ctrl+S). Help > About lists credits for every library, model, font and inspiration.

**Planned:** see [docs/design.md](docs/design.md).
- Model download on first run, from a `models-v1` release.
- Solid-fill cleaning, then LaMa inpainting.
- JPG, CBZ and layered PSD export.
- Glossary editor and QC checks.

## Build

```powershell
cargo run --release -p asagitype
```

You need Rust 1.85 or newer on Windows 10 or 11. Comic fonts are loaded from the folder set in Tools > Settings.

Optional: `manga109-segmentation-bubble/model.safetensors` gives exact balloon masks for cleaning. It is never bundled, because it was trained on the academic-use Manga109 dataset.

Until the weights release exists, set Tools > Settings > Models folder to a folder that holds:

- `bubble-detector/model.safetensors`, from [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector).
- `manga-ocr/model.safetensors` and `manga-ocr/vocab.txt`, from [kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base).

## Layout

| Path | Contents |
| --- | --- |
| `apps/asagitype` | egui app: window, theme, canvas, dialogs, About |
| `crates/at2k` | `.at2k` project format: a zip holding `manifest.json` and the page images |
| `crates/provider` | OpenAI-compatible translation and key storage |
| `crates/render` | cosmic-text and tiny-skia lettering and fitting |
| `crates/vision` | candle RT-DETR v2 detector and manga-ocr, plus page analysis |

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. Third-party credits are in [CREDITS.md](CREDITS.md).
