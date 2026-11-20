# AsagiType 2000: design

Pure-Rust manga scanlation editor. Decisions were made with the owner on 2026-11-21 in two grilling rounds. The second round replaced the first draft, which used Tauri with a Python worker.

## Goals

- A single Windows `.exe` written in Rust: no web view, no Python.
- The whole job in one place: open a chapter, detect balloons and text, OCR, translate with an LLM, clean, letter, and export.
- A boxy Windows 2000 look. Title bars use asagi teal (`#00A3AF`) where Windows 2000 used navy.

## Decisions

| Topic | Decision |
| --- | --- |
| Name | AsagiType 2000 (short: AsagiType) |
| Repository | `github.com/Aischii/AsagiType`, public, `MIT OR Apache-2.0` |
| UI | `egui`/`eframe` on wgpu, the same approach as PhotoCraft. Custom Win2000 visuals and our own title bar |
| ML runtime | `candle`, pure Rust. Models are ported in stages (see below) |
| Text | `cosmic-text` shaping with `tiny-skia` rasterising. The canvas and the export use the same renderer |
| Translation | Any OpenAI-compatible chat endpoint (DeepSeek, OpenRouter, Ollama, LM Studio). Default is DeepSeek `deepseek-chat` |
| API keys | Windows Credential Manager through `keyring`. Never stored in files |
| Context sent to the LLM | Every balloon on the page in reading order, with source text, position, the last lines of the previous page, the glossary and the rules, all in one call |
| Languages | Source: Japanese first. Target: English and Indonesian |
| Project format | `.at2k`: a zip holding `manifest.json`, the page images, and later the clean layers. It is the master copy. JPG, PSD and CBZ are exports |
| Model weights | Published as GitHub Release assets and downloaded on first run, checked by SHA-256. Never committed |
| Fonts | Only OFL fonts are bundled, each with its licence. Blambot and Microsoft fonts load from a folder set by the user |
| Credits | Help > About lists every library, model, font and inspiration, with author and licence. `CREDITS.md` carries the same list |
| Relation to the scanlation workspace | Fully independent. A workspace laid out as `series/<S>/chapters/<C>/source` can be an import source for chapters, glossaries and fonts. Nothing in it is changed |

## Model plan (candle)

| Stage | Task | Model | Licence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Balloon, text and SFX detection | ogkalu comic-speech-bubble-detector (RT-DETR) | Apache-2.0 | Convert the weights once to safetensors, then write RT-DETR in candle |
| 1 | Japanese OCR | kha-white manga-ocr (ViT encoder with BERT decoder) | Apache-2.0 | Encoder and decoder written in candle |
| 1 | Cleaning | Solid fill inside the balloon, sampled from the balloon colour | — | Art SFX are never touched |
| 2 | Inpainting | LaMa (manga fine-tune) | Apache-2.0 | Needs an FFT op in candle |
| 3 | Glyph masks | Hi-SAM / Text-SAM | Apache-2.0 | Optional |

Models avoided for licence reasons: YOLO26 comic-layout (Ultralytics AGPL-3.0 underneath) and the Manga109 bubble segmenter (trained on the non-commercial Manga109 dataset).

## Lettering (port of the translate tools)

These rules come from `lettering.py`, `refit_page.py` and `typography.json`:

- Scans are 168 DPI, so point sizes convert at 168/72.
- Fit text inside the balloon shape (an ellipse or polygon inset), not just its bounding box. Shrink until it fits, with a minimum line height of 16 px.
- Font roles: dialogue, shout, thought, narration, legacy-dialogue. Each role maps to a font family in the project's settings.
- Dialogue is uppercase.
- A watermark goes on every third numbered page: editable, 12% opacity.

## Translation rules sent with every prompt (editable per project)

- Never translate SFX or onomatopoeia drawn on the art.
- Dialogue is UPPERCASE. Profile and prose captions may use sentence case.
- No parentheses in dialogue or thoughts.
- Keep Japanese honorifics and kinship terms as they are (-sama, -san, -chan, -kun, -dono, -senpai, sensei, Onii-chan, Onee-san and so on).
- The series glossary overrides the global one. Never use rejected variants.
- Give exactly one output per balloon id. Never merge, split or drop balloons, and never invent lines.

## Crates

```
apps/asagitype      egui application (window, theme, canvas, panels, dialogs, About)
crates/at2k         .at2k manifest types and zip read/write, chapter import
crates/provider     OpenAI-compatible translation; key storage
crates/render       cosmic-text + tiny-skia lettering and fitting
crates/vision       (stage 1) candle detection, OCR and cleaning
```

## Roadmap

1. **Skeleton (this session):** window and theme. Import a chapter from a folder or the workspace. Open and save `.at2k`. Draw balloons by hand and type the source text. Translate a page. Render the lettering. About and credits.
2. Candle detection and OCR. Solid-fill cleaning. Weights release `models-v1`.
3. JPG, CBZ and layered PSD export. Glossary editor. QC checks.
4. LaMa inpainting. Batch mode. Vision check of speakers.
