# Credits

This list is shown in the app under Help > About. Keep it in sync with `apps/asagitype/src/credits.rs`.

## Libraries

| Name | Author | Licence | Used for |
| --- | --- | --- | --- |
| [egui / eframe](https://github.com/emilk/egui) | Emil Ernerfeldt and contributors | MIT OR Apache-2.0 | User interface |
| [wgpu](https://github.com/gfx-rs/wgpu) | gfx-rs contributors | MIT OR Apache-2.0 | GPU drawing |
| [cosmic-text](https://github.com/pop-os/cosmic-text) | System76 | MIT OR Apache-2.0 | Text shaping |
| [tiny-skia](https://github.com/linebender/tiny-skia) | Yevhenii Reizner and contributors | BSD-3-Clause | Rasterising lettering |
| [image](https://github.com/image-rs/image) | image-rs contributors | MIT OR Apache-2.0 | Decoding pages |
| [reqwest](https://github.com/seanmonstar/reqwest) | Sean McArthur | MIT OR Apache-2.0 | Translation requests |
| [keyring](https://github.com/hwchen/keyring-rs) | Walther Chen, Daniel Brotsky | MIT OR Apache-2.0 | API keys in Credential Manager |
| [zip](https://github.com/zip-rs/zip2) | zip-rs contributors | MIT | `.at2k` projects |
| [rfd](https://github.com/PolyMeilex/rfd) | Poly Meilex and contributors | MIT | File dialogs |
| serde, serde_json, anyhow | David Tolnay and contributors | MIT OR Apache-2.0 | Data and errors |
| imagesize, dirs | Their authors | MIT / MIT OR Apache-2.0 | Utilities |
| [candle](https://github.com/huggingface/candle) | Hugging Face | MIT OR Apache-2.0 | ML runtime (detection, OCR) |

## Models

Weights are published as release assets and downloaded on first run. They are never committed to this repository.

| Model | Author | Licence | Used for |
| --- | --- | --- | --- |
| [comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) (RT-DETR v2) | ogkalu | Apache-2.0 | Detecting balloons and text |
| [manga-ocr](https://github.com/kha-white/manga-ocr) | Maciej Budyś (kha-white) | Apache-2.0 | Japanese OCR |
| [LaMa](https://github.com/advimman/lama) | Suvorov et al., Samsung AI Center | Apache-2.0 | Planned: inpainting |

These were left out for licence reasons:

- YOLO26 comic-layout: Ultralytics, AGPL-3.0.
- The Manga109 bubble segmenter: trained on the non-commercial Manga109 dataset.

## Fonts

Only OFL fonts are bundled, in `assets/fonts/`. Blambot fonts (Anime Ace 2.0 BB, Badaboom BB, Mouth Breather BB, all by Nate Piekos), CC Wild Words and Microsoft fonts are loaded from the user's own fonts folder. They are not redistributed.

## Inspiration

These projects shaped the ideas. No code was copied from any of them.

- [PhotoCraft](https://github.com/storytold/photocraft) (MIT OR Apache-2.0): a photo editor in pure Rust, built on egui.
- [Koharu](https://github.com/koharu-rs/koharu) (MIT OR Apache-2.0): running stages separately, and layered export.
- [Comic Translate](https://github.com/ogkalu2/comic-translate) (Apache-2.0): giving the LLM the whole page as context.
- [Chisa Translator](https://chisa.id): one canvas for the whole job, with its own project format.

## Translation services

DeepSeek, OpenRouter, Ollama and LM Studio are names of their owners. You supply your own key or a local model.
