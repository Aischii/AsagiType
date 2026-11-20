//! Help > About credits. Keep in sync with CREDITS.md.

pub struct Credit {
    pub name: &'static str,
    pub author: &'static str,
    pub licence: &'static str,
    pub url: &'static str,
    /// What AsagiType uses it for, or "planned".
    pub role: &'static str,
}

pub struct Section {
    pub title: &'static str,
    pub note: &'static str,
    pub items: &'static [Credit],
}

const fn c(
    name: &'static str,
    author: &'static str,
    licence: &'static str,
    url: &'static str,
    role: &'static str,
) -> Credit {
    Credit {
        name,
        author,
        licence,
        url,
        role,
    }
}

pub const SECTIONS: &[Section] = &[
    Section {
        title: "Libraries",
        note: "Rust crates compiled into AsagiType 2000.",
        items: &[
            c("egui / eframe", "Emil Ernerfeldt and contributors", "MIT OR Apache-2.0", "https://github.com/emilk/egui", "user interface"),
            c("wgpu", "gfx-rs contributors", "MIT OR Apache-2.0", "https://github.com/gfx-rs/wgpu", "GPU drawing"),
            c("cosmic-text", "System76", "MIT OR Apache-2.0", "https://github.com/pop-os/cosmic-text", "text shaping"),
            c("tiny-skia", "Yevhenii Reizner and contributors", "BSD-3-Clause", "https://github.com/linebender/tiny-skia", "lettering raster"),
            c("image", "image-rs contributors", "MIT OR Apache-2.0", "https://github.com/image-rs/image", "page decoding"),
            c("reqwest", "Sean McArthur", "MIT OR Apache-2.0", "https://github.com/seanmonstar/reqwest", "translation requests"),
            c("keyring", "Walther Chen, Daniel Brotsky", "MIT OR Apache-2.0", "https://github.com/hwchen/keyring-rs", "API keys in Credential Manager"),
            c("zip", "zip-rs contributors", "MIT", "https://github.com/zip-rs/zip2", ".at2k projects"),
            c("rfd", "Poly Meilex and contributors", "MIT", "https://github.com/PolyMeilex/rfd", "file dialogs"),
            c("serde, serde_json, anyhow", "David Tolnay and contributors", "MIT OR Apache-2.0", "https://github.com/serde-rs", "data and errors"),
            c("imagesize, dirs", "their authors", "MIT / MIT OR Apache-2.0", "https://crates.io", "utilities"),
            c("candle", "Hugging Face", "MIT OR Apache-2.0", "https://github.com/huggingface/candle", "planned: ML runtime"),
        ],
    },
    Section {
        title: "Models",
        note: "Weights are downloaded from the AsagiType releases, never bundled in the source.",
        items: &[
            c("comic-speech-bubble-detector (RT-DETR)", "ogkalu", "Apache-2.0", "https://huggingface.co/ogkalu", "planned: balloon, text and SFX detection"),
            c("manga-ocr", "Maciej Budys (kha-white)", "Apache-2.0", "https://github.com/kha-white/manga-ocr", "planned: Japanese OCR"),
            c("LaMa", "Suvorov et al., Samsung AI Center", "Apache-2.0", "https://github.com/advimman/lama", "planned: inpainting"),
        ],
    },
    Section {
        title: "Fonts",
        note: "Only OFL fonts are bundled. Blambot and Microsoft fonts are loaded from your own fonts folder and are not distributed.",
        items: &[
            c("Anime Ace 2.0 BB, Badaboom BB, Mouth Breather BB", "Blambot (Nate Piekos)", "Blambot licence, not redistributed", "https://blambot.com", "your fonts folder"),
        ],
    },
    Section {
        title: "Inspiration",
        note: "Ideas only; no code was copied.",
        items: &[
            c("PhotoCraft", "Storytold / ArtCraft team", "MIT OR Apache-2.0", "https://github.com/storytold/photocraft", "pure-Rust egui editor approach"),
            c("Koharu", "koharu-rs contributors", "MIT OR Apache-2.0", "https://github.com/koharu-rs/koharu", "run stages separately, layered export"),
            c("Comic Translate", "ogkalu2", "Apache-2.0", "https://github.com/ogkalu2/comic-translate", "whole-page LLM context"),
            c("Chisa Translator", "Chisa team", "proprietary", "https://chisa.id", "single canvas, own project format"),
        ],
    },
    Section {
        title: "Translation services",
        note: "Names belong to their owners. You bring your own key or local model.",
        items: &[
            c("DeepSeek", "DeepSeek", "service", "https://api.deepseek.com", "default provider"),
            c("OpenRouter, Ollama, LM Studio", "their owners", "service / local", "https://openrouter.ai", "other providers"),
        ],
    },
];
