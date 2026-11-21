//! `.at2k` project files: a zip holding `manifest.json` and `pages/<id>.<ext>`.
//! The project is the master copy; JPG, PSD and CBZ are exports.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipArchive, ZipWriter};

pub const MANIFEST: &str = "manifest.json";
pub const EXTENSION: &str = "at2k";
/// v2 adds typography, Ignore and presets. v1 files still open.
pub const FORMAT_VERSION: u32 = 3;

fn default_role() -> String {
    "dialogue".into()
}

/// A text area to translate: a speech balloon, caption or free aside.
/// Art SFX are kept separately and never translated.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Balloon {
    pub id: u32,
    /// "dialogue", "aside" or "caption".
    pub kind: String,
    /// [x1, y1, x2, y2] in page pixels.
    pub text_box: [i32; 4],
    #[serde(default)]
    pub bubble: Option<[i32; 4]>,
    #[serde(default)]
    pub source_text: String,
    #[serde(default)]
    pub text: String,
    #[serde(default = "default_role")]
    pub role: String,
    /// Point size at 168 DPI; 0 means fit automatically.
    #[serde(default)]
    pub size: f32,
    /// Kept on the page but skipped by translate, clean and lettering (credits, false hits).
    #[serde(default)]
    pub ignore: bool,
    /// Font family overriding the role's font.
    #[serde(default)]
    pub font: Option<String>,
    #[serde(default)]
    pub bold: bool,
    #[serde(default)]
    pub italic: bool,
    #[serde(default)]
    pub case: TextCase,
    #[serde(default)]
    pub align: Align,
    /// Index of the preset last applied, for display only.
    #[serde(default)]
    pub preset: Option<u8>,
    /// How the clean layer treats this balloon: Auto erases the source text inside the
    /// existing bubble; a shape paints a new bubble with `fill` and `outline`.
    #[serde(default)]
    pub shape: BubbleShape,
    /// Fill colour (RGBA). None: sampled from the page for Auto, white for shapes.
    #[serde(default)]
    pub fill: Option<[u8; 4]>,
    #[serde(default)]
    pub outline: Option<Outline>,
    /// Hidden in the Translated pane and exports (Layers tab).
    #[serde(default)]
    pub hidden: bool,
    /// Locked boxes cannot be moved, resized or deleted.
    #[serde(default)]
    pub locked: bool,
    /// Text angle in degrees, clockwise; the lettering is drawn at this angle.
    #[serde(default)]
    pub rotation: f32,
    /// Lettering frame for rotated text as a fraction of the box [w, h]; None uses the box.
    #[serde(default)]
    pub frame: Option<[f32; 2]>,
    /// The amber "needs touch-up" mark was dealt with (painted over or marked fixed).
    #[serde(default)]
    pub touch_fixed: bool,
}

impl Balloon {
    pub fn new(id: u32, kind: &str, text_box: [i32; 4]) -> Self {
        Self {
            id,
            kind: kind.into(),
            text_box,
            bubble: None,
            source_text: String::new(),
            text: String::new(),
            role: default_role(),
            size: 0.0,
            ignore: false,
            font: None,
            bold: false,
            italic: false,
            case: TextCase::default(),
            align: Align::default(),
            preset: None,
            shape: BubbleShape::default(),
            fill: None,
            outline: None,
            hidden: false,
            locked: false,
            rotation: 0.0,
            frame: None,
            touch_fixed: false,
        }
    }

    /// Copy a preset's typography onto this balloon.
    pub fn apply_preset(&mut self, index: u8, p: &Preset) {
        self.font = p.font.clone();
        self.size = p.size;
        self.bold = p.bold;
        self.italic = p.italic;
        self.case = p.case;
        self.align = p.align;
        self.preset = Some(index);
    }

    /// The text as it should be lettered.
    pub fn display_text(&self) -> String {
        self.case.apply(&self.text)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TextCase {
    #[default]
    AsTyped,
    Upper,
    Title,
    Lower,
}

impl TextCase {
    pub const ALL: [TextCase; 4] = [
        TextCase::AsTyped,
        TextCase::Upper,
        TextCase::Title,
        TextCase::Lower,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TextCase::AsTyped => "As typed",
            TextCase::Upper => "UPPER",
            TextCase::Title => "Title",
            TextCase::Lower => "lower",
        }
    }

    pub fn apply(self, text: &str) -> String {
        match self {
            TextCase::AsTyped => text.to_string(),
            TextCase::Upper => text.to_uppercase(),
            TextCase::Lower => text.to_lowercase(),
            TextCase::Title => {
                let mut out = String::with_capacity(text.len());
                let mut start = true;
                for c in text.chars() {
                    if c.is_alphabetic() {
                        if start {
                            out.extend(c.to_uppercase());
                        } else {
                            out.extend(c.to_lowercase());
                        }
                        start = false;
                    } else {
                        out.push(c);
                        // Apostrophes stay inside a word (DON'T -> Don't).
                        start = !(c == '\'' || c == '\u{2019}');
                    }
                }
                out
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BubbleShape {
    /// Erase the source text inside the detected bubble.
    #[default]
    Auto,
    Rect,
    Rounded,
    Ellipse,
}

impl BubbleShape {
    pub const ALL: [BubbleShape; 4] = [
        BubbleShape::Auto,
        BubbleShape::Rect,
        BubbleShape::Rounded,
        BubbleShape::Ellipse,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BubbleShape::Auto => "Auto (erase)",
            BubbleShape::Rect => "Rectangle",
            BubbleShape::Rounded => "Rounded",
            BubbleShape::Ellipse => "Ellipse",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Outline {
    pub color: [u8; 4],
    /// Width in page pixels.
    pub width: f32,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Align {
    Left,
    #[default]
    Center,
    Right,
}

/// A saved typography style; projects keep five.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Preset {
    pub name: String,
    #[serde(default)]
    pub font: Option<String>,
    #[serde(default)]
    pub size: f32,
    #[serde(default)]
    pub bold: bool,
    #[serde(default)]
    pub italic: bool,
    #[serde(default)]
    pub case: TextCase,
    #[serde(default)]
    pub align: Align,
}

impl Preset {
    pub fn from_balloon(name: &str, b: &Balloon) -> Self {
        Self {
            name: name.into(),
            font: b.font.clone(),
            size: b.size,
            bold: b.bold,
            italic: b.italic,
            case: b.case,
            align: b.align,
        }
    }
}

pub const PRESET_COUNT: usize = 5;

/// Dialogue, shout, thought, narration and aside starting points.
pub fn default_presets() -> Vec<Preset> {
    let p = |name: &str, bold: bool, italic: bool, case: TextCase| Preset {
        name: name.into(),
        font: None,
        size: 0.0,
        bold,
        italic,
        case,
        align: Align::Center,
    };
    vec![
        p("Dialogue", false, false, TextCase::Upper),
        p("Shout", true, false, TextCase::Upper),
        p("Thought", false, true, TextCase::Upper),
        p("Narration", false, false, TextCase::AsTyped),
        p("Aside", false, true, TextCase::Upper),
    ]
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Page {
    pub id: String,
    pub image: String,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub balloons: Vec<Balloon>,
    #[serde(default)]
    pub sfx: Vec<[i32; 4]>,
    /// Covers, contents pages and logos stay untouched.
    #[serde(default)]
    pub raw_only: bool,
    /// Archive entry of the clean layer (`clean/<id>.png`, RGBA, page size), once cleaned.
    #[serde(default)]
    pub clean: Option<String>,
    /// Archive entry of the healing-brush paint layer (`paint/<id>.png`, RGBA, page size).
    #[serde(default)]
    pub paint: Option<String>,
}

impl Page {
    pub fn clean_entry_name(&self) -> String {
        format!("clean/{}.png", self.id)
    }

    pub fn paint_entry_name(&self) -> String {
        format!("paint/{}.png", self.id)
    }
}

impl Page {
    pub fn next_balloon_id(&self) -> u32 {
        self.balloons.iter().map(|b| b.id).max().unwrap_or(0) + 1
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub title: String,
    pub series: String,
    pub chapter: String,
    pub source_lang: String,
    pub target_lang: String,
    #[serde(default)]
    pub glossary: String,
    #[serde(default)]
    pub rules: Vec<String>,
    /// Five typography presets; empty in v1 files until the app seeds them.
    #[serde(default)]
    pub presets: Vec<Preset>,
    /// Layers tab visibility and opacity. View only: exports always use every layer.
    #[serde(default)]
    pub view: LayerView,
    pub pages: Vec<Page>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(default)]
pub struct LayerState {
    pub visible: bool,
    /// 0.0 to 1.0.
    pub opacity: f32,
}

impl Default for LayerState {
    fn default() -> Self {
        Self {
            visible: true,
            opacity: 1.0,
        }
    }
}

/// The fixed layer stack, bottom to top.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(default)]
pub struct LayerView {
    pub source: LayerState,
    pub clean: LayerState,
    pub paint: LayerState,
    pub text: LayerState,
}

pub fn default_rules() -> Vec<String> {
    [
        "Never translate sound effects or onomatopoeia drawn on the art.",
        "Dialogue is UPPERCASE; profile and novel prose captions may use sentence case.",
        "No parentheses in dialogue or thoughts.",
        "Keep Japanese honorifics and kinship terms as-is (-sama, -san, -chan, -kun, -dono, -senpai, sensei, Onii-chan, Onee-san, Okaa-san, Otou-san). Never localize them.",
        "The series glossary overrides the global glossary; never use rejected variants.",
        "Translate each balloon from its own source text and position. One output per balloon id; never merge, split or drop balloons, never invent lines.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

pub struct ProjectInfo {
    pub path: PathBuf,
    pub name: String,
    pub modified: SystemTime,
}

/// `.at2k` files in `dir`, newest first.
pub fn list_projects(dir: &Path) -> Vec<ProjectInfo> {
    let mut out: Vec<ProjectInfo> = fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(EXTENSION))
                .map(|p| ProjectInfo {
                    name: p
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                    modified: fs::metadata(&p)
                        .and_then(|m| m.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH),
                    path: p,
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    out
}

pub fn image_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| {
                            matches!(
                                e.to_ascii_lowercase().as_str(),
                                "jpg" | "jpeg" | "png" | "webp"
                            )
                        })
                        .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

pub struct NewProject {
    pub series: String,
    pub chapter: String,
    pub source_lang: String,
    pub target_lang: String,
    pub glossary: String,
    pub presets: Vec<Preset>,
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || " -._".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim()
        .to_string()
}

pub fn project_file_name(series: &str, chapter: &str, lang: &str) -> String {
    let short = series.rsplit('/').next().unwrap_or(series);
    format!(
        "{} - {} [{}].{EXTENSION}",
        sanitize(short),
        sanitize(chapter),
        sanitize(lang)
    )
}

/// Copy page images into a new `.at2k`. Source files are only read.
pub fn create(out: &Path, meta: NewProject, images: &[PathBuf]) -> Result<Manifest> {
    if out.exists() {
        bail!("Project already exists: {}", out.display());
    }
    if images.is_empty() {
        bail!("No page images (jpg, png, webp) found");
    }
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = out.with_extension("at2k.tmp");
    let mut zw = ZipWriter::new(File::create(&tmp)?);
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    let mut pages = Vec::with_capacity(images.len());
    for img in images {
        let stem = img
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let ext = img
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
            .to_ascii_lowercase();
        let name = format!("pages/{stem}.{ext}");
        let bytes = fs::read(img).with_context(|| img.display().to_string())?;
        let size = imagesize::blob_size(&bytes).map_err(|e| anyhow!("{}: {e}", img.display()))?;
        zw.start_file(&name, stored)?;
        zw.write_all(&bytes)?;
        pages.push(Page {
            id: stem,
            image: name,
            width: size.width as u32,
            height: size.height as u32,
            balloons: vec![],
            sfx: vec![],
            raw_only: false,
            clean: None,
            paint: None,
        });
    }
    let manifest = Manifest {
        format: EXTENSION.into(),
        version: FORMAT_VERSION,
        title: format!("{} {}", meta.series, meta.chapter),
        series: meta.series,
        chapter: meta.chapter,
        source_lang: meta.source_lang,
        target_lang: meta.target_lang,
        glossary: meta.glossary,
        rules: default_rules(),
        presets: meta.presets,
        view: LayerView::default(),
        pages,
    };
    zw.start_file(MANIFEST, SimpleFileOptions::default())?;
    zw.write_all(serde_json::to_string_pretty(&manifest)?.as_bytes())?;
    zw.finish()?;
    fs::rename(&tmp, out)?;
    Ok(manifest)
}

pub fn read_entry(path: &Path, name: &str) -> Result<Vec<u8>> {
    let mut za = ZipArchive::new(File::open(path)?)?;
    let mut f = za
        .by_name(name)
        .with_context(|| format!("{name} in {}", path.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

pub fn open(path: &Path) -> Result<Manifest> {
    let m: Manifest =
        serde_json::from_slice(&read_entry(path, MANIFEST)?).context("bad manifest")?;
    if m.format != EXTENSION {
        bail!("Not an AsagiType project");
    }
    if m.version > FORMAT_VERSION {
        bail!(
            "Project was made by a newer AsagiType (format {})",
            m.version
        );
    }
    Ok(m)
}

/// Rewrite the archive with a new manifest; other entries are copied raw.
/// The file is written in the current format version.
pub fn save(path: &Path, manifest: &Manifest) -> Result<()> {
    save_with_files(path, manifest, &[])
}

/// Like [`save`], also adding or replacing archive entries (for example clean layers).
pub fn save_with_files(
    path: &Path,
    manifest: &Manifest,
    files: &[(String, Vec<u8>)],
) -> Result<()> {
    let mut manifest = manifest.clone();
    manifest.version = FORMAT_VERSION;
    let manifest = &manifest;
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    {
        let mut za = ZipArchive::new(File::open(path)?)?;
        let mut zw = ZipWriter::new(File::create(&tmp)?);
        for i in 0..za.len() {
            let f = za.by_index_raw(i)?;
            let orphan = (f.name().starts_with("paint/") || f.name().starts_with("clean/"))
                && !manifest.pages.iter().any(|p| {
                    p.paint.as_deref() == Some(f.name()) || p.clean.as_deref() == Some(f.name())
                });
            if f.name() != MANIFEST && !orphan && !files.iter().any(|(n, _)| n == f.name()) {
                zw.raw_copy_file(f)?;
            }
        }
        let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        for (name, bytes) in files {
            zw.start_file(name.as_str(), stored)?;
            zw.write_all(bytes)?;
        }
        zw.start_file(MANIFEST, SimpleFileOptions::default())?;
        zw.write_all(serde_json::to_string_pretty(manifest)?.as_bytes())?;
        zw.finish()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// A chapter in a scanlation workspace laid out as `series/<S>/chapters/<C>/source/`.
pub struct WorkspaceChapter {
    pub series: String,
    pub series_dir: PathBuf,
    pub chapter: String,
    pub source_dir: PathBuf,
    pub pages: usize,
}

fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

pub fn workspace_chapters(workspace: &Path) -> Vec<WorkspaceChapter> {
    let root = workspace.join("series");
    let mut series_dirs = vec![];
    for d in subdirs(&root) {
        if d.join("chapters").is_dir() {
            series_dirs.push(d);
        } else {
            series_dirs.extend(
                subdirs(&d)
                    .into_iter()
                    .filter(|d2| d2.join("chapters").is_dir()),
            );
        }
    }
    let mut out = vec![];
    for sd in series_dirs {
        let name = sd
            .strip_prefix(&root)
            .unwrap_or(&sd)
            .to_string_lossy()
            .replace('\\', "/");
        for ch in subdirs(&sd.join("chapters")) {
            let src = ch.join("source");
            let pages = image_files(&src).len();
            if pages > 0 {
                out.push(WorkspaceChapter {
                    series: name.clone(),
                    series_dir: sd.clone(),
                    chapter: ch
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                    source_dir: src,
                    pages,
                });
            }
        }
    }
    out
}

/// Series glossary followed by the workspace's global glossary.
pub fn workspace_glossary(workspace: &Path, series_dir: &Path) -> String {
    let mut out = String::new();
    if let Ok(t) = fs::read_to_string(series_dir.join("glossary.md")) {
        out.push_str("# Series glossary (overrides global)\n");
        out.push_str(&t);
        out.push('\n');
    }
    if let Ok(t) = fs::read_to_string(workspace.join("glossary").join("global.md")) {
        out.push_str("# Global glossary\n");
        out.push_str(&t);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_open_save_roundtrip() {
        let dir = std::env::temp_dir().join(format!("at2k-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Minimal 1x1 PNG.
        let png: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52, 0,
            0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 0x1F, 0x15, 0xC4, 0x89, 0, 0, 0, 13, 0x49, 0x44,
            0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0x0F, 0x00, 0x00, 0x01, 0x01, 0x00, 0x05, 0x18,
            0xD8, 0x4E, 0, 0, 0, 0, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let img = dir.join("001.png");
        fs::write(&img, png).unwrap();
        let out = dir.join("t.at2k");
        let meta = NewProject {
            series: "S".into(),
            chapter: "c1".into(),
            source_lang: "ja".into(),
            target_lang: "en".into(),
            glossary: String::new(),
            presets: default_presets(),
        };
        let mut m = create(&out, meta, &[img]).unwrap();
        assert_eq!(m.pages[0].width, 1);
        let mut b = Balloon::new(1, "dialogue", [0, 0, 1, 1]);
        b.source_text = "こんにちは".into();
        b.text = "hello.".into();
        b.apply_preset(1, &m.presets[1]);
        m.pages[0].balloons.push(b);
        save(&out, &m).unwrap();
        let back = open(&out).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.pages[0].balloons[0].display_text(), "HELLO.");
        assert!(back.pages[0].balloons[0].bold);
        // Clean layers are added, then replaced in place.
        let mut m2 = back.clone();
        let name = m2.pages[0].clean_entry_name();
        m2.pages[0].clean = Some(name.clone());
        save_with_files(&out, &m2, &[(name.clone(), vec![1, 2, 3])]).unwrap();
        save_with_files(&out, &m2, &[(name.clone(), vec![4, 5])]).unwrap();
        assert_eq!(read_entry(&out, &name).unwrap(), vec![4, 5]);
        assert_eq!(
            open(&out).unwrap().pages[0].clean.as_deref(),
            Some("clean/001.png")
        );
        assert_eq!(read_entry(&out, "pages/001.png").unwrap(), png);
        // Paint layers are stored, and dropped once no page refers to them.
        let mut m3 = open(&out).unwrap();
        let paint = m3.pages[0].paint_entry_name();
        m3.pages[0].paint = Some(paint.clone());
        m3.view.paint.opacity = 0.5;
        save_with_files(&out, &m3, &[(paint.clone(), vec![9])]).unwrap();
        let back3 = open(&out).unwrap();
        assert_eq!(back3.pages[0].paint.as_deref(), Some("paint/001.png"));
        assert_eq!(back3.view.paint.opacity, 0.5);
        assert_eq!(read_entry(&out, &paint).unwrap(), vec![9]);
        m3.pages[0].paint = None;
        m3.pages[0].clean = None;
        save(&out, &m3).unwrap();
        assert!(read_entry(&out, &paint).is_err());
        assert!(read_entry(&out, &name).is_err());
        assert_eq!(read_entry(&out, "pages/001.png").unwrap(), png);
        fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod v2_tests {
    use super::*;

    #[test]
    fn v1_balloon_gets_v2_defaults() {
        let v1 = r#"{"id":3,"kind":"dialogue","text_box":[1,2,3,4],"text":"HI","role":"dialogue","size":0}"#;
        let b: Balloon = serde_json::from_str(v1).unwrap();
        assert!(!b.ignore && !b.bold && b.font.is_none());
        assert!(!b.hidden && !b.locked && b.fill.is_none() && b.outline.is_none());
        assert_eq!((b.rotation, b.frame), (0.0, None));
        assert_eq!(b.shape, BubbleShape::Auto);
        assert_eq!((b.case, b.align), (TextCase::AsTyped, Align::Center));
        let m: Manifest = serde_json::from_str(
            r#"{"format":"at2k","version":1,"title":"t","series":"s","chapter":"c","source_lang":"ja","target_lang":"en","pages":[]}"#,
        )
        .unwrap();
        assert!(m.presets.is_empty());
        assert_eq!(m.view, LayerView::default());
        assert!(m.view.text.visible && m.view.text.opacity == 1.0);
        assert!(!b.touch_fixed);
    }

    #[test]
    fn title_case_keeps_apostrophes_inside_words() {
        assert_eq!(
            TextCase::Title.apply("DON'T GO, ONII-CHAN!"),
            "Don't Go, Onii-Chan!"
        );
        assert_eq!(TextCase::Lower.apply("WAIT"), "wait");
    }
}
