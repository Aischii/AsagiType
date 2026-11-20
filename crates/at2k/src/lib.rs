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
pub const FORMAT_VERSION: u32 = 1;

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
    pub pages: Vec<Page>,
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
pub fn save(path: &Path, manifest: &Manifest) -> Result<()> {
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    {
        let mut za = ZipArchive::new(File::open(path)?)?;
        let mut zw = ZipWriter::new(File::create(&tmp)?);
        for i in 0..za.len() {
            let f = za.by_index_raw(i)?;
            if f.name() != MANIFEST {
                zw.raw_copy_file(f)?;
            }
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
        };
        let mut m = create(&out, meta, &[img]).unwrap();
        assert_eq!(m.pages[0].width, 1);
        m.pages[0].balloons.push(Balloon {
            id: 1,
            kind: "dialogue".into(),
            text_box: [0, 0, 1, 1],
            bubble: None,
            source_text: "こんにちは".into(),
            text: "HELLO.".into(),
            role: "dialogue".into(),
            size: 0.0,
        });
        save(&out, &m).unwrap();
        assert_eq!(open(&out).unwrap(), m);
        assert_eq!(read_entry(&out, "pages/001.png").unwrap(), png);
        fs::remove_dir_all(&dir).unwrap();
    }
}
