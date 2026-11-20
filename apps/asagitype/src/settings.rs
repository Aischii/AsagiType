//! App settings in %APPDATA%\AsagiType\settings.json. API keys are not stored here.

use provider::ProviderConfig;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Settings {
    pub workspace_root: String,
    pub projects_dir: String,
    pub fonts_dir: String,
    pub provider: ProviderConfig,
    pub target_lang: String,
    /// Lettering role -> font family.
    pub role_fonts: BTreeMap<String, String>,
}

pub const ROLES: &[&str] = &[
    "dialogue",
    "shout",
    "thought",
    "narration",
    "legacy-dialogue",
];

impl Default for Settings {
    fn default() -> Self {
        let role_fonts = [
            ("dialogue", "Anime Ace 2.0 BB"),
            ("shout", "Badaboom BB"),
            ("thought", "Anime Ace 2.0 BB"),
            ("narration", "CC Wild Words"),
            ("legacy-dialogue", "SS Soapy Hands"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        Self {
            workspace_root: "C:/Projects/translate".into(),
            projects_dir: dirs::document_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("AsagiType")
                .to_string_lossy()
                .into_owned(),
            fonts_dir: "C:/Projects/translate/tools/fonts".into(),
            provider: ProviderConfig::default(),
            target_lang: "en".into(),
            role_fonts,
        }
    }
}

fn path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("AsagiType")
        .join("settings.json")
}

impl Settings {
    pub fn load() -> Self {
        fs::read_to_string(path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let p = path();
        fs::create_dir_all(p.parent().unwrap())?;
        fs::write(p, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn font_for(&self, role: &str) -> String {
        self.role_fonts
            .get(role)
            .cloned()
            .unwrap_or_else(|| "Anime Ace 2.0 BB".into())
    }
}
