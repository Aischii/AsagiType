//! OpenAI-compatible chat translation (DeepSeek, OpenRouter, Ollama, LM Studio).
//! API keys live in Windows Credential Manager, never in files.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

const KEY_SERVICE: &str = "AsagiType2000";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Preset id, also the key name in Credential Manager.
    pub name: String,
    pub base_url: String,
    pub model: String,
}

pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    pub base_url: &'static str,
    pub model: &'static str,
    pub needs_key: bool,
}

pub const PRESETS: &[Preset] = &[
    Preset {
        id: "deepseek",
        label: "DeepSeek",
        base_url: "https://api.deepseek.com",
        model: "deepseek-chat",
        needs_key: true,
    },
    Preset {
        id: "openrouter",
        label: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        model: "deepseek/deepseek-chat",
        needs_key: true,
    },
    Preset {
        id: "ollama",
        label: "Ollama (local)",
        base_url: "http://localhost:11434/v1",
        model: "qwen3:8b",
        needs_key: false,
    },
    Preset {
        id: "lmstudio",
        label: "LM Studio (local)",
        base_url: "http://localhost:1234/v1",
        model: "local-model",
        needs_key: false,
    },
    Preset {
        id: "custom",
        label: "Custom (OpenAI-compatible)",
        base_url: "",
        model: "",
        needs_key: true,
    },
];

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            name: "deepseek".into(),
            base_url: PRESETS[0].base_url.into(),
            model: PRESETS[0].model.into(),
        }
    }
}

fn is_local(name: &str) -> bool {
    matches!(name, "ollama" | "lmstudio")
}

pub fn api_key(provider: &str) -> Option<String> {
    keyring::Entry::new(KEY_SERVICE, provider)
        .ok()?
        .get_password()
        .ok()
}

/// An empty key removes the stored credential.
pub fn set_api_key(provider: &str, key: &str) -> Result<()> {
    let entry = keyring::Entry::new(KEY_SERVICE, provider)?;
    if key.trim().is_empty() {
        let _ = entry.delete_credential();
    } else {
        entry.set_password(key.trim())?;
    }
    Ok(())
}

#[derive(Serialize, Clone, Debug)]
pub struct BalloonIn {
    pub id: u32,
    pub kind: String,
    pub source_text: String,
    pub text_box: [i32; 4],
}

#[derive(Clone, Debug)]
pub struct PageRequest {
    pub series: String,
    pub page_id: String,
    pub source_lang: String,
    pub target_lang: String,
    pub glossary: String,
    pub rules: Vec<String>,
    pub balloons: Vec<BalloonIn>,
    pub previous_lines: Vec<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct BalloonOut {
    pub id: u32,
    pub text: String,
}

pub fn lang_name(code: &str) -> &str {
    match code {
        "en" => "English",
        "id" => "Indonesian",
        "ja" => "Japanese",
        "zh" => "Chinese",
        "ko" => "Korean",
        other => other,
    }
}

const GLOSSARY_LIMIT: usize = 16_000;

pub fn system_prompt(req: &PageRequest) -> String {
    let mut glossary = req.glossary.clone();
    if glossary.len() > GLOSSARY_LIMIT {
        let mut cut = GLOSSARY_LIMIT;
        while !glossary.is_char_boundary(cut) {
            cut -= 1;
        }
        glossary.truncate(cut);
    }
    format!(
        "You are a professional manga scanlation translator. Translate {src} manga text into natural {dst} for the series \"{series}\".\n\
         Rules:\n- {rules}\n\n\
         Balloons are listed in manga reading order (right to left, top to bottom) with pixel boxes [x1,y1,x2,y2]; use positions to keep speakers consistent.\n\
         Reply with JSON only: {{\"balloons\": [{{\"id\": <id>, \"text\": \"<translation>\"}}]}} with exactly one entry per input id.\n\n\
         {glossary}",
        src = lang_name(&req.source_lang),
        dst = lang_name(&req.target_lang),
        series = req.series,
        rules = req.rules.join("\n- "),
    )
}

/// Pull the balloon list out of a model reply, tolerating code fences.
pub fn parse_reply(content: &str) -> Result<Vec<BalloonOut>> {
    let trimmed = content.trim();
    let body = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed)
        .trim_end_matches("```")
        .trim();
    let v: Value = serde_json::from_str(body).context("reply was not JSON")?;
    let list = v
        .get("balloons")
        .cloned()
        .ok_or_else(|| anyhow!("reply has no \"balloons\" list"))?;
    Ok(serde_json::from_value(list).context("unexpected balloon shape")?)
}

/// Blocking: call from a worker thread.
pub fn translate_page(cfg: &ProviderConfig, req: &PageRequest) -> Result<Vec<BalloonOut>> {
    if req.balloons.is_empty() {
        return Ok(vec![]);
    }
    let local = is_local(&cfg.name);
    let key = api_key(&cfg.name);
    if key.is_none() && !local {
        bail!(
            "No API key stored for '{}'. Set it in Tools > Settings.",
            cfg.name
        );
    }
    let user = json!({
        "page": req.page_id,
        "previous_page_last_lines": req.previous_lines,
        "balloons": req.balloons,
    });
    let mut body = json!({
        "model": cfg.model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt(req)},
            {"role": "user", "content": user.to_string()},
        ],
    });
    if !local {
        body["response_format"] = json!({"type": "json_object"});
    }
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()?;
    let mut http = client.post(url).json(&body);
    if let Some(k) = key {
        http = http.bearer_auth(k);
    }
    let resp = http.send().context("request failed")?;
    let status = resp.status();
    let v: Value = resp.json().context("bad response body")?;
    if !status.is_success() {
        bail!(
            "{status}: {}",
            v["error"]["message"].as_str().unwrap_or("provider error")
        );
    }
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("empty reply"))?;
    parse_reply(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_fenced_replies() {
        let plain = r#"{"balloons":[{"id":1,"text":"HI."}]}"#;
        let fenced = "```json\n{\"balloons\":[{\"id\":2,\"text\":\"YES.\"}]}\n```";
        assert_eq!(
            parse_reply(plain).unwrap(),
            vec![BalloonOut {
                id: 1,
                text: "HI.".into()
            }]
        );
        assert_eq!(parse_reply(fenced).unwrap()[0].id, 2);
        assert!(parse_reply("nope").is_err());
    }

    #[test]
    fn glossary_is_truncated_on_char_boundary() {
        let req = PageRequest {
            series: "S".into(),
            page_id: "001".into(),
            source_lang: "ja".into(),
            target_lang: "en".into(),
            glossary: "語".repeat(10_000),
            rules: vec!["Rule.".into()],
            balloons: vec![],
            previous_lines: vec![],
        };
        let p = system_prompt(&req);
        assert!(p.contains("natural English"));
        assert!(p.len() < GLOSSARY_LIMIT + 2_000);
    }
}
