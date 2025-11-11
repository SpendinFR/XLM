use crate::ingestion::response::{LLMExtractionRequest, LLMExtractionResponse};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use reqwest::blocking::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

pub trait LLMClient: Send + Sync {
    fn extract_relations(&self, request: &LLMExtractionRequest) -> Result<LLMExtractionResponse>;
}

pub struct OllamaLLMClient {
    endpoint: String,
    model: String,
    http: HttpClient,
}

impl OllamaLLMClient {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("impossible d'initialiser le client HTTP pour Ollama")?;

        Ok(Self {
            endpoint: endpoint.into(),
            model: model.into(),
            http,
        })
    }
}

impl LLMClient for OllamaLLMClient {
    fn extract_relations(&self, request: &LLMExtractionRequest) -> Result<LLMExtractionResponse> {
        let payload = OllamaGenerateRequest {
            model: &self.model,
            prompt: &request.user_prompt,
            system: &request.system_prompt,
            stream: false,
        };

        let response = self
            .http
            .post(&self.endpoint)
            .json(&payload)
            .send()
            .context("appel HTTP au serveur Ollama impossible")?
            .error_for_status()
            .context("le serveur Ollama a renvoyé un statut d'erreur")?;

        let raw: OllamaGenerateResponse = response
            .json()
            .context("réponse du serveur Ollama illisible")?;

        let value = extract_json_from_text(&raw.response)?;
        let response: LLMExtractionResponse =
            serde_json::from_value(value).context("format JSON des relations inattendu")?;
        Ok(response)
    }
}

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    system: &'a str,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

pub fn extract_json_from_text(s: &str) -> Result<Value> {
    let t = s.trim().trim_matches(|c| c == '\u{feff}').to_string();

    if let Ok(v) = serde_json::from_str::<Value>(&t) {
        return Ok(v);
    }

    if let Some(i) = t.find('{') {
        if let Some(j) = t.rfind('}') {
            if i < j {
                let slice = &t[i..=j];
                if let Ok(v) = serde_json::from_str::<Value>(slice) {
                    return Ok(v);
                }
            }
        }
    }

    if let Some(start) = t.find("```json") {
        if let Some(end) = t[start + 7..].find("```") {
            let block = &t[start + 7..start + 7 + end];
            if let Ok(v) = serde_json::from_str::<Value>(block) {
                return Ok(v);
            }
        }
    }

    if let Some(start) = t.find("```") {
        if let Some(end) = t[start + 3..].find("```") {
            let block = &t[start + 3..start + 3 + end];
            if let Ok(v) = serde_json::from_str::<Value>(block) {
                return Ok(v);
            }
        }
    }

    Err(anyhow::anyhow!("no valid JSON found in LLM output"))
}

#[derive(Clone, Default)]
pub struct MockLLMClient {
    responses: Arc<Mutex<VecDeque<LLMExtractionResponse>>>,
}

impl MockLLMClient {
    pub fn push_response(&self, response: LLMExtractionResponse) {
        self.responses.lock().push_back(response);
    }
}

impl LLMClient for MockLLMClient {
    fn extract_relations(&self, _: &LLMExtractionRequest) -> Result<LLMExtractionResponse> {
        self.responses
            .lock()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("aucune réponse mock disponible"))
    }
}
