use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMExtractionRequest {
    pub concept_slug: String,
    pub system_prompt: String,
    pub user_prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRelationCandidate {
    pub subject: String,
    pub relation: String,
    pub object: String,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub justification: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LLMExtractionResponse {
    pub concept_label: Option<String>,
    #[serde(default)]
    pub relations: Vec<LLMRelationCandidate>,
    #[serde(default)]
    pub notes: Option<String>,
}
