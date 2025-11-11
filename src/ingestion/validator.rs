use crate::domain::RelationKind;
use anyhow::{Context, Result};
use serde::Serialize;
use std::str::FromStr;

use super::response::{LLMExtractionResponse, LLMRelationCandidate};

#[derive(Debug, Clone, Serialize)]
pub struct NormalizedRelation {
    pub subject_label: String,
    pub relation_kind: RelationKind,
    pub object_label: String,
    pub confidence: Option<f32>,
    pub justification: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidatedExtraction {
    pub concept_label: String,
    pub relations: Vec<NormalizedRelation>,
    pub notes: Option<String>,
}

fn normalize_relation(candidate: &LLMRelationCandidate) -> Result<NormalizedRelation> {
    let relation_kind = RelationKind::from_str(&candidate.relation)
        .with_context(|| format!("relation inconnue: {}", candidate.relation))?;

    let confidence = candidate.confidence.map(|c| c.clamp(0.0, 1.0));
    Ok(NormalizedRelation {
        subject_label: candidate.subject.trim().to_string(),
        relation_kind,
        object_label: candidate.object.trim().to_string(),
        confidence,
        justification: candidate.justification.clone(),
    })
}

pub fn validate_response(
    response: LLMExtractionResponse,
    fallback_label: &str,
) -> Result<ValidatedExtraction> {
    let concept_label = response
        .concept_label
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_label.to_string());

    let mut relations = Vec::new();
    for relation in &response.relations {
        relations.push(normalize_relation(relation)?);
    }

    Ok(ValidatedExtraction {
        concept_label,
        relations,
        notes: response.notes,
    })
}
