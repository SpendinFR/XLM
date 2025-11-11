use super::llm_client::LLMClient;
use super::prompt::{PromptBuilder, PromptTemplate};
use super::response::LLMExtractionRequest;
use super::validator::{validate_response, ValidatedExtraction};
use crate::domain::{AssertionSource, ConceptKind, RelationKind};
use crate::memory::KnowledgeGraph;
use crate::utils::hash_text;
use anyhow::{Context, Result};
use serde_json;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tracing::info;

pub struct IngestionPipeline<C: LLMClient> {
    llm: Arc<C>,
    prompt_builder: PromptBuilder,
}

#[derive(Debug, Clone)]
pub struct IngestionOutcome {
    pub concept_label: String,
    pub relations_added: usize,
    pub notes: Option<String>,
}

impl<C: LLMClient> IngestionPipeline<C> {
    pub fn new(llm: Arc<C>, template: PromptTemplate) -> Self {
        Self {
            llm,
            prompt_builder: PromptBuilder::new(template),
        }
    }

    pub fn ingest_file(
        &self,
        graph: &mut KnowledgeGraph,
        concept_slug: &str,
        file: impl AsRef<Path>,
    ) -> Result<IngestionOutcome> {
        let file_path = file.as_ref();
        let content = fs::read_to_string(file_path)
            .with_context(|| format!("impossible de lire le fichier {:?}", file_path))?;
        let (system_prompt, user_prompt) = self.prompt_builder.build(concept_slug, &content);

        let request = LLMExtractionRequest {
            concept_slug: concept_slug.to_string(),
            system_prompt: system_prompt.clone(),
            user_prompt: user_prompt.clone(),
        };

        let response = self
            .llm
            .extract_relations(&request)
            .context("échec lors de l'appel au LLM")?;

        let validated = validate_response(response, concept_slug)?;

        let prompt_hash = hash_text(&format!("{}\n{}", system_prompt, user_prompt));
        self.apply_to_graph(graph, &validated, file_path, prompt_hash)?;

        Ok(IngestionOutcome {
            concept_label: validated.concept_label,
            relations_added: validated.relations.len(),
            notes: validated.notes,
        })
    }

    fn apply_to_graph(
        &self,
        graph: &mut KnowledgeGraph,
        extraction: &ValidatedExtraction,
        file_path: &Path,
        prompt_hash: String,
    ) -> Result<()> {
        let mut total = 0;
        let concept_handle = graph.ensure_concept(&extraction.concept_label, ConceptKind::Entity);
        graph.add_alias(&concept_handle.id, extraction.concept_label.clone());
        let serialized = serde_json::to_string(extraction)?;

        for relation in &extraction.relations {
            let subject_kind = infer_kind(relation.relation_kind, true);
            let object_kind = infer_kind(relation.relation_kind, false);
            let subject_handle = if relation
                .subject_label
                .eq_ignore_ascii_case(&extraction.concept_label)
            {
                concept_handle.clone()
            } else {
                graph.ensure_concept(&relation.subject_label, subject_kind)
            };
            let object_handle = graph.ensure_concept(&relation.object_label, object_kind);

            let source = AssertionSource {
                file_path: Some(file_path.to_string_lossy().to_string()),
                llm_model: None,
                prompt_hash: Some(prompt_hash.clone()),
                llm_response: Some(serialized.clone()),
            };

            graph.add_relation(
                subject_handle,
                relation.relation_kind,
                object_handle,
                relation.confidence,
                relation.justification.clone(),
                source,
            )?;
            total += 1;
        }

        info!("relations_integrées" = total, concept = %extraction.concept_label);
        Ok(())
    }
}

fn infer_kind(relation: RelationKind, is_subject: bool) -> ConceptKind {
    match relation {
        RelationKind::CauseEffect => ConceptKind::Process,
        RelationKind::ActionObject => {
            if is_subject {
                ConceptKind::Action
            } else {
                ConceptKind::Entity
            }
        }
        RelationKind::AgentAction => {
            if is_subject {
                ConceptKind::Entity
            } else {
                ConceptKind::Action
            }
        }
        RelationKind::PropertyEntity => {
            if is_subject {
                ConceptKind::Property
            } else {
                ConceptKind::Entity
            }
        }
        RelationKind::CategoryInstance => {
            if is_subject {
                ConceptKind::Property
            } else {
                ConceptKind::Entity
            }
        }
        RelationKind::PartWhole => ConceptKind::Entity,
        RelationKind::LocationEntity => ConceptKind::Entity,
        RelationKind::TimeEvent => {
            if is_subject {
                ConceptKind::Abstract
            } else {
                ConceptKind::Process
            }
        }
        RelationKind::MaterialObject => ConceptKind::Entity,
        RelationKind::GoalAction => {
            if is_subject {
                ConceptKind::Action
            } else {
                ConceptKind::Abstract
            }
        }
        RelationKind::Possession => ConceptKind::Entity,
        RelationKind::Comparison => ConceptKind::Entity,
        RelationKind::InitialFinalState => ConceptKind::Process,
        RelationKind::FunctionUsage => {
            if is_subject {
                ConceptKind::Entity
            } else {
                ConceptKind::Action
            }
        }
        RelationKind::Condition => ConceptKind::Abstract,
        RelationKind::Opposition => ConceptKind::Entity,
        RelationKind::Similarity => ConceptKind::Entity,
        RelationKind::Dependence => ConceptKind::Entity,
        RelationKind::ProcessResult => {
            if is_subject {
                ConceptKind::Process
            } else {
                ConceptKind::Entity
            }
        }
        RelationKind::SocialConstruct => ConceptKind::Abstract,
    }
}
