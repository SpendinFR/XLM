use crate::domain::{
    AssertionSource, Concept, ConceptId, ConceptKind, ConceptLabel, Relation, RelationKind,
    SourceAttribution,
};
use crate::utils::slugify;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct GraphData {
    concepts: HashMap<Uuid, Concept>,
    relations: Vec<Relation>,
}

#[derive(Debug, Clone)]
pub struct ConceptHandle {
    pub id: ConceptId,
    pub label: ConceptLabel,
    pub kind: ConceptKind,
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub relation: Relation,
    pub subject: Concept,
    pub object: Concept,
}

pub struct KnowledgeGraph {
    data: GraphData,
    label_index: HashMap<String, Uuid>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            data: GraphData::default(),
            label_index: HashMap::new(),
        }
    }

    pub fn from_parts(data: GraphData) -> Self {
        let mut label_index = HashMap::new();
        for (uuid, concept) in &data.concepts {
            label_index.insert(concept.label.primary.to_lowercase(), *uuid);
            for alias in &concept.label.aliases {
                label_index.insert(alias.to_lowercase(), *uuid);
            }
        }
        Self { data, label_index }
    }

    pub fn concepts(&self) -> impl Iterator<Item = &Concept> {
        self.data.concepts.values()
    }

    pub fn relations(&self) -> impl Iterator<Item = &Relation> {
        self.data.relations.iter()
    }

    pub fn get_concept_by_label(&self, label: &str) -> Option<&Concept> {
        let key = label.to_lowercase();
        self.label_index
            .get(&key)
            .and_then(|uuid| self.data.concepts.get(uuid))
    }

    pub fn get_concept_by_id(&self, id: &Uuid) -> Option<&Concept> {
        self.data.concepts.get(id)
    }

    pub fn ensure_concept(&mut self, label: &str, preferred_kind: ConceptKind) -> ConceptHandle {
        if let Some(concept) = self.get_concept_by_label(label) {
            return ConceptHandle {
                id: concept.id.clone(),
                label: concept.label.clone(),
                kind: concept.kind,
            };
        }

        let slug = slugify(label);
        let mut concept = Concept::new(slug, ConceptLabel::new(label), preferred_kind);
        concept.properties.description =
            Some(format!("Concept ajouté automatiquement depuis ingestion"));
        let uuid = concept.id.uuid;
        self.label_index
            .insert(concept.label.primary.to_lowercase(), uuid);
        for alias in &concept.label.aliases {
            self.label_index.insert(alias.to_lowercase(), uuid);
        }
        self.data.concepts.insert(uuid, concept.clone());
        ConceptHandle {
            id: concept.id,
            label: concept.label,
            kind: concept.kind,
        }
    }

    pub fn add_alias(&mut self, concept_id: &ConceptId, alias: String) {
        if let Some(concept) = self.data.concepts.get_mut(&concept_id.uuid) {
            concept.label.aliases.insert(alias.to_owned());
            self.label_index
                .insert(alias.to_lowercase(), concept_id.uuid);
            concept.touch();
        }
    }

    pub fn upsert_concept(&mut self, concept: Concept) {
        let uuid = concept.id.uuid;
        for alias in concept
            .label
            .aliases
            .iter()
            .chain(std::iter::once(&concept.label.primary))
        {
            self.label_index.insert(alias.to_lowercase(), uuid);
        }
        self.data.concepts.insert(uuid, concept);
    }

    pub fn add_relation(
        &mut self,
        subject: ConceptHandle,
        relation_kind: RelationKind,
        object: ConceptHandle,
        confidence: Option<f32>,
        justification: Option<String>,
        source: AssertionSource,
    ) -> Result<()> {
        let existing = self.data.relations.iter_mut().find(|rel| {
            rel.subject.uuid == subject.id.uuid
                && rel.object.uuid == object.id.uuid
                && rel.kind == relation_kind
        });

        if let Some(rel) = existing {
            rel.confidence = match (rel.confidence, confidence) {
                (Some(a), Some(b)) => Some((a + b) / 2.0),
                (None, Some(b)) => Some(b),
                (Some(a), None) => Some(a),
                (None, None) => None,
            };
            if let Some(justification) = justification {
                rel.justification = Some(match &rel.justification {
                    Some(existing) if existing != &justification => {
                        format!("{}\n---\n{}", existing, justification)
                    }
                    _ => justification,
                });
            }
            rel.source.push(source);
            return Ok(());
        }

        let relation = Relation {
            subject: subject.id.clone(),
            object: object.id.clone(),
            kind: relation_kind,
            confidence,
            justification,
            source: SourceAttribution::new(source),
        };
        self.data.relations.push(relation);
        Ok(())
    }

    pub fn has_relation(
        &self,
        subject: &ConceptId,
        relation_kind: RelationKind,
        object: &ConceptId,
    ) -> bool {
        self.data.relations.iter().any(|rel| {
            rel.subject.uuid == subject.uuid
                && rel.object.uuid == object.uuid
                && rel.kind == relation_kind
        })
    }

    pub fn query_by_concept(&self, label: &str) -> Vec<QueryResult> {
        let Some(subject) = self.get_concept_by_label(label) else {
            return Vec::new();
        };
        self.data
            .relations
            .iter()
            .filter(|rel| rel.subject.uuid == subject.id.uuid || rel.object.uuid == subject.id.uuid)
            .filter_map(|relation| {
                let subject_concept = self.data.concepts.get(&relation.subject.uuid)?;
                let object_concept = self.data.concepts.get(&relation.object.uuid)?;
                Some(QueryResult {
                    relation: relation.clone(),
                    subject: subject_concept.clone(),
                    object: object_concept.clone(),
                })
            })
            .collect()
    }

    pub fn into_inner(self) -> GraphData {
        self.data
    }

    pub fn snapshot(&self) -> GraphData {
        self.data.clone()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}
