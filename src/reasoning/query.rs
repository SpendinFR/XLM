use crate::domain::{Concept, Relation, RelationKind};
use crate::memory::KnowledgeGraph;

#[derive(Debug, Clone)]
pub struct ConceptRelations {
    pub concept: Concept,
    pub outgoing: Vec<Relation>,
    pub incoming: Vec<Relation>,
}

pub struct RelationQueryService<'a> {
    graph: &'a KnowledgeGraph,
}

impl<'a> RelationQueryService<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        Self { graph }
    }

    pub fn relations_for(&self, label: &str) -> Option<ConceptRelations> {
        let concept = self.graph.get_concept_by_label(label)?.clone();
        let mut outgoing = Vec::new();
        let mut incoming = Vec::new();
        for relation in self.graph.relations() {
            if relation.subject.uuid == concept.id.uuid {
                outgoing.push(relation.clone());
            } else if relation.object.uuid == concept.id.uuid {
                incoming.push(relation.clone());
            }
        }
        Some(ConceptRelations {
            concept,
            outgoing,
            incoming,
        })
    }

    pub fn filter_by_kind(&self, label: &str, kind: RelationKind) -> Vec<Relation> {
        let Some(concept) = self.graph.get_concept_by_label(label) else {
            return Vec::new();
        };
        self.graph
            .relations()
            .filter(|rel| {
                (rel.subject.uuid == concept.id.uuid || rel.object.uuid == concept.id.uuid)
                    && rel.kind == kind
            })
            .cloned()
            .collect()
    }
}
