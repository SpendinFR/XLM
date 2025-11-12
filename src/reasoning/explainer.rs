use crate::domain::{Relation, RelationKind};
use crate::memory::KnowledgeGraph;
use serde::Serialize;
use std::collections::{HashSet, VecDeque};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RelationDirection {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExplanationNode {
    pub from: String,
    pub to: String,
    pub relation: RelationKind,
    pub direction: RelationDirection,
    pub justification: Option<String>,
}

pub struct PathExplainer<'a> {
    graph: &'a KnowledgeGraph,
}

impl<'a> PathExplainer<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        Self { graph }
    }

    pub fn explain(&self, from: &str, to: &str, max_depth: usize) -> Option<Vec<ExplanationNode>> {
        if max_depth == 0 {
            return None;
        }
        let start = self.graph.get_concept_by_label(from)?;
        let target = self.graph.get_concept_by_label(to)?;

        let mut queue: VecDeque<(Uuid, Vec<(Relation, RelationDirection)>)> = VecDeque::new();
        let mut visited: HashSet<Uuid> = HashSet::new();
        queue.push_back((start.id.uuid, Vec::new()));
        visited.insert(start.id.uuid);

        while let Some((current_id, path)) = queue.pop_front() {
            if current_id == target.id.uuid {
                return Some(self.materialize_path(path));
            }
            if path.len() >= max_depth {
                continue;
            }
            for relation in self.graph.relations() {
                if relation.subject.uuid == current_id {
                    let mut next_path = path.clone();
                    next_path.push((relation.clone(), RelationDirection::Forward));
                    let next_id = relation.object.uuid;
                    if visited.insert(next_id) {
                        queue.push_back((next_id, next_path));
                    }
                } else if relation.object.uuid == current_id {
                    let mut next_path = path.clone();
                    next_path.push((relation.clone(), RelationDirection::Backward));
                    let next_id = relation.subject.uuid;
                    if visited.insert(next_id) {
                        queue.push_back((next_id, next_path));
                    }
                }
            }
        }
        None
    }

    fn materialize_path(&self, path: Vec<(Relation, RelationDirection)>) -> Vec<ExplanationNode> {
        let mut explanation = Vec::new();
        for (relation, direction) in path {
            let from_concept = match direction {
                RelationDirection::Forward => self.graph.get_concept_by_id(&relation.subject.uuid),
                RelationDirection::Backward => self.graph.get_concept_by_id(&relation.object.uuid),
            };
            let to_concept = match direction {
                RelationDirection::Forward => self.graph.get_concept_by_id(&relation.object.uuid),
                RelationDirection::Backward => self.graph.get_concept_by_id(&relation.subject.uuid),
            };
            if let (Some(from), Some(to)) = (from_concept, to_concept) {
                explanation.push(ExplanationNode {
                    from: from.label.primary.clone(),
                    to: to.label.primary.clone(),
                    relation: relation.kind,
                    direction,
                    justification: relation.justification.clone(),
                });
            }
        }
        explanation
    }
}
