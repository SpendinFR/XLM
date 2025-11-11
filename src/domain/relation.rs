use anyhow::{self, Error};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use super::{concept::ConceptId, source::SourceAttribution};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationAxis {
    Physical,
    LogicalSocial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationMetadata {
    pub axis: RelationAxis,
    pub name: &'static str,
    pub description: &'static str,
    pub template: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    CauseEffect,
    ActionObject,
    AgentAction,
    PropertyEntity,
    CategoryInstance,
    PartWhole,
    LocationEntity,
    TimeEvent,
    MaterialObject,
    GoalAction,
    Possession,
    Comparison,
    InitialFinalState,
    FunctionUsage,
    Condition,
    Opposition,
    Similarity,
    Dependence,
    ProcessResult,
    SocialConstruct,
}

impl RelationKind {
    pub fn metadata(self) -> RelationMetadata {
        use RelationAxis::{LogicalSocial, Physical};
        match self {
            RelationKind::CauseEffect => RelationMetadata {
                axis: Physical,
                name: "cause_effect",
                description: "X provoque ou déclenche Y",
                template: "{subject} CAUSE {object}",
            },
            RelationKind::ActionObject => RelationMetadata {
                axis: Physical,
                name: "action_object",
                description: "Une action s'applique sur un objet ou patient",
                template: "{subject} agit sur {object}",
            },
            RelationKind::AgentAction => RelationMetadata {
                axis: Physical,
                name: "agent_action",
                description: "Un agent réalise une action",
                template: "{subject} est l'agent de {object}",
            },
            RelationKind::PropertyEntity => RelationMetadata {
                axis: Physical,
                name: "property_entity",
                description: "Une propriété caractérise une entité",
                template: "{subject} est une propriété de {object}",
            },
            RelationKind::CategoryInstance => RelationMetadata {
                axis: Physical,
                name: "category_instance",
                description: "Une catégorie et ses instances",
                template: "{subject} est un type de {object}",
            },
            RelationKind::PartWhole => RelationMetadata {
                axis: Physical,
                name: "part_whole",
                description: "Une partie qui compose un tout",
                template: "{subject} est une partie de {object}",
            },
            RelationKind::LocationEntity => RelationMetadata {
                axis: Physical,
                name: "location_entity",
                description: "Localisation d'une entité",
                template: "{subject} se trouve dans/sur {object}",
            },
            RelationKind::TimeEvent => RelationMetadata {
                axis: Physical,
                name: "time_event",
                description: "Lien temporel entre un moment et un évènement",
                template: "{subject} est le moment de {object}",
            },
            RelationKind::MaterialObject => RelationMetadata {
                axis: Physical,
                name: "material_object",
                description: "Composition matérielle d'un objet",
                template: "{subject} est fait de {object}",
            },
            RelationKind::GoalAction => RelationMetadata {
                axis: Physical,
                name: "goal_action",
                description: "But poursuivi par une action",
                template: "{subject} a pour but {object}",
            },
            RelationKind::Possession => RelationMetadata {
                axis: LogicalSocial,
                name: "possession",
                description: "Relation de possession ou d'appartenance",
                template: "{subject} possède {object}",
            },
            RelationKind::Comparison => RelationMetadata {
                axis: LogicalSocial,
                name: "comparison",
                description: "Comparaison entre deux entités",
                template: "{subject} est comparé à {object}",
            },
            RelationKind::InitialFinalState => RelationMetadata {
                axis: LogicalSocial,
                name: "initial_final_state",
                description: "Transformation d'un état initial vers un état final",
                template: "{subject} devient {object}",
            },
            RelationKind::FunctionUsage => RelationMetadata {
                axis: LogicalSocial,
                name: "function_usage",
                description: "Fonction ou usage d'une entité",
                template: "{subject} sert à {object}",
            },
            RelationKind::Condition => RelationMetadata {
                axis: LogicalSocial,
                name: "condition",
                description: "Condition nécessaire ou suffisante",
                template: "si {subject} alors {object}",
            },
            RelationKind::Opposition => RelationMetadata {
                axis: LogicalSocial,
                name: "opposition",
                description: "Conflit ou opposition",
                template: "{subject} s'oppose à {object}",
            },
            RelationKind::Similarity => RelationMetadata {
                axis: LogicalSocial,
                name: "similarity",
                description: "Analogie ou similarité",
                template: "{subject} est similaire à {object}",
            },
            RelationKind::Dependence => RelationMetadata {
                axis: LogicalSocial,
                name: "dependence",
                description: "Relation de dépendance ou de besoin",
                template: "{subject} dépend de {object}",
            },
            RelationKind::ProcessResult => RelationMetadata {
                axis: LogicalSocial,
                name: "process_result",
                description: "Résultat d'un processus",
                template: "{subject} produit {object}",
            },
            RelationKind::SocialConstruct => RelationMetadata {
                axis: LogicalSocial,
                name: "social_construct",
                description: "Existence par convention sociale",
                template: "{subject} existe par {object}",
            },
        }
    }

    pub fn all() -> &'static [RelationKind] {
        const ALL: &[RelationKind] = &[
            RelationKind::CauseEffect,
            RelationKind::ActionObject,
            RelationKind::AgentAction,
            RelationKind::PropertyEntity,
            RelationKind::CategoryInstance,
            RelationKind::PartWhole,
            RelationKind::LocationEntity,
            RelationKind::TimeEvent,
            RelationKind::MaterialObject,
            RelationKind::GoalAction,
            RelationKind::Possession,
            RelationKind::Comparison,
            RelationKind::InitialFinalState,
            RelationKind::FunctionUsage,
            RelationKind::Condition,
            RelationKind::Opposition,
            RelationKind::Similarity,
            RelationKind::Dependence,
            RelationKind::ProcessResult,
            RelationKind::SocialConstruct,
        ];
        ALL
    }
}

impl fmt::Display for RelationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.metadata().name)
    }
}

impl FromStr for RelationKind {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalized = s.trim().to_lowercase().replace([' ', '-', '>'], "_");
        let normalized = normalized.as_str();
        let candidate = match normalized {
            "cause_effect" | "cause_to_effect" => RelationKind::CauseEffect,
            "action_object" | "action_to_object" => RelationKind::ActionObject,
            "agent_action" | "agent_to_action" => RelationKind::AgentAction,
            "property_entity" | "property_to_entity" => RelationKind::PropertyEntity,
            "category_instance" | "category_to_instance" => RelationKind::CategoryInstance,
            "part_whole" | "part_to_whole" => RelationKind::PartWhole,
            "location_entity" | "location_to_entity" => RelationKind::LocationEntity,
            "time_event" | "time_to_event" => RelationKind::TimeEvent,
            "material_object" | "material_to_object" => RelationKind::MaterialObject,
            "goal_action" | "goal_to_action" => RelationKind::GoalAction,
            "possession" | "possess" => RelationKind::Possession,
            "comparison" | "compare" => RelationKind::Comparison,
            "initial_final_state" | "state_transition" => RelationKind::InitialFinalState,
            "function_usage" | "function" | "usage" => RelationKind::FunctionUsage,
            "condition" | "conditional" => RelationKind::Condition,
            "opposition" | "conflict" => RelationKind::Opposition,
            "similarity" | "analogy" => RelationKind::Similarity,
            "dependence" | "dependency" => RelationKind::Dependence,
            "process_result" | "process_to_result" => RelationKind::ProcessResult,
            "social_construct" | "social_construction" => RelationKind::SocialConstruct,
            _ => anyhow::bail!("relation inconnue: {}", s),
        };
        Ok(candidate)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub subject: ConceptId,
    pub object: ConceptId,
    pub kind: RelationKind,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub justification: Option<String>,
    #[serde(default)]
    pub source: SourceAttribution,
}
