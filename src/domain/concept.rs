use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConceptId {
    pub uuid: Uuid,
    pub slug: String,
}

impl ConceptId {
    pub fn new<S: Into<String>>(slug: S) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            slug: slug.into(),
        }
    }

    pub fn with_uuid<S: Into<String>>(uuid: Uuid, slug: S) -> Self {
        Self {
            uuid,
            slug: slug.into(),
        }
    }
}

impl fmt::Display for ConceptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.slug, self.uuid)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConceptLabel {
    pub primary: String,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub aliases: BTreeSet<String>,
}

impl ConceptLabel {
    pub fn new<S: Into<String>>(primary: S) -> Self {
        Self {
            primary: primary.into(),
            aliases: BTreeSet::new(),
        }
    }

    pub fn with_aliases(
        primary: impl Into<String>,
        aliases: impl IntoIterator<Item = String>,
    ) -> Self {
        let mut label = Self::new(primary);
        for alias in aliases {
            label.aliases.insert(alias);
        }
        label
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConceptKind {
    Entity,
    Action,
    Process,
    Property,
    Abstract,
}

impl Default for ConceptKind {
    fn default() -> Self {
        ConceptKind::Entity
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConceptProperties {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub id: ConceptId,
    pub label: ConceptLabel,
    #[serde(default)]
    pub kind: ConceptKind,
    #[serde(default)]
    pub properties: ConceptProperties,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub updated_at: Option<DateTime<Utc>>,
}

impl Concept {
    pub fn new<S: Into<String>>(slug: S, label: ConceptLabel, kind: ConceptKind) -> Self {
        Self {
            id: ConceptId::new(slug.into()),
            label,
            kind,
            properties: ConceptProperties::default(),
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    pub fn with_uuid(uuid: Uuid, label: ConceptLabel, kind: ConceptKind) -> Self {
        Self {
            id: ConceptId::with_uuid(uuid, label.primary.clone()),
            label,
            kind,
            properties: ConceptProperties::default(),
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = Some(Utc::now());
    }
}
