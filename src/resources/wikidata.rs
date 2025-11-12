use std::collections::{HashMap, HashSet};
use std::thread::sleep;
use std::time::{Duration, Instant};

use chrono::{DateTime, NaiveDate, NaiveDateTime, SecondsFormat};

use anyhow::{anyhow, Context, Result};
use reqwest::blocking::Client as HttpClient;
use reqwest::StatusCode;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE, USER_AGENT};
use reqwest::StatusCode;
use serde::Deserialize;
use tracing::{info, warn};

use crate::domain::{AssertionSource, ConceptKind, RelationKind};
use crate::memory::KnowledgeGraph;

const DEFAULT_ENDPOINT: &str = "https://query.wikidata.org/sparql";
const DEFAULT_USER_AGENT: &str = "XLM-Concept-Graph/0.1 (+https://github.com/)";
const MAX_FETCH_ATTEMPTS: u32 = 4;
const RETRY_BASE_DELAY_MS: u64 = 1_000;

const SENSE_PROPERTY_IDS: &[&str] = &[
    "P5137", // item for this sense
    "P5973", // synonym of sense
    "P5974", // antonym of sense
    "P5975", // troponym of
    "P5976", // false friend
    "P5978", // classifier
    "P6084", // location of sense usage
    "P6593", // hypernym of sense
];

const ITEM_PROPERTY_IDS: &[&str] = &[
    "P31",   // instance of
    "P279",  // subclass of
    "P361",  // part of
    "P527",  // has part
    "P276",  // location
    "P7153", // significant place
    "P585",  // point in time
    "P580",  // start time
    "P582",  // end time
    "P186",  // made from material
    "P127",  // owned by
    "P828",  // has cause
    "P1542", // has effect
    "P366",  // has use
    "P2283", // uses
    "P3712", // has goal
    "P1552", // has quality
    "P460",  // said to be the same as
    "P461",  // opposite of
    "P1889", // different from
];

#[derive(Debug, Clone)]
pub struct WikidataImportConfig {
    pub language_code: String,
    pub language_qid: String,
    pub endpoint: String,
    pub batch_size: usize,
    pub max_records: usize,
    pub pause: Duration,
    pub user_agent: String,
}

impl WikidataImportConfig {
    pub fn new(language_code: impl Into<String>) -> Result<Self> {
        let language_code = language_code.into();
        let language_qid = language_qid(&language_code)
            .ok_or_else(|| anyhow!("langue non prise en charge: {}", language_code))?
            .to_string();
        Ok(Self {
            language_code,
            language_qid,
            endpoint: DEFAULT_ENDPOINT.to_string(),
            batch_size: 500,
            max_records: 35_000,
            pause: Duration::from_millis(250),
            user_agent: DEFAULT_USER_AGENT.to_string(),
        })
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    pub fn with_max_records(mut self, max: usize) -> Self {
        self.max_records = max.max(1);
        self
    }

    pub fn with_pause(mut self, pause: Duration) -> Self {
        self.pause = pause;
        self
    }

    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }
}

#[derive(Debug)]
pub struct ResourceImportStats {
    pub lemmas_added: usize,
    pub concepts_created: usize,
    pub relations_added: usize,
    pub batches_processed: usize,
    pub elapsed: Duration,
}

pub struct WikidataImporter {
    client: HttpClient,
    config: WikidataImportConfig,
}

impl WikidataImporter {
    pub fn new(config: WikidataImportConfig) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&config.user_agent)
                .unwrap_or_else(|_| HeaderValue::from_static(DEFAULT_USER_AGENT)),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );

        let client = HttpClient::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(120))
            .build()
            .context("impossible de créer le client HTTP pour Wikidata")?;

        Ok(Self { client, config })
    }

    pub fn import_into(&self, graph: &mut KnowledgeGraph) -> Result<ResourceImportStats> {
        let mut lemmas_seen: HashSet<String> = HashSet::new();
        let mut stats = ResourceImportStats {
            lemmas_added: 0,
            concepts_created: 0,
            relations_added: 0,
            batches_processed: 0,
            elapsed: Duration::default(),
        };
        let start = Instant::now();

        let mut offset = 0usize;
        let mut imported = 0usize;

        while imported < self.config.max_records {
            let remaining = self.config.max_records - imported;
            let limit = remaining.min(self.config.batch_size);
            let records = match self.fetch_batch(limit, offset) {
                Ok(records) => records,
                Err(err) => {
                    warn!("erreur" = %err, offset, "message" = "échec d'une requête Wikidata");
                    break;
                }
            };

            if records.is_empty() {
                break;
            }

            stats.batches_processed += 1;

            for record in records {
                if imported >= self.config.max_records {
                    break;
                }

                let lemma = record.lemma.trim();
                if lemma.is_empty() {
                    continue;
                }

                let Some(relation_kind) = record.relation_kind() else {
                    continue;
                };

                let lemma_key = lemma.to_lowercase();
                let existed = graph.get_concept_by_label(lemma).is_some();
                let subject = graph.ensure_concept(lemma, ConceptKind::Entity);
                if !existed && lemmas_seen.insert(lemma_key) {
                    stats.lemmas_added += 1;
                    stats.concepts_created += 1;
                }

                if let Some(alias) = record
                    .lexeme_label
                    .as_ref()
                    .map(|value| value.trim())
                    .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case(lemma))
                {
                    graph.add_alias(&subject.id, alias.to_string());
                }

                let Some((object_label, object_alias)) = record.object_label_and_alias() else {
                    continue;
                };
                let object_existed = graph.get_concept_by_label(&object_label).is_some();
                let object_kind = record.object_concept_kind(relation_kind);
                let object = graph.ensure_concept(&object_label, object_kind);
                if !object_existed {
                    stats.concepts_created += 1;
                }

                if let Some(alias) = object_alias {
                    graph.add_alias(&object.id, alias);
                }

                let already = graph.has_relation(&subject.id, relation_kind, &object.id);
                let justification = record.build_justification();
                let mut source = AssertionSource::default();
                source.dataset = Some("wikidata_lexeme".to_string());
                source.external_id = Some(format!("{}#{}", record.lexeme_id, record.sense_id));
                source.url = Some(record.entity_url());

                graph.add_relation(
                    subject.clone(),
                    relation_kind,
                    object,
                    Some(1.0),
                    justification,
                    source,
                )?;

                if !already {
                    stats.relations_added += 1;
                }

                imported += 1;
            }

            offset += limit;
            sleep(self.config.pause);

            info!(
                "wikidata_import_progress" = stats.lemmas_added,
                relations = stats.relations_added,
                batches = stats.batches_processed,
                offset,
                "message" = "lot Wikidata traité"
            );
        }

        stats.elapsed = start.elapsed();
        Ok(stats)
    }

    fn fetch_batch(&self, limit: usize, offset: usize) -> Result<Vec<SenseRelationRecord>> {
        let query = self.build_query(limit, offset);
        let response = self
            .client
            .post(&self.config.endpoint)
            .body(query)
            .send()
            .context("échec lors de l'appel SPARQL à Wikidata")?
            .error_for_status()
            .context("réponse HTTP invalide depuis Wikidata")?;

        let parsed: SparqlResponse = response
            .json()
            .context("impossible d'analyser la réponse SPARQL de Wikidata")?;
        Ok(parsed.into_records())
    }

    fn fetch_relations(&self, senses: &[SenseEntry]) -> Result<Vec<RelationRow>> {
        if senses.is_empty() {
            return Ok(Vec::new());
        }

        let query = self.build_relation_query(senses);
        let response: RelationQueryResponse = self.execute_sparql(&query)?;
        Ok(response
            .results
            .bindings
            .into_iter()
            .filter_map(|binding| binding.try_into().ok())
            .collect())
    }

    fn execute_sparql<T>(&self, query: &str) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut attempt = 0u32;

        loop {
            attempt += 1;

            match self
                .client
                .post(&self.config.endpoint)
                .body(query.to_string())
                .send()
            {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return response
                            .json()
                            .context("impossible d'analyser la réponse SPARQL de Wikidata");
                    }

                    if should_retry_status(status) && attempt < MAX_FETCH_ATTEMPTS {
                        warn!(
                            attempt,
                            status = status.as_u16(),
                            "message" = "réponse HTTP non valide, nouvelle tentative"
                        );
                        sleep(retry_delay(attempt));
                        continue;
                    }

                    return Err(anyhow!(
                        "réponse HTTP invalide depuis Wikidata (statut {})",
                        status
                    ));
                }
                Err(err) => {
                    if attempt < MAX_FETCH_ATTEMPTS {
                        warn!(
                            attempt,
                            "message" = "erreur lors de l'appel SPARQL à Wikidata, nouvelle tentative",
                            "erreur" = %err
                        );
                        sleep(retry_delay(attempt));
                        continue;
                    }

                    return Err(anyhow!("échec lors de l'appel SPARQL à Wikidata: {err}"));
                }
            }
        }
    }

    fn build_sense_query(&self, limit: usize, offset: usize) -> String {
        format!(
            r#"PREFIX dct: <http://purl.org/dc/terms/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX ontolex: <http://www.wikidata.org/ns/lemon/ontolex#>

SELECT DISTINCT ?lexeme ?lexemeLabel ?lemma ?sense ?senseDescription
WHERE {{
  ?lexeme a ontolex:LexicalEntry ;
          dct:language wd:{language_qid} ;
          wikibase:lemma ?lemma ;
          ontolex:sense ?sense .
  FILTER(LANG(?lemma) = "{language_code}")
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language_code},en". }}
}}
ORDER BY ?lemma
LIMIT {limit}
OFFSET {offset}
"#,
            language_qid = self.config.language_qid,
            language_code = self.config.language_code,
            limit = limit,
            offset = offset,
        )
    }

    fn build_relation_query(&self, senses: &[SenseEntry]) -> String {
        let sense_values = senses
            .iter()
            .map(|sense| format!("wd:{}", sense.sense_id))
            .collect::<Vec<_>>()
            .join(" ");
        let sense_properties = sense_property_values();
        let item_properties = item_property_values();
        format!(
            r#"PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?sense ?property ?propertyId ?propertyLabel ?target ?targetLabel ?targetDescription
WHERE {{
  VALUES ?sense {{ {sense_values} }}
  {{
    VALUES ?propertyDirect {{ {sense_properties} }}
    ?sense ?propertyDirect ?target .
  }}
  UNION
  {{
    ?sense wdt:P5137 ?item .
    VALUES ?propertyDirect {{ {item_properties} }}
    ?item ?propertyDirect ?target .
  }}
  BIND(STRAFTER(STR(?propertyDirect), "http://www.wikidata.org/prop/direct/") AS ?propertyId)
  BIND(IRI(CONCAT("http://www.wikidata.org/entity/", ?propertyId)) AS ?property)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language_code},en". }}
}}
"#,
            sense_values = sense_values,
            sense_properties = sense_properties,
            item_properties = item_properties,
            language_code = self.config.language_code,
        )
    }
}

fn should_retry_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
            | StatusCode::BAD_GATEWAY
            | StatusCode::REQUEST_TIMEOUT
            | StatusCode::INTERNAL_SERVER_ERROR
    )
}

fn retry_delay(attempt: u32) -> Duration {
    let step = 1u64 << (attempt.saturating_sub(1));
    Duration::from_millis(RETRY_BASE_DELAY_MS.saturating_mul(step))
}

#[derive(Debug, Deserialize)]
struct SenseQueryResponse {
    results: SenseQueryResults,
}

#[derive(Debug, Deserialize)]
struct SenseQueryResults {
    bindings: Vec<SenseQueryBinding>,
}

#[derive(Debug, Deserialize)]
struct SenseQueryBinding {
    lexeme: SparqlValue,
    #[serde(rename = "lexemeLabel")]
    lexeme_label: Option<SparqlValue>,
    lemma: SparqlValue,
    sense: SparqlValue,
    #[serde(rename = "senseDescription")]
    sense_description: Option<SparqlValue>,
}

#[derive(Debug, Clone)]
struct SenseEntry {
    lexeme_id: String,
    lemma: String,
    lexeme_label: Option<String>,
    sense_id: String,
    sense_description: Option<String>,
}

impl TryFrom<SenseQueryBinding> for SenseEntry {
    type Error = anyhow::Error;

    fn try_from(value: SenseQueryBinding) -> Result<Self> {
        let lexeme_id = extract_entity_id(&value.lexeme.value)
            .ok_or_else(|| {
                anyhow!(
                    "identifiant de lexème introuvable dans {}",
                    value.lexeme.value
                )
            })?
            .to_string();
        let sense_id = extract_entity_id(&value.sense.value)
            .ok_or_else(|| anyhow!("identifiant de sens introuvable dans {}", value.sense.value))?
            .to_string();
        Ok(SenseEntry {
            lexeme_id,
            lemma: value.lemma.value,
            lexeme_label: value.lexeme_label.map(|v| v.value),
            sense_id,
            sense_description: value.sense_description.map(|v| v.value),
        })
    }
}

impl SenseEntry {
    fn combine(&self, relation: RelationRow) -> SenseRelationRecord {
        SenseRelationRecord {
            lexeme_id: self.lexeme_id.clone(),
            lemma: self.lemma.clone(),
            lexeme_label: self.lexeme_label.clone(),
            sense_id: self.sense_id.clone(),
            _sense_label: None,
            sense_description: self.sense_description.clone(),
            property_id: relation.property_id,
            property_label: relation.property_label,
            target_id: relation.target_id,
            target_literal: relation.target_literal,
            target_datatype: relation.target_datatype,
            target_label: relation.target_label,
            target_description: relation.target_description,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RelationQueryResponse {
    results: RelationQueryResults,
}

#[derive(Debug, Deserialize)]
struct RelationQueryResults {
    bindings: Vec<RelationBinding>,
}

#[derive(Debug, Deserialize)]
struct RelationBinding {
    sense: SparqlValue,
    #[serde(rename = "propertyId")]
    property_id: SparqlValue,
    #[serde(rename = "propertyLabel")]
    property_label: Option<SparqlValue>,
    target: SparqlValue,
    #[serde(rename = "targetLabel")]
    target_label: Option<SparqlValue>,
    #[serde(rename = "targetDescription")]
    target_description: Option<SparqlValue>,
}

#[derive(Debug)]
struct RelationRow {
    sense_id: String,
    property_id: String,
    property_label: Option<String>,
    target_id: Option<String>,
    target_literal: Option<String>,
    target_datatype: Option<String>,
    target_label: Option<String>,
    target_description: Option<String>,
}

impl TryFrom<RelationBinding> for RelationRow {
    type Error = anyhow::Error;

    fn try_from(value: RelationBinding) -> Result<Self> {
        let sense_id = extract_entity_id(&value.sense.value)
            .ok_or_else(|| anyhow!("identifiant de sens introuvable dans {}", value.sense.value))?
            .to_string();
        let property_id = value.property_id.value;
        let property_label = value.property_label.map(|v| v.value);
        let (target_id, target_literal, target_datatype) = match value.target.value_type.as_str() {
            "uri" => (
                extract_entity_id(&value.target.value).map(|id| id.to_string()),
                None,
                None,
            ),
            _ => (
                None,
                Some(value.target.value.trim().to_string()),
                value.target.datatype.clone(),
            ),
        };
        let target_label = value.target_label.map(|v| v.value);
        let target_description = value.target_description.map(|v| v.value);

        Ok(RelationRow {
            sense_id,
            property_id,
            property_label,
            target_id,
            target_literal,
            target_datatype,
            target_label,
            target_description,
        })
    }
}

#[derive(Debug)]
struct SenseRelationRecord {
    lexeme_id: String,
    lemma: String,
    lexeme_label: Option<String>,
    sense_id: String,
    _sense_label: Option<String>,
    sense_description: Option<String>,
    property_id: String,
    property_label: Option<String>,
    target_id: Option<String>,
    target_literal: Option<String>,
    target_datatype: Option<String>,
    target_label: Option<String>,
    target_description: Option<String>,
}

impl SenseRelationRecord {
    fn object_label_and_alias(&self) -> Option<(String, Option<String>)> {
        if let Some(id) = &self.target_id {
            if let Some(label) = self
                .target_label
                .as_ref()
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
            {
                return Some((format!("{} ({})", label, id), Some(label.to_string())));
            }
            return Some((id.to_string(), None));
        }

        if let Some(label) = self
            .target_label
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        {
            return Some((label.to_string(), None));
        }

        self.formatted_literal().map(|literal| (literal, None))
    }

    fn relation_kind(&self) -> Option<RelationKind> {
        infer_relation_kind(&self.property_id, self.property_label.as_deref())
    }

    fn object_concept_kind(&self, relation_kind: RelationKind) -> ConceptKind {
        match relation_kind {
            RelationKind::CategoryInstance
            | RelationKind::FunctionUsage
            | RelationKind::GoalAction
            | RelationKind::ProcessResult
            | RelationKind::Condition
            | RelationKind::Similarity
            | RelationKind::Comparison
            | RelationKind::SocialConstruct
            | RelationKind::PropertyEntity
            | RelationKind::MaterialObject
            | RelationKind::PartWhole
            | RelationKind::InitialFinalState
            | RelationKind::TimeEvent
            | RelationKind::LocationEntity => ConceptKind::Abstract,
            RelationKind::CauseEffect
            | RelationKind::AgentAction
            | RelationKind::ActionObject
            | RelationKind::Opposition
            | RelationKind::Dependence
            | RelationKind::Possession => ConceptKind::Entity,
        }
    }

    fn build_justification(&self) -> Option<String> {
        let property_label = self.property_label.as_deref().unwrap_or(&self.property_id);
        let description = self
            .target_description
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                self.sense_description
                    .as_deref()
                    .filter(|value| !value.trim().is_empty())
            });
        let target = self
            .object_label_and_alias()
            .map(|(label, _)| label)
            .or_else(|| self.target_id.clone());

        match (description, target) {
            (Some(description), Some(target)) => {
                Some(format!("{} — {} — {}", property_label, target, description))
            }
            (Some(description), None) => Some(format!("{} — {}", property_label, description)),
            (None, Some(target)) => Some(format!("{} — {}", property_label, target)),
            _ => Some(property_label.to_string()),
        }
    }

    fn entity_url(&self) -> String {
        format!("https://www.wikidata.org/wiki/{}", self.lexeme_id)
    }

    fn formatted_literal(&self) -> Option<String> {
        let literal = self.target_literal.as_ref()?.trim();
        if literal.is_empty() {
            return None;
        }

        if let Some(datatype) = self.target_datatype.as_deref() {
            if datatype.ends_with("dateTime") {
                if let Ok(parsed) = DateTime::parse_from_rfc3339(literal) {
                    return Some(parsed.to_rfc3339_opts(SecondsFormat::Secs, true));
                }
                let normalized = literal.trim_start_matches('+');
                if let Ok(parsed) = DateTime::parse_from_rfc3339(normalized) {
                    return Some(parsed.to_rfc3339_opts(SecondsFormat::Secs, true));
                }
                if let Ok(parsed) = NaiveDateTime::parse_from_str(normalized, "%Y-%m-%dT%H:%M:%S") {
                    return Some(parsed.format("%Y-%m-%dT%H:%M:%S").to_string());
                }
            } else if datatype.ends_with("date") {
                let normalized = literal.trim_start_matches('+');
                if let Ok(parsed) = DateTime::parse_from_rfc3339(normalized) {
                    return Some(parsed.date_naive().format("%Y-%m-%d").to_string());
                }
                if let Ok(parsed) = NaiveDate::parse_from_str(normalized, "%Y-%m-%d") {
                    return Some(parsed.format("%Y-%m-%d").to_string());
                }
                if normalized.len() >= 10 {
                    return Some(normalized[..10].to_string());
                }
            } else if datatype.ends_with("gYear") {
                return Some(literal.trim_start_matches('+').to_string());
            }
        }

        Some(literal.to_string())
    }
}

#[derive(Debug, Deserialize)]
struct SparqlValue {
    #[serde(rename = "type")]
    value_type: String,
    value: String,
    #[serde(rename = "xml:lang", default)]
    _lang: Option<String>,
    #[serde(default)]
    datatype: Option<String>,
}

fn extract_entity_id(uri: &str) -> Option<&str> {
    uri.rsplit('/').next()
}

fn infer_relation_kind(property_id: &str, property_label: Option<&str>) -> Option<RelationKind> {
    match property_id {
        // Sense-level statements
        "P5137" => Some(RelationKind::CategoryInstance),
        "P5973" => Some(RelationKind::Similarity),
        "P5974" => Some(RelationKind::Opposition),
        "P5975" => Some(RelationKind::CategoryInstance),
        "P5976" => Some(RelationKind::Opposition),
        "P5978" => Some(RelationKind::Condition),
        "P6084" => Some(RelationKind::LocationEntity),
        "P6593" => Some(RelationKind::CategoryInstance),
        // Item-level statements on the linked concept
        "P31" | "P279" => Some(RelationKind::CategoryInstance),
        "P361" | "P527" => Some(RelationKind::PartWhole),
        "P276" | "P7153" => Some(RelationKind::LocationEntity),
        "P585" | "P580" | "P582" => Some(RelationKind::TimeEvent),
        "P186" => Some(RelationKind::MaterialObject),
        "P127" => Some(RelationKind::Possession),
        "P828" | "P1542" => Some(RelationKind::CauseEffect),
        "P366" | "P2283" => Some(RelationKind::FunctionUsage),
        "P3712" => Some(RelationKind::GoalAction),
        "P1552" => Some(RelationKind::PropertyEntity),
        "P460" => Some(RelationKind::Similarity),
        "P461" | "P1889" => Some(RelationKind::Opposition),
        _ => property_label
            .map(|label| label.to_lowercase())
            .and_then(|lower| match lower {
                value
                    if value.contains("synonym")
                        || value.contains("equivalent")
                        || value.contains("variant")
                        || value.contains("similar") =>
                {
                    Some(RelationKind::Similarity)
                }
                value
                    if value.contains("antonym")
                        || value.contains("opposite")
                        || value.contains("contraire")
                        || value.contains("different") =>
                {
                    Some(RelationKind::Opposition)
                }
                value
                    if value.contains("hypernym")
                        || value.contains("hyperonyme")
                        || value.contains("superordinate")
                        || value.contains("broader concept")
                        || value.contains("is a") =>
                {
                    Some(RelationKind::CategoryInstance)
                }
                value
                    if value.contains("hyponym")
                        || value.contains("hyponyme")
                        || value.contains("narrower concept")
                        || value.contains("type of") =>
                {
                    Some(RelationKind::CategoryInstance)
                }
                value
                    if value.contains("holonym")
                        || value.contains("meronym")
                        || value.contains("part of")
                        || value.contains("composant")
                        || value.contains("membre de") =>
                {
                    Some(RelationKind::PartWhole)
                }
                value
                    if value.contains("used for")
                        || value.contains("use")
                        || value.contains("utilisé pour")
                        || value.contains("purpose")
                        || value.contains("fonction")
                        || value.contains("usage") =>
                {
                    Some(RelationKind::FunctionUsage)
                }
                value
                    if value.contains("requires")
                        || value.contains("requires the condition")
                        || value.contains("condition")
                        || value.contains("nécessite") =>
                {
                    Some(RelationKind::Condition)
                }
                value
                    if value.contains("depends on")
                        || value.contains("dépend")
                        || value.contains("needs") =>
                {
                    Some(RelationKind::Dependence)
                }
                value
                    if value.contains("cause")
                        || value.contains("causes")
                        || value.contains("résultat")
                        || value.contains("produces") =>
                {
                    Some(RelationKind::CauseEffect)
                }
                value
                    if value.contains("result")
                        || value.contains("produces")
                        || value.contains("gives") =>
                {
                    Some(RelationKind::ProcessResult)
                }
                value
                    if value.contains("agent")
                        || value.contains("performed by")
                        || value.contains("acteur") =>
                {
                    Some(RelationKind::AgentAction)
                }
                value
                    if value.contains("object")
                        || value.contains("patient")
                        || value.contains("cible") =>
                {
                    Some(RelationKind::ActionObject)
                }
                value
                    if value.contains("property")
                        || value.contains("quality")
                        || value.contains("characteristic") =>
                {
                    Some(RelationKind::PropertyEntity)
                }
                value
                    if value.contains("location")
                        || value.contains("lieu")
                        || value.contains("place") =>
                {
                    Some(RelationKind::LocationEntity)
                }
                value
                    if value.contains("time")
                        || value.contains("moment")
                        || value.contains("date") =>
                {
                    Some(RelationKind::TimeEvent)
                }
                value
                    if value.contains("material")
                        || value.contains("made of")
                        || value.contains("matière") =>
                {
                    Some(RelationKind::MaterialObject)
                }
                value
                    if value.contains("goal")
                        || value.contains("purpose")
                        || value.contains("objectif") =>
                {
                    Some(RelationKind::GoalAction)
                }
                value
                    if value.contains("possession")
                        || value.contains("owner")
                        || value.contains("possède") =>
                {
                    Some(RelationKind::Possession)
                }
                value if value.contains("comparison") || value.contains("comparé à") => {
                    Some(RelationKind::Comparison)
                }
                value if value.contains("social") || value.contains("institution") => {
                    Some(RelationKind::SocialConstruct)
                }
                _ => None,
            }),
    }
}

fn sense_property_values() -> String {
    SENSE_PROPERTY_IDS
        .iter()
        .map(|id| format!("wdt:{}", id))
        .collect::<Vec<_>>()
        .join(" ")
}

fn item_property_values() -> String {
    ITEM_PROPERTY_IDS
        .iter()
        .map(|id| format!("wdt:{}", id))
        .collect::<Vec<_>>()
        .join(" ")
}

fn language_qid(code: &str) -> Option<&'static str> {
    match code {
        "fr" => Some("Q150"),
        "en" => Some("Q1860"),
        "es" => Some("Q1321"),
        "de" => Some("Q188"),
        "it" => Some("Q652"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sample_response() {
        let sense_json = r#"{
            "results": {
                "bindings": [
                    {
                        "lexeme": {"type": "uri", "value": "http://www.wikidata.org/entity/L123"},
                        "lexemeLabel": {"type": "literal", "value": "chat"},
                        "lemma": {"type": "literal", "xml:lang": "fr", "value": "chat"},
                        "sense": {"type": "uri", "value": "http://www.wikidata.org/entity/L123-S1"},
                        "senseDescription": {"type": "literal", "xml:lang": "fr", "value": "animal domestique"}
                    }
                ]
            }
        }"#;

        let sense_resp: SenseQueryResponse = serde_json::from_str(sense_json).unwrap();
        let senses: Vec<SenseEntry> = sense_resp
            .results
            .bindings
            .into_iter()
            .map(|binding| binding.try_into().unwrap())
            .collect();
        assert_eq!(senses.len(), 1);
        let sense = &senses[0];

        let relation_json = r#"{
            "results": {
                "bindings": [
                    {
                        "sense": {"type": "uri", "value": "http://www.wikidata.org/entity/L123-S1"},
                        "propertyId": {"type": "literal", "value": "P5137"},
                        "propertyLabel": {"type": "literal", "xml:lang": "en", "value": "item for this sense"},
                        "target": {"type": "uri", "value": "http://www.wikidata.org/entity/Q146"},
                        "targetLabel": {"type": "literal", "xml:lang": "fr", "value": "chat"},
                        "targetDescription": {"type": "literal", "xml:lang": "fr", "value": "animal domestique"}
                    }
                ]
            }
        }"#;

        let relation_resp: RelationQueryResponse = serde_json::from_str(relation_json).unwrap();
        let mut relations: Vec<RelationRow> = relation_resp
            .results
            .bindings
            .into_iter()
            .map(|binding| binding.try_into().unwrap())
            .collect();
        assert_eq!(relations.len(), 1);
        let record = sense.combine(relations.remove(0));
        assert_eq!(record.lemma, "chat");
        assert_eq!(record.lexeme_id, "L123");
        assert_eq!(record.sense_id, "L123-S1");
        assert_eq!(record.property_id, "P5137");
        assert_eq!(record.target_id.as_deref(), Some("Q146"));
        let relation_kind = record.relation_kind().unwrap();
        assert_eq!(relation_kind, RelationKind::CategoryInstance);
        assert_eq!(
            record.object_concept_kind(relation_kind),
            ConceptKind::Abstract
        );
        let (label, alias) = record
            .object_label_and_alias()
            .expect("object label should exist");
        assert_eq!(label, "chat (Q146)");
        assert_eq!(alias.as_deref(), Some("chat"));
        let justification = record.build_justification().unwrap();
        assert!(justification.contains("item for this sense"));
        assert!(justification.contains("Q146"));
        assert!(record.entity_url().contains("L123"));
    }
}
