use std::collections::HashSet;
use std::thread::sleep;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use reqwest::blocking::Client as HttpClient;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE, USER_AGENT};
use serde::Deserialize;
use tracing::{info, warn};

use crate::domain::{AssertionSource, ConceptKind, RelationKind};
use crate::memory::KnowledgeGraph;

const DEFAULT_ENDPOINT: &str = "https://query.wikidata.org/sparql";
const DEFAULT_USER_AGENT: &str = "XLM-Concept-Graph/0.1 (+https://github.com/)";

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
            let query = self.build_query(limit, offset);
            let records = match self.fetch_batch(&query) {
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

                let Some(object_label) = record.object_label() else {
                    continue;
                };
                let object_existed = graph.get_concept_by_label(&object_label).is_some();
                let object = graph.ensure_concept(&object_label, ConceptKind::Abstract);
                if !object_existed {
                    stats.concepts_created += 1;
                }

                let already =
                    graph.has_relation(&subject.id, RelationKind::CategoryInstance, &object.id);
                let justification = record.build_justification();
                let mut source = AssertionSource::default();
                source.dataset = Some("wikidata_lexeme".to_string());
                source.external_id = Some(record.lexeme_id.clone());
                source.url = Some(record.entity_url());

                graph.add_relation(
                    subject.clone(),
                    RelationKind::CategoryInstance,
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

    fn fetch_batch(&self, query: &str) -> Result<Vec<LexemeRecord>> {
        let response = self
            .client
            .post(&self.config.endpoint)
            .body(query.to_string())
            .send()
            .context("échec lors de l'appel SPARQL à Wikidata")?
            .error_for_status()
            .context("réponse HTTP invalide depuis Wikidata")?;

        let parsed: SparqlResponse = response
            .json()
            .context("impossible d'analyser la réponse SPARQL de Wikidata")?;
        Ok(parsed.into_records())
    }

    fn build_query(&self, limit: usize, offset: usize) -> String {
        format!(
            r#"PREFIX dct: <http://purl.org/dc/terms/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?lexeme ?lexemeLabel ?lemma ?item ?itemLabel ?itemDescription
WHERE {{
  ?lexeme a ontolex:LexicalEntry ;
          dct:language wd:{language_qid} ;
          wikibase:lemma ?lemma ;
          ontolex:sense ?sense .
  ?sense wdt:P5137 ?item .
  FILTER(LANG(?lemma) = "{language_code}")
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language_code}". }}
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
}

#[derive(Debug, Deserialize)]
struct SparqlResponse {
    results: SparqlResults,
}

impl SparqlResponse {
    fn into_records(self) -> Vec<LexemeRecord> {
        self.results
            .bindings
            .into_iter()
            .filter_map(|binding| binding.try_into().ok())
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct SparqlResults {
    bindings: Vec<SparqlBinding>,
}

#[derive(Debug, Deserialize)]
struct SparqlBinding {
    lexeme: SparqlValue,
    #[serde(rename = "lexemeLabel")]
    lexeme_label: Option<SparqlValue>,
    lemma: SparqlValue,
    item: SparqlValue,
    #[serde(rename = "itemLabel")]
    item_label: Option<SparqlValue>,
    #[serde(rename = "itemDescription")]
    item_description: Option<SparqlValue>,
}

impl TryFrom<SparqlBinding> for LexemeRecord {
    type Error = anyhow::Error;

    fn try_from(value: SparqlBinding) -> Result<Self> {
        let lemma = value.lemma.value;
        let lexeme_id = extract_entity_id(&value.lexeme.value)
            .ok_or_else(|| {
                anyhow!(
                    "identifiant de lexème introuvable dans {}",
                    value.lexeme.value
                )
            })?
            .to_string();
        let lexeme_label = value.lexeme_label.map(|v| v.value);
        let item_id = extract_entity_id(&value.item.value)
            .ok_or_else(|| anyhow!("identifiant d'item introuvable dans {}", value.item.value))?
            .to_string();
        let item_label = value.item_label.map(|v| v.value);
        let item_description = value.item_description.map(|v| v.value);

        Ok(LexemeRecord {
            lexeme_id,
            lemma,
            lexeme_label,
            sense_item_id: Some(item_id),
            sense_label: item_label,
            sense_description: item_description,
        })
    }
}

#[derive(Debug)]
struct LexemeRecord {
    lexeme_id: String,
    lemma: String,
    lexeme_label: Option<String>,
    sense_item_id: Option<String>,
    sense_label: Option<String>,
    sense_description: Option<String>,
}

impl LexemeRecord {
    fn object_label(&self) -> Option<String> {
        if let Some(label) = &self.sense_label {
            let trimmed = label.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        self.sense_item_id.clone()
    }

    fn build_justification(&self) -> Option<String> {
        match (&self.sense_description, &self.sense_item_id) {
            (Some(description), Some(item)) if !description.trim().is_empty() => {
                Some(format!("{} — {}", item, description.trim()))
            }
            (Some(description), _) if !description.trim().is_empty() => {
                Some(description.trim().to_string())
            }
            (_, Some(item)) => Some(format!("Référence Wikidata {}", item)),
            _ => None,
        }
    }

    fn entity_url(&self) -> String {
        format!("https://www.wikidata.org/wiki/{}", self.lexeme_id)
    }
}

#[derive(Debug, Deserialize)]
struct SparqlValue {
    value: String,
    #[serde(rename = "xml:lang", default)]
    _lang: Option<String>,
}

fn extract_entity_id(uri: &str) -> Option<&str> {
    uri.rsplit('/').next()
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
        let json = r#"{
            "results": {
                "bindings": [
                    {
                        "lexeme": {"type": "uri", "value": "http://www.wikidata.org/entity/L123"},
                        "lexemeLabel": {"type": "literal", "value": "chat"},
                        "lemma": {"type": "literal", "xml:lang": "fr", "value": "chat"},
                        "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q146"},
                        "itemLabel": {"type": "literal", "xml:lang": "fr", "value": "chat"},
                        "itemDescription": {"type": "literal", "xml:lang": "fr", "value": "animal domestique"}
                    }
                ]
            }
        }"#;

        let parsed: SparqlResponse = serde_json::from_str(json).unwrap();
        let records = parsed.into_records();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.lemma, "chat");
        assert_eq!(record.lexeme_id, "L123");
        assert_eq!(record.sense_item_id.as_deref(), Some("Q146"));
        assert_eq!(record.object_label().as_deref(), Some("chat"));
        assert!(record.build_justification().unwrap().contains("Q146"));
        assert!(record.entity_url().contains("L123"));
    }
}
