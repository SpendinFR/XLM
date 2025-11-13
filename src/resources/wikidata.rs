
use std::collections::{HashMap, HashSet};
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

/// Identifiants des propriétés à propager depuis l'item Q... (cible de P5137).
const PROPAGATE_PIDS: &[&str] = &[
    "P31",  // instance of
    "P279", // subclass of
    "P361", // part of
    "P527", // has part
    "P276", // location
    "P7153",// significant place
    "P186", // made from material
    "P127", // owned by
    "P460", // said to be the same as
    "P461", // opposite of
    "P1889",// different from
    "P828", // has cause
    "P1542",// has effect
    "P366", // has use
    "P2283",// uses
    "P3712",// has goal
    "P1552",// has characteristic
];

/// QIDs trop génériques à ignorer lors de la propagation (bruit).
const SKIP_PROPAGATED_QIDS: &[&str] = &[
    "Q111352", // lexeme — peu utile pour structurer tes données
    // ajoute ici d'autres QIDs génériques si besoin (ex: "Q35120" concept ?) — laissé volontairement de côté
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
    /// Propage des relations depuis l'item de P5137 (P31/P279 et autres PIDs choisis).
    pub propagate_item_properties: bool,
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
            propagate_item_properties: true, // activé par défaut
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

    pub fn with_propagate_item_properties(mut self, value: bool) -> Self {
        self.propagate_item_properties = value;
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
        // cache des propriétés déjà récupérées pour un item Q...
        let mut item_cache: HashMap<String, Vec<ItemProp>> = HashMap::new();

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

                let Some(relation_kind) = record.relation_kind() else {
                    continue;
                };

                // Sujet = concept pour le lemme
                let lemma_key = lemma.to_lowercase();
                let existed = graph.get_concept_by_label(lemma).is_some();
                let subject = graph.ensure_concept(lemma, ConceptKind::Entity);
                if !existed && lemmas_seen.insert(lemma_key) {
                    stats.lemmas_added += 1;
                    stats.concepts_created += 1;
                }

                // Alias éventuel
                if let Some(alias) = record
                    .lexeme_label
                    .as_ref()
                    .map(|value| value.trim())
                    .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case(lemma))
                {
                    graph.add_alias(&subject.id, alias.to_string());
                }

                // Objet direct (depuis la ligne SPARQL principale)
                let Some(object_label) = record.object_label() else {
                    continue;
                };
                let object_existed = graph.get_concept_by_label(&object_label).is_some();
                let object_kind = record.object_concept_kind(relation_kind);
                let object = graph.ensure_concept(&object_label, object_kind);
                if !object_existed {
                    stats.concepts_created += 1;
                }

                // Ajout de la relation directe
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

                // === Propagation depuis l'item cible de P5137 =====================
                if self.config.propagate_item_properties
                    && record.property_id == "P5137"
                    && record.target_id.is_some()
                {
                    let item_qid = record.target_id.as_ref().unwrap().clone(); // Qxxxx

                    // Récupération (avec cache) des propriétés de l'item
                    let props = if let Some(cached) = item_cache.get(&item_qid) {
                        cached.clone()
                    } else {
                        match self.fetch_item_props(&item_qid) {
                            Ok(v) => {
                                item_cache.insert(item_qid.clone(), v.clone());
                                v
                            }
                            Err(err) => {
                                warn!(%item_qid, "échec de propagation depuis l'item: {}", err);
                                Vec::new()
                            }
                        }
                    };

                    for ItemProp { pid, target_qid, target_label } in props {
                        if SKIP_PROPAGATED_QIDS.contains(&target_qid.as_str()) {
                            continue;
                        }

                        // Déduire le RelationKind pour ce pid
                        let rk = match infer_relation_kind(&pid, Some(property_label_for(&pid))) {
                            Some(kind) => kind,
                            None => continue,
                        };

                        // Objet = classe/cible de l'item
                        let label = target_label.clone().unwrap_or_else(|| target_qid.clone());
                        let existed = graph.get_concept_by_label(&label).is_some();
                        let obj_kind = SenseRelationRecord::fake_object_concept_kind(rk);
                        let obj = graph.ensure_concept(&label, obj_kind);
                        if !existed {
                            stats.concepts_created += 1;
                        }

                        // Ajoute relation propagée (faible bruit)
                        let already = graph.has_relation(&subject.id, rk, &obj.id);

                        let justification = Some(format!(
                            "{} — {} ({}) — propagé depuis l'item {} (item for this sense)",
                            property_label_for(&pid), label, target_qid, item_qid
                        ));

                        let mut source = AssertionSource::default();
                        source.dataset = Some("wikidata_lexeme_derived".to_string());
                        source.external_id = Some(format!("{}#{}", record.lexeme_id, record.sense_id));
                        source.url = Some(format!("https://www.wikidata.org/wiki/{}", item_qid));

                        graph.add_relation(
                            subject.clone(),
                            rk,
                            obj,
                            Some(0.9),
                            justification,
                            source,
                        )?;

                        if !already {
                            stats.relations_added += 1;
                        }
                    }
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

    fn fetch_batch(&self, query: &str) -> Result<Vec<SenseRelationRecord>> {
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

SELECT DISTINCT ?lexeme ?lexemeLabel ?lemma ?sense ?senseLabel ?senseDescription ?property ?propertyId ?propertyLabel ?target ?targetLabel ?targetDescription
WHERE {{
  ?lexeme a ontolex:LexicalEntry ;
          dct:language wd:{language_qid} ;
          wikibase:lemma ?lemma ;
          ontolex:sense ?sense .
  FILTER(LANG(?lemma) = "{language_code}")
  ?sense ?prop ?target .
  FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
  BIND(STRAFTER(STR(?prop), "http://www.wikidata.org/prop/direct/") AS ?propertyId)
  BIND(IRI(CONCAT("http://www.wikidata.org/entity/", ?propertyId)) AS ?property)
  FILTER(!isLiteral(?target))
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

    /// Récupère un grand éventail de propriétés de wd:Q... (cible de P5137).
    fn fetch_item_props(&self, qid: &str) -> Result<Vec<ItemProp>> {
        let joined = PROPAGATE_PIDS
            .iter()
            .map(|pid| format!("wdt:{}", pid))
            .collect::<Vec<_>>()
            .join(" ");
        let query = format!(
            r#"
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd:  <http://www.wikidata.org/entity/>
SELECT ?p ?target ?targetLabel ?targetDescription WHERE {{
  VALUES ?p {{ {joined} }}
  wd:{qid} ?p ?target .
  FILTER(!isLiteral(?target))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language},en". }}
}}
"#,
            joined = joined,
            qid = qid,
            language = self.config.language_code
        );

        let response = self
            .client
            .post(&self.config.endpoint)
            .body(query)
            .send()
            .context("échec lors de la récupération des propriétés de l'item")?
            .error_for_status()
            .context("réponse HTTP invalide depuis Wikidata (item props)")?;

        let parsed: ItemPropResponse = response
            .json()
            .context("impossible d'analyser la réponse SPARQL (item props)")?;

        let mut out = Vec::new();
        for b in parsed.results.bindings {
            let pid = extract_entity_id(&b.p.value).unwrap_or("").to_string(); // P31, ...
            let target_qid = extract_entity_id(&b.target.value).unwrap_or("").to_string(); // Qxxxx
            if pid.is_empty() || target_qid.is_empty() {
                continue;
            }
            let target_label = b.target_label.map(|v| v.value);
            out.push(ItemProp { pid, target_qid, target_label });
        }
        Ok(out)
    }
}

#[derive(Debug, Deserialize)]
struct SparqlResponse {
    results: SparqlResults,
}

impl SparqlResponse {
    fn into_records(self) -> Vec<SenseRelationRecord> {
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
    sense: SparqlValue,
    #[serde(rename = "senseLabel")]
    sense_label: Option<SparqlValue>,
    #[serde(rename = "senseDescription")]
    sense_description: Option<SparqlValue>,
    #[serde(rename = "property")]
    _property_entity: Option<SparqlValue>,
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

impl TryFrom<SparqlBinding> for SenseRelationRecord {
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
        let sense_id = extract_entity_id(&value.sense.value)
            .ok_or_else(|| anyhow!("identifiant de sens introuvable dans {}", value.sense.value))?
            .to_string();
        let lexeme_label = value.lexeme_label.map(|v| v.value);
        let _sense_label = value.sense_label.map(|v| v.value);
        let sense_description = value.sense_description.map(|v| v.value);
        let property_id = value.property_id.value;
        let property_label = value.property_label.map(|v| v.value);
        let target_id = extract_entity_id(&value.target.value).map(|id| id.to_string());
        let target_label = value.target_label.map(|v| v.value);
        let target_description = value.target_description.map(|v| v.value);

        Ok(SenseRelationRecord {
            lexeme_id,
            lemma,
            lexeme_label,
            sense_id,
            _sense_label,
            sense_description,
            property_id,
            property_label,
            target_id,
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
    target_label: Option<String>,
    target_description: Option<String>,
}

impl SenseRelationRecord {
    fn object_label(&self) -> Option<String> {
        if let Some(label) = &self.target_label {
            let trimmed = label.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        self.target_id.clone()
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
            | RelationKind::InitialFinalState => ConceptKind::Abstract,
            RelationKind::TimeEvent => ConceptKind::Abstract,
            RelationKind::LocationEntity => ConceptKind::Abstract,
            RelationKind::CauseEffect
            | RelationKind::AgentAction
            | RelationKind::ActionObject
            | RelationKind::Opposition
            | RelationKind::Dependence
            | RelationKind::Possession => ConceptKind::Entity,
        }
    }

    /// Variante "statique" pour l'objet lors des propagations (on n'a pas l'enregistrement complet).
    fn fake_object_concept_kind(relation_kind: RelationKind) -> ConceptKind {
        Self::object_concept_kind_for(relation_kind)
    }

    fn object_concept_kind_for(relation_kind: RelationKind) -> ConceptKind {
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
            | RelationKind::InitialFinalState => ConceptKind::Abstract,
            RelationKind::TimeEvent => ConceptKind::Abstract,
            RelationKind::LocationEntity => ConceptKind::Abstract,
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
        let target = match (
            self.target_label
                .as_deref()
                .map(|value| value.trim())
                .filter(|value| !value.is_empty()),
            self.target_id.as_deref(),
        ) {
            (Some(label), Some(id)) => Some(format!("{} ({})", label, id)),
            (Some(label), None) => Some(label.to_string()),
            (None, Some(id)) => Some(id.to_string()),
            _ => None,
        };

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

/// Table de correspondance (propriétés Wikidata -> RelationKind)
/// - Ajouts : P5973, P5976 (Similarity) ; P5974, P461, P1889 (Opposition) ; P6593, P5975, P31, P279 (CategoryInstance)
///   P5978 (Condition) ; P6084, P276, P7153 (LocationEntity) ; P361, P527 (PartWhole) ;
///   P186 (MaterialObject) ; P127 (Possession) ; P460 (Similarity) ;
///   P828, P1542 (CauseEffect) ; P3712 (GoalAction) ; P1552 (PropertyEntity).
/// - Retirés : P6091, P7948, P9444, P5257.
fn infer_relation_kind(property_id: &str, property_label: Option<&str>) -> Option<RelationKind> {
    match property_id {
        // === CategoryInstance ===================================================
        "P5137" => Some(RelationKind::CategoryInstance), // item for this sense
        "P6593" => Some(RelationKind::CategoryInstance), // hyperonym
        "P5975" => Some(RelationKind::CategoryInstance), // troponym of
        "P31"   => Some(RelationKind::CategoryInstance), // instance of
        "P279"  => Some(RelationKind::CategoryInstance), // subclass of

        // === PartWhole ==========================================================
        "P5971" => Some(RelationKind::PartWhole), // (historique, conservé)
        "P361"  => Some(RelationKind::PartWhole), // part of
        "P527"  => Some(RelationKind::PartWhole), // has part

        // === FunctionUsage ======================================================
        "P5972" => Some(RelationKind::FunctionUsage), // (historique)
        "P366"  => Some(RelationKind::FunctionUsage), // has use
        "P2283" => Some(RelationKind::FunctionUsage), // uses

        // === Similarity =========================================================
        "P5973" => Some(RelationKind::Similarity), // synonym
        "P5976" => Some(RelationKind::Similarity), // false friend
        "P460"  => Some(RelationKind::Similarity), // said to be the same as

        // === Opposition / Différence ============================================
        "P5974" => Some(RelationKind::Opposition), // antonym
        "P461"  => Some(RelationKind::Opposition), // opposite of
        "P1889" => Some(RelationKind::Opposition), // different from

        // === Condition ==========================================================
        "P5978" => Some(RelationKind::Condition), // classifier (usage constraints)

        // === LocationEntity =====================================================
        "P6084" => Some(RelationKind::LocationEntity), // location of sense usage
        "P276"  => Some(RelationKind::LocationEntity), // location
        "P7153" => Some(RelationKind::LocationEntity), // significant place

        // === MaterialObject =====================================================
        "P186"  => Some(RelationKind::MaterialObject), // made from material

        // === Possession =========================================================
        "P127"  => Some(RelationKind::Possession), // owned by

        // === CauseEffect ========================================================
        "P828"  => Some(RelationKind::CauseEffect), // has cause
        "P1542" => Some(RelationKind::CauseEffect), // has effect

        // === GoalAction =========================================================
        "P3712" => Some(RelationKind::GoalAction), // has goal

        // === PropertyEntity =====================================================
        "P1552" => Some(RelationKind::PropertyEntity), // has characteristic

        // === Autres conservés de la table originale ============================
        "P9987" | "P6887" | "P3831" => Some(RelationKind::CauseEffect),

        // === Retirés (ne plus mapper) ==========================================
        "P6091" | "P7948" | "P9444" | "P5257" => None,

        // === Heuristiques sur les libellés =====================================
        _ => property_label
            .map(|label| label.to_lowercase())
            .and_then(|lower| match lower.as_str() {
                // Similarité
                v if v.contains("synonym")
                    || v.contains("synonyme")
                    || v.contains("equivalent")
                    || v.contains("variant")
                    || v.contains("similar")
                    || v.contains("false friend")
                    || v.contains("faux ami") => Some(RelationKind::Similarity),

                // Opposition / différence
                v if v.contains("antonym")
                    || v.contains("antonyme")
                    || v.contains("opposite")
                    || v.contains("oppose")
                    || v.contains("contraire")
                    || v.contains("different from")
                    || v.contains("différent de")
                    || v.contains("contraste") => Some(RelationKind::Opposition),

                // Catégorie / hyper-/hyponymie
                v if v.contains("hypernym")
                    || v.contains("hyperonyme")
                    || v.contains("superordinate")
                    || v.contains("broader concept")
                    || v.contains("is a")
                    || v.contains("instance of")
                    || v.contains("subclass")
                    || v.contains("troponym") => Some(RelationKind::CategoryInstance),

                v if v.contains("hyponym")
                    || v.contains("hyponyme")
                    || v.contains("narrower concept")
                    || v.contains("type of") => Some(RelationKind::CategoryInstance),

                // Part–Whole
                v if v.contains("holonym")
                    || v.contains("meronym")
                    || v.contains("part of")
                    || v.contains("has part")
                    || v.contains("composant")
                    || v.contains("membre de") => Some(RelationKind::PartWhole),

                // Usage / fonction
                v if v.contains("used for")
                    || v.contains("use")
                    || v.contains("utilisé pour")
                    || v.contains("purpose")
                    || v.contains("fonction")
                    || v.contains("usage")
                    || v.contains("has use")
                    || v.contains("uses") => Some(RelationKind::FunctionUsage),

                // Condition / contrainte
                v if v.contains("requires")
                    || v.contains("requires the condition")
                    || v.contains("condition")
                    || v.contains("nécessite")
                    || v.contains("classifier") => Some(RelationKind::Condition),

                // Dépendance
                v if v.contains("depends on")
                    || v.contains("dépend")
                    || v.contains("needs") => Some(RelationKind::Dependence),

                // Cause / effet
                v if v.contains("cause")
                    || v.contains("causes")
                    || v.contains("résultat")
                    || v.contains("produces")
                    || v.contains("has cause")
                    || v.contains("has effect") => Some(RelationKind::CauseEffect),

                // Résultat de processus
                v if v.contains("result")
                    || v.contains("produces")
                    || v.contains("gives") => Some(RelationKind::ProcessResult),

                // Agent / action / patient
                v if v.contains("agent")
                    || v.contains("performed by")
                    || v.contains("acteur") => Some(RelationKind::AgentAction),
                v if v.contains("object")
                    || v.contains("patient")
                    || v.contains("cible") => Some(RelationKind::ActionObject),

                // Propriété / qualité
                v if v.contains("property")
                    || v.contains("quality")
                    || v.contains("characteristic")
                    || v.contains("has characteristic") => Some(RelationKind::PropertyEntity),

                // Lieu
                v if v.contains("location")
                    || v.contains("lieu")
                    || v.contains("place")
                    || v.contains("significant place")
                    || v.contains("location of sense usage") => Some(RelationKind::LocationEntity),

                // Temps
                v if v.contains("time")
                    || v.contains("moment")
                    || v.contains("date") => Some(RelationKind::TimeEvent),

                // Matière / matériau
                v if v.contains("material")
                    || v.contains("made of")
                    || v.contains("made from material")
                    || v.contains("matière") => Some(RelationKind::MaterialObject),

                // But
                v if v.contains("goal")
                    || v.contains("purpose")
                    || v.contains("objectif")
                    || v.contains("has goal") => Some(RelationKind::GoalAction),

                // Possession
                v if v.contains("possession")
                    || v.contains("owner")
                    || v.contains("owned by")
                    || v.contains("possède") => Some(RelationKind::Possession),

                // Comparaison
                v if v.contains("comparison") || v.contains("comparé à") => {
                    Some(RelationKind::Comparison)
                }

                // Constructions sociales
                v if v.contains("social") || v.contains("institution") => {
                    Some(RelationKind::SocialConstruct)
                }

                _ => None,
            }),
    }
}

fn property_label_for(pid: &str) -> &'static str {
    match pid {
        "P5137" => "item for this sense",
        "P6593" => "hyperonym",
        "P5975" => "troponym of",
        "P31" => "instance of",
        "P279" => "subclass of",
        "P5971" => "part-whole (legacy)",
        "P361" => "part of",
        "P527" => "has part",
        "P5972" => "translation to sense (legacy)",
        "P366" => "has use",
        "P2283" => "uses",
        "P5973" => "synonym",
        "P5976" => "false friend",
        "P460" => "said to be the same as",
        "P5974" => "antonym",
        "P461" => "opposite of",
        "P1889" => "different from",
        "P5978" => "classifier",
        "P6084" => "location of sense usage",
        "P276" => "location",
        "P7153" => "significant place",
        "P186" => "made from material",
        "P127" => "owned by",
        "P828" => "has cause",
        "P1542" => "has effect",
        "P3712" => "has goal",
        "P1552" => "has characteristic",
        _ => "property",
    }
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

/// Propriété d'item (propagée) -> cible Q et label éventuel.
#[derive(Clone, Debug)]
struct ItemProp {
    pid: String,            // "P31" etc.
    target_qid: String,     // "Qxxxx"
    target_label: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ItemPropResponse {
    results: ItemPropResults,
}

#[derive(Debug, Deserialize)]
struct ItemPropResults {
    bindings: Vec<ItemPropBinding>,
}

#[derive(Debug, Deserialize)]
struct ItemPropBinding {
    #[serde(rename = "p")]
    p: SparqlValue,
    #[serde(rename = "target")]
    target: SparqlValue,
    #[serde(rename = "targetLabel")]
    target_label: Option<SparqlValue>,
    #[serde(rename = "targetDescription")]
    _target_description: Option<SparqlValue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_new_properties() {
        // mappages ajoutés
        assert_eq!(infer_relation_kind("P5973", Some("synonym")), Some(RelationKind::Similarity));
        assert_eq!(infer_relation_kind("P5976", Some("false friend")), Some(RelationKind::Similarity));
        assert_eq!(infer_relation_kind("P5974", Some("antonym")), Some(RelationKind::Opposition));
        assert_eq!(infer_relation_kind("P6593", Some("hyperonym")), Some(RelationKind::CategoryInstance));
        assert_eq!(infer_relation_kind("P5975", Some("troponym of")), Some(RelationKind::CategoryInstance));
        assert_eq!(infer_relation_kind("P5978", Some("classifier")), Some(RelationKind::Condition));
        assert_eq!(infer_relation_kind("P6084", Some("location of sense usage")), Some(RelationKind::LocationEntity));
        assert_eq!(infer_relation_kind("P31", Some("instance of")), Some(RelationKind::CategoryInstance));
        assert_eq!(infer_relation_kind("P279", Some("subclass of")), Some(RelationKind::CategoryInstance));
        assert_eq!(infer_relation_kind("P361", Some("part of")), Some(RelationKind::PartWhole));
        assert_eq!(infer_relation_kind("P527", Some("has part")), Some(RelationKind::PartWhole));
        assert_eq!(infer_relation_kind("P276", Some("location")), Some(RelationKind::LocationEntity));
        assert_eq!(infer_relation_kind("P7153", Some("significant place")), Some(RelationKind::LocationEntity));
        assert_eq!(infer_relation_kind("P186", Some("made from material")), Some(RelationKind::MaterialObject));
        assert_eq!(infer_relation_kind("P127", Some("owned by")), Some(RelationKind::Possession));
        assert_eq!(infer_relation_kind("P461", Some("opposite of")), Some(RelationKind::Opposition));
        assert_eq!(infer_relation_kind("P460", Some("said to be the same as")), Some(RelationKind::Similarity));
        assert_eq!(infer_relation_kind("P1889", Some("different from")), Some(RelationKind::Opposition));
        assert_eq!(infer_relation_kind("P828", Some("has cause")), Some(RelationKind::CauseEffect));
        assert_eq!(infer_relation_kind("P1542", Some("has effect")), Some(RelationKind::CauseEffect));
        assert_eq!(infer_relation_kind("P366", Some("has use")), Some(RelationKind::FunctionUsage));
        assert_eq!(infer_relation_kind("P2283", Some("uses")), Some(RelationKind::FunctionUsage));
        assert_eq!(infer_relation_kind("P3712", Some("has goal")), Some(RelationKind::GoalAction));
        assert_eq!(infer_relation_kind("P1552", Some("has characteristic")), Some(RelationKind::PropertyEntity));

        // IDs retirés
        assert_eq!(infer_relation_kind("P6091", Some("MGG Online ID")), None);
        assert_eq!(infer_relation_kind("P7948", Some("GreatSchools ID")), None);
        assert_eq!(infer_relation_kind("P9444", Some("CAOI person ID")), None);
        assert_eq!(infer_relation_kind("P5257", Some("identifiant taxonomique")), None);
    }
}
