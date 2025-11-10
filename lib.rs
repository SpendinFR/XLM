use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::collections::{BTreeSet, HashSet, BTreeMap};
use std::fs;
use regex::{Regex, RegexBuilder};
use rusqlite::{Connection, types::ValueRef};
#[cfg(feature = "rag_syn")]
use xlm_rag as rag;
use reqwest::blocking::Client;
use once_cell::sync::Lazy;
use chrono::{Utc, NaiveDate};

//========================//
//   Core data structs    //
//========================//

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capsule {
    pub typ: String,
    pub val: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: Option<String>,
    pub op: String,
    #[serde(default)]
    pub r#in: Value,
    #[serde(default)]
    pub out: Value,
    #[serde(default)]
    pub params: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Program {
    pub program: Vec<Node>,
}

pub trait Kernel: Send + Sync {
    fn name(&self) -> &'static str;
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, params: &Value) -> Result<Capsule>;
}

pub struct Vm {
    kernels: HashMap<String, Box<dyn Kernel>>,
}

impl Vm {
    pub fn new() -> Self {
        let mut vm = Vm { kernels: HashMap::new() };
        vm.register(SumMoney);
        vm.register(Vat);
        vm.register(CheckEquals);
        vm.register(RenderText);
        vm.register(CheckJsonSchema);
        vm.register(CheckMoney);
        vm.register(CheckDateBetween);
        vm.register(SqliteCall);
        vm.register(RagQuery);
        vm.register(PlanCompile);
        vm.register(PlanRun);
        vm.register(CriticCompact);
        vm.register(CriticGuardJson);
        vm.register(PlanLint);
        vm.register(RenderInvoiceJson);
        vm.register(RulesApply);
        vm.register(RulesGet);
        vm.register(CriticApplyRules);
        vm.register(ToolsRoute);
        vm.register(EvalAddCase);
        vm.register(EvalRun);
        vm.register(PlanAutogen);
        vm.register(PlanPatch);
        vm.register(ConceptLoad);
        vm.register(ConceptBind);
        vm.register(ConceptQuery);
        vm.register(SurfaceRealize);
        vm.register(ConceptAbstract);
        vm.register(TransferClassify);
        vm.register(ReasonAbduce);
        vm.register(ReasonCounterfactual);
        vm.register(InstanceApplyPatch);
        vm.register(ReasonPlan);
        vm.register(InstanceGetRole);
        vm.register(InstanceNew);
        vm.register(CdlInduce);
        vm.register(ConceptSave);
        vm.register(PlanApply);
        vm.register(StoryCompose);
        vm.register(PlanWrapText);
        vm.register(ReasonCausalize);
        vm.register(PlanConcat);
        vm.register(PlanFilter);
        vm.register(CdlFuse);
        vm.register(AnalogyMap);
        vm.register(AnalogyRemapInstance);
        vm.register(AnalogyRemap);
        vm.register(ConceptReason);
        vm.register(ConceptSpecialize);
        vm.register(ConceptValidateInstance);
        vm.register(LearnFromCorpus);
        vm.register(ReasonGraphActivate);
        vm.register(ComposeGeneric);
        vm.register(CdlAutoCurate);
        vm.register(AutoGrow);
        vm.register(CuesLearn);
        vm.register(CuesFuse);
        vm.register(EdgesUpdate);
        vm.register(GraphEdgesUpdate);
        vm.register(ComposeFromActivate);
        vm.register(ReasonGraphObserve);
        vm.register(LexiconUpdate);
        vm.register(EntitiesLearn);
        vm.register(FactsLearn);
        vm.register(DefaultsSuggest);

                        // 🔥 nouvelles ops
        vm.register(ComputeAddMoney);
        vm.register(ProveAll);
        vm.register(CiteRequire);
        vm
    }
    pub fn register<K: Kernel + 'static>(&mut self, k: K) {
        self.kernels.insert(k.name().to_string(), Box::new(k));
    }
    pub fn run(&self, p: &Program) -> Result<Vec<Capsule>> {
        let mut env: HashMap<String, Capsule> = HashMap::new();
        let mut outs = Vec::new();
        for node in &p.program {
            let op = &node.op;
            let k = self.kernels.get(op).ok_or_else(|| anyhow!("unknown op: {}", op))?;
            let cap = k.run(&env, &node.r#in, &node.params)?;
            if let Some(id) = &node.id {
                env.insert(id.clone(), cap.clone());
            }
            outs.push(cap);
        }
        Ok(outs)
    }
}

//========================//
//   Helpers              //
//========================//
fn read_json(path: &str) -> Option<Value> {
    fs::read_to_string(path).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok())
}
fn write_json(path: &str, v: &Value) -> Result<()> {
    let s = serde_json::to_string_pretty(v)?;
    fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(std::path::Path::new(".")))?;
    fs::write(path, s)?;
    Ok(())
}
fn today_iso() -> String {
    chrono::Utc::now().naive_utc().date().format("%Y-%m-%d").to_string()
}
fn exp_decay(lambda: f64, last: &str) -> f64 {
    if let Ok(d) = chrono::NaiveDate::parse_from_str(last, "%Y-%m-%d") {
        let t = chrono::Utc::now().naive_utc().date();
        (-lambda * (t - d).num_days() as f64).exp()
    } else { 1.0 }
}
fn incr_count(map: &mut serde_json::Map<String, Value>, key: &str, add: f64, last_seen: &str) {
    use serde_json::json;
    let entry = map.entry(key.to_string()).or_insert(json!({"count":0.0,"last_seen":"1970-01-01"}));
    let c0 = entry.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let l0 = entry.get("last_seen").and_then(|x| x.as_str()).unwrap_or("1970-01-01");
    let last = if last_seen > l0 { last_seen } else { l0 };
    *entry = json!({"count": c0 + add, "last_seen": last});
}
fn union_str_vec(a: Option<&Value>, b: Option<&Value>) -> Vec<Value> {
    let mut set = BTreeSet::<String>::new();
    for src in [a, b] {
        if let Some(Value::Array(arr)) = src {
            for it in arr {
                if let Some(s) = it.as_str() { set.insert(s.to_string()); }
            }
        } else if let Some(Value::Object(m)) = src {
            for k in m.keys() { set.insert(k.to_string()); }
        }
    }
    set.into_iter().map(Value::String).collect()
}
fn tokenize_fr(s: &str) -> Vec<String> {
    let mut v = vec![];
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() || "éèàùâêîôûçÉÈÀÙÂÊÎÔÛÇ'".contains(ch) {
            cur.push(ch);
        } else {
            if !cur.is_empty() { v.push(cur.to_lowercase()); cur.clear(); }
        }
    }
    if !cur.is_empty() { v.push(cur.to_lowercase()); }
    v
}
fn head_verb_guess(tokens: &[String]) -> Option<String> {
    // Heuristique ultra simple : premier verbe candidat hors stoplist
    let stop = ["le","la","les","un","une","des","à","pour","de","du","au","aux","et","en","sur","dans","par","chez","avec"];
    for t in tokens {
        if stop.contains(&t.as_str()) { continue; }
        // si ça finit par "er/ir/re" (très grossier) on le prend comme "lemme candidat"
        if t.ends_with("er") || t.ends_with("ir") || t.ends_with("re") || t.ends_with("dre") { return Some(t.clone()); }
        // sinon on retourne le premier token non stop (verbe conjugué probable)
        return Some(t.clone());
    }
    None
}
fn frame_signature_from_roles(inst: &serde_json::Value) -> String {
    let has_cons = inst.pointer("/roles/Consideration").is_some();
    if has_cons {
        "Agent V Theme à Receiver pour Consideration".to_string()
    } else {
        "Agent V Theme à Receiver".to_string()
    }
}
fn merge_sorted_unique(mut base: Vec<String>, adds: &[String]) -> Vec<String> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<String> = base.into_iter().collect();
    for s in adds { set.insert(s.clone()); }
    set.into_iter().collect()
}

#[cfg(feature = "rag_syn")]
fn rag_synonyms_fr(lemma: &str, k: usize) -> Vec<String> {
    // Exemple minimal : on récupère 1–2 variantes plausibles depuis l’index local
    // (adapte si ton xlm_rag expose une autre API).
    let q = format!("synonymes du verbe '{}'", lemma);
    if let Ok(docs) = rag::query(&q, k) {
        let mut out = Vec::new();
        for d in docs {
            // Très permissif : prend des mots uniques en minuscules dans le titre
            for w in d.title.split(|c: char| !c.is_alphabetic()) {
                let w = w.trim().to_lowercase();
                if !w.is_empty() && w != lemma { out.push(w); }
            }
        }
        out
    } else { Vec::new() }
}

#[cfg(not(feature = "rag_syn"))]

fn rag_synonyms_fr(_lemma: &str, _k: usize) -> Vec<String> { Vec::new() }

fn tool_registry() -> Value {
    // Registre minimal : nom d'op + exemples d'input/params
    json!([
      {"op":"compute.sum_money","in":{"table":[{"amount_cents":1500,"currency":"EUR"},{"amount_cents":3500,"currency":"EUR"}],"currency":"EUR"}},
      {"op":"compute.add_money","in":{"items":["$subtotal","$vat"],"currency":"EUR"}},
      {"op":"compute.vat","in":{"money":"$subtotal","rate":0.20}},
      {"op":"check.equals","in":{"lhs":"$a","rhs":"$b"}},
      {"op":"prove.all","in":{"tests":[{"op":"equals","lhs":"$x","rhs":"$y"}]}},
      {"op":"render.text","in":{"subtotal":"$subtotal","vat":"$vat","total":"$total"},"params":{"style":"neutral|brief"}},
      {"op":"render.invoice.json","in":{"subtotal":"$subtotal","vat":"$vat","total":"$total","currency":"EUR","issued_at":"YYYY-MM-DD"}},
      {"op":"critic.compact","in":{"text":"$text","max_chars":80}},
      {"op":"critic.guard.json","in":{"json":"$obj","required_keys":["..."]}},
      {"op":"rag.query","in":{"q":"...","k":2}},
      {"op":"cite.require","in":{"sources":"$sources","min":1}},
      {"op":"plan.compile","in":{"steps":[{"id":"...","op":"...","in":{}}]}},
      {"op":"plan.run","in":{"program":"$compiled"}},
      {"op":"plan.lint","in":{"program":"$compiled"}},
      {"op":"tools.route","in":{"program":"$compiled","budget_ms":120}}
    ])
}

fn extract_json_from_text(s: &str) -> Result<Value> {
    // tente d'abord un parse direct
    if let Ok(v) = serde_json::from_str::<Value>(s) { return Ok(v); }
    // cherche un bloc ```json ... ```
    if let Some(start) = s.find("```json") {
        if let Some(end) = s[start+7..].find("```") {
            let block = &s[start+7 .. start+7+end];
            if let Ok(v) = serde_json::from_str::<Value>(block) { return Ok(v); }
        }
    }
    // fallback : entre la première '{' et la dernière '}'
    if let (Some(i), Some(j)) = (s.find('{'), s.rfind('}')) {
        if i < j {
            let slice = &s[i..=j];
            if let Ok(v) = serde_json::from_str::<Value>(slice) { return Ok(v); }
        }
    }
    Err(anyhow!("plan.autogen: no valid JSON found in LLM output"))
}

// petit filtre "allow" : préfixe* ou nom exact
fn allow_filter<'a>(ops: &'a [Value], allow: &Option<Vec<String>>) -> Vec<&'a Value> {
    if let Some(list) = allow {
        let mut out = Vec::new();
        for op in ops {
            let name = op.get("op").and_then(|v| v.as_str()).unwrap_or("");
            if list.iter().any(|p| {
                if let Some(pref) = p.strip_suffix('*') { name.starts_with(pref) } else { name == p }
            }) {
                out.push(op);
            }
        }
        out
    } else {
        ops.iter().collect()
    }
}

fn resolve(env: &std::collections::HashMap<String, Capsule>, v: &serde_json::Value) -> Result<serde_json::Value> {
    if let Some(s) = v.as_str() {
        if let Some(rest) = s.strip_prefix('$') {
            let mut parts = rest.split('.');
            let id = parts.next().unwrap_or("");
            let cap = env.get(id).ok_or_else(|| anyhow!("missing ref ${}", rest))?;
            let mut cur = cap.val.clone();

            for seg in parts {
                if seg == "out" {               // <-- on ignore ce segment (style $node.out)
                    continue;
                }
                if let Ok(idx) = seg.parse::<usize>() {
                    let arr = cur.as_array().ok_or_else(|| anyhow!("ref ${}: '{}' is not an array", rest, seg))?;
                    cur = arr.get(idx).cloned().ok_or_else(|| anyhow!("ref ${}: index {} out of bounds", rest, idx))?;
                } else {
                    let obj = cur.as_object().ok_or_else(|| anyhow!("ref ${}: '{}' is not an object", rest, seg))?;
                    cur = obj.get(seg).cloned().ok_or_else(|| anyhow!("ref ${}: missing field '{}'", rest, seg))?;
                }
            }
            return Ok(cur);
        }
    }
    Ok(v.clone())
}
fn load_concept_by_id(id: &str, lang: &str) -> Result<Value> {
    let path = format!("./concepts/{}.{}.json", id, lang);
    let s = fs::read_to_string(&path).map_err(|e| anyhow!("concept.load: read {}: {}", path, e))?;
    let v: Value = serde_json::from_str(&s).map_err(|e| anyhow!("concept.load: parse {}: {}", path, e))?;
    Ok(v)
}

fn collect_defaults_with_parents(concept: &Value, lang: &str, seen: &mut HashSet<String>) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    if let Some(cid) = concept.get("id").and_then(|v| v.as_str()) {
        if !seen.insert(cid.to_string()) {
            return Ok(out); // évite cycles
        }
    }
    if let Some(defs) = concept.get("defaults").and_then(|v| v.as_array()) {
        for d in defs { out.push(d.clone()); }
    }
    if let Some(parents) = concept.get("parents").and_then(|v| v.as_array()) {
        for pidv in parents {
            if let Some(pid) = pidv.as_str() {
                if let Ok(pc) = load_concept_by_id(pid, lang) {
                    let mut sub = collect_defaults_with_parents(&pc, lang, seen)?;
                    out.append(&mut sub);
                }
            }
        }
    }
    Ok(out)
}

fn role_present(roles: &serde_json::Map<String, Value>, name: &str) -> bool {
    roles.get(name)
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
}



//========================//
//   Kernels              //
//========================//

// -- Money ops --

struct SumMoney;
impl Kernel for SumMoney {
    fn name(&self) -> &'static str { "compute.sum_money" }

    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        use serde_json::Value;

        fn parse_amount_to_cents(v: &Value) -> Result<i64> {
            if let Some(n) = v.as_f64() {
                return Ok((n * 100.0).round() as i64);
            }
            if let Some(s) = v.as_str() {
                let s2 = s.trim()
                    .replace(",", ".")
                    .replace("€", "")
                    .replace("EUR", "")
                    .trim()
                    .to_string();
                let n: f64 = s2.parse().map_err(|_| anyhow!("amount parse error: {}", s))?;
                return Ok((n * 100.0).round() as i64);
            }
            Err(anyhow!("unsupported amount type: {}", v))
        }

        // Essaie d’extraire une liste de "rows" depuis divers formats
        fn coerce_rows(v: &Value) -> Option<Vec<Value>> {
            // 1) Déjà un array
            if let Some(a) = v.as_array() { return Some(a.clone()); }

            // 2) Objet avec clés usuelles: table/rows/items/values/lines/amounts/entries/list
            if let Some(obj) = v.as_object() {
                for k in ["table","rows","items","values","lines","amounts","entries","list"] {
                    if let Some(a) = obj.get(k).and_then(|x| x.as_array()) {
                        return Some(a.clone());
                    }
                }
                // Objet unique {amount_cents|amount|value|price}
                if obj.contains_key("amount_cents") || obj.contains_key("amount")
                    || obj.contains_key("value") || obj.contains_key("price")
                {
                    return Some(vec![Value::Object(obj.clone())]);
                }
            }

            // 3) String style "15,35" ou "15+35" ou "15 35" → split
            if let Some(s) = v.as_str() {
                let mut out = Vec::new();
                let norm = s.replace(|c: char| c == '+' || c == ';' || c == '|' || c.is_whitespace(), ",");
                for tok in norm.split(',') {
                    let t = tok.trim();
                    if !t.is_empty() { out.push(Value::String(t.to_string())); }
                }
                if !out.is_empty() { return Some(out); }
            }
            None
        }

        // 1) currency (top-level param prioritaire)
        let currency = input.get("currency").and_then(|c| c.as_str()).unwrap_or("EUR");

        // 2) rows : d’abord input["table"], sinon l’objet `input` lui-même
        let rows_opt = if input.get("table").is_some() {
            let table_v = resolve(env, &input["table"]).ok();
            table_v.and_then(|tv| coerce_rows(&tv))
        } else {
            coerce_rows(input)
        };

        let rows = rows_opt.ok_or_else(|| anyhow!("sum_money expects array of rows"))?;

        // 3) agrégation
        let mut total: i64 = 0;
        for row in &rows {
            match row {
                Value::Object(obj) => {
                    // amount_cents direct
                    if let Some(c) = obj.get("amount_cents").and_then(|x| x.as_i64()) {
                        let cur = obj.get("currency").and_then(|v| v.as_str()).unwrap_or(currency);
                        if cur != currency { return Err(anyhow!("mixed currencies: {} vs {}", cur, currency)); }
                        total += c;
                        continue;
                    }
                    // amount/value/price (nombre ou string) → cents
                    if let Some(a) = obj.get("amount").or_else(|| obj.get("value")).or_else(|| obj.get("price")) {
                        let cents = parse_amount_to_cents(a)?;
                        let cur = obj.get("currency").and_then(|v| v.as_str()).unwrap_or(currency);
                        if cur != currency { return Err(anyhow!("mixed currencies: {} vs {}", cur, currency)); }
                        total += cents;
                        continue;
                    }
                    return Err(anyhow!("row object must contain amount_cents OR amount/value/price"));
                }
                // formats “naturels” : 15, "15.00", etc.
                Value::Number(_) | Value::String(_) => {
                    let cents = parse_amount_to_cents(row)?;
                    total += cents;
                }
                _ => return Err(anyhow!("unsupported row type in table")),
            }
        }

        Ok(Capsule {
            typ: "Money".into(),
            val: json!({"amount_cents": total, "currency": currency})
        })
    }
}





struct ComputeAddMoney;
impl Kernel for ComputeAddMoney {
    fn name(&self) -> &'static str { "compute.add_money" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "items": [ "$a", "$b", {...} ], "currency": "EUR" }
        let items_v = input.get("items").ok_or_else(|| anyhow!("items required"))?;
        let arr = match resolve(env, items_v)? {
            Value::Array(a) => a,
            _ => return Err(anyhow!("items must be array")),
        };
        let currency = input.get("currency").and_then(|v| v.as_str()).unwrap_or("EUR");
        let mut total: i64 = 0;
        for it in arr {
            let m = match it {
                Value::String(s) if s.starts_with('$') => resolve(env, &Value::String(s))?,
                other => other,
            };
            let cents = m.get("amount_cents").and_then(|v| v.as_i64()).ok_or_else(|| anyhow!("money.amount_cents"))?;
            let cur = m.get("currency").and_then(|v| v.as_str()).unwrap_or(currency);
            if cur != currency { return Err(anyhow!("currency mismatch {} != {}", cur, currency)); }
            total += cents;
        }
        Ok(Capsule{ typ: "Money".into(), val: json!({"amount_cents": total, "currency": currency}) })
    }
}

struct Vat;
impl Kernel for Vat {
    fn name(&self) -> &'static str { "compute.vat" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        fn parse_rate(v: &Value) -> Result<f64> {
            if let Some(f) = v.as_f64() {
                return Ok(if f > 1.0 { f / 100.0 } else { f });
            }
            if let Some(s) = v.as_str() {
                let s2 = s.trim().trim_end_matches('%').trim();
                let f: f64 = s2.parse().map_err(|_| anyhow!("rate parse error: {}", s))?;
                return Ok(if s.ends_with('%') || f > 1.0 { f / 100.0 } else { f });
            }
            Err(anyhow!("rate must be number or string"))
        }

        let money_v = resolve(env, &input["money"])?;
        let rate = parse_rate(input.get("rate").ok_or_else(|| anyhow!("missing rate"))?)?;
        let cents = money_v.get("amount_cents").and_then(|v| v.as_i64()).ok_or_else(|| anyhow!("money missing amount_cents"))?;
        let vat_cents = ((cents as f64) * rate).round() as i64;
        let currency = money_v.get("currency").and_then(|v| v.as_str()).unwrap_or("EUR");
        Ok(Capsule { typ: "Money".into(), val: json!({"amount_cents": vat_cents, "currency": currency}) })
    }
}

// -- RAG --

struct RagQuery;

impl Kernel for RagQuery {
    fn name(&self) -> &'static str { "rag.query" }

    fn run(
        &self,
        _env: &HashMap<String, Capsule>,
        input: &Value,
        _params: &Value
    ) -> Result<Capsule> {
        let q = input.get("q").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("q required"))?;
        let k = input.get("k").and_then(|v| v.as_u64()).unwrap_or(3) as usize;

        #[cfg(feature = "rag_syn")]
        {
            let docs = rag::query(q, k)?; // lit rag_index.json
            let arr: Vec<Value> = docs.into_iter().map(|d| {
                json!({"uri": d.uri, "title": d.title, "snippet": d.text.chars().take(240).collect::<String>()})
            }).collect();
            return Ok(Capsule{ typ: "Sources".into(), val: Value::Array(arr) });
        }

        #[cfg(not(feature = "rag_syn"))]
        {
            return Err(anyhow!("rag.query: build sans feature 'rag_syn' (active-la ou retire cet op)"));
        }
    }
}

// -- Plan compile --

struct PlanCompile;
impl Kernel for PlanCompile {
    fn name(&self) -> &'static str { "plan.compile" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input:
        //   { "steps": [ ... ] }
        //   ou
        //   { "steps": "$draft" }   where $draft = { "program":[ ... ] }
        //   ou
        //   { "steps": "$draft.program" }  (ancienne tentative)
        let raw = input.get("steps").ok_or_else(|| anyhow!("steps required"))?;

        // 1) resolve $ref
        let mut resolved = resolve(env, raw)?;

        // 2) si c’est un objet {program:[...]} -> extraire program
        if resolved.get("program").and_then(|v| v.as_array()).is_some() {
            resolved = resolved.get("program").cloned().unwrap();
        }

        // 3) à ce stade on veut un array de steps
        let steps = resolved.as_array().ok_or_else(|| anyhow!("steps must be array"))?;

        // 4) validation: chaque op existe
        let vm = Vm::new();
        for (i, st) in steps.iter().enumerate() {
            let op = st.get("op").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("steps[{}].op missing", i))?;
            if !vm.kernels.contains_key(op) {
                return Err(anyhow!("unknown op in steps[{}]: {}", i, op));
            }
        }

        Ok(Capsule{ typ: "Program".into(), val: json!({ "program": steps }) })
    }
}


// -- Plan run --

struct PlanRun;
impl Kernel for PlanRun {
    fn name(&self) -> &'static str { "plan.run" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "program": "$compiled" | { "program": [...] } }
        let prog_v_resolved = resolve(env, input.get("program").ok_or_else(|| anyhow!("program required"))?)?;
        let sub: Program = serde_json::from_value(prog_v_resolved)?;
        let vm = Vm::new();
        let outs = vm.run(&sub)?;
        let trace: Vec<Value> = outs.into_iter().map(|c| serde_json::json!({"typ": c.typ, "val": c.val})).collect();
        Ok(Capsule{ typ: "Trace".into(), val: serde_json::json!({ "outputs": trace }) })
    }
}


// -- Critic compact (anti-blabla simple) --

struct CriticCompact;
impl Kernel for CriticCompact {
    fn name(&self) -> &'static str { "critic.compact" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "text": "$text_full" | "$text_full.text" | "..." , "max_chars"?: 300 }
        let txt_v = resolve(env, &input["text"])?;
        let s = if let Some(s) = txt_v.as_str() {
            s.to_string()
        } else if let Some(obj) = txt_v.as_object() {
            obj.get("text").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("text must be string or {{text: ...}}"))?.to_string()
        } else {
            return Err(anyhow!("text must be string or {{text: ...}}"));
        };

        let max = input.get("max_chars").and_then(|v| v.as_u64()).unwrap_or(280) as usize;

        // heuristiques light : espaces, ponctuation, abréviations simples
        let mut t = s.replace("\r\n", " ").replace('\n', " ");
        t = t.split_whitespace().collect::<Vec<_>>().join(" ");
        t = t.replace("Sous-total", "ST").replace("Total", "T");

        if t.len() > max {
            t.truncate(max.saturating_sub(1));
            t.push('…');
        }
        Ok(Capsule{ typ: "Text".into(), val: serde_json::json!({ "text": t }) })
    }
}
// -- Critic guard JSON (fail-closed structure) --

struct CriticGuardJson;
impl Kernel for CriticGuardJson {
    fn name(&self) -> &'static str { "critic.guard.json" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: l'un des deux:
        //   { "json": { ... } }  |  { "text": "...json..." }
        // + contraintes optionnelles:
        //   { "required_keys": ["a","b"], "numeric_keys": ["x","y"], "non_negative_keys": ["x"], "exact_keys": true }
        let root_v = if input.get("json").is_some() {
            resolve(env, &input["json"])?
        } else if input.get("text").is_some() {
        let txt_v = resolve(env, &input["text"])?;                    // <-- garde en vie
        let s = txt_v.as_str().ok_or_else(|| anyhow!("text must be string JSON"))?
            .to_string();                                            // <-- on possède la String
        serde_json::from_str::<Value>(&s)?
        } else {
            return Err(anyhow!("critic.guard.json: provide either 'json' or 'text'"));
        };

        let obj = root_v.as_object().ok_or_else(|| anyhow!("guard expects a JSON object at root"))?;

        // required keys
        if let Some(req) = input.get("required_keys").and_then(|v| v.as_array()) {
            for k in req {
                let ks = k.as_str().ok_or_else(|| anyhow!("required_keys must be strings"))?;
                if !obj.contains_key(ks) { return Err(anyhow!("missing required key: {}", ks)); }
            }
        }
        // numeric keys
        if let Some(nums) = input.get("numeric_keys").and_then(|v| v.as_array()) {
            for k in nums {
                let ks = k.as_str().ok_or_else(|| anyhow!("numeric_keys must be strings"))?;
                let v = obj.get(ks).ok_or_else(|| anyhow!("numeric key missing: {}", ks))?;
                if !(v.is_number() || (v.is_string() && v.as_str().unwrap().parse::<f64>().is_ok())) {
                    return Err(anyhow!("key '{}' must be numeric", ks));
                }
            }
        }
        // non-negative checks
        if let Some(nn) = input.get("non_negative_keys").and_then(|v| v.as_array()) {
            for k in nn {
                let ks = k.as_str().ok_or_else(|| anyhow!("non_negative_keys must be strings"))?;
                let v = obj.get(ks).ok_or_else(|| anyhow!("non-negative key missing: {}", ks))?;
                let val = if v.is_string() { v.as_str().unwrap().parse::<f64>().ok() } else { v.as_f64() }
                    .ok_or_else(|| anyhow!("key '{}' must be numeric for non-negative check", ks))?;
                if val < 0.0 { return Err(anyhow!("key '{}' must be ≥ 0", ks)); }
            }
        }
        // exact keys (aucune clé en plus)
        if input.get("exact_keys").and_then(|v| v.as_bool()).unwrap_or(false) {
            let req_vec: Vec<String> = input.get("required_keys")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter()
                     .filter_map(|k| k.as_str().map(|s| s.to_string()))
                     .collect())
                .unwrap_or_else(|| Vec::new());

            let set: std::collections::HashSet<String> = req_vec.into_iter().collect();
            if obj.keys().any(|k| !set.contains(k)) {
                return Err(anyhow!("unexpected extra keys present"));
            }
        }

        Ok(Capsule { typ: "JSON".into(), val: Value::Object(obj.clone()) })
    }
}

// -- Plan lint (estimation coût/latence, lisibilité) --

struct PlanLint;
impl Kernel for PlanLint {
    fn name(&self) -> &'static str { "plan.lint" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "program": "$compiled" | { "program":[...] } }
        let prog_v = resolve(env, input.get("program").ok_or_else(|| anyhow!("program required"))?)?;
        let prog: Program = serde_json::from_value(prog_v)?;
        let mut by_op: HashMap<String, usize> = HashMap::new();
        for n in &prog.program {
            *by_op.entry(n.op.clone()).or_insert(0) += 1;
        }
        // coûts arbitraires mais utiles pour routing
        let w = |op: &str| -> u64 {
            match op {
                "call.sqlite"         => 300,
                "rag.query"           => 200,
                "prove.all"           => 40,
                "compute.sum_money"   => 50,
                "compute.add_money"   => 30,
                "compute.vat"         => 30,
                "render.text"         => 10,
                "critic.compact"      => 5,
                "cite.require"        => 20,
                "plan.compile"        => 15,
                "plan.run"            => 25,
                "critic.guard.json"   => 20,
                _ => 25,
            }
        };
        let mut est_ms: u64 = 0;
        for (op, n) in &by_op {
            est_ms += (*n as u64) * w(op);
        }
        Ok(Capsule {
            typ: "Lint".into(),
            val: json!({ "ops_count": prog.program.len(), "by_op": by_op, "est_cost_ms": est_ms }),
        })
    }
}

// -- Render JSON spécifique facture (exemple) --

struct RenderInvoiceJson;
impl Kernel for RenderInvoiceJson {
    fn name(&self) -> &'static str { "render.invoice.json" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "subtotal":"$subtotal","vat":"$vat","total":"$total","currency":"EUR","issued_at":"YYYY-MM-DD" }
        let sub = resolve(env, &input["subtotal"])?;
        let vat = resolve(env, &input["vat"])?;
        let tot = resolve(env, &input["total"])?;
        let cur = input.get("currency").and_then(|v| v.as_str()).unwrap_or("EUR");
        let issued = input.get("issued_at").and_then(|v| v.as_str()).unwrap_or("2025-01-01");

        let sub_c = sub.get("amount_cents").and_then(|v| v.as_i64()).ok_or_else(|| anyhow!("subtotal bad"))?;
        let vat_c = vat.get("amount_cents").and_then(|v| v.as_i64()).ok_or_else(|| anyhow!("vat bad"))?;
        let tot_c = tot.get("amount_cents").and_then(|v| v.as_i64()).ok_or_else(|| anyhow!("total bad"))?;

        let obj = json!({
            "subtotal_cents": sub_c,
            "vat_cents": vat_c,
            "total_cents": tot_c,
            "currency": cur,
            "issued_at": issued
        });
        Ok(Capsule { typ: "JSON".into(), val: obj })
    }
}
// ===== Rules persistence helpers =====
fn rules_path() -> std::path::PathBuf { std::path::PathBuf::from("rules.json") }

fn rules_load() -> Result<std::collections::HashMap<String, Value>> {
    use std::fs;
    let p = rules_path();
    if !p.exists() { return Ok(std::collections::HashMap::new()); }
    let s = fs::read_to_string(p)?;
    let m: std::collections::HashMap<String, Value> = serde_json::from_str(&s)?;
    Ok(m)
}
fn rules_save(m: &std::collections::HashMap<String, Value>) -> Result<()> {
    use std::fs;
    let s = serde_json::to_string_pretty(m)?;
    fs::write(rules_path(), s)?;
    Ok(())
}

// -- rules.apply --
struct RulesApply;
impl Kernel for RulesApply {
    fn name(&self) -> &'static str { "rules.apply" }
    fn run(&self, _env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "name": "brevity", "rule": { "max_chars"?: 80, "replace"?: [ ["A","B"], ... ] } }
        let name = input.get("name").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("name required"))?.to_string();
        let rule = input.get("rule").ok_or_else(|| anyhow!("rule required"))?.clone();
        let mut m = rules_load()?;
        m.insert(name.clone(), rule);
        rules_save(&m)?;
        Ok(Capsule{ typ: "Rules".into(), val: json!({"ok": true, "count": m.len()}) })
    }
}

// -- rules.get --
struct RulesGet;
impl Kernel for RulesGet {
    fn name(&self) -> &'static str { "rules.get" }
    fn run(&self, _env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "name": "brevity" }
        let name = input.get("name").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("name required"))?.to_string();
        let m = rules_load()?;
        let rule = m.get(&name).ok_or_else(|| anyhow!("rule '{}' not found", name))?.clone();
        Ok(Capsule{ typ: "Rule".into(), val: rule })
    }
}

// -- critic.apply_rules --
struct CriticApplyRules;
impl Kernel for CriticApplyRules {
    fn name(&self) -> &'static str { "critic.apply_rules" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "text": "$text_full" | {"text":"..."}, "rules": ["brevity","styleX"] }
        let txt_v = resolve(env, &input["text"])?;
        let mut s = if let Some(st) = txt_v.as_str() {
            st.to_string()
        } else if let Some(obj) = txt_v.as_object() {
            obj.get("text").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("text must be string or {{text: ...}}"))?.to_string()
        } else {
            return Err(anyhow!("text must be string or {{text: ...}}"));
        };

        let names: Vec<String> = input.get("rules").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_else(|| vec![]);

        let m = rules_load()?;
        let mut max_chars: Option<usize> = None;

        for n in names {
            if let Some(r) = m.get(&n) {
                if let Some(mc) = r.get("max_chars").and_then(|v| v.as_u64()) { max_chars = Some(max_chars.unwrap_or(usize::MAX).min(mc as usize)); }
                if let Some(rep) = r.get("replace").and_then(|v| v.as_array()) {
                    for pair in rep {
                        if let (Some(from), Some(to)) = (pair.get(0).and_then(|x| x.as_str()), pair.get(1).and_then(|x| x.as_str())) {
                            s = s.replace(from, to);
                        }
                    }
                }
            }
        }
        // Normalisations légères
        s = s.replace("\r\n", " ").replace('\n', " ");
        s = s.split_whitespace().collect::<Vec<_>>().join(" ");
        if let Some(mx) = max_chars {
            if s.len() > mx {
                let mut t = s;
                t.truncate(mx.saturating_sub(1));
                t.push('…');
                s = t;
            }
        }
        Ok(Capsule{ typ: "Text".into(), val: json!({"text": s}) })
    }
}

// -- tools.route --
struct ToolsRoute;
impl Kernel for ToolsRoute {
    fn name(&self) -> &'static str { "tools.route" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "program": "$compiled"|{program:[...]}, "budget_ms": 80 }
        let prog_v = resolve(env, input.get("program").ok_or_else(|| anyhow!("program required"))?)?;
        let mut prog: Program = serde_json::from_value(prog_v)?;

        let weight = |op: &str| -> u64 {
            match op {
                "call.sqlite"         => 300,
                "rag.query"           => 200,
                "cite.require"        => 120,
                "prove.all"           => 40,
                "compute.sum_money"   => 50,
                "compute.add_money"   => 30,
                "compute.vat"         => 30,
                "render.text"         => 10,
                "critic.compact"      => 5,
                "critic.apply_rules"  => 8,
                "critic.guard.json"   => 20,
                "plan.compile"        => 15,
                "plan.run"            => 25,
                _ => 25,
            }
        };
        let budget = input.get("budget_ms").and_then(|v| v.as_u64()).unwrap_or(100);

        let mut cost = prog.program.iter().map(|n| weight(&n.op)).sum::<u64>();
        if cost <= budget {
            return Ok(Capsule{ typ: "Program".into(), val: serde_json::to_value(&prog)? });
        }

        // stratégie simple: on coupe d'abord rag.query puis cite.require jusqu’à rentrer dans le budget
        let mut steps = prog.program.clone();
        let drop_list = ["rag.query", "cite.require"];
        for target in &drop_list {
            if cost <= budget { break; }
            let before = steps.len();
            steps.retain(|n| n.op != *target);
            if steps.len() != before {
                // recalc cost
                cost = steps.iter().map(|n| weight(&n.op)).sum::<u64>();
            }
        }
        // si toujours trop cher, on tente de supprimer critic.compact (peu impactant)
        if cost > budget {
            let before = steps.len();
            steps.retain(|n| n.op != "critic.compact");
            if steps.len() != before {
                cost = steps.iter().map(|n| weight(&n.op)).sum::<u64>();
            }
        }
        prog.program = steps;
        Ok(Capsule{ typ: "Program".into(), val: serde_json::to_value(&prog)? })
    }
}
// ===== Eval persistence helpers =====
fn eval_path() -> std::path::PathBuf { std::path::PathBuf::from("eval_cases.json") }

fn eval_load() -> Result<Vec<serde_json::Value>> {
    use std::fs;
    let p = eval_path();
    if !p.exists() { return Ok(vec![]); }
    let s = fs::read_to_string(p)?;
    let v: Vec<serde_json::Value> = serde_json::from_str(&s)?;
    Ok(v)
}
fn eval_save(v: &Vec<serde_json::Value>) -> Result<()> {
    use std::fs;
    let s = serde_json::to_string_pretty(v)?;
    fs::write(eval_path(), s)?;
    Ok(())
}

// petite exécution locale qui CAPTURE aussi l'env {id -> Capsule}
pub fn run_with_env(vm: &Vm, p: &Program) -> Result<(Vec<Capsule>, HashMap<String, Capsule>)> {
    let mut env: HashMap<String, Capsule> = HashMap::new();
    let mut outs = Vec::new();
    for node in &p.program {
        let k = vm.kernels.get(&node.op).ok_or_else(|| anyhow!("unknown op: {}", node.op))?;
        let cap = k.run(&env, &node.r#in, &node.params)?;
        if let Some(id) = &node.id { env.insert(id.clone(), cap.clone()); }
        outs.push(cap);
    }
    Ok((outs, env))
}

// -- eval.add_case --
struct EvalAddCase;
impl Kernel for EvalAddCase {
    fn name(&self) -> &'static str { "eval.add_case" }
    fn run(&self, _env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "name": "invoice_20pct", "program": {program:[...]}, "tests": [ ... as in prove.all ... ] }
        let name = input.get("name").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("name required"))?;
        let program = input.get("program").ok_or_else(|| anyhow!("program required"))?.clone();
        let tests   = input.get("tests").ok_or_else(|| anyhow!("tests required"))?.clone();

        // validation légère
        let _ : Program = serde_json::from_value(program.clone())?;
        let _ : Vec<Value> = serde_json::from_value(tests.clone()).map_err(|_| anyhow!("tests must be array"))?;

        let mut cases = eval_load()?;
        cases.retain(|c| c.get("name").and_then(|x| x.as_str()) != Some(name)); // overwrite
        cases.push(json!({"name": name, "program": program, "tests": tests}));
        eval_save(&cases)?;
        Ok(Capsule{ typ: "Eval".into(), val: json!({"ok": true, "cases": cases.len()}) })
    }
}

// -- eval.run --
struct EvalRun;
impl Kernel for EvalRun {
    fn name(&self) -> &'static str { "eval.run" }
    fn run(&self, _env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "filter"?: "prefix*" }
        let filt = input.get("filter").and_then(|v| v.as_str()).unwrap_or("*");
        let cases = eval_load()?;
        let vm = Vm::new();
        let mut total = 0usize;
        let mut ok = 0usize;
        let mut reports: Vec<Value> = vec![];

        for case in cases {
            let name = case.get("name").and_then(|v| v.as_str()).unwrap_or("<noname>");
            // filtre simple prefix*
            if let Some(prefix) = filt.strip_suffix('*') {
                if !name.starts_with(prefix) { continue; }
            }
            total += 1;
            let prog_v = case.get("program").cloned().ok_or_else(|| anyhow!("case missing program"))?;
            let tests  = case.get("tests").cloned().unwrap_or(json!([]));

            let prog: Program = serde_json::from_value(prog_v)?;
            let (_outs, env) = run_with_env(&vm, &prog)?;

            // réutilise la logique de ProveAll
            let prover = ProveAll;
            let res = prover.run(&env, &json!({"tests": tests}), &Value::Null);

            match res {
                Ok(_) => { ok += 1; reports.push(json!({"name": name, "status":"ok"})); }
                Err(e)=> { reports.push(json!({"name": name, "status":"fail", "error": e.to_string()})); }
            }
        }

        Ok(Capsule{
            typ: "EvalReport".into(),
            val: json!({"total": total, "ok": ok, "fail": (total - ok), "reports": reports}),
        })
    }
}

struct PlanAutogen;
impl Kernel for PlanAutogen {
    fn name(&self) -> &'static str { "plan.autogen" }
    fn run(&self, _env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: {
        //   "instruction": "texte utilisateur",
        //   "model"?: "qwen2.5:7b-instruct",        // par défaut "qwen2.5:7b"
        //   "allow"?: ["compute.*","render.text"],  // filtre d'ops
        //   "budget_ms"?: 120
        // }
        let instruction = input.get("instruction").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("instruction required"))?;
        let model = input.get("model").and_then(|v| v.as_str()).unwrap_or("qwen2.5:7b");
        let budget = input.get("budget_ms").and_then(|v| v.as_u64()).unwrap_or(150) as u64;

        let all_tools = tool_registry();
        let tools_arr = all_tools.as_array().ok_or_else(|| anyhow!("internal tools registry bad"))?;
        let allow = input.get("allow").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect::<Vec<_>>());
        let tools_used = allow_filter(tools_arr, &allow);

        let tools_str = serde_json::to_string_pretty(&tools_used.iter().cloned().collect::<Vec<_>>())?;

        let sys = format!(
"Tu es un planneur outillé. Tu dois produire un JSON **strict** qui décrit un programme exécutable par une VM.
Contraintes:
- Utilise UNIQUEMENT les outils listés (voir TOOLS) et leurs schémas.
- Les dépendances se passent via `id` et références `\"$id\"`.
- Le DERNIER nœud doit rendre un texte avec `render.text` ou un JSON via `render.invoice.json`.
- Ajoute une étape `prove.all` si un calcul est fait (ex: total attendu).
- Pour tout fait externe, insère `rag.query` suivi de `cite.require`.
- Respecte un budget approx. de {budget} ms (tu peux réduire rag/cite si nécessaire).

Sortie STRICTE attendue (pas de prose, pas de backticks):
{{\"program\":[{{\"id\":\"...\",\"op\":\"...\",\"in\":{{}},\"params\":{{}}}}]}}
"
        );

        let user = format!(
"INSTRUCTION:
{}

TOOLS (autorisés):
{}

RAPPEL: Retourne UNIQUEMENT l'objet JSON du programme, rien d'autre.", instruction, tools_str);

        let prompt = format!("{sys}\n{user}");

        // Appel Ollama /api/generate (non-streaming)
        let url = "http://127.0.0.1:11434/api/generate";
        let body = json!({
            "model": model,
            "prompt": prompt,
            "stream": false
        });
        let cli = Client::new();
        let resp = cli.post(url).json(&body).send().map_err(|e| anyhow!("ollama http error: {}", e))?;
        let json_resp: Value = resp.json().map_err(|e| anyhow!("ollama decode error: {}", e))?;
        let txt = json_resp.get("response").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("ollama: missing 'response'"))?;
        let plan_v = extract_json_from_text(txt)?;

        // On ne valide pas ici : la pipeline doit appeler plan.compile après.
        Ok(Capsule{ typ: "Program".into(), val: plan_v })
    }
}
// ------------ Concepts: kernels de base ----------------
// -- concept.validate_instance --
struct ConceptValidateInstance;
impl Kernel for ConceptValidateInstance {
    fn name(&self) -> &'static str { "concept.validate_instance" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { concept: <ref|obj>, instance: <ref|obj>, allow_partial?: bool }
        let c = resolve(env, input.get("concept").ok_or_else(|| anyhow!("concept required"))?)?;
        let i = resolve(env, input.get("instance").ok_or_else(|| anyhow!("instance required"))?)?;
        let allow_partial = input.get("allow_partial").and_then(|v| v.as_bool()).unwrap_or(false);

        if !c.is_object() { return Err(anyhow!("concept.validate_instance: 'concept' must be an object")); }
        if !i.is_object() { return Err(anyhow!("concept.validate_instance: 'instance' must be an object")); }

        validate_instance_value(&c, &i, allow_partial)?;
        Ok(Capsule{ typ:"Validated".into(), val: json!({"ok": true}) })
    }
}


struct ConceptLoad;
impl Kernel for ConceptLoad {
    fn name(&self) -> &'static str { "concept.load" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // id/lang/path peuvent être des refs: "$sub", etc.
        let id_v   = resolve(env, input.get("id").ok_or_else(|| anyhow!("concept.load: id required"))?)?;
        let id     = id_v.as_str().ok_or_else(|| anyhow!("concept.load: id must be string"))?;

        let lang_v = if input.get("lang").is_some() {
            Some(resolve(env, &input["lang"])?)
        } else { None };
        let lang   = lang_v.as_ref().and_then(|v| v.as_str()).unwrap_or("fr");

        let path_v = if input.get("path").is_some() {
            Some(resolve(env, &input["path"])?)
        } else { None };

        let path = path_v
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| format!("./concepts/{}.{}.json", id, lang));

        let s = std::fs::read_to_string(&path).map_err(|e| anyhow!("concept.load: read {}: {}", path, e))?;
        let concept: Value = serde_json::from_str(&s).map_err(|e| anyhow!("concept.load: parse {}: {}", path, e))?;
        Ok(Capsule { typ: "Concept".to_string(), val: concept })
    }
}


// --- Binder FR tolérant (Agent VERBE Theme [à X] [pour Y] [pendant Z]) ---
struct ConceptBind;

impl Kernel for ConceptBind {
    fn name(&self) -> &'static str { "concept.bind_text" }

    fn run(
        &self,
        env: &HashMap<String, Capsule>,
        input: &Value,
        _params: &Value
    ) -> Result<Capsule> {

        let concept_v = resolve(env, input.get("concept")
            .ok_or_else(|| anyhow!("concept.bind_text: concept required"))?)?;
        let text = input.get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("concept.bind_text: text required"))?;
        let lang = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");

        // Flags
        let allow_partial = input.get("allow_partial").and_then(|v| v.as_bool()).unwrap_or(false);
        let use_rag_syn   = input.get("use_rag_syn").and_then(|v| v.as_bool()).unwrap_or(false);

        // ID & rôles (si dispo)
        let cid   = concept_v.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let roles_declared: HashSet<String> = concept_v.get("roles")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        // ---- Lexèmes (FR) : lemma, frames, syn ----
        let lex_fr = concept_v.get("lexemes")
            .and_then(|lx| lx.get(lang));

        // lemma (essaie verb_lemma puis lemma)
        let lemma = lex_fr
            .and_then(|m| m.get("verb_lemma").and_then(|v| v.as_str()))
            .or_else(|| lex_fr.and_then(|m| m.get("lemma").and_then(|v| v.as_str())))
            .unwrap_or("");

        // frames
        let frames: Vec<String> = lex_fr
            .and_then(|m| m.get("frames").and_then(|v| v.as_array()))
            .map(|arr| arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        // syn depuis le concept
        let mut syn: Vec<String> = lex_fr
            .and_then(|m| m.get("syn").and_then(|v| v.as_array()))
            .map(|arr| arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        // syn additionnels passés dans l'input (facultatif)
        if let Some(extra) = input.get("syn").and_then(|v| v.as_array()) {
            for s in extra {
                if let Some(w) = s.as_str() {
                    syn.push(w.to_string());
                }
            }
        }

        // Si demandé et que syn est vide -> “piocher” 1–2 variantes via rag (soft, best-effort)
        if use_rag_syn && syn.is_empty() && !lemma.is_empty() {
            if let Some(rag_syns) = try_rag_synonyms(lemma) {
                for s in rag_syns { syn.push(s); }
            }
        }

        // Construire les formes verbales autorisées (lemma + 3ᵉ pers + variantes + leurs 3ᵉ pers)
        let verb_forms = make_verb_forms(lemma, &syn);
        let verb_alts  = alt_union(&verb_forms); // "(?:vendre|vend|céder|cède|...)"

        // Si on a des frames, on essaie de binder via frames → regex nommés
        let mut best_capture: Option<HashMap<String, String>> = None;
        let mut best_score: usize = 0;

        for fr in &frames {
            let (re_str, roles_in_frame) = build_regex_from_frame(fr, &verb_alts);
            let re = RegexBuilder::new(&re_str).case_insensitive(true).unicode(true).build();
            let re = match re { Ok(r) => r, Err(_) => continue };

            if let Some(caps) = re.captures(text) {
                let mut got: HashMap<String, String> = HashMap::new();
                for rname in &roles_in_frame {
                    if let Some(m) = caps.name(rname) {
                        let val = normalize_role_value(m.as_str());
                        if !val.is_empty() {
                            got.insert(rname.clone(), val);
                        }
                    }
                }

                // Score = nb de rôles capturés
                let score = got.len();

                // Si on n’autorise pas le partiel, il faut tous les rôles cités dans la frame
                if !allow_partial && score < roles_in_frame.len() {
                    continue;
                }

                if score > best_score {
                    best_score = score;
                    best_capture = Some(got);
                }
            }
        }

        // Si rien capturé via frames → fallback heuristique pour TRANSFER (robuste à la démo classify)
        if best_capture.is_none() && cid == "TRANSFER" && lang == "fr" {
            if let Some(got) = fallback_transfer_fr(text) {
                best_capture = Some(got);
                best_score = best_capture.as_ref().map(|m| m.len()).unwrap_or(0);
            }
        }

        // Si toujours rien → erreur
        if best_capture.is_none() {
            return Err(anyhow!("concept.bind_text: aucun patron reconnu pour {}", cid));
        }

        // Filtrage léger par rôles déclarés s'ils existent (on garde les capturés quand même)
        let mut roles_map: serde_json::Map<String, Value> = serde_json::Map::new();
        for (k, v) in best_capture.unwrap().into_iter() {
            if roles_declared.is_empty() || roles_declared.contains(&k) {
                roles_map.insert(k, Value::String(v));
            } else {
                // laisse passer aussi les alias fréquents Theme/Patient etc.
                roles_map.insert(k, Value::String(v));
            }
        }

        let inst = json!({
            "concept_id": cid,
            "roles": roles_map,
            "meta": { "lang": lang, "source": "bind_text" }
        });
        let allow_partial = input.get("allow_partial").and_then(|v| v.as_bool()).unwrap_or(false);
        if !allow_partial {
            if let Ok(concept_v) = resolve(env, &input["concept"]) {
                if concept_v.is_object() {
                    validate_instance_value(&concept_v, &inst, false)?;
                }
            }
        }
        Ok(Capsule { typ: "Instance".into(), val: inst })
    }
}

/* ----------------------------- Helpers ----------------------------- */

fn required_roles_for(concept_id: &str) -> &'static [&'static str] {
    match concept_id {
        "SELL"   => &["Giver","Receiver","Theme"],
        "GIVE"   => &["Giver","Receiver","Theme"],
        "LEND"   => &["Giver","Receiver","Theme"],   // Duration souvent utile mais pas strict
        "RETURN" => &["Agent","Receiver","Theme"],
        "PAY"    => &["Agent","Receiver","Amount"],  // ou "Consideration" (géré plus bas)
        "THANK"  => &["Agent","Receiver"],
        "EAT"    => &["Agent","Patient"],
        _        => &[], // par défaut: pas de rôle obligatoire
    }
}

// Vérif basique "montant": accepte "20", "20.00", "20 EUR", "20 euros", "$20", "USD 20"
fn looks_like_money(s: &str) -> bool {
    static PAT: once_cell::sync::Lazy<Regex> = once_cell::sync::Lazy::new(|| {
        Regex::new(r"(?i)^\s*(\$|usd|eur|€)?\s*\d+(?:[.,]\d{1,2})?\s*(€|eur|usd|dollars?|euros?)?\s*$").unwrap()
    });
    PAT.is_match(s)
}

// - allow_partial => n’impose pas les rôles obligatoires (utile pour bind_text partiel)
fn validate_instance_value(concept: &Value, instance: &Value, allow_partial: bool) -> Result<()> {
    let cid = concept.get("id").and_then(|v| v.as_str()).unwrap_or("<unknown>");
    let inst_roles = instance
        .get("roles")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("validate_instance: instance.roles doit être un objet"))?;

    // 1) Rôles obligatoires (déclarés dans concept.constraints: [{kind:"required", roles:[...]}])
    if let Some(cons) = concept.get("constraints").and_then(|v| v.as_array()) {
        for c in cons {
            if c.get("kind").and_then(|v| v.as_str()) == Some("required") {
                let req = c.get("roles")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| anyhow!("validate_instance({}): required.roles must be array", cid))?;
                for r in req {
                    let rname = r.as_str()
                        .ok_or_else(|| anyhow!("validate_instance({}): role name must be string", cid))?;

                    // tolérance spéciale PAY: Amount ou Consideration (si tu veux la garder)
                    if cid == "PAY" && rname == "Amount" && inst_roles.contains_key("Consideration") {
                        continue;
                    }

                    let missing = !inst_roles.contains_key(rname)
                        || inst_roles.get(rname)
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().is_empty())
                            .unwrap_or(false);

                    if missing && !allow_partial {
                        return Err(anyhow!("validate_instance({}): rôle {} obligatoire manquant", cid, rname));
                    }
                }
            }
        }
    }
    // 2) Cardinalités triviales: chaque rôle présent doit être scalaire (string/number/bool)
    for (k, v) in inst_roles {
        match v {
            Value::Array(_) | Value::Object(_) => {
                return Err(anyhow!("validate_instance({}): rôle '{}' ne doit pas être un objet/tableau", cid, k));
            }
            _ => {}
        }
    }

    // 3) Asymétries simples
    let same = |a:&str,b:&str| -> bool {
        inst_roles.get(a).and_then(|v| v.as_str())
            .zip(inst_roles.get(b).and_then(|v| v.as_str()))
            .map(|(x,y)| x==y).unwrap_or(false)
    };
    if same("Agent","Receiver") {
        return Err(anyhow!("validate_instance({}): Agent ≠ Receiver attendu", cid));
    }
    if same("Giver","Receiver") {
        return Err(anyhow!("validate_instance({}): Giver ≠ Receiver attendu", cid));
    }

    // 4) Distinct via concept.constraints: [{kind:"distinct", on:["RoleA","RoleB",...]}]
    if let Some(constraints) = concept.get("constraints").and_then(|v| v.as_array()) {
        for c in constraints {
            if c.get("kind").and_then(|v| v.as_str()) == Some("distinct") {
                if let Some(on) = c.get("on").and_then(|v| v.as_array()) {
                    // si au moins 2 présents et identiques -> erreur
                    let mut seen: Vec<String> = vec![];
                    for r in on {
                        if let Some(val) = r.as_str()
                            .and_then(|rk| inst_roles.get(rk))
                            .and_then(|v| v.as_str()) {
                                seen.push(val.to_string());
                            }
                    }
                    if seen.len() >= 2 {
                        // tous distincts
                        for i in 0..seen.len() {
                            for j in (i+1)..seen.len() {
                                if seen[i] == seen[j] {
                                    return Err(anyhow!("validate_instance({}): contraintes distinct violées sur {:?}", cid, on));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 5) Unit checks basiques
    if let Some(v) = inst_roles.get("Consideration").and_then(|v| v.as_str()) {
        if !looks_like_money(v) {
            return Err(anyhow!("validate_instance({}): Consideration n’a pas l’air d’un montant («{}»)", cid, v));
        }
    }
    if let Some(v) = inst_roles.get("Amount").and_then(|v| v.as_str()) {
        if !looks_like_money(v) {
            return Err(anyhow!("validate_instance({}): Amount n’a pas l’air d’un montant («{}»)", cid, v));
        }
    }

    Ok(())
}

fn normalize_role_value(s: &str) -> String {
    // Trim, retire ponctuation finale, et déterm. français en tête
    let s = s.trim();
    let s = s.trim_matches(|c: char| c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?');
    // boucle pour enlever des déterminants en tête
    let det_re = Regex::new(r"(?i)^(?:l'|d'|de l'|de la|du|des|un|une|le|la|les)\s+").unwrap();
    let mut t = s.to_string();
    loop {
        let u = det_re.replace(&t, "").to_string();
        if u == t { break; }
        t = u;
    }
    t.trim().to_string()
}

fn alt_union(words: &[String]) -> String {
    let mut uniq = HashSet::<String>::new();
    for w in words {
        let w = w.trim();
        if !w.is_empty() { uniq.insert(regex::escape(w)); }
    }
    let mut v: Vec<_> = uniq.into_iter().collect();
    v.sort();
    format!("(?:{})", v.join("|"))
}

// Génère lemma + 3ᵉ pers + syn + leur 3ᵉ pers (naïf mais efficace)
fn make_verb_forms(lemma: &str, syn: &[String]) -> Vec<String> {
    let mut out = Vec::<String>::new();
    if !lemma.is_empty() {
        out.push(lemma.to_string());
        if let Some(f) = fr_3sg(lemma) { out.push(f); }
    }
    for s in syn {
        if s.is_empty() { continue; }
        out.push(s.clone());
        if let Some(f) = fr_3sg(s) { out.push(f); }
    }
    out
}

// 3ᵉ pers du singulier (présent indicatif) extrêmement simplifiée
fn fr_3sg(lemma: &str) -> Option<String> {
    let l = lemma.trim().to_lowercase();
    if l.ends_with("er") {
        let stem = &l[..l.len()-2];
        return Some(format!("{}e", stem));
    }
    if l.ends_with("ir") {
        let stem = &l[..l.len()-2];
        return Some(format!("{}it", stem));
    }
    if l.ends_with("re") {
        let stem = &l[..l.len()-2];
        return Some(stem.to_string()); // vendre -> vend ; prendre -> prend
    }
    None
}

// Construit un regex à partir d’une frame du type:
// "Agent V Theme à Receiver pour Consideration"
//
// - 'V' => alternatives verbales
// - RoleName => (?P<RoleName>...)
// - mots "à/pour/pendant/avec/de" => littéraux
// - espace flexible, casse insensible
fn build_regex_from_frame(frame: &str, verb_alts: &str) -> (String, Vec<String>) {
    let mut pattern = String::from(r"^\s*");
    let mut roles = Vec::<String>::new();

    // Mots outillés fréquents (on élargit un peu les séparateurs)
    let literals = ["à", "pour", "pendant", "avec", "de", "au", "aux", "chez", "vers"];

    // Tokenisation très simple par espaces
    for (i, tok) in frame.split_whitespace().enumerate() {
        if i > 0 { pattern.push_str(r"\s+"); }

        match tok {
            "V" => {
                pattern.push_str(verb_alts); // pas de capture, juste alternance
            }
            t if literals.contains(&t) => {
                pattern.push_str(&regex::escape(t));
            }
            role if is_role_like(role) => {
                roles.push(role.to_string());
                // déterminant optionnel + capture jusqu’à ponctuation/délimiteur courant
                pattern.push_str(r"(?:(?:l'|d'|de l'|de la|du|des|un|une|le|la|les)\s+)?");
                pattern.push_str(&format!(r"(?P<{}>[^.,;:!?]+?)", role));
            }
            other => {
                // Mot fixe dans la frame (p. ex. “un/une”? normalement on évite)
                pattern.push_str(&regex::escape(other));
            }
        }
    }

    pattern.push_str(r"\s*[.,;:!?]?\s*$");

    (pattern, roles)
}

fn is_role_like(s: &str) -> bool {
    // Heuristique: Maj initiale + minuscules => "Agent", "Theme", "Receiver", ...
    if s.is_empty() { return false; }
    let mut ch = s.chars();
    match ch.next() {
        Some(c) if c.is_uppercase() => (),
        _ => return false,
    }
    ch.all(|c| c.is_alphabetic() && !c.is_uppercase())
}

// Fallback dédié TRANSFER (FR) pour sécuriser la démo classify
fn fallback_transfer_fr(text: &str) -> Option<HashMap<String, String>> {
    let txt = text.trim();
    // 1) SELL
    let re_sell = RegexBuilder::new(
        r"(?i)^\s*(?P<Agent>[^,.;:!?]+?)\s+vend\s+(?P<Theme>[^,.;:!?]+?)\s+à\s+(?P<Receiver>[^,.;:!?]+?)(?:\s+pour\s+(?P<Consideration>[^,.;:!?]+?))?\s*[.,;:!?]?\s*$"
    ).case_insensitive(true).unicode(true).build().ok()?;

    if let Some(c) = re_sell.captures(txt) {
        return Some(capt_map(&c, &["Agent","Theme","Receiver","Consideration"]));
    }

    // 2) GIVE
    let re_give = RegexBuilder::new(
        r"(?i)^\s*(?P<Giver>[^,.;:!?]+?)\s+donne\s+(?P<Theme>[^,.;:!?]+?)\s+à\s+(?P<Receiver>[^,.;:!?]+?)\s*[.,;:!?]?\s*$"
    ).case_insensitive(true).unicode(true).build().ok()?;
    if let Some(c) = re_give.captures(txt) {
        let mut m = capt_map(&c, &["Giver","Theme","Receiver"]);
        // alias courant pour la suite du pipeline
        if let Some(g) = m.get("Giver").cloned() { m.insert("Agent".into(), g); }
        return Some(m);
    }

    // 3) LEND
    let re_lend = RegexBuilder::new(
        r"(?i)^\s*(?P<Giver>[^,.;:!?]+?)\s+prête\s+(?P<Theme>[^,.;:!?]+?)\s+à\s+(?P<Receiver>[^,.;:!?]+?)(?:\s+pendant\s+(?P<Duration>[^,.;:!?]+?))?\s*[.,;:!?]?\s*$"
    ).case_insensitive(true).unicode(true).build().ok()?;
    if let Some(c) = re_lend.captures(txt) {
        let mut m = capt_map(&c, &["Giver","Theme","Receiver","Duration"]);
        if let Some(g) = m.get("Giver").cloned() { m.insert("Agent".into(), g); }
        return Some(m);
    }

    // 4) BORROW
    let re_borrow = RegexBuilder::new(
        r"(?i)^\s*(?P<Agent>[^,.;:!?]+?)\s+emprunte\s+(?P<Theme>[^,.;:!?]+?)\s+à\s+(?P<Giver>[^,.;:!?]+?)(?:\s+pendant\s+(?P<Duration>[^,.;:!?]+?))?\s*[.,;:!?]?\s*$"
    ).case_insensitive(true).unicode(true).build().ok()?;
    if let Some(c) = re_borrow.captures(txt) {
        return Some(capt_map(&c, &["Agent","Theme","Giver","Duration"]));
    }

    None
}

fn capt_map(c: &regex::Captures<'_>, names: &[&str]) -> HashMap<String, String> {
    let mut m = HashMap::<String, String>::new();
    for &n in names {
        if let Some(v) = c.name(n) {
            let t = normalize_role_value(v.as_str());
            if !t.is_empty() { m.insert(n.to_string(), t); }
        }
    }
    m
}

// "RAG syn" opportuniste : on grignote 1–2 variantes depuis l’index (si dispo)
// -> nécessite le crate xlm_rag (déjà utilisé côté CLI)
fn try_rag_synonyms(lemma: &str) -> Option<Vec<String>> {
    // Best effort : si la lib n’est pas linkée, retourne None.
    // Ici on fait un appel direct, très simple.
    #[allow(unused_mut)]
    let mut out: Vec<String> = Vec::new();
    #[cfg(feature = "rag_syn")]
    {
        if let Ok(docs) = xlm_rag::query(&format!("synonymes du verbe {} en français", lemma), 3) {
            for d in docs.iter().take(3) {
                // On découpe le titre, on prend 1–2 mots pertinents
                let title = d.title.to_lowercase();
                for w in title.split(|c: char| c.is_whitespace() || ",;:()[]{}".contains(c)) {
                    let w = w.trim();
                    if w.len() >= 4 && w != lemma && w.chars().all(|ch| ch.is_alphabetic()) {
                        out.push(w.to_string());
                        if out.len() >= 2 { break; }
                    }
                }
                if out.len() >= 2 { break; }
            }
        }
    }
    if out.is_empty() { None } else { Some(out) }
}



struct ConceptQuery;
impl Kernel for ConceptQuery {
    fn name(&self) -> &'static str { "concept.query" }

    fn run(
        &self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _params: &serde_json::Value
    ) -> anyhow::Result<Capsule> {
        use anyhow::{anyhow, Result};
        use serde_json::{json, Value};
        use regex::Regex;
        use std::collections::{HashMap, HashSet};

        let op = input.get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("concept.query: op required"))?;

        // ✅ concept devient OPTIONNEL : on ne l’exige plus ici
        //    - Si présent: on le résout.
        //    - Si absent: None (et on sautera juste la collecte des defaults hérités).
        let concept_v_opt: Option<Value> = if let Some(c) = input.get("concept") {
            Some(resolve(env, c)?)
        } else {
            None
        };

        let inst_v = resolve(env, input.get("instance")
            .ok_or_else(|| anyhow!("concept.query: instance required"))?)?;

        let roles = inst_v.get("roles")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("concept.query: instance.roles object required"))?;

        // helper: premier rôle présent parmi une liste d'alias
        let pick = |names: &[&str]| -> Value {
            for n in names {
                if let Some(v) = roles.get(*n) { return v.clone(); }
            }
            Value::Null
        };

        // ── CAS COURANTS WHO/WHAT/... ───────────────────────────────────────────────
        if op != "WHY" {
            let answer = match op {
                "WHO"   => pick(&["Agent","Giver","Actor","Subject","Experiencer","Sender","Source"]),
                "WHAT"  => pick(&["Patient","Theme","Object","Content","Message"]),
                "WHERE" => pick(&["Loc","Location","Place","Dest","GoalLoc"]),
                "WHEN"  => pick(&["Time","Date","Tense"]),
                "HOW"   => pick(&["Instr","Instrument","Manner","Means"]),
                _       => Value::Null,
            };
            return Ok(Capsule {
                typ: "Answer".to_string(),
                val: json!({ "op": op, "answer": answer }),
            });
        }

        // ── WHY (fusion logique riche + compat), concept optionnel ──────────────────

        // 1) Contexte optionnel
        let mut ctx: Vec<Value> = Vec::new();
        if let Some(arr) = input.get("context").and_then(|v| v.as_array()) {
            for it in arr {
                if let Ok(cv) = resolve(env, it) { ctx.push(cv); }
            }
        }

        // 2) Exceptions optionnelles
        let exceptions: HashSet<String> = input.get("exceptions")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        // 3) Utilitaires
        let cid = inst_v.get("concept_id").and_then(|v| v.as_str()).unwrap_or("");
        let roles_map = roles.clone();
        let get = |m:&serde_json::Map<String,Value>, k:&str|
            m.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();

        // normalisation FR pour comparaisons souples
        let det_re   = Regex::new(r"(?i)^(?:l'|d'|de l'|de la|du|des|un|une|le|la|les)\s+").unwrap();
        let punct_re = Regex::new(r"[.,;:!?]$").unwrap();
        let normalize = |s: &str| -> String {
            let s = s.trim().to_lowercase();
            let s = punct_re.replace(&s, "").to_string();
            let mut t = s;
            loop {
                let u = det_re.replace(&t, "").to_string();
                if u == t { break; }
                t = u;
            }
            t.trim().to_string()
        };
        let equiv = |a: &str, b: &str| -> bool {
            let na = normalize(a);
            let nb = normalize(b);
            na == nb || na.contains(&nb) || nb.contains(&na)
        };

        // 4) Hypothèses textuelles (WHY ciblé)
        let mut hyps: Vec<Value> = Vec::new();
        match cid {
            "RETURN" => {
                let agent = get(&roles_map, "Agent");
                let theme = get(&roles_map, "Theme");
                let recv  = get(&roles_map, "Receiver");

                let mut best_dur = String::new();
                let mut matched = false;

                for c in ctx {
                    let ccid = c.get("concept_id").and_then(|v| v.as_str()).unwrap_or("");
                    if ccid == "LEND" {
                        if let Some(r) = c.get("roles").and_then(|v| v.as_object()) {
                            let lender   = get(r, "Giver");
                            let borrower = get(r, "Receiver");
                            let theme2   = if r.get("Theme").is_some() { get(r,"Theme") } else { get(r,"Patient") };
                            let dur      = get(r, "Duration");

                            if !agent.is_empty() && !recv.is_empty() && !theme.is_empty()
                               && equiv(&agent, &borrower)
                               && equiv(&recv,  &lender)
                               && equiv(&theme, &theme2) {
                                matched = true;
                                if !dur.is_empty() { best_dur = dur; }
                                break;
                            }
                        }
                    }
                }

                if matched {
                    let mut score = 0.9;
                    if exceptions.contains("NO_RETURN_EXPECTED") { score *= 0.5; }
                    let mut t = format!(
                        "Parce que {} avait emprunté {} à {} et devait le rendre",
                        agent, theme, recv
                    );
                    if !best_dur.is_empty() { t.push_str(&format!(" après {}", best_dur)); }
                    t.push('.');
                    hyps.push(json!({ "text": t, "score": score }));
                } else {
                    let mut score_base = 0.55;
                    if exceptions.contains("NO_RETURN_EXPECTED") { score_base *= 0.6; }
                    if !agent.is_empty() && !recv.is_empty() && !theme.is_empty() {
                        hyps.push(json!({
                            "text": format!("Parce que {} avait emprunté {} et qu’il était attendu de le rendre à {}.", agent, theme, recv),
                            "score": score_base
                        }));
                        hyps.push(json!({
                            "text": format!("Parce que la restitution de {} à {} était la norme dans ce contexte.", theme, recv),
                            "score": score_base - 0.05
                        }));
                    } else if !agent.is_empty() && !recv.is_empty() {
                        hyps.push(json!({
                            "text": format!("Parce que {} devait rendre l’objet à {}.", agent, recv),
                            "score": score_base - 0.05
                        }));
                        hyps.push(json!({
                            "text": "Parce que la restitution était attendue suite à un prêt.",
                            "score": score_base - 0.1
                        }));
                    } else {
                        hyps.push(json!({
                            "text": "Parce que la restitution était typiquement attendue.",
                            "score": 0.5
                        }));
                    }
                }
            }
            "EAT" => {
                let agent = get(&roles_map, "Agent");
                if !agent.is_empty() {
                    hyps.push(json!({ "text": format!("Parce que {} avait faim.", agent), "score": 0.7 }));
                } else {
                    hyps.push(json!({ "text": "Parce qu’il y avait faim.", "score": 0.6 }));
                }
            }
                _ => {
                    // Ne rien pousser ici : on laisse les defaults hérités
                    // (et/ou d’autres signaux) décider en aval.
                }
        }

        // dédup textuelle
        let mut seen_h = HashSet::new();
        let mut hyps_out: Vec<Value> = Vec::new();
        for h in hyps {
            if let Some(t) = h.get("text").and_then(|v| v.as_str()) {
                let k = normalize(t);
                if seen_h.insert(k) { hyps_out.push(h); }
            } else {
                hyps_out.push(h);
            }
        }

        // 5) Defaults hérités — ✅ seulement si `concept` est fourni
        let lang = inst_v.get("meta")
            .and_then(|m| m.get("lang"))
            .and_then(|v| v.as_str())
            .unwrap_or("fr");

        let mut defaults_out: Vec<Value> = Vec::new();
        if let Some(concept_v) = &concept_v_opt {
            let mut seen_concepts = HashSet::new();
            let defs = collect_defaults_with_parents(concept_v, lang, &mut seen_concepts)?;
            let mut seen_then: HashSet<String> = HashSet::new();
            for d in defs {
                let mut ok = true;

                if let Some(when_has) = d.get("when_has").and_then(|v| v.as_array()) {
                    for r in when_has {
                        if let Some(rn) = r.as_str() {
                            if !role_present(&roles_map, rn) { ok = false; break; }
                        }
                    }
                }
                if ok {
                    if let Some(unless_has) = d.get("unless_has").and_then(|v| v.as_array()) {
                        for r in unless_has {
                            if let Some(rn) = r.as_str() {
                                if role_present(&roles_map, rn) { ok = false; break; }
                            }
                        }
                    }
                }

                if ok {
                    if let Some(th) = d.get("then") {
                        let key = th.as_str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| th.to_string());
                        if seen_then.insert(key) {
                            defaults_out.push(th.clone());
                        }
                    }
                }
            }
        }

        // 6) Compat "answer" (ne casse pas l’API existante)
        let answer_compat: Value = if !defaults_out.is_empty() {
            defaults_out[0].clone()
        } else if !hyps_out.is_empty() {
            hyps_out[0].get("text").cloned().unwrap_or(Value::Null)
        } else {
            Value::Null
        };

        Ok(Capsule{
            typ: "Answer".into(),
            val: json!({
                "op": "WHY",
                "hypotheses": hyps_out,
                "defaults": defaults_out,
                "answer": answer_compat
            })
        })
    }
}



// mini-conjugueur FR (démo)
fn conj_fr_present_3s(lemma: &str) -> String {
    let l = lemma.to_lowercase();
    match l.as_str() {
        // irréguliers / fréquents
        "être" | "etre" => "est".into(),
        "avoir" => "a".into(),
        "aller" => "va".into(),
        "faire" => "fait".into(),
        "dire" => "dit".into(),
        "voir" => "voit".into(),
        "savoir" => "sait".into(),
        "pouvoir" => "peut".into(),
        "vouloir" => "veut".into(),
        "devoir" => "doit".into(),
        "prendre" => "prend".into(),
        "mettre" => "met".into(),
        "venir" => "vient".into(),
        "tenir" => "tient".into(),
        "rendre" => "rend".into(),
        "attendre" => "attend".into(),
        "payer" => "paie".into(),       // <- pour éviter "paye"
        "écrire" | "ecrire" => "écrit".into(),

        // règles génériques
        _ => {
            if l.ends_with("er") {
                let stem = &l[..l.len()-2];
                format!("{stem}e")        // parler -> parle
            } else if l.ends_with("re") {
                let stem = &l[..l.len()-2];
                stem.to_string()          // vendre -> vend, entendre -> entend
            } else if l.ends_with("ir") {
                let stem = &l[..l.len()-2];
                format!("{stem}it")       // finir -> finit
            } else {
                l
            }
        }
    }
}



struct TransferClassify;
impl Kernel for TransferClassify {
    fn name(&self) -> &'static str { "transfer.classify" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // ← résout la ref "$inst"
        let inst_v = resolve(env, input.get("instance").ok_or_else(|| anyhow!("transfer.classify: instance required"))?)?;
        let roles = inst_v.get("roles").and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("transfer.classify: instance.roles required"))?;

        let has_consideration = roles.get("Consideration").and_then(|v| v.as_str()).map(|s| !s.trim().is_empty()).unwrap_or(false);
        let has_duration      = roles.get("Duration").and_then(|v| v.as_str()).map(|s| !s.trim().is_empty()).unwrap_or(false);

        let subtype = if has_consideration && !has_duration { "SELL" }
                      else if has_duration { "LEND" }
                      else { "GIVE" };

        // ← renvoie juste la chaîne
        Ok(Capsule { typ: "Subtype".to_string(), val: serde_json::json!(subtype) })
    }
}


struct ConceptAbstract;
impl Kernel for ConceptAbstract {
    fn name(&self) -> &'static str { "concept.abstract" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &Value, _params: &Value) -> Result<Capsule> {
        let concept_v = resolve(env, input.get("concept").ok_or_else(|| anyhow!("concept.abstract: concept required"))?)?;
        let parents = concept_v.get("parents").cloned().unwrap_or(json!([]));
        Ok(Capsule { typ: "Parents".to_string(), val: json!({"parents": parents}) })
    }
}

// -- dans crates/runtime/src/lib.rs (ou où tu définis tes kernels) --
struct ConceptReason;

// -- dans runtime, près de ConceptReason existant --
impl Kernel for ConceptReason {
    fn name(&self) -> &'static str { "concept.reason" }

    fn run(&self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _params: &serde_json::Value
    ) -> anyhow::Result<Capsule> {
        use serde_json::{json, Value};

        let mut instances: Vec<Value> = Vec::new();
        if let Some(arr_v) = input.get("instances") {
            let arr = resolve(env, arr_v)?;
            if let Some(a) = arr.as_array() { instances = a.clone(); }
        }

        let exceptions: std::collections::HashSet<String> = input.get("params")
            .and_then(|p| p.get("exceptions"))
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        // Effets/attentes déduits (très simple pour la démo)
        let mut notes: Vec<Value> = Vec::new();

        for (idx, inst) in instances.iter().enumerate() {
            let cid = inst.get("concept_id").and_then(|v| v.as_str()).unwrap_or("");
            if cid == "LEND" {
                // default: EXPECT(Receiver -> RETURN(Theme to Giver)) avec score 0.9
                let mut score = 0.9;
                if exceptions.contains("NO_RETURN_EXPECTED") {
                    score *= 0.3; // on affaiblit fortement l’attente
                }
                notes.push(json!({
                    "type":"EXPECT_RETURN",
                    "instance_index": idx,
                    "score": score
                }));
            }
        }

        Ok(Capsule{
            typ: "Reason".into(),
            val: json!({
                "graph": { "instances": instances },
                "notes": notes
            })
        })
    }
}




struct SurfaceRealize;
impl Kernel for SurfaceRealize {
    fn name(&self) -> &'static str { "surface.realize" }

    fn run(
        &self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _params: &serde_json::Value
    ) -> anyhow::Result<Capsule> {
        use serde_json::json;

        // --- 1) Rendu direct d'un plan de discours ---
        if let Some(plan_v) = input.get("plan") {
            let plan = resolve(env, plan_v)?;
            let props = plan.get("propositions")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("surface.realize: plan.propositions array required"))?;

            // strings ou objets {text:"..."}
            let mut lines: Vec<String> = Vec::new();
            for p in props {
                if let Some(s) = p.as_str() {
                    let s2 = s.trim().trim_end_matches('.');
                    if !s2.is_empty() { lines.push(s2.to_string()); }
                } else if let Some(obj) = p.as_object() {
                    if let Some(s) = obj.get("text").and_then(|v| v.as_str()) {
                        let s2 = s.trim().trim_end_matches('.');
                        if !s2.is_empty() { lines.push(s2.to_string()); }
                    }
                }
            }
            // Si aucune proposition valide → renvoyer texte vide (évite le ".")
            if lines.is_empty() {
                return Ok(Capsule { typ: "Text".into(), val: json!({ "text": "" }) });
            }

            let mut text = lines.join(". ");
            if !text.ends_with('.') { text.push('.'); }
            return Ok(Capsule { typ: "Text".into(), val: json!({ "text": text }) });
        }

        // --- 2) Chemin classique concept/instance ---
        // in: { "concept":"$c", "instance":"$inst", "style": {"lang":"fr","frame_idx"?:0} }
        let concept_v = resolve(env, input.get("concept").ok_or_else(|| anyhow::anyhow!("surface.realize: concept required"))?)?;
        let inst_v    = resolve(env, input.get("instance").ok_or_else(|| anyhow::anyhow!("surface.realize: instance required"))?)?;
        let style     = input.get("style").cloned().unwrap_or(json!({}));
        let lang      = style.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let frame_idx = style.get("frame_idx").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let roles = inst_v.get("roles").and_then(|v| v.as_object()).ok_or_else(|| anyhow::anyhow!("surface.realize: instance.roles required"))?;
        let lemma = concept_v.get("lexemes").and_then(|lx| lx.get(lang)).and_then(|l| l.get("verb_lemma")).and_then(|v| v.as_str()).unwrap_or("faire");
        let vform = if lang == "fr" { conj_fr_present_3s(lemma) } else { lemma.to_string() };

        // Helper pour récupérer un rôle (avec alias)
        let get_role = |name: &str| -> String {
            let cands: &[&str] = match name {
                "Agent"        => &["Agent","Giver","Subject","Actor"],
                "Giver"        => &["Giver","Agent","Subject","Actor"],
                "Patient"      => &["Patient","Theme","Object"],
                "Theme"        => &["Theme","Patient","Object"],
                "Receiver"     => &["Receiver","Goal","Dest","Beneficiary"],
                "Consideration"=> &["Consideration","Amount"],
                "Amount"       => &["Amount","Consideration"],
                "Loc"          => &["Loc","Location","Place"],
                "Time"         => &["Time","Date","Tense"],
                "Instr"        => &["Instr","Instrument","Means"],
                "Goal"         => &["Goal","Dest","Receiver"],
                "Duration"     => &["Duration"],
                other          => &[other],
            };
            for k in cands {
                if let Some(s) = roles.get(*k).and_then(|v| v.as_str()) {
                    if !s.is_empty() { return s.to_string(); }
                }
            }
            String::new()
        };

        // Try frames d'abord
        if let Some(frames) = concept_v.get("lexemes")
            .and_then(|lx| lx.get(lang))
            .and_then(|l| l.get("frames"))
            .and_then(|v| v.as_array())
        {
            if let Some(frame) = frames.get(frame_idx).and_then(|v| v.as_str()) {
                use std::collections::HashSet;

                // rôles explicitement présents dans la frame
                let mut roles_in_frame: HashSet<&str> = HashSet::new();
                for tok in frame.split(' ') {
                    match tok {
                        "Agent" | "Patient" | "Theme" | "Giver" | "Receiver" |
                        "Consideration" | "Amount" | "Loc" | "Time" | "Instr" | "Goal" | "Duration" => {
                            roles_in_frame.insert(tok);
                        }
                        _ => {}
                    }
                }

                // remplacement
                let mut out_tokens: Vec<String> = Vec::new();
                for tok in frame.split(' ') {
                    let repl = match tok {
                        "V" => vform.as_str().to_string(),
                        "Agent" | "Patient" | "Theme" | "Giver" | "Receiver" |
                        "Consideration" | "Amount" | "Loc" | "Time" | "Instr" | "Goal" | "Duration" => {
                            get_role(tok)
                        }
                        _ => tok.to_string(),
                    };
                    if !repl.is_empty() { out_tokens.push(repl); }
                }
                let mut text = out_tokens.join(" ");

                // ajouts optionnels si absents de la frame (évite les doublons)
                if !roles_in_frame.contains("Loc") {
                    let l = get_role("Loc");
                    if !l.is_empty() { text.push_str(&format!(" dans {}", l)); }
                }
                if !roles_in_frame.contains("Time") {
                    let t = get_role("Time");
                    if !t.is_empty() { text.push_str(&format!(" à {}", t)); }
                }
                if !roles_in_frame.contains("Duration") {
                    let d = get_role("Duration");
                    if !d.is_empty() { text.push_str(&format!(" pendant {}", d)); }
                }

                text.push('.');
                return Ok(Capsule { typ: "Text".to_string(), val: json!({"text": text}) });
            }
        }

        // Fallback générique
        let agent = {
            let a = get_role("Agent");
            if a.is_empty() { get_role("Giver") } else { a }
        };
        let patient = {
            let t = get_role("Patient");
            if t.is_empty() { get_role("Theme") } else { t }
        };

        let mut s = String::new();
        s.push_str(&agent);
        if !agent.is_empty() { s.push(' '); }
        s.push_str(&vform);
        if !patient.is_empty() { s.push(' '); s.push_str(&patient); }
        s.push('.');

        Ok(Capsule { typ: "Text".to_string(), val: json!({"text": s}) })
    }
}


struct ReasonAbduce;
impl Kernel for ReasonAbduce {
    fn name(&self) -> &'static str { "reason.abduce" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // in: { "concept":"$c", "instance":"$inst", "for_subtype"?: "LEND"|"GIVE"|"SELL", "target_contains"?: "EXPECT(" }
        let concept_v = resolve(env, input.get("concept").ok_or_else(|| anyhow!("reason.abduce: concept required"))?)?;
        let inst_v    = resolve(env, input.get("instance").ok_or_else(|| anyhow!("reason.abduce: instance required"))?)?;
        let roles_obj_owned = inst_v.get("roles").and_then(|v| v.as_object()).cloned().unwrap_or_default();

        let lang = inst_v.get("meta").and_then(|m| m.get("lang")).and_then(|v| v.as_str()).unwrap_or("fr");
        let mut seen = HashSet::new();
        let defs = collect_defaults_with_parents(&concept_v, lang, &mut seen)?;

        let for_subtype     = input.get("for_subtype").and_then(|v| v.as_str());
        let target_contains = input.get("target_contains").and_then(|v| v.as_str());

        let mut suggestions: Vec<Value> = Vec::new();

        for d in defs {
            // filtre éventuel par sous-type (le champ "if" porte l'id du concept concerné)
            if let Some(fs) = for_subtype {
                if d.get("if").and_then(|v| v.as_str()) != Some(fs) { continue; }
            }
            // filtre éventuel par motif dans "then"
            if let Some(tc) = target_contains {
                let th_str = d.get("then").and_then(|v| v.as_str()).unwrap_or("");
                if !th_str.contains(tc) { continue; }
            }

            let mut add: Vec<String> = Vec::new();
            let mut remove: Vec<String> = Vec::new();

            if let Some(when_has) = d.get("when_has").and_then(|v| v.as_array()) {
                for r in when_has {
                    if let Some(rn) = r.as_str() {
                        if !role_present(&roles_obj_owned, rn) { add.push(rn.to_string()); }
                    }
                }
            }
            if let Some(unless_has) = d.get("unless_has").and_then(|v| v.as_array()) {
                for r in unless_has {
                    if let Some(rn) = r.as_str() {
                        if role_present(&roles_obj_owned, rn) { remove.push(rn.to_string()); }
                    }
                }
            }

            let then_val = d.get("then").cloned().unwrap_or(json!(null));
            suggestions.push(json!({ "then": then_val, "add": add, "remove": remove }));
        }

        Ok(Capsule { typ: "Abduce".to_string(), val: json!({ "suggestions": suggestions }) })
    }
}
// Helper: calc WHY (dédupliqué) pour un (concept, roles, lang)
fn compute_why(concept_v: &Value, roles_map: &serde_json::Map<String, Value>, lang: &str) -> Result<Vec<Value>> {
    let mut seen_concepts = std::collections::HashSet::new();
    let defs = collect_defaults_with_parents(concept_v, lang, &mut seen_concepts)?;
    let mut out: Vec<Value> = Vec::new();
    let mut seen_then: std::collections::HashSet<String> = std::collections::HashSet::new();

    for d in defs {
        let mut ok = true;
        if let Some(when_has) = d.get("when_has").and_then(|v| v.as_array()) {
            for r in when_has {
                if let Some(rn) = r.as_str() {
                    if !role_present(roles_map, rn) { ok = false; break; }
                }
            }
        }
        if ok {
            if let Some(unless_has) = d.get("unless_has").and_then(|v| v.as_array()) {
                for r in unless_has {
                    if let Some(rn) = r.as_str() {
                        if role_present(roles_map, rn) { ok = false; break; }
                    }
                }
            }
        }
        if ok {
            if let Some(th) = d.get("then") {
                let key = th.as_str().map(|s| s.to_string()).unwrap_or_else(|| th.to_string());
                if seen_then.insert(key) { out.push(th.clone()); }
            }
        }
    }
    Ok(out)
}

struct ReasonCounterfactual;
impl Kernel for ReasonCounterfactual {
    fn name(&self) -> &'static str { "reason.counterfactual" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // in: { "concept":"$c", "instance":"$inst", "set"?: {Role: "val", ...}, "unset"?: [Role,...] }
        let concept_v = resolve(env, input.get("concept").ok_or_else(|| anyhow!("reason.counterfactual: concept required"))?)?;
        let inst_v    = resolve(env, input.get("instance").ok_or_else(|| anyhow!("reason.counterfactual: instance required"))?)?;
        let lang = inst_v.get("meta").and_then(|m| m.get("lang")).and_then(|v| v.as_str()).unwrap_or("fr");

        let mut roles_before = inst_v.get("roles").and_then(|v| v.as_object()).cloned()
            .ok_or_else(|| anyhow!("reason.counterfactual: instance.roles required"))?;

        // Compute WHY before
        let why_before = compute_why(&concept_v, &roles_before, lang)?;

        // Apply patch
        if let Some(set) = input.get("set").and_then(|v| v.as_object()) {
            for (k, v) in set { roles_before.insert(k.clone(), v.clone()); }
        }
        if let Some(unset) = input.get("unset").and_then(|v| v.as_array()) {
            for r in unset {
                if let Some(k) = r.as_str() { roles_before.remove(k); }
            }
        }

        // Compute WHY after
        let why_after = compute_why(&concept_v, &roles_before, lang)?;

        // Diff
        let to_key = |x: &Value| x.as_str().map(|s| s.to_string()).unwrap_or_else(|| x.to_string());
        let set_before: std::collections::HashSet<String> = why_before.iter().map(|v| to_key(v)).collect();
        let set_after : std::collections::HashSet<String> = why_after.iter().map(|v| to_key(v)).collect();

        let mut added = Vec::new();
        for k in set_after.difference(&set_before) {
            let v = if k.starts_with('"') { serde_json::from_str::<Value>(k).unwrap_or(json!(k)) } else { json!(k) };
            added.push(v);
        }
        let mut removed = Vec::new();
        for k in set_before.difference(&set_after) {
            let v = if k.starts_with('"') { serde_json::from_str::<Value>(k).unwrap_or(json!(k)) } else { json!(k) };
            removed.push(v);
        }

        Ok(Capsule { typ: "Counterfactual".to_string(),
            val: json!({ "before": why_before, "after": why_after, "added": added, "removed": removed }) })
    }
}
struct InstanceApplyPatch;
impl Kernel for InstanceApplyPatch {
    fn name(&self) -> &'static str { "instance.apply_patch" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        let inst_v = resolve(env, input.get("instance").ok_or_else(|| anyhow!("instance.apply_patch: instance required"))?)?;
        let mut inst  = inst_v.as_object().cloned().ok_or_else(|| anyhow!("instance.apply_patch: instance must be object"))?;
        let mut roles = inst.get("roles").and_then(|v| v.as_object()).cloned().unwrap_or_default();

        // RESOUDRE les refs "$plan.set" / "$plan.unset"
        let set_obj = if input.get("set").is_some() {
            let v = resolve(env, &input["set"])?;
            v.as_object().cloned()
        } else { None };

        let unset_arr = if input.get("unset").is_some() {
            let v = resolve(env, &input["unset"])?;
            v.as_array().cloned()
        } else { None };

        if let Some(set) = set_obj {
            for (k, v) in set { roles.insert(k.clone(), v.clone()); }
        }
        if let Some(unset) = unset_arr {
            for r in unset {
                if let Some(k) = r.as_str() { roles.remove(k); }
            }
        }

        inst.insert("roles".into(), Value::Object(roles));
        Ok(Capsule { typ: "Instance".into(), val: Value::Object(inst) })
    }
}

struct ReasonPlan;
impl Kernel for ReasonPlan {
    fn name(&self) -> &'static str { "reason.plan" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // in: { "concept":"$c", "instance":"$inst", "goal": { "subtype"?: "LEND"|"GIVE"|"SELL",
        //                                                    "activate_contains"?: "EXPECT(" } }
        let concept_v = resolve(env, input.get("concept").ok_or_else(|| anyhow!("reason.plan: concept required"))?)?;
        let inst_v    = resolve(env, input.get("instance").ok_or_else(|| anyhow!("reason.plan: instance required"))?)?;
        let roles     = inst_v.get("roles").and_then(|v| v.as_object()).cloned().unwrap_or_default();

        let lang = inst_v.get("meta").and_then(|m| m.get("lang")).and_then(|v| v.as_str()).unwrap_or("fr");
        let mut seen = std::collections::HashSet::new();
        let defs = collect_defaults_with_parents(&concept_v, lang, &mut seen)?;

        let goal = input.get("goal").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let want_subtype = goal.get("subtype").and_then(|v| v.as_str());
        let want_contains = goal.get("activate_contains").and_then(|v| v.as_str());

        // Filtre les règles pertinentes au but
        let mut candidates: Vec<Value> = Vec::new();
        for d in &defs {
            // ---- branche spéciale: objectif par SOUS-TYPE (rejoue la logique du classifieur) ----
            if let Some(st) = want_subtype {
                // état courant
                let has_consideration = roles.get("Consideration").and_then(|v| v.as_str()).map(|s| !s.trim().is_empty()).unwrap_or(false);
                let has_duration      = roles.get("Duration").and_then(|v| v.as_str()).map(|s| !s.trim().is_empty()).unwrap_or(false);

                // set/unset minimaux pour atteindre le sous-type visé
                let mut set_map = serde_json::Map::new();
                let mut unset = Vec::<Value>::new();

                // valeurs par défaut
                fn default_val(role: &str) -> &'static str {
                    match role {
                        "Duration" => "deux semaines",
                        "Consideration" => "20 euros",
                        _ => "X"
                    }
                }

                match st {
                    "SELL" => {
                        if !has_consideration { set_map.insert("Consideration".into(), json!(default_val("Consideration"))); }
                        if has_duration { unset.push(json!("Duration")); }
                    }
                    "LEND" => {
                        if !has_duration { set_map.insert("Duration".into(), json!(default_val("Duration"))); }
                        // pas d'exigence sur Consideration ici
                    }
                    "GIVE" => {
                        if has_consideration { unset.push(json!("Consideration")); }
                        if has_duration { unset.push(json!("Duration")); }
                    }
                    _ => {}
                }

                return Ok(Capsule { typ:"Plan".into(), val: json!({ "set": Value::Object(set_map), "unset": Value::Array(unset) }) });
            }
            if let Some(tc) = want_contains {
                if !d.get("then").and_then(|v| v.as_str()).unwrap_or("").contains(tc) { continue; }
            }
            candidates.push(d.clone());
        }
        if candidates.is_empty() { return Ok(Capsule{ typ:"Plan".into(), val: json!({"set":{}, "unset":[]}) }); }

        // Score minimal (|add| + |remove|)
        let mut best: Option<(usize, Value)> = None;
        for d in candidates {
            let mut add: Vec<String> = Vec::new();
            let mut rem: Vec<String> = Vec::new();
            if let Some(wh) = d.get("when_has").and_then(|v| v.as_array()) {
                for r in wh { if let Some(k)=r.as_str() { if !role_present(&roles, k) { add.push(k.to_string()); } } }
            }
            if let Some(uh) = d.get("unless_has").and_then(|v| v.as_array()) {
                for r in uh { if let Some(k)=r.as_str() { if role_present(&roles, k) { rem.push(k.to_string()); } } }
            }
            let cost = add.len() + rem.len();
            let plan = json!({"add": add, "remove": rem});
            if best.as_ref().map(|(c,_)| cost < *c).unwrap_or(true) {
                best = Some((cost, plan));
            }
        }

        // Filler valeurs par défaut pour quelques rôles courants
        fn default_val(role: &str) -> &'static str {
            match role {
                "Duration" => "deux semaines",
                "Consideration" => "20 euros",
                _ => "X"
            }
        }

        let (_cost, plan) = best.unwrap();
        let mut set_map = serde_json::Map::new();
        if let Some(add) = plan.get("add").and_then(|v| v.as_array()) {
            for r in add {
                if let Some(k)=r.as_str() { set_map.insert(k.to_string(), json!(default_val(k))); }
            }
        }
        let unset = plan.get("remove").cloned().unwrap_or(json!([]));

        Ok(Capsule{ typ:"Plan".into(), val: json!({ "set": Value::Object(set_map), "unset": unset }) })
    }
}
struct InstanceGetRole;
impl Kernel for InstanceGetRole {
    fn name(&self) -> &'static str { "instance.get_role" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &serde_json::Value, _: &serde_json::Value) -> Result<Capsule> {
        // in: { "instance":"$inst", "role":"Receiver" }
        let inst_v = resolve(env, input.get("instance").ok_or_else(|| anyhow::anyhow!("instance.get_role: instance"))?)?;
        let role   = input.get("role").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("instance.get_role: role"))?;
        let s = inst_v.get("roles")
            .and_then(|v| v.as_object())
            .and_then(|m| m.get(role))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok(Capsule{ typ: "Str".into(), val: serde_json::json!(s) })
    }
}
struct InstanceNew;
impl Kernel for InstanceNew {
    fn name(&self) -> &'static str { "instance.new" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &serde_json::Value, _: &serde_json::Value) -> Result<Capsule> {
        // in: { "concept":"$c", "roles": { "Agent":"...", ... }, "lang"?: "fr" }
        let concept_v = resolve(env, input.get("concept").ok_or_else(|| anyhow::anyhow!("instance.new: concept"))?)?;
        let lang = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let roles_in = input.get("roles").and_then(|v| v.as_object()).ok_or_else(|| anyhow::anyhow!("instance.new: roles object"))?;

        // ↓↓↓ Résout CHAQUE valeur (références "$recv", etc.)
        let mut roles = serde_json::Map::new();
        for (k, v) in roles_in {
            let vr = resolve(env, v)?;
            let s = if let Some(s) = vr.as_str() {
                s.to_string()
            } else {
                vr.to_string() // au pire stringify
            };
            roles.insert(k.clone(), serde_json::Value::String(s));
        }

        let inst = serde_json::json!({
            "concept_id": concept_v.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            "roles": roles,
            "meta": { "lang": lang }
        });
        // Validation automatique (sauf si allow_partial true)
        let allow_partial = input.get("allow_partial").and_then(|v| v.as_bool()).unwrap_or(false);
        if !allow_partial {
            if let Ok(concept_v) = resolve(env, &input["concept"]) {
                if concept_v.is_object() {
                    validate_instance_value(&concept_v, &inst, allow_partial)?;
                }
            }
        }
        Ok(Capsule{ typ: "Instance".into(), val: inst })
    }
}

struct CdlInduce;
impl Kernel for CdlInduce {
    fn name(&self) -> &'static str { "cdl.induce" }
    fn run(&self, _env: &std::collections::HashMap<String, Capsule>, input: &serde_json::Value, _: &serde_json::Value) -> Result<Capsule> {
        // in: { "id":"BORROW", "lemma":"emprunter", "sentence":"Marie emprunte un livre à Paul pendant deux semaines", "lang":"fr" }
        let id     = input.get("id").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("cdl.induce: id required"))?;
        let lemma  = input.get("lemma").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("cdl.induce: lemma required"))?;
        let sent   = input.get("sentence").and_then(|v| v.as_str()).unwrap_or("");
        let lang   = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");

        // Heuristiques FR très simples pour extraire "à <X>" (donneur) et "pendant <Y>" (durée)
        let lower = sent.to_lowercase();
        let giver = lower.split(" à ").nth(1).map(|s| s.split_whitespace().take_while(|w| *w != "pour" && *w != "pendant" && *w != ".").collect::<Vec<_>>().join(" "))
            .filter(|s| !s.is_empty()).unwrap_or("X".into());
        let duration = if lower.contains("pendant ") {
            lower.split("pendant ").nth(1).map(|s| s.trim_end_matches('.').to_string()).unwrap_or("deux semaines".into())
        } else { "".into() };

        // Frame FR
        let frames = vec!["Agent V Theme à Giver".to_string()];

        // Concept JSON
        let mut concept = serde_json::json!({
            "id": id,
            "type": "EVENT",
            "parents": ["TRANSFER"],
            "roles": ["Agent","Giver","Theme","Duration","Loc","Time","Instr","Goal"],
            "lexemes": { lang: { "verb_lemma": lemma, "frames": frames } },
            "defaults": [
                { "if": id, "then": "EXPECT(Agent,RETURN(Theme,to=Giver))", "strength": 0.8 }
            ]
        });

        // Option: stocker un exemple (pas obligatoire)
        concept.as_object_mut().unwrap().insert("example".into(), serde_json::json!(sent));

        Ok(Capsule{ typ: "Concept".into(), val: concept })
    }
}
struct ConceptSave;
impl Kernel for ConceptSave {
    fn name(&self) -> &'static str { "concept.save" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &serde_json::Value, _: &serde_json::Value) -> Result<Capsule> {
        use std::fs;
        use std::path::Path;

        // ✅ Résoudre la référence "$c"
        let c = resolve(env, input.get("concept").ok_or_else(|| anyhow::anyhow!("concept.save: concept required"))?)?;
        let id = c.get("id").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("concept.save: concept.id required"))?;

        let path = format!("./concepts/{}.fr.json", id);
        fs::create_dir_all(Path::new(&path).parent().unwrap())?;
        fs::write(&path, serde_json::to_string_pretty(&c)?)?;
        Ok(Capsule{ typ: "Saved".into(), val: serde_json::json!({ "path": path }) })
    }
}
// Applique un plan {set:{...}, unset:[...]} à une instance, en FUSIONNANT les rôles.
struct PlanApply;
impl Kernel for PlanApply {
    fn name(&self) -> &'static str { "plan.apply" }

    fn run(
        &self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _: &serde_json::Value
    ) -> anyhow::Result<Capsule> {
        use serde_json::{json, Value};

        let inst_v = resolve(env, input.get("instance").ok_or_else(|| anyhow::anyhow!("plan.apply: instance required"))?)?;
        let plan_v = resolve(env, input.get("plan").ok_or_else(|| anyhow::anyhow!("plan.apply: plan required"))?)?;

        let mut inst = inst_v.as_object()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("plan.apply: instance must be object"))?;

        // roles cible
        let roles_val = inst.entry("roles".to_string()).or_insert(json!({}));
        let roles_obj = roles_val
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("plan.apply: instance.roles must be object"))?;

        // unset
        if let Some(unset) = plan_v.get("unset").and_then(|v| v.as_array()) {
            for k in unset.iter().filter_map(|x| x.as_str()) {
                roles_obj.remove(k);
            }
        }
        // set (fusion/upsert)
        if let Some(set) = plan_v.get("set").and_then(|v| v.as_object()) {
            for (k, v) in set {
                roles_obj.insert(k.clone(), v.clone());
            }
        }

        Ok(Capsule { typ: "Instance".into(), val: Value::Object(inst) })
    }
}
struct StoryCompose;
impl Kernel for StoryCompose {
    fn name(&self) -> &'static str { "story.compose" }

    fn run(
        &self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _params: &serde_json::Value
    ) -> anyhow::Result<Capsule> {
        use serde_json::{json, Value};
        // optionnel: lire un Reason capsule (si fourni) pour piloter les phrases
        let mut expect_return_score: Option<f32> = None;
        if let Some(reason_v) = input.get("reason") {
            if let Ok(rv) = resolve(env, reason_v) {
                if let Some(arr) = rv.get("notes").and_then(|v| v.as_array()) {
                    for n in arr {
                        if n.get("type").and_then(|v| v.as_str()) == Some("EXPECT_RETURN") {
                            if let Some(s) = n.get("score").and_then(|v| v.as_f64()) {
                                expect_return_score = Some(s as f32);
                            }
                        }
                    }
                }
            }
        }
        // seuil simple
        let allow_return = expect_return_score.unwrap_or(0.9) >= 0.6;

        // inputs: { "instances":[ "$i0", ... ] }  ou  { "seed":"$i0" }
        let mut insts: Vec<Value> = Vec::new();
        if let Some(a) = input.get("instances") {
            let arr = resolve(env, a)?;
            if let Some(v) = arr.as_array() { insts = v.clone(); }
        }
        if insts.is_empty() {
            if let Some(s) = input.get("seed") {
                let v = resolve(env, s)?;
                insts.push(v);
            } else {
                return Err(anyhow::anyhow!("story.compose: provide 'instances' or 'seed'"));
            }
        }

        // petit helper rôle
        let get_role = |roles: &serde_json::Map<String, Value>, name: &str| -> String {
            let cands: &[&str] = match name {
                "Agent"        => &["Agent","Giver","Subject","Actor"],
                "Giver"        => &["Giver","Agent","Subject","Actor"],
                "Patient"      => &["Patient","Theme","Object"],
                "Theme"        => &["Theme","Patient","Object"],
                "Receiver"     => &["Receiver","Goal","Dest","Beneficiary"],
                "Consideration"=> &["Consideration","Amount"],
                "Amount"       => &["Amount","Consideration"],
                "Duration"     => &["Duration"],
                _              => &[name],
            };
            for k in cands {
                if let Some(s) = roles.get(*k).and_then(|v| v.as_str()) {
                    if !s.is_empty() { return s.to_string(); }
                }
            }
            String::new()
        };

        // capitalise première lettre (pour "Deux semaines plus tard")
        let cap_first = |s: &str| -> String {
            let mut ch = s.chars();
            match ch.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + ch.as_str(),
            }
        };
        // "un/une/des X" -> "le/la/les X" (très simple)
        let to_definite = |np: &str| -> String {
            let s = np.trim();
            if s.starts_with("un ")  { return format!("le {}",  &s[3..]); }
            if s.starts_with("une ") { return format!("la {}",  &s[4..]); }
            if s.starts_with("des ") { return format!("les {}", &s[4..]); }
            s.to_string() // déjà "le/la/les" ou nom propre
        };

        let mut props: Vec<Value> = Vec::new();

        for inst in insts {
            let roles = inst.get("roles").and_then(|v| v.as_object()).cloned().unwrap_or_default();
            let cid   = inst.get("concept_id").and_then(|v| v.as_str()).unwrap_or("");

            match cid {
                "LEND" => {
                    let giver = get_role(&roles, "Giver");
                    let recv  = get_role(&roles, "Receiver");
                    let theme = get_role(&roles, "Theme");
                    let dur   = get_role(&roles, "Duration");

                    if !giver.is_empty() && !recv.is_empty() && !theme.is_empty() {
                        // 1) prêt
                        let mut s1 = format!("{giver} prête {theme} à {recv}");
                        if !dur.is_empty() { s1.push_str(&format!(" pendant {}", dur)); }
                        s1.push('.');
                        // 1) prêt (inchangé)
                        props.push(json!(s1));

                        // 2) restitution (conditionnée)
                        if allow_return {
                            let mut s2 = String::new();
                            if !dur.is_empty() {
                                s2.push_str(&format!("{} plus tard, ", cap_first(&dur)));
                            }
                            s2.push_str(&format!("{recv} rend {} à {}", theme, giver));
                            s2.push('.');
                            props.push(json!(s2));

                            // 3) remerciement (optionnel si restitution écrite)
                            props.push(json!(format!("{giver} remercie {} pour sa ponctualité.", recv)));
                        }
                    }
                }
                "SELL" => {
                    let giver = get_role(&roles, "Giver");
                    let recv  = get_role(&roles, "Receiver");
                    let theme = get_role(&roles, "Theme");
                    let amt   = get_role(&roles, "Consideration");
                    if !giver.is_empty() && !recv.is_empty() && !theme.is_empty() {
                        let mut s1 = format!("{giver} vend {theme} à {recv}");
                        if !amt.is_empty() { s1.push_str(&format!(" pour {}", amt)); }
                        s1.push('.');
                        props.push(json!(s1));

                        if !amt.is_empty() {
                            let s2 = format!("{recv} paie {} à {}.", amt, giver);
                            props.push(json!(s2));
                        }
                        // petit bonus poli si on veut
                        let s3 = format!("{recv} remercie {}.", giver);
                        props.push(json!(s3));
                    }
                }
                _ => {
                    // fallback muet (ou on pourrait générer une phrase minimale)
                }
            }
        }

        Ok(Capsule{ typ:"Plan".into(), val: json!({ "propositions": props }) })
    }
}
struct PlanWrapText;
impl Kernel for PlanWrapText {
    fn name(&self) -> &'static str { "plan.wrap_text" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &serde_json::Value, _: &serde_json::Value)
        -> anyhow::Result<Capsule>
    {
        use serde_json::{json, Value};
        let texts_v = input.get("texts").ok_or_else(|| anyhow::anyhow!("plan.wrap_text: texts required"))?;
        let arr = match texts_v {
            Value::Array(a) => a.clone(),
            _ => vec![texts_v.clone()],
        };

        let kind = input.get("kind").and_then(|v| v.as_str()).unwrap_or("COMPOSE");
        let mut props: Vec<Value> = Vec::new();
        for it in arr {
            let v = resolve(env, &it)?;
            let s = if let Some(st) = v.as_str() { st.to_string() }
                    else if let Some(o) = v.as_object() {
                        o.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string()
                    } else { "".to_string() };
            if !s.is_empty() { props.push(json!({ "text": s, "kind": kind })); }
        }
        Ok(Capsule{ typ:"Plan".into(), val: json!({ "propositions": props }) })
    }
}
struct ReasonCausalize;
impl Kernel for ReasonCausalize {
    fn name(&self) -> &'static str { "reason.causalize" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &serde_json::Value, _: &serde_json::Value)
        -> anyhow::Result<Capsule>
    {
        use serde_json::{json, Value};

        let inst_v = resolve(env, input.get("instance").ok_or_else(|| anyhow::anyhow!("reason.causalize: instance required"))?)?;
        let roles = inst_v.get("roles").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let cid   = inst_v.get("concept_id").and_then(|v| v.as_str()).unwrap_or("");

        // exceptions simples: ["RETURN","PAY"] pour skipper certains défauts
        let skip: std::collections::HashSet<String> = input.get("skip").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        // petits helpers
        let get = |names: &[&str]| -> String {
            for k in names {
                if let Some(s) = roles.get(*k).and_then(|v| v.as_str()) { if !s.is_empty() { return s.to_string(); } }
            }
            String::new()
        };
        let giver = || get(&["Giver","Agent"]);
        let recv  = || get(&["Receiver","Goal","Dest"]);
        let theme = || get(&["Theme","Patient","Object"]);
        let amt   = || get(&["Consideration","Amount"]);
        let dur   = || get(&["Duration"]);

        // “un/une/des X” → “le/la/les X” avec élision l’
        let starts_with_vowelish = |w: &str| -> bool {
            let ch = w.trim().chars().next().unwrap_or('x').to_ascii_lowercase();
            matches!(ch, 'a'|'e'|'i'|'o'|'u'|'y'|'h')
        };
        let to_definite = |np: &str| -> String {
            let s = np.trim();
            let lower = s.to_lowercase();
            if lower.starts_with("un ")  { let r=&s[3..].trim(); return if starts_with_vowelish(r){ format!("l’{r}") } else { format!("le {r}") }; }
            if lower.starts_with("une ") { let r=&s[4..].trim(); return if starts_with_vowelish(r){ format!("l’{r}") } else { format!("la {r}") }; }
            if lower.starts_with("des ") { let r=&s[4..].trim(); return format!("les {r}"); }
            s.to_string()
        };
        let cap_first = |s: &str| -> String {
            let mut ch = s.chars();
            match ch.next() { None => String::new(), Some(f) => f.to_uppercase().collect::<String>() + ch.as_str() }
        };

        let mut props: Vec<Value> = Vec::new();

        match cid {
            // LEND/BORROW → RETURN par défaut
            "LEND" | "BORROW" => {
                if !skip.contains("RETURN") {
                    let g = giver(); let r = recv(); let th = theme(); let d = dur();
                    if !g.is_empty() && !r.is_empty() && !th.is_empty() {
                        let th_def = to_definite(&th);
                        let mut s = String::new();
                        if !d.is_empty() { s.push_str(&format!("{} plus tard, ", cap_first(&d))); }
                        else { s.push_str("Ensuite, "); }
                        s.push_str(&format!("{r} rend {th_def} à {g}."));
                        props.push(json!({ "text": s, "kind": "RETURN" }));
                    }
                }
            }
            "EAT" | "EAT_POISONED" => {
                if !skip.contains("ILLNESS") {
                    let ag = get(&["Agent","Giver","Subject","Actor"]);
                    let th = theme();
                    let poisoned = cid == "EAT_POISONED" || th.to_lowercase().contains("empoison");
                    if poisoned && !ag.is_empty() {
                        let s = format!("{ag} tombe malade.");
                        props.push(json!({ "text": s, "kind": "ILLNESS" }));
                    }
                }
            }
            // SELL → PAY par défaut
            "SELL" => {
                if !skip.contains("PAY") {
                    let g = giver(); let r = recv(); let a = amt();
                    if !g.is_empty() && !r.is_empty() && !a.is_empty() {
                        let s = format!("{r} paie {a} à {g}.");
                        props.push(json!({ "text": s, "kind": "PAY" }));
                    }
                }
            }
            _ => {}
        }

        Ok(Capsule{ typ:"Plan".into(), val: json!({ "propositions": props }) })
    }
}
struct PlanConcat;
impl Kernel for PlanConcat {
    fn name(&self) -> &'static str { "plan.concat" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &serde_json::Value, _: &serde_json::Value)
        -> anyhow::Result<Capsule>
    {
        use serde_json::{json, Value};
        let arr = input.get("plans").and_then(|v| v.as_array()).ok_or_else(|| anyhow::anyhow!("plan.concat: plans[] required"))?;
        let mut props: Vec<Value> = Vec::new();
        for p in arr {
            let v = resolve(env, p)?;
            if let Some(a) = v.get("propositions").and_then(|x| x.as_array()) {
                for it in a { props.push(it.clone()); }
            }
        }
        Ok(Capsule{ typ:"Plan".into(), val: json!({ "propositions": props }) })
    }
}
struct PlanFilter;
impl Kernel for PlanFilter {
    fn name(&self) -> &'static str { "plan.filter" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &serde_json::Value, _: &serde_json::Value)
        -> anyhow::Result<Capsule>
    {
        use serde_json::{json, Value};
        let p = resolve(env, input.get("plan").ok_or_else(|| anyhow::anyhow!("plan.filter: plan required"))?)?;
        let mut props = p.get("propositions").and_then(|v| v.as_array()).cloned().unwrap_or_default();

        let skip: std::collections::HashSet<String> = input.get("skip_kinds").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        props.retain(|it| {
            let k = it.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            !skip.contains(k)
        });

        Ok(Capsule{ typ:"Plan".into(), val: json!({ "propositions": props }) })
    }
}
// --- add this kernel ---
// =====================
// cdl.fuse (safe merge)
// =====================
struct CdlFuse;
impl Kernel for CdlFuse {
    fn name(&self) -> &'static str { "cdl.fuse" }

    fn run(&self,
           env: &std::collections::HashMap<String, Capsule>,
           input: &serde_json::Value,
           _: &serde_json::Value) -> anyhow::Result<Capsule>
    {
        fn union_str_vec(a: Option<&Value>, b: Option<&Value>) -> Vec<Value> {
            let mut set = BTreeSet::<String>::new();
            for src in [a, b] {
                if let Some(Value::Array(arr)) = src {
                    for it in arr {
                        if let Some(s) = it.as_str() { set.insert(s.to_string()); }
                    }
                } else if let Some(Value::Object(m)) = src {
                    for k in m.keys() { set.insert(k.to_string()); }
                }
            }
            set.into_iter().map(Value::String).collect()
        }

        fn union_defaults(a: Option<&Value>, b: Option<&Value>) -> Vec<Value> {
            let mut seen = BTreeSet::<String>::new();
            let mut out = Vec::<Value>::new();
            for src in [a, b] {
                if let Some(arr) = src.and_then(|v| v.as_array()) {
                    for d in arr {
                        let key = d.get("then").and_then(|v| v.as_str()).unwrap_or(&d.to_string()).to_string();
                        if seen.insert(key) { out.push(d.clone()); }
                    }
                }
            }
            out
        }

        // Merge de deux objets "stats" par catégorie (frames|syn), en cumulant samples
        fn merge_stats_category(base: Option<&serde_json::Map<String, Value>>,
                                delta: Option<&serde_json::Map<String, Value>>) -> serde_json::Map<String, Value>
        {
            let mut out = base.cloned().unwrap_or_default();
            if let Some(d) = delta {
                for (k, dv) in d {
                    match (out.get(k), dv) {
                        (Some(Value::Object(bo)), Value::Object(do_)) => {
                            let mut merged = bo.clone();
                            for (sk, sv) in do_ {
                                if sk == "samples" {
                                    // union des samples (Array<String>), cap à 10
                                    let mut set = BTreeSet::<String>::new();
                                    if let Some(Value::Array(arr)) = merged.get("samples") {
                                        for it in arr { if let Some(s) = it.as_str() { set.insert(s.to_string()); } }
                                    }
                                    if let Value::Array(arr) = sv {
                                        for it in arr { if let Some(s) = it.as_str() { set.insert(s.to_string()); } }
                                    }
                                    let mut v: Vec<Value> = set.into_iter().map(Value::String).collect();
                                    if v.len() > 10 { v.truncate(10); }
                                    merged.insert("samples".to_string(), Value::Array(v));
                                } else {
                                    merged.insert(sk.clone(), sv.clone());
                                }
                            }
                            out.insert(k.clone(), Value::Object(merged));
                        }
                        _ => { out.insert(k.clone(), dv.clone()); }
                    }
                }
            }
            out
        }

        // Fusion lexemes pour 1 langue
        fn merge_lang(base_l: Option<&Value>, delta_l: Option<&Value>) -> Value {
            let mut out = base_l.cloned().unwrap_or(json!({}));
            let mut map = out.as_object().cloned().unwrap_or_default();

            // frames union
            let frames = union_str_vec(base_l.and_then(|l| l.get("frames")), delta_l.and_then(|l| l.get("frames")));
            if !frames.is_empty() { map.insert("frames".into(), Value::Array(frames)); }

            // syn union
            let syn = union_str_vec(base_l.and_then(|l| l.get("syn")), delta_l.and_then(|l| l.get("syn")));
            if !syn.is_empty() { map.insert("syn".into(), Value::Array(syn)); }

            // verb_lemma : garder celui du base si présent; sinon prendre delta
            if !map.contains_key("verb_lemma") {
                if let Some(vl) = delta_l.and_then(|l| l.get("verb_lemma")).and_then(|v| v.as_str()) {
                    map.insert("verb_lemma".into(), json!(vl));
                }
            }

            // stats : deep-merge
            if base_l.and_then(|l| l.get("stats")).is_some() || delta_l.and_then(|l| l.get("stats")).is_some() {
                let base_stats = base_l.and_then(|l| l.get("stats")).and_then(|v| v.as_object()).cloned();
                let delta_stats= delta_l.and_then(|l| l.get("stats")).and_then(|v| v.as_object()).cloned();

                let mut stats_out = base_stats.clone().unwrap_or_default();

                // frames
                let merged_frames = merge_stats_category(
                    base_stats.as_ref().and_then(|m| m.get("frames")).and_then(|v| v.as_object()),
                    delta_stats.as_ref().and_then(|m| m.get("frames")).and_then(|v| v.as_object())
                );
                if !merged_frames.is_empty() {
                    stats_out.insert("frames".into(), Value::Object(merged_frames));
                }

                // syn
                let merged_syn = merge_stats_category(
                    base_stats.as_ref().and_then(|m| m.get("syn")).and_then(|v| v.as_object()),
                    delta_stats.as_ref().and_then(|m| m.get("syn")).and_then(|v| v.as_object())
                );
                if !merged_syn.is_empty() {
                    stats_out.insert("syn".into(), Value::Object(merged_syn));
                }

                // autres clés au niveau de stats: override shallow par delta
                if let Some(ds) = delta_stats {
                    for (k, vv) in ds {
                        if k != "frames" && k != "syn" {
                            stats_out.insert(k, vv);
                        }
                    }
                }

                if !stats_out.is_empty() {
                    map.insert("stats".into(), Value::Object(stats_out));
                }
            }

            Value::Object(map)
        }

        // --- entrée
        let base = resolve(env, input.get("base").ok_or_else(|| anyhow!("cdl.fuse: base required"))?)?;
        let delta= resolve(env, input.get("delta").ok_or_else(|| anyhow!("cdl.fuse: delta required"))?)?;

        let mut merged = base.as_object().cloned().ok_or_else(|| anyhow!("base must be object"))?;

        // id/type
        if merged.get("id").is_none() {
            if let Some(id) = delta.get("id") { merged.insert("id".into(), id.clone()); }
        }

        // roles / parents / defaults — unions
        let union_str_vec_top = |k: &str| -> Option<Value> {
            let v = {
                let a = base.get(k);
                let b = delta.get(k);
                let u = {
                    let mut set = BTreeSet::<String>::new();
                    for src in [a, b] {
                        if let Some(Value::Array(arr)) = src {
                            for it in arr { if let Some(s) = it.as_str() { set.insert(s.to_string()); } }
                        } else if let Some(Value::Object(m)) = src {
                            for key in m.keys() { set.insert(key.to_string()); }
                        }
                    }
                    set
                };
            Value::Array(u.into_iter().map(Value::String).collect())
            };
            if matches!(v, Value::Array(ref a) if !a.is_empty()) { Some(v) } else { None }
        };
        if let Some(v) = union_str_vec_top("roles")    { merged.insert("roles".into(), v); }
        if let Some(v) = union_str_vec_top("parents")  { merged.insert("parents".into(), v); }
        let defaults = union_defaults(base.get("defaults"), delta.get("defaults"));
        if !defaults.is_empty() { merged.insert("defaults".into(), Value::Array(defaults)); }

        // lexemes — support multi-lang : on fusionne pour chaque langue vu dans base|delta
        let mut out_lex = base.get("lexemes").cloned().unwrap_or(json!({}));
        let mut out_map = out_lex.as_object().cloned().unwrap_or_default();

        // collecter toutes les langues
        let mut langs = BTreeSet::<String>::new();
        if let Some(obj) = base.get("lexemes").and_then(|v| v.as_object()) {
            langs.extend(obj.keys().cloned());
        }
        if let Some(obj) = delta.get("lexemes").and_then(|v| v.as_object()) {
            langs.extend(obj.keys().cloned());
        }
        // rétro-compat si delta est uniquement {lexemes:{fr:{...}}} et base vide
        if langs.is_empty() { langs.insert("fr".into()); }

        for lang in langs {
            let b = base.get("lexemes").and_then(|v| v.get(&lang));
            let d = delta.get("lexemes").and_then(|v| v.get(&lang));
            let merged_lang = merge_lang(b, d);
            if !merged_lang.as_object().map(|m| m.is_empty()).unwrap_or(true) {
                out_map.insert(lang, merged_lang);
            }
        }
        merged.insert("lexemes".into(), Value::Object(out_map));

        Ok(Capsule { typ: "Concept".into(), val: Value::Object(merged) })
    }
}

// ===== lexicon.update =====
struct LexiconUpdate;
impl Kernel for LexiconUpdate {
    fn name(&self) -> &'static str { "lexicon.update" }

    fn run(&self, _env:&std::collections::HashMap<String,Capsule>, input:&serde_json::Value, _:&serde_json::Value) -> anyhow::Result<Capsule> {
        use serde_json::{json, Value};
        use std::{fs, path::Path};

        let lang  = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let lemma = input.get("lemma").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("lemma required"))?;
        let forms = input.get("forms").and_then(|v| v.as_array()).cloned().unwrap_or_else(|| vec![json!(lemma)]);
        let syn   = input.get("syn").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let count = input.get("count").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let today = chrono::Utc::now().naive_utc().date().format("%Y-%m-%d").to_string();

        fs::create_dir_all("./lexicon").ok();
        let path = format!("./lexicon/lemmas.{}.json", lang);

        let mut root: serde_json::Map<String,Value> = if Path::new(&path).exists() {
            fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default()
        } else { serde_json::Map::new() };

        // décroissance simple (lambda=0.02/jour) si présent
        let decay = |last:&str, c:f64| -> f64 {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(last, "%Y-%m-%d") {
                let days = (chrono::Utc::now().naive_utc().date() - d).num_days() as f64;
                (-(0.02_f64) * days).exp() * c
            } else { c }
        };

        let entry = root.entry(lemma.to_string()).or_insert_with(|| json!({"forms":[],"syn":[],"count":0.0,"last_seen":today}));
        let mut obj = entry.as_object().cloned().unwrap_or_default();

        // merge forms (dédup)
        let mut form_set = std::collections::BTreeSet::<String>::new();
        for v in obj.get("forms").and_then(|a| a.as_array()).unwrap_or(&vec![]) {
            if let Some(s) = v.as_str() { form_set.insert(s.to_string()); }
        }
        for v in &forms {
            if let Some(s) = v.as_str() { form_set.insert(s.to_string()); }
        }
        obj.insert("forms".into(), Value::Array(form_set.into_iter().map(Value::String).collect()));

        // merge syn (dédup)
        let mut syn_set = std::collections::BTreeSet::<String>::new();
        for v in obj.get("syn").and_then(|a| a.as_array()).unwrap_or(&vec![]) {
            if let Some(s) = v.as_str() { syn_set.insert(s.to_string()); }
        }
        for v in &syn {
            if let Some(s) = v.as_str() { syn_set.insert(s.to_string()); }
        }
        obj.insert("syn".into(), Value::Array(syn_set.into_iter().map(Value::String).collect()));

        // count + decay
        let prev = obj.get("count").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let prev_last = obj.get("last_seen").and_then(|v| v.as_str()).unwrap_or("2025-01-01");
        let decayed = decay(prev_last, prev);
        obj.insert("count".into(), json!(decayed + count));
        obj.insert("last_seen".into(), json!(today));

        root.insert(lemma.to_string(), serde_json::Value::Object(obj));
        fs::write(&path, serde_json::to_string_pretty(&serde_json::Value::Object(root))?)?;

        Ok(Capsule{ typ:"Lexicon".into(), val: json!({"ok":true,"lemma":lemma,"lang":lang})})
    }
}
// ===== nlp.entities.learn =====
struct EntitiesLearn;
impl Kernel for EntitiesLearn {
    fn name(&self) -> &'static str { "nlp.entities.learn" }

    fn run(&self,
           _env: &std::collections::HashMap<String, Capsule>,
           input: &serde_json::Value,
           _params: &serde_json::Value) -> Result<Capsule>
    {
        use regex::Regex;
        use std::fs;
        use std::path::Path;

        let lang   = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let folder = input.get("folder").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("folder required"))?;
        let lambda = input.get("lambda").and_then(|v| v.as_f64()).unwrap_or(0.02_f64);
        let sample_cap = input.get("sample_cap").and_then(|v| v.as_u64()).unwrap_or(4) as usize;

        let today  = chrono::Utc::now().naive_utc().date();
        let today_s = today.format("%Y-%m-%d").to_string();
        let decay = |last:&str| -> f64 {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(last, "%Y-%m-%d") {
                (-lambda * (today - d).num_days() as f64).exp()
            } else { 1.0 }
        };

        // Store: ./stores/entities.<lang>.json
        fs::create_dir_all("./stores").ok();
        let store_path = format!("./stores/entities.{}.json", lang);
        let mut store = if Path::new(&store_path).exists() {
            let s = fs::read_to_string(&store_path).unwrap_or_else(|_| "{}".into());
            serde_json::from_str::<serde_json::Value>(&s).unwrap_or_else(|_| json!({}))
        } else { json!({}) };

        // Décroissance préalable
        if let Some(map) = store.as_object_mut() {
            for (_k, v) in map.iter_mut() {
                if let Some(obj) = v.as_object_mut() {
                    let last = obj.get("last_seen").and_then(|x| x.as_str()).unwrap_or("2025-01-01");
                    let cnt  = obj.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0) * decay(last);
                    obj.insert("count".to_string(), json!(cnt));
                }
            }
        }

        // Regex FR basiques
        let re_amount = Regex::new(r"(?i)\b(\d{1,3}(?:[ \u00A0.,]\d{3})*|\d+)(?:[.,]\d+)?\s?(€|eur|euros|\$|usd)\b").unwrap();
        let re_date_iso = Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap();
        let re_date_fr  = Regex::new(r"(?i)\b(\d{1,2}\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre))\b").unwrap();
        // NPs simples: déterminant + 1-3 tokens
        let re_np_det  = Regex::new(r"(?i)\b(le|la|les|un|une|des|ce|cet|cette|mon|ma|mes|ton|ta|tes|son|sa|ses|notre|votre)\s+([[:alpha:]][[:alpha:]\-']{1,})(?:\s+[[:alpha:]\-']{1,}){0,2}\b").unwrap();
        // Séquence de capitalisés (Person/Org)
        let re_caps    = Regex::new(r"\b(\p{Lu}\p{L}+)(?:\s+\p{Lu}\p{L}+){0,3}\b").unwrap();

        // Extraction
        let mut added = 0usize;
        for entry in fs::read_dir(folder)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("txt") { continue; }
            let content = fs::read_to_string(&path)?;
            for line in content.lines().map(|s| s.trim()).filter(|s| !s.is_empty()) {
                let low = line.to_lowercase();

                let mut push_ent = |text:&str, typ:&str| {
                    if text.is_empty() { return; }
                    let key = text.to_string();
                    let e = store.get(key.as_str()).cloned().unwrap_or_else(|| json!({"type":typ,"count":0.0,"last_seen":"2025-01-01","samples":[]}));
                    let mut obj = e.as_object().cloned().unwrap_or_default();
                    let prev = obj.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    obj.insert("type".to_string(), json!(typ));
                    obj.insert("count".to_string(), json!(prev + 1.0));
                    obj.insert("last_seen".to_string(), json!(today_s.clone()));

                    // samples
                    let mut samples = obj.get("samples").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                    if samples.len() < sample_cap { samples.push(json!(line)); }
                    obj.insert("samples".to_string(), json!(samples));

                    store.as_object_mut().unwrap().insert(key, json!(obj));
                    added += 1;
                };

                for m in re_amount.find_iter(&line) {
                    push_ent(m.as_str(), "Amount");
                }
                for m in re_date_iso.find_iter(&line) { push_ent(m.as_str(), "Date"); }
                for m in re_date_fr.find_iter(&line)  { push_ent(m.as_str(), "Date"); }

                for cap in re_caps.captures_iter(&line) {
                    if let Some(m0) = cap.get(0) {
                        // évite début de phrase générique (“Lundi”, “Bonjour”)
                        let s = m0.as_str();
                        // mini-filtre: ≥ 2 tokens ou token unique non “Bonjour”, “Merci”, etc.
                        if s.split_whitespace().count() >= 2 {
                            push_ent(s, "PersonOrOrg");
                        }
                    }
                }
                for cap in re_np_det.captures_iter(&line) {
                    if let Some(m0) = cap.get(0) {
                        // item “la commande”, “le colis”, “la facture…”
                        push_ent(m0.as_str(), "Item");
                    }
                }
            }
        }

        std::fs::write(&store_path, serde_json::to_string_pretty(&store)?)?;
        Ok(Capsule{ typ:"EntitiesReport".into(), val: json!({"updated": added, "store": store_path}) })
    }
}


// ===== nlp.facts.learn =====
struct FactsLearn;
impl Kernel for FactsLearn {
    fn name(&self) -> &'static str { "nlp.facts.learn" }

    fn run(&self,
           _env: &std::collections::HashMap<String, Capsule>,
           input: &serde_json::Value,
           _params: &serde_json::Value) -> Result<Capsule>
    {
        use regex::Regex;
        use std::fs;
        use std::path::Path;

        let lang   = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let folder = input.get("folder").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("folder required"))?;
        let lambda = input.get("lambda").and_then(|v| v.as_f64()).unwrap_or(0.02_f64);
        let sample_cap = input.get("sample_cap").and_then(|v| v.as_u64()).unwrap_or(4) as usize;

        let today  = chrono::Utc::now().naive_utc().date();
        let today_s = today.format("%Y-%m-%d").to_string();
        let decay = |last:&str| -> f64 {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(last, "%Y-%m-%d") {
                (-lambda * (today - d).num_days() as f64).exp()
            } else { 1.0 }
        };

        // Store: ./stores/facts.<lang>.json  (clé: "subj|pred|obj")
        fs::create_dir_all("./stores").ok();
        let store_path = format!("./stores/facts.{}.json", lang);
        let mut store = if Path::new(&store_path).exists() {
            let s = fs::read_to_string(&store_path).unwrap_or_else(|_| "{}".into());
            serde_json::from_str::<serde_json::Value>(&s).unwrap_or_else(|_| json!({}))
        } else { json!({}) };

        // Décroissance préalable
        if let Some(map) = store.as_object_mut() {
            for (_k, v) in map.iter_mut() {
                if let Some(obj) = v.as_object_mut() {
                    let last = obj.get("last_seen").and_then(|x| x.as_str()).unwrap_or("2025-01-01");
                    let cnt  = obj.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0) * decay(last);
                    obj.insert("count".to_string(), json!(cnt));
                }
            }
        }

        // Motifs robustes: "X est Y", "X a Y", "X appartient à Y", "X cause Y"
        // On capture des NPs larges de part et d’autre (sans gourmandise infinie).
        let re_est  = Regex::new(r"(?i)\b(.+?)\s+est\s+(.+?)(?:[.?!]|$)").unwrap();
        let re_a    = Regex::new(r"(?i)\b(.+?)\s+a\s+(.+?)(?:[.?!]|$)").unwrap();
        let re_app  = Regex::new(r"(?i)\b(.+?)\s+appartient\s+à\s+(.+?)(?:[.?!]|$)").unwrap();
        let re_caus = Regex::new(r"(?i)\b(.+?)\s+(?:cause|entra[iî]ne|provoque)\s+(.+?)(?:[.?!]|$)").unwrap();

        // Nettoyages simples pour sujets/objets (trim ponctuation)
        let clean = |s:&str| -> String {
            s.trim().trim_matches(|c:char| c==':' || c==';' || c==',' || c=='"' || c=='\'' )
             .trim().to_string()
        };

        let mut added = 0usize;
        for entry in fs::read_dir(folder)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("txt") { continue; }
            let content = fs::read_to_string(&path)?;
            for line in content.lines().map(|s| s.trim()).filter(|s| !s.is_empty()) {

                let mut push_fact = |subj:&str, pred:&str, obj:&str| {
                    let subj = clean(subj);
                    let obj  = clean(obj);
                    if subj.is_empty() || obj.is_empty() { return; }
                    let key = format!("{}|{}|{}", subj, pred, obj);
                    let e = store.get(&key).cloned().unwrap_or_else(|| json!({"count":0.0,"last_seen":"2025-01-01","samples":[]}));
                    let mut objv = e.as_object().cloned().unwrap_or_default();
                    let prev = objv.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    objv.insert("count".to_string(), json!(prev + 1.0));
                    objv.insert("last_seen".to_string(), json!(today_s.clone()));
                    let mut samples = objv.get("samples").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                    if samples.len() < sample_cap { samples.push(json!(line)); }
                    objv.insert("samples".to_string(), json!(samples));
                    store.as_object_mut().unwrap().insert(key, json!(objv));
                    added += 1;
                };

                if let Some(c) = re_est.captures(line) {
                    push_fact(c.get(1).map(|m| m.as_str()).unwrap_or(""), "est", c.get(2).map(|m| m.as_str()).unwrap_or(""));
                }
                if let Some(c) = re_a.captures(line) {
                    push_fact(c.get(1).map(|m| m.as_str()).unwrap_or(""), "a", c.get(2).map(|m| m.as_str()).unwrap_or(""));
                }
                if let Some(c) = re_app.captures(line) {
                    push_fact(c.get(1).map(|m| m.as_str()).unwrap_or(""), "appartient_à", c.get(2).map(|m| m.as_str()).unwrap_or(""));
                }
                if let Some(c) = re_caus.captures(line) {
                    push_fact(c.get(1).map(|m| m.as_str()).unwrap_or(""), "cause", c.get(2).map(|m| m.as_str()).unwrap_or(""));
                }
            }
        }

        std::fs::write(&store_path, serde_json::to_string_pretty(&store)?)?;
        Ok(Capsule{ typ:"FactsReport".into(), val: json!({"updated": added, "store": store_path}) })
    }
}


// ===== cdl.defaults.suggest =====
struct DefaultsSuggest;
impl Kernel for DefaultsSuggest {
    fn name(&self) -> &'static str { "cdl.defaults.suggest" }

    fn run(&self,
           _env: &std::collections::HashMap<String, Capsule>,
           input: &serde_json::Value,
           _params: &serde_json::Value) -> Result<Capsule>
    {
        use std::fs;
        use std::path::Path;

        let lang      = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let min_supp  = input.get("min_support").and_then(|v| v.as_f64()).unwrap_or(4.0);
        let min_w     = input.get("min_weight").and_then(|v| v.as_f64()).unwrap_or(0.8);
        let max_prom  = input.get("max_promote").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
        let today_s   = chrono::Utc::now().naive_utc().date().format("%Y-%m-%d").to_string();

        // edges store: ./stores/graph.edges.json (fallback ./graph.edges.json)
        let paths = ["./stores/graph.edges.json", "./graph.edges.json"];
        let mut edges: Vec<serde_json::Value> = Vec::new();
        for p in &paths {
            if Path::new(p).exists() {
                if let Ok(s) = fs::read_to_string(p) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                        if let Some(arr) = v.get("edges").and_then(|x| x.as_array()) {
                            edges = arr.clone(); break;
                        } else if v.is_array() {
                            edges = v.as_array().cloned().unwrap_or_default();
                            break;
                        }
                    }
                }
            }
        }
        if edges.is_empty() {
            return Ok(Capsule{ typ:"DefaultsSuggest".into(), val: json!({"promoted":0, "reason":"no edges"}) });
        }

        // Bascule THEN en fonction du type d’arête
        let mk_then = |typ:&str, b:&str| -> String {
            match typ {
                "CAUSES"    => format!("EXPECT({})", b),
                "ENABLES"   => format!("EXPECT({})", b),
                "PREVENTS"  => format!("PREVENT({})", b),
                "MITIGATES" => format!("MITIGATE({})", b),
                "RESOLVES"  => format!("RESOLVE({})", b),
                _ => format!("EXPECT({})", b),
            }
        };

        // Regroupe par concept cible (to)
        use std::collections::BTreeMap;
        let mut per_concept: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
        for e in edges {
            let from = e.get("from").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let to   = e.get("to").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let typ  = e.get("type").and_then(|v| v.as_str()).unwrap_or("CAUSES").to_string();
            let guard= e.get("guard").and_then(|v| v.as_str()).unwrap_or("IF").to_string();
            let supp = e.get("support").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let w    = e.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.0);

            if from.is_empty() || to.is_empty() { continue; }
            if supp < min_supp || w < min_w { continue; }

            // guard -> champ conditionnel
            let mut d = serde_json::Map::new();
            match guard.as_str() {
                "UNLESS"    => { d.insert("unless".into(), json!(from)); }
                "OTHERWISE" => { d.insert("otherwise".into(), json!(from)); }
                _           => { d.insert("if".into(), json!(from)); }
            }
            let then_s = mk_then(&typ, &to);
            d.insert("then".into(), json!(then_s));
            let strength = if w < 0.1 { 0.1 } else if w > 5.0 { 5.0 } else { w };
            d.insert("strength".into(), json!(strength));
            d.insert("meta".into(), json!({"source":"edges","support":supp,"weight":w,"last_seen":today_s}));

            per_concept.entry(to).or_default().push(json!(d));
        }

        // Promeut via cdl.fuse sur chaque concept cible
        let mut promoted = 0usize;
        let mut details: Vec<serde_json::Value> = Vec::new();
        for (cid, defs) in per_concept {
            if defs.is_empty() { continue; }
            // On tronque si max_promote
            let defs_trunc = if defs.len() > max_prom { defs.into_iter().take(max_prom).collect() } else { defs };

            let prog = Program { program: vec![
                Node{ id: Some("c".into()), op:"concept.load".into(),
                      r#in: json!({"id": cid, "lang": lang}), out: json!({}), params: json!({}) },
                Node{ id: Some("f".into()), op:"cdl.fuse".into(),
                      r#in: json!({"base":"$c","delta":{"id":cid, "defaults": defs_trunc}}),
                      out: json!({}), params: json!({}) },
                Node{ id: Some("s".into()), op:"concept.save".into(),
                      r#in: json!({"concept":"$f","lang": lang}), out: json!({}), params: json!({}) },
            ]};
            if let Ok((_o,_env)) = run_with_env(&Vm::new(), &prog) {
                promoted += 1;
                details.push(json!({"concept":cid}));
            }
        }

        Ok(Capsule{ typ:"DefaultsSuggest".into(), val: json!({"promoted": promoted, "details": details}) })
    }
}


// ==============================
// cdl.learn_from_corpus (robust)
// ==============================
struct LearnFromCorpus;
impl Kernel for LearnFromCorpus {
    fn name(&self) -> &'static str { "cdl.learn_from_corpus" }

    fn run(
        &self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _: &serde_json::Value
    ) -> anyhow::Result<Capsule>
    {
        use serde_json::{json, Value};
        use std::collections::{HashMap, HashSet};
        use std::fs;

        let lang             = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let folder           = input.get("folder").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("folder required"))?;
        let save             = input.get("save").and_then(|v| v.as_bool()).unwrap_or(false);
        let lambda           = input.get("lambda").and_then(|v| v.as_f64()).unwrap_or(0.02_f64);
        let top_k_frames     = input.get("top_k_frames").and_then(|v| v.as_u64()).unwrap_or(8)  as usize;
        let top_k_syn        = input.get("top_k_syn").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
        let sample_cap       = input.get("sample_cap").and_then(|v| v.as_u64()).unwrap_or(5)  as usize;

        // NEW: liste optionnelle de fichiers à traiter (active learning)
        let files_opt: Option<Vec<String>> = input.get("files")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect());

        let today  = chrono::Utc::now().naive_utc().date();
        let decay = |last:&str| -> f64 {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(last, "%Y-%m-%d") {
                (-lambda * (today - d).num_days() as f64).exp()
            } else { 1.0 }
        };
        let today_s = today.format("%Y-%m-%d").to_string();

        // concept: resolve (accept id or full object)
        let concept_v = resolve(env, input.get("concept").unwrap_or(&json!(null)))?;
        let concept: Value = if concept_v.get("id").is_some() && concept_v.get("lexemes").is_some() {
            concept_v
        } else {
            let prog = Program { program: vec![
                Node{ id: Some("c".into()), op:"concept.load".into(),
                      r#in: json!({"id": concept_v.get("id").and_then(|x| x.as_str()).ok_or_else(|| anyhow::anyhow!("concept.id required"))?, "lang":lang}),
                      out: json!({}), params: json!({}) },
            ]};
            let (_outs, subenv) = run_with_env(&Vm::new(), &prog)?;
            subenv.get("c").ok_or_else(|| anyhow::anyhow!("failed to load concept"))?.val.clone()
        };

        let lemma_ptr = format!("/lexemes/{}/verb_lemma", lang);
        let lemma = concept.pointer(&lemma_ptr).and_then(|v| v.as_str()).unwrap_or("");

        let mut frame_counts : HashMap<String, f64> = HashMap::new();
        let mut syn_counts   : HashMap<String, f64> = HashMap::new();
        let mut frame_samples: HashMap<String, Vec<String>> = HashMap::new();
        let mut syn_samples  : HashMap<String, Vec<String>> = HashMap::new();
        let mut exceptions   : HashSet<String>      = HashSet::new();

        // Charger stats précédentes avec décroissance
        let stats_frames_path = format!("/lexemes/{}/stats/frames", lang);
        let stats_syn_path    = format!("/lexemes/{}/stats/syn", lang);
        if let Some(prev) = concept.pointer(&stats_frames_path).and_then(|v| v.as_object()) {
            for (k, v) in prev {
                let last = v.get("last_seen").and_then(|x| x.as_str()).unwrap_or("2025-01-01");
                let cnt  = v.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0) * decay(last);
                *frame_counts.entry(k.clone()).or_insert(0.0) += cnt;
                if let Some(arr) = v.get("samples").and_then(|x| x.as_array()) {
                    let mut vec = Vec::new();
                    for it in arr.iter().filter_map(|x| x.as_str()) {
                        vec.push(it.to_string());
                    }
                    frame_samples.entry(k.clone()).or_insert_with(Vec::new).extend(vec);
                }
            }
        }
        if let Some(prev) = concept.pointer(&stats_syn_path).and_then(|v| v.as_object()) {
            for (k, v) in prev {
                let last = v.get("last_seen").and_then(|x| x.as_str()).unwrap_or("2025-01-01");
                let cnt  = v.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0) * decay(last);
                *syn_counts.entry(k.clone()).or_insert(0.0) += cnt;
                if let Some(arr) = v.get("samples").and_then(|x| x.as_array()) {
                    let mut vec = Vec::new();
                    for it in arr.iter().filter_map(|x| x.as_str()) {
                        vec.push(it.to_string());
                    }
                    syn_samples.entry(k.clone()).or_insert_with(Vec::new).extend(vec);
                }
            }
        }

        // helper de filtrage: si files[] fourni, n'autoriser que les chemins correspondants
        let wanted_file = |p: &std::path::Path| -> bool {
            if let Some(ref lst) = files_opt {
                let ps = p.to_string_lossy().to_string();
                // match exact ou suffixe
                lst.iter().any(|f| f == &ps || ps.ends_with(f))
            } else { true }
        };

        // iterate *.txt files
        for entry in fs::read_dir(folder)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("txt") { continue; }
            if !wanted_file(&path) { continue; }

            let content = fs::read_to_string(&path)?;
            for line in content.lines().map(|s| s.trim()).filter(|s| !s.is_empty()) {
                // bind_text (allow_partial)
                let prog = Program { program: vec![
                    Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
                          r#in: json!({"concept": concept, "text": line, "lang": lang, "allow_partial": true}),
                          out: json!({}), params: json!({}) },
                ]};
                if let Ok((_outs, subenv)) = run_with_env(&Vm::new(), &prog) {
                    if let Some(inst) = subenv.get("inst") {
                        let sig = frame_signature_from_roles(&inst.val);
                        *frame_counts.entry(sig.clone()).or_insert(0.0) += 1.0;

                        // samples pour ce frame
                        let s = line.to_string();
                        let e = frame_samples.entry(sig).or_insert_with(Vec::new);
                        if e.len() < sample_cap { e.push(s.clone()); }

                        let toks = tokenize_fr(line);
                        if let Some(v) = head_verb_guess(&toks) {
                            if !lemma.is_empty() && v != lemma {
                                *syn_counts.entry(v.clone()).or_insert(0.0) += 1.0;
                                let es = syn_samples.entry(v).or_insert_with(Vec::new);
                                if es.len() < sample_cap { es.push(s); }
                            }
                        }
                        let low = line.to_lowercase();
                        if low.contains("sans retour") || low.contains("aucun retour") || low.contains("pas de retour") {
                            exceptions.insert("NO_RETURN_EXPECTED".to_string());
                        }
                    }
                }
            }
        }

        // sélection top-N
        let mut frames_sorted: Vec<(String,f64)> = frame_counts.into_iter().collect();
        frames_sorted.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if frames_sorted.len() > top_k_frames { frames_sorted.truncate(top_k_frames); }

        let mut syn_sorted: Vec<(String,f64)> = syn_counts.into_iter().collect();
        syn_sorted.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if syn_sorted.len() > top_k_syn { syn_sorted.truncate(top_k_syn); }

        // construire delta (frames + syn + stats) — objet bien imbriqué
        let mut delta = json!({"id": concept.get("id").and_then(|v| v.as_str()).unwrap_or("UNKNOWN")});
        delta["lexemes"] = delta.get("lexemes").cloned().unwrap_or_else(|| json!({}));
        if !delta["lexemes"].is_object() { delta["lexemes"] = json!({}); }
        if !delta["lexemes"].get(lang).is_some() { delta["lexemes"][lang] = json!({}); }

        if !frames_sorted.is_empty() {
            let frs: Vec<String> = frames_sorted.iter().map(|(k,_)| k.clone()).collect();
            delta["lexemes"][lang]["frames"] = json!(frs);
        }
        if !syn_sorted.is_empty() {
            let syns: Vec<String> = syn_sorted.iter().map(|(k,_)| k.clone()).collect();
            delta["lexemes"][lang]["syn"] = json!(syns);
        }

        // stats avec échantillons
        let mut frames_stats = serde_json::Map::new();
        for (f,cnt) in &frames_sorted {
            let mut samples = frame_samples.get(f).cloned().unwrap_or_default();
            samples.sort(); samples.dedup();
            if samples.len() > sample_cap { samples.truncate(sample_cap); }
            frames_stats.insert(f.clone(), json!({"count": cnt, "last_seen": today_s, "samples": samples}));
        }
        let mut syn_stats = serde_json::Map::new();
        for (s,cnt) in &syn_sorted {
            let mut samples = syn_samples.get(s).cloned().unwrap_or_default();
            samples.sort(); samples.dedup();
            if samples.len() > sample_cap { samples.truncate(sample_cap); }
            syn_stats.insert(s.clone(), json!({"count": cnt, "last_seen": today_s, "samples": samples}));
        }
        if !frames_stats.is_empty() || !syn_stats.is_empty() {
            delta["lexemes"][lang]["stats"] = json!({"frames": frames_stats, "syn": syn_stats});
        }

        // fusion (+ save si demandé)
        let fused = if delta.get("lexemes").is_some() {
            let prog = Program { program: vec![
                Node{ id: Some("f".into()), op:"cdl.fuse".into(),
                      r#in: json!({"base": concept, "delta": delta}), out: json!({}), params: json!({}) },
            ]};
            let (_outs, subenv) = run_with_env(&Vm::new(), &prog)?;
            let fv = subenv.get("f").ok_or_else(|| anyhow::anyhow!("fuse failed"))?.val.clone();
            if save {
                let _ = Vm::new().run(&Program{ program: vec![
                    Node{ id: Some("s".into()), op:"concept.save".into(),
                          r#in: json!({"concept": fv}), out: json!({}), params: json!({}) },
                ]});
            }
            fv
        } else { concept.clone() };

        // NEW: hook lexicon.update (best-effort, ignore erreurs)
        if !lemma.is_empty() {
            let syns_vec: Vec<String> = syn_sorted.iter().map(|(k,_)| k.clone()).collect();
            let _ = Vm::new().run(&Program{ program: vec![
                Node{ id: Some("lx".into()), op:"lexicon.update".into(),
                      r#in: json!({"lang": lang, "lemma": lemma, "forms":[lemma], "syn": syns_vec, "count": 1.0}),
                      out: json!({}), params: json!({}) },
            ]});
        }

        let af = fused.pointer(&format!("/lexemes/{}/frames", lang))
            .and_then(|v| v.as_array()).unwrap_or(&vec![])
            .iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect::<Vec<_>>();
        let asy = fused.pointer(&format!("/lexemes/{}/syn", lang))
            .and_then(|v| v.as_array()).unwrap_or(&vec![])
            .iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect::<Vec<_>>();

        Ok(Capsule{
            typ:"LearnReport".into(),
            val: json!({
                "added_frames": af,
                "added_syn":    asy,
                "exceptions":   exceptions.into_iter().collect::<Vec<_>>(),
                "fused":        fused
            })
        })
    }
}


struct CuesLearn;
impl Kernel for CuesLearn {
    fn name(&self) -> &'static str { "cdl.cues.learn" }
    fn run(&self, _env:&HashMap<String, Capsule>, input:&Value, _:&Value) -> Result<Capsule> {
        let lang = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let today = today_iso();
        let lambda = 0.02_f64;

        // 0) charge stats existantes (avec décroissance)
        let cues_path = format!("./cues.{}.json", lang);
        let mut store = read_json(&cues_path).unwrap_or(json!({"lang":lang,"cues":{}}));
        let mut cues = store.get("cues").cloned().unwrap_or(json!({}));
        let mut cues_obj = cues.as_object().cloned().unwrap_or_default(); // class -> map

        // 1) collecte de textes
        let mut texts: Vec<String> = vec![];
        if let Some(arr) = input.get("lines").and_then(|v| v.as_array()) {
            for s in arr { if let Some(t) = s.as_str() { texts.push(t.to_string()); } }
        }
        if texts.is_empty() {
            if let Some(folder) = input.get("folder").and_then(|v| v.as_str()) {
                if let Ok(rd) = fs::read_dir(folder) {
                    for e in rd {
                        let p = e?.path();
                        if p.extension().and_then(|x| x.to_str()) == Some("txt") {
                            if let Ok(s) = fs::read_to_string(&p) {
                                for line in s.lines().map(|x| x.trim()).filter(|x| !x.is_empty()) {
                                    texts.push(line.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
        if texts.is_empty() {
            return Ok(Capsule{ typ:"CuesReport".into(), val: json!({"added":{}, "stats": store}) });
        }

        // 2) règles fr simples (GO-lite) -> classe & lexème de cue
        // NB: on normalise en minuscule, on prend des motifs robustes
        let patterns: Vec<(&str, &str)> = vec![
            // (regex, classe)
            (r"\bsi\b.*\balors\b", "IF"),
            (r"\bsauf si\b", "UNLESS"),
            (r"\bsinon\b", "OTHERWISE"),
            (r"\bà défaut de\b", "OTHERWISE"),
            (r"\bavant\b", "BEFORE"),
            (r"\baprès\b", "AFTER"),
            (r"\bquand\b", "WHEN"),
            (r"\blorsque\b", "WHEN"),
        ];
        let regs: Vec<(regex::Regex, &str)> = patterns.into_iter()
            .map(|(re, cls)| (regex::Regex::new(re).unwrap(), cls)).collect();

        // 3) comptage
        let mut added: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for t in texts {
            let low = t.to_lowercase();
            for (re, cls) in &regs {
                if re.is_match(&low) {
                    // cue lexème grossier = 1er match (tronqué à 5 mots)
                    if let Some(m) = re.find(&low) {
                        let mut cue = low[m.start()..m.end()].to_string();
                        // tronque "si ... alors" -> "si ... alors" (sans tout le reste)
                        if cue.len() > 80 { cue.truncate(80); }
                        let class_map = cues_obj.entry(cls.to_string()).or_insert(json!({})).as_object_mut().unwrap();
                        // applique décroissance au chargement (lazy) : on multiplie c existant par decay
                        // (pour éviter un passage O(n) séparé)
                        for (_k, v) in class_map.iter_mut() {
                            let last = v.get("last_seen").and_then(|x| x.as_str()).unwrap_or("2025-01-01");
                            let c = v.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0) * exp_decay(lambda, last);
                            *v = json!({"count": c, "last_seen": last});
                        }
                        incr_count(class_map, &cue, 1.0, &today);
                        added.entry(cls.to_string()).or_default().push(cue);
                    }
                }
            }
        }

        // 4) persiste
        store["cues"] = Value::Object(cues_obj);
        write_json(&cues_path, &store)?;

        Ok(Capsule{
            typ:"CuesReport".into(),
            val: json!({"added": added, "stats": store})
        })
    }
}

// ========== 2) cdl.cues.fuse ==========
// Promotion simple (GO-lite) : freq>=15 et récence non nulle (décroissance implicite).
// Garde champs pmi/gain_mdl si fournis, sans les exiger.
struct CuesFuse;
impl Kernel for CuesFuse {
    fn name(&self) -> &'static str { "cdl.cues.fuse" }
    fn run(&self, _env:&HashMap<String, Capsule>, input:&Value, _:&Value) -> Result<Capsule> {
        let lang = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let freq_min = input.get("freq_min").and_then(|v| v.as_f64()).unwrap_or(15.0);
        let cues_path = format!("./cues.{}.json", lang);
        let store = read_json(&cues_path).unwrap_or(json!({"lang":lang,"cues":{}}));
        let cues = store.get("cues").and_then(|v| v.as_object()).cloned().unwrap_or_default();

        let mut promoted: Vec<Value> = vec![];
        for (class, map) in cues {
            if let Some(m) = map.as_object() {
                for (cue, stat) in m {
                    let cnt = stat.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    if cnt >= freq_min {
                        let last = stat.get("last_seen").and_then(|x| x.as_str()).unwrap_or("1970-01-01").to_string();
                        promoted.push(json!({"cue": cue, "class": class, "count": cnt, "last_seen": last}));
                    }
                }
            }
        }

        // Exporte une vue “promoted” minimale (optionnelle)
        let promoted_path = format!("./cues.promoted.{}.json", lang);
        write_json(&promoted_path, &json!({"lang":lang,"promoted":promoted}))?;

        Ok(Capsule{ typ:"CuesFuseReport".into(), val: json!({"promoted": promoted}) })
    }
}

// ========== 3) cdl.edges.update ==========
// Entrée flexible :
//  - pairs: [{from:"CID", to:"CID", guard:"IF|UNLESS|OTHERWISE|...", etype:"CAUSES|ENABLES|PREVENTS|MITIGATES|RESOLVES", weight:f64}]
//  - OU cooccs: [{a:{concept_id:..,roles:{...}}, b:{...}, guard:"IF|.."}] -> type & weight heuristiques
// Merge avec décroissance + last_seen. Persiste ./graph.edges.json
struct EdgesUpdate;
impl Kernel for EdgesUpdate {
    fn name(&self) -> &'static str { "cdl.edges.update" }
    fn run(&self, env:&HashMap<String, Capsule>, input:&Value, _:&Value) -> Result<Capsule> {
        let today = today_iso();
        let lambda = 0.02_f64;

        let mut edges = read_json("./graph.edges.json").unwrap_or(json!({"edges":[]}));
        let mut list = edges.get("edges").and_then(|v| v.as_array()).cloned().unwrap_or_default();

        // fonction clé pour dédupliquer
        let edge_key = |e:&Value| -> String {
            format!("{}|{}|{}|{}",
                e.get("from").and_then(|x| x.as_str()).unwrap_or(""),
                e.get("to").and_then(|x| x.as_str()).unwrap_or(""),
                e.get("etype").and_then(|x| x.as_str()).unwrap_or(""),
                e.get("guard").and_then(|x| x.as_str()).unwrap_or("NONE"))
        };

        // index existant
        let mut idx: HashMap<String, usize> = HashMap::new();
        for (i, e) in list.iter().enumerate() {
            idx.insert(edge_key(e), i);
        }

        // helper pour upsert + décroissance
        let mut upsert = |from:&str, to:&str, guard:&str, etype:&str, w:f64| {
            let key = format!("{}|{}|{}|{}", from,to,etype,guard);
            let (mut count, mut weight, mut last) = (0.0, 0.0, "1970-01-01".to_string());
            if let Some(i) = idx.get(&key).cloned() {
                if let Some(e) = list.get(i) {
                    let l = e.get("last_seen").and_then(|x| x.as_str()).unwrap_or("2025-01-01");
                    let decay = exp_decay(lambda, l);
                    count = e.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0) * decay;
                    weight = e.get("weight").and_then(|x| x.as_f64()).unwrap_or(0.0) * decay;
                    last = l.to_string();
                }
            }
            count += 1.0;
            weight = (weight + w).min(10.0);
            let new = json!({"from":from,"to":to,"etype":etype,"guard":guard,"count":count,"weight":weight,"last_seen":today});
            if let Some(i) = idx.get(&key).cloned() {
                list[i] = new;
            } else {
                idx.insert(key, list.len());
                list.push(new);
            }
        };

        // 1) pairs brutes
        if let Some(arr) = input.get("pairs").and_then(|v| v.as_array()) {
            for it in arr {
                let from = it.get("from").and_then(|x| x.as_str()).unwrap_or("");
                let to   = it.get("to").and_then(|x| x.as_str()).unwrap_or("");
                if from.is_empty() || to.is_empty() { continue; }
                let guard = it.get("guard").and_then(|x| x.as_str()).unwrap_or("NONE");
                let etype = it.get("etype").and_then(|x| x.as_str()).unwrap_or("ENABLES");
                let w     = it.get("weight").and_then(|x| x.as_f64()).unwrap_or(1.0);
                upsert(from, to, guard, etype, w);
            }
        }

        // 2) cooccurrences d’instances (heuristique)
        if let Some(arr) = input.get("cooccs").and_then(|v| v.as_array()) {
            let role = |i:&Value, k:&str| -> String {
                i.get("roles").and_then(|r| r.get(k)).and_then(|v| v.as_str()).unwrap_or("").to_string()
            };
            for it in arr {
                let a = resolve(env, it.get("a").unwrap_or(&json!(null)))?;
                let b = resolve(env, it.get("b").unwrap_or(&json!(null)))?;
                let ga = it.get("guard").and_then(|x| x.as_str()).unwrap_or("NONE");
                let from = a.get("concept_id").and_then(|x| x.as_str()).unwrap_or("");
                let to   = b.get("concept_id").and_then(|x| x.as_str()).unwrap_or("");
                if from.is_empty() || to.is_empty() { continue; }

                // compat de rôles simple -> spécificité bonus
                let spec = (role(&a,"Receiver") != "" && role(&a,"Receiver") == role(&b,"Agent")) as i32 as f64
                         + (role(&a,"Theme")    != "" && role(&a,"Theme")    == role(&b,"Theme")) as i32 as f64;

                // map guard -> type
                let etype = match ga {
                    "UNLESS" => "PREVENTS",
                    "OTHERWISE" => "RESOLVES",
                    "BEFORE" => "ENABLES",
                    "AFTER" => "CAUSES",
                    "WHEN" | "IF" => "CAUSES",
                    _ => "ENABLES",
                };
                let w = 1.0 + 0.5 * spec;
                upsert(from, to, ga, etype, w);
            }
        }

        edges["edges"] = json!(list);
        write_json("./graph.edges.json", &edges)?;
        Ok(Capsule{ typ:"EdgesReport".into(), val: edges })
    }
}

// ===== cdl.autogrow =====
struct AutoGrow;
impl Kernel for AutoGrow {
    fn name(&self) -> &'static str { "cdl.autogrow" }

    fn run(&self,
           _env:&std::collections::HashMap<String, Capsule>,
           input:&serde_json::Value,
           _:&serde_json::Value) -> anyhow::Result<Capsule>
    {
        use serde_json::{json, Value};
        use std::fs;

        let folder = input.get("folder").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("folder required"))?;
        let lang   = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let epochs = input.get("epochs").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let save   = input.get("save").and_then(|v| v.as_bool()).unwrap_or(true);

        // métriques
        let mut m_frames = 0usize;
        let mut m_syn    = 0usize;
        let mut m_cues   = 0usize;
        let mut m_edges  = 0usize;
        let mut created_concepts = 0usize;

        // au début de run(), près des autres paramètres
        let freq_min = input.get("freq_min").and_then(|v| v.as_f64()).unwrap_or(15.0);

        // util: ensure concept from lemma+sentence (induce if missing)
        let mut ensure_concept = |lemma:&str, sentence:&str| -> anyhow::Result<Value> {
            // try load
            let try_load = Program { program: vec![
                Node{ id: Some("c".into()), op:"concept.load".into(),
                    r#in: json!({"id": lemma.to_uppercase(), "lang": lang}), out: json!({}), params: json!({}) },
            ]};
            if let Ok((_outs, env)) = run_with_env(&Vm::new(), &try_load) {
                if let Some(c) = env.get("c") { return Ok(c.val.clone()); }
            }
            // induce
            let induce = Program { program: vec![
                Node{ id: Some("c".into()), op:"cdl.induce".into(),
                    r#in: json!({"id": lemma.to_uppercase(), "lemma": lemma, "sentence": sentence, "lang": lang}),
                    out: json!({}), params: json!({}) },
                Node{ id: Some("s".into()), op:"concept.save".into(),
                    r#in: json!({"concept":"$c","lang":lang}), out: json!({}), params: json!({}) },
            ]};
            let (_outs, env) = run_with_env(&Vm::new(), &induce)?;
            created_concepts += 1;
            Ok(env.get("c").unwrap().val.clone())
        };

        // very small head-verb guesser (reuse your tokenizer if available)
        let head_guess = |line:&str| -> String {
            // trivial: 1er mot alpha → lower
            line.split(|c:char| !c.is_alphabetic())
                .find(|w| !w.is_empty())
                .unwrap_or("action").to_lowercase()
        };

        // EPOCHS
        for _ in 0..epochs {
            // 1) learn + fuse per concept encountered
            let mut touched: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

            for entry in fs::read_dir(folder)? {
                let path = entry?.path();
                if path.extension().and_then(|s| s.to_str()) != Some("txt") { continue; }
                let content = fs::read_to_string(&path)?;
                let mut prev_inst: Option<Value> = None;
                let mut prev_line: String = String::new();

                for line in content.lines().map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    let lemma = head_guess(line);
                    let concept = ensure_concept(&lemma, line)?;

                    // learn_from_corpus (unitaire: folder complet + decay cumule)
                    let prog_l = Program { program: vec![
                        Node{ id: Some("rep".into()), op:"cdl.learn_from_corpus".into(),
                            r#in: json!({"concept": concept, "lang":lang, "folder": folder, "save": false}),
                            out: json!({}), params: json!({}) },
                        Node{ id: Some("f".into()), op:"cdl.fuse".into(),
                            r#in: json!({"base": concept, "delta": "$rep.fused"}), out: json!({}), params: json!({}) },
                        Node{ id: Some("s".into()), op:"concept.save".into(),
                            r#in: json!({"concept":"$f","lang":lang}), out: json!({}), params: json!({}) },
                        // bind instance courant (servira à observe)
                        Node{ id: Some("i".into()), op:"concept.bind_text".into(),
                            r#in: json!({"concept":"$f","text": line, "lang": lang, "allow_partial": true}),
                            out: json!({}), params: json!({}) },
                    ]};
                    let (_outs, subenv) = run_with_env(&Vm::new(), &prog_l)?;
                    // métriques incrementales (frames/syn ajoutées dans rep.added_*)
                    if let Some(rep) = subenv.get("rep") {
                        m_frames += rep.val.get("added_frames").and_then(|a| a.as_array()).map(|a| a.len()).unwrap_or(0);
                        m_syn    += rep.val.get("added_syn").and_then(|a| a.as_array()).map(|a| a.len()).unwrap_or(0);
                    }
                    touched.insert(concept.get("id").and_then(|v| v.as_str()).unwrap_or("?").to_string());

                    // 2-lignes: observe edges
                    if let Some(prev) = prev_inst.take() {
                        let window = format!("{} /// {}", prev_line, line);
                        let prog_e = Program { program: vec![
                            Node{ id: Some("obs".into()), op:"reason.graph.observe".into(),
                                r#in: json!({"a": prev, "b":"$i", "window":window, "lang":lang}),
                                out: json!({}), params: json!({}) },
                        ]};
                        let (_o, _env_e) = run_with_env(&Vm::new(), &prog_e)?;
                        m_edges += 1;
                    }
                    prev_inst = subenv.get("i").map(|c| c.val.clone());
                    prev_line = line.to_string();
                }
            }

            // 2) cues: learn + fuse (après un tour de corpus)
            let prog_cues = Program { program: vec![
                Node{ id: Some("c1".into()), op:"cdl.cues.learn".into(),
                    r#in: json!({ "lang": lang, "folder": folder }),
                    out: json!({}), params: json!({}) },
                Node{ id: Some("c2".into()), op:"cdl.cues.fuse".into(),
                    r#in: json!({ "lang": lang, "freq_min": freq_min }),
                    out: json!({}), params: json!({}) },
            ]};
            let (_outs_c, env_c) = run_with_env(&Vm::new(), &prog_cues)?;
            if let Some(c2) = env_c.get("c2") {
                m_cues += c2.val.get("promoted").and_then(|a| a.as_array()).map(|a| a.len()).unwrap_or(0);
            }
        }

        Ok(Capsule{
            typ:"AutogrowReport".into(),
            val: json!({
                "frames_added": m_frames,
                "syn_promoted": m_syn,
                "cues_promoted": m_cues,
                "edges_observed": m_edges,
                "concepts_created": created_concepts
            })
        })
    }
}




// -----------------------------------------------
// reason.graph.activate  (scoring + ALT, style possédé)
// -----------------------------------------------
struct ReasonGraphActivate;
impl Kernel for ReasonGraphActivate {
    fn name(&self) -> &'static str { "reason.graph.activate" }

    fn run(&self, env:&HashMap<String, Capsule>, input:&Value, _:&Value) -> Result<Capsule> {
        // ==== (A) code existant inchangé (légères variables renom) ====
        // instances / seed
        let mut instances: Vec<Value> = vec![];
        if let Some(arr) = input.get("instances").and_then(|v| v.as_array()) {
            for it in arr { instances.push(resolve(env, it)?); }
        } else if input.get("seed").is_some() {
            instances.push(resolve(env, &input["seed"])?);
        } else { return Err(anyhow!("instances or seed required")); }

        // exceptions
        let exceptions: std::collections::HashSet<String> = input.get("exceptions")
            .and_then(|v| v.as_array()).unwrap_or(&Vec::new())
            .iter().filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect();

        // helpers
        let role = |i:&Value, k:&str| -> String {
            i.get("roles").and_then(|r| r.get(k)).and_then(|v| v.as_str()).unwrap_or("").to_string()
        };
        let count_fixed_roles = |i:&Value| -> usize {
            i.get("roles").and_then(|r| r.as_object()).map(|m| {
                m.iter().filter(|(_,v)| v.is_string() || v.is_number() || v.is_boolean()).count()
            }).unwrap_or(0)
        };
        let has_guard = |d:&Value| -> bool {
            d.get("when_has").is_some() || d.get("if").is_some() || d.get("unless").is_some() || d.get("otherwise").is_some()
        };
        let get_defaults_for = |cid:&str| -> Vec<Value> {
            for cap in env.values() {
                if cap.typ == "Concept" && cap.val.get("id").and_then(|v| v.as_str()) == Some(cid) {
                    if let Some(arr) = cap.val.get("defaults").and_then(|v| v.as_array()) { return arr.to_vec(); }
                }
            }
            let paths = [format!("./concepts/{}.fr.json", cid), format!("./concepts/{}.json",cid)];
            for p in &paths {
                if let Ok(s) = std::fs::read_to_string(p) {
                    if let Ok(v) = serde_json::from_str::<Value>(&s) {
                        if let Some(arr) = v.get("defaults").and_then(|x| x.as_array()) { return arr.to_vec(); }
                    }
                }
            }
            Vec::new()
        };
        let lambda = 0.02_f64;
        let today  = chrono::Utc::now().naive_utc().date();
        let age_days = |iso:&str| -> Option<f64> {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(iso, "%Y-%m-%d") {
                return Some((today - d).num_days() as f64);
            }
            None
        };

        let mut props: Vec<Value> = vec![];
        let mut alts:  Vec<Value> = vec![];

        for inst in &instances {
            let cid = inst.get("concept_id").and_then(|v| v.as_str()).unwrap_or("<unknown>");
            let defaults = get_defaults_for(cid);

            // bucket par THEN
            let mut buckets: HashMap<String, Vec<(f64,Value)>> = HashMap::new();

            for d in defaults {
                let then_s = d.get("then").and_then(|v| v.as_str()).unwrap_or("").to_string();
                if then_s.is_empty() { continue; }

                // when_has: tous les rôles requis doivent être présents
                if let Some(wh) = d.get("when_has").and_then(|v| v.as_array()) {
                    let mut ok = true;
                    for rname in wh {
                        if let Some(rn) = rname.as_str() {
                            if role(&inst, rn).is_empty() { ok = false; break; }
                        }
                    }
                    if !ok { continue; }
                }

                // strength via support
                let meta = d.get("meta").cloned().unwrap_or(json!({}));
                let support = meta.get("support").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let strength = (1.0 + support).ln();

                // spécificité
                let spec = 1.0
                    + 0.2 * (count_fixed_roles(&inst) as f64)
                    + if has_guard(&d) { 0.5 } else { 0.0 };

                // récence
                let recency = if let Some(ad) = meta.get("last_seen").and_then(|v| v.as_str()).and_then(|s| age_days(s)) {
                    (-lambda * ad).exp()
                } else { 1.0 };

                let score = spec * strength * recency;

                // gestion exception
                let is_expect_return = then_s.starts_with("EXPECT(") && then_s.contains("RETURN(");
                if is_expect_return && exceptions.contains("NO_RETURN_EXPECTED") {
                    alts.push(json!({ "kind":"ACTIVATE", "text": then_s, "score": score }));
                    continue;
                }

                buckets.entry(then_s.clone()).or_default().push((score, json!({
                    "kind":"ACTIVATE", "text": then_s, "score": score
                })));
            }

            for (_k, mut vecs) in buckets {
                if vecs.is_empty() { continue; }
                vecs.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                props.push(vecs[0].1.clone());
                for alt in vecs.into_iter().skip(1) {
                    let t = alt.1.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    alts.push(json!({ "kind":"ALT", "text": t, "score": alt.0 }));
                }
            }
        }

        // ==== (B) AJOUT : edges.json → propositions EDGE ====
        if let Some(g) = read_json("./graph.edges.json") {
            if let Some(arr) = g.get("edges").and_then(|v| v.as_array()) {
                // seed set pour filtrer edges sortants pertinents
                let mut seed: BTreeSet<String> = BTreeSet::new();
                for inst in &instances {
                    if let Some(cid) = inst.get("concept_id").and_then(|v| v.as_str()) {
                        seed.insert(cid.to_string());
                    }
                }
                for e in arr {
                    let from = e.get("from").and_then(|x| x.as_str()).unwrap_or("");
                    if !seed.contains(from) { continue; }
                    let to    = e.get("to").and_then(|x| x.as_str()).unwrap_or("");
                    let etype = e.get("etype").and_then(|x| x.as_str()).unwrap_or("ENABLES");
                    let guard = e.get("guard").and_then(|x| x.as_str()).unwrap_or("NONE");
                    let count = e.get("count").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    let w     = e.get("weight").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    let rec   = e.get("last_seen").and_then(|x| x.as_str()).map(|s| {
                        if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                            let ad = (chrono::Utc::now().naive_utc().date() - d).num_days() as f64;
                            (-0.02_f64 * ad).exp()
                        } else { 1.0 }
                    }).unwrap_or(1.0);
                    let score = (1.0 + count.ln()) * (1.0 + w) * rec;

                    props.push(json!({
                        "kind":"EDGE",
                        "from": from, "to": to, "etype": etype, "guard": guard,
                        "score": score,
                        "text": format!("EDGE({} --{}[{}]--> {})", from, guard, etype, to)
                    }));
                }
            }
        }

        // tri scoré (desc)
        props.sort_by(|a,b| {
            let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Capsule{
            typ:"Plan".into(),
            val: json!({ "propositions": props, "alternatives": alts })
        })
    }
}

struct GraphEdgesUpdate;
impl Kernel for GraphEdgesUpdate {
    fn name(&self) -> &'static str { "graph.edges.update" }
    fn run(&self, _env:&std::collections::HashMap<String, Capsule>, input:&serde_json::Value, _:&serde_json::Value) -> anyhow::Result<Capsule> {
        use serde_json::{json, Value};
        use std::fs;

        let from  = input.get("from").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("from required"))?;
        let to    = input.get("to").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("to required"))?;
        let typ   = input.get("type").and_then(|v| v.as_str()).unwrap_or("CAUSES");
        let guard = input.get("guard").and_then(|v| v.as_str()).unwrap_or("NONE");
        let delta_support = input.get("delta_support").and_then(|v| v.as_f64()).unwrap_or(1.0);

        // récence
        let lambda = 0.02_f64;
        let today  = chrono::Utc::now().naive_utc().date();
        let age_days = |iso:&str| -> Option<f64> {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(iso, "%Y-%m-%d") {
                return Some((today - d).num_days() as f64);
            }
            None
        };
        let today_s = today.format("%Y-%m-%d").to_string();

        // charge store
        let path = "./graph.edges.json";
        let mut store = if let Ok(s) = fs::read_to_string(path) {
            serde_json::from_str::<Value>(&s).unwrap_or(json!({"edges":[]}))
        } else { json!({"edges":[]}) };

        // index -> map clé
        let edges = store.get("edges").cloned().unwrap_or(json!([]));
        let mut arr = edges.as_array().cloned().unwrap_or_default();

        // cherche la clé
        let mut found = None;
        for (i,e) in arr.iter().enumerate() {
            let kf = e.get("from").and_then(|v| v.as_str()).unwrap_or("");
            let kt = e.get("to").and_then(|v| v.as_str()).unwrap_or("");
            let ktp= e.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let kg = e.get("guard").and_then(|v| v.as_str()).unwrap_or("");
            if kf==from && kt==to && ktp==typ && kg==guard {
                found = Some(i); break;
            }
        }

        if let Some(i) = found {
            let mut e = arr[i].clone();
            let last = e.get("last_seen").and_then(|v| v.as_str()).unwrap_or("2025-01-01");
            let prev = e.get("support").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let dec  = age_days(last).map(|ad| (-lambda * ad).exp()).unwrap_or(1.0);
            let new_support = prev * dec + delta_support;
            e.as_object_mut().unwrap().insert("support".into(), json!(new_support));
            e.as_object_mut().unwrap().insert("last_seen".into(), json!(today_s));
            arr[i] = e;
        } else {
            arr.push(json!({
                "from": from, "to": to, "type": typ, "guard": guard,
                "support": delta_support, "last_seen": today_s
            }));
        }

        store.as_object_mut().unwrap().insert("edges".into(), Value::Array(arr));
        // écriture atomique
        let tmp = format!("{}.tmp", path);
        fs::write(&tmp, serde_json::to_string_pretty(&store)?)?;
        fs::rename(&tmp, path)?;

        Ok(Capsule{ typ:"EdgeUpdated".into(), val: json!({"from":from,"to":to,"type":typ,"guard":guard}) })
    }
}

// ===== reason.graph.observe =====
struct ReasonGraphObserve;
impl Kernel for ReasonGraphObserve {
    fn name(&self) -> &'static str { "reason.graph.observe" }

    fn run(&self,
           env:&std::collections::HashMap<String, Capsule>,
           input:&serde_json::Value,
           _:&serde_json::Value) -> anyhow::Result<Capsule>
    {
        use serde_json::{json, Value};
        use std::collections::HashMap;
        use std::fs;

        let lang = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let inst_a = resolve(env, input.get("a").ok_or_else(|| anyhow::anyhow!("a required"))?)?;
        let inst_b = resolve(env, input.get("b").ok_or_else(|| anyhow::anyhow!("b required"))?)?;

        let cid_a = inst_a.get("concept_id").and_then(|v| v.as_str()).unwrap_or("?");
        let cid_b = inst_b.get("concept_id").and_then(|v| v.as_str()).unwrap_or("?");

        // 1) guard via cues promues
        let cues_path = format!("./cues.{}.json", lang);
        let mut guard = "NONE".to_string();
        if let Ok(s) = fs::read_to_string(&cues_path) {
            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                // format attendu: { "promoted":[{"form":"sauf si","kind":"UNLESS"}, ...] }
                let promoted = v.get("promoted").and_then(|x| x.as_array()).cloned().unwrap_or_default();
                let window = input.get("window").and_then(|x| x.as_str()).unwrap_or("").to_lowercase();
                for p in promoted {
                    if let (Some(form), Some(kind)) = (p.get("form").and_then(|x| x.as_str()),
                                                       p.get("kind").and_then(|x| x.as_str())) {
                        if window.contains(&form.to_lowercase()) { guard = kind.to_string(); break; }
                    }
                }
            }
        }

        // 2) type depuis defaults(A) si on voit B dans THEN
        let mut rel_type = "CAUSES".to_string();
        let defaults_of = |cid:&str| -> Vec<Value> {
            // cherche en env, sinon ./concepts/<id>.(fr.)json
            for cap in env.values() {
                if cap.typ=="Concept" && cap.val.get("id").and_then(|x| x.as_str())==Some(cid) {
                    if let Some(a) = cap.val.get("defaults").and_then(|x| x.as_array()) { return a.to_vec(); }
                }
            }
            for p in [format!("./concepts/{}.fr.json", cid), format!("./concepts/{}.json", cid)] {
                if let Ok(s) = fs::read_to_string(&p) {
                    if let Ok(v) = serde_json::from_str::<Value>(&s) {
                        if let Some(a) = v.get("defaults").and_then(|x| x.as_array()) { return a.to_vec(); }
                    }
                }
            }
            vec![]
        };
        for d in defaults_of(cid_a) {
            if let Some(t) = d.get("then").and_then(|x| x.as_str()) {
                if t.contains(cid_b) {
                    if t.starts_with("PREVENT(") || t.contains("NOT_") { rel_type = "PREVENTS".into(); break; }
                    if t.starts_with("ENABLE(")  { rel_type = "ENABLES".into();  break; }
                    if t.starts_with("MITIGATE("){ rel_type = "MITIGATES".into(); break; }
                    if t.starts_with("RESOLVE(") { rel_type = "RESOLVES".into(); break; }
                    rel_type = "CAUSES".into(); break;
                }
            }
        }

        // 3) spécificité simple via compat rôles
        let role = |i:&Value, k:&str| -> String {
            i.get("roles").and_then(|r| r.get(k)).and_then(|v| v.as_str()).unwrap_or("").to_string()
        };
        let compat = {
            let mut s: f64 = 0.0;
            if !role(&inst_a,"Receiver").is_empty() && role(&inst_a,"Receiver")==role(&inst_b,"Agent") { s += 1.0; }
            if !role(&inst_a,"Theme").is_empty()    && role(&inst_a,"Theme")==role(&inst_b,"Theme")   { s += 0.6; }
            if !role(&inst_a,"Agent").is_empty()    && role(&inst_a,"Agent")==role(&inst_b,"Receiver"){ s += 0.4; }
            s.max(0.1_f64)
        };

        // 4) force & récence (décroissance)
        let lambda = 0.02_f64;
        let today  = chrono::Utc::now().naive_utc().date();
        let decay  = |last:&str| -> f64 {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(last, "%Y-%m-%d") {
                (-lambda * (today - d).num_days() as f64).exp()
            } else { 1.0 }
        };

        // 5) edge store
        let edges_path = "./graph.edges.json";
        let mut edges: Vec<Value> = if let Ok(s) = fs::read_to_string(edges_path) {
            serde_json::from_str(&s).unwrap_or_else(|_| vec![])
        } else { vec![] };

        // recherche d’un edge équivalent
        let mut found = None;
        for (idx, e) in edges.iter().enumerate() {
            if e.get("src").and_then(|x| x.as_str())==Some(cid_a)
            && e.get("dst").and_then(|x| x.as_str())==Some(cid_b)
            && e.get("type").and_then(|x| x.as_str())==Some(&rel_type)
            && e.get("guard").and_then(|x| x.as_str())==Some(&guard)
            {
                found = Some(idx); break;
            }
        }

        let mut support = 1.0;
        let mut last_s = today.format("%Y-%m-%d").to_string();
        if let Some(idx) = found {
            let prev = edges[idx].clone();
            support = prev.get("support").and_then(|x| x.as_f64()).unwrap_or(0.0) + 1.0;
            last_s  = today.format("%Y-%m-%d").to_string();
            let prev_last = prev.get("last_seen").and_then(|x| x.as_str()).unwrap_or("2025-01-01");
            let prev_w = prev.get("weight").and_then(|x| x.as_f64()).unwrap_or(0.0) * decay(prev_last);
            let weight = (1.0 + support).ln() * compat + prev_w;
            edges[idx] = json!({"src":cid_a,"dst":cid_b,"type":rel_type,"guard":guard,
                                "support": support, "weight": weight, "last_seen": last_s});
        } else {
            let weight = (1.0 + support).ln() * compat;
            edges.push(json!({"src":cid_a,"dst":cid_b,"type":rel_type,"guard":guard,
                              "support": support, "weight": weight, "last_seen": last_s}));
        }

        // persist (atomique best-effort)
        let tmp = format!("{}.tmp", edges_path);
        fs::write(&tmp, serde_json::to_vec_pretty(&edges)?)?;
        fs::rename(&tmp, edges_path)?;

        Ok(Capsule{ typ:"EdgeUpdate".into(), val: json!({"updated": true, "src":cid_a, "dst":cid_b}) })
    }
}

// ===== compose.generic =====
struct ComposeGeneric;
impl Kernel for ComposeGeneric {
    fn name(&self) -> &'static str { "compose.generic" }

    fn run(&self, env:&HashMap<String,Capsule>, input:&Value, _: &Value) -> Result<Capsule> {
        let arr = input.get("instances")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("instances array required"))?;

        // Résout toutes les références d'instances
        let mut insts: Vec<Value> = Vec::with_capacity(arr.len());
        for it in arr { insts.push(resolve(env, it)?); }

        // Helper : récupérer un rôle sous forme possédée (évite les lifetimes)
        let role = |i:&Value, k:&str| -> String {
            i.get("roles")
                .and_then(|r| r.get(k))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };

        // Si exactement 3 évts → garde ton heuristique d’origine
        let mut ordered = insts.clone();
        if insts.len() == 3 {
            let a0 = role(&insts[0], "Receiver"); // acheteur (SELL)
            let a1 = role(&insts[1], "Agent");    // payeur  (PAY)
            let g0 = role(&insts[0], "Giver");    // vendeur (SELL)
            let r2 = role(&insts[2], "Receiver"); // remercié (THANK)
            if !a0.is_empty() && a0 == a1 && !g0.is_empty() && r2 == g0 {
                ordered = insts.clone(); // déjà dans le bon ordre
            }
        } else if insts.len() > 3 {
            // v1: tri glouton basé sur une compat stricte + petits bonus
            let link_score = |a:&Value, b:&Value| -> f64 {
                let a_recv = role(a,"Receiver");
                let b_ag   = role(b,"Agent");
                let mut s = 0.0;
                if !a_recv.is_empty() && a_recv == b_ag { s += 1.0; }

                let a_th = role(a,"Theme");
                let b_th = role(b,"Theme");
                if !a_th.is_empty() && a_th == b_th { s += 0.6; }

                let a_cons = a.get("roles").and_then(|r| r.get("Consideration")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                let b_amt  = b.get("roles").and_then(|r| r.get("Amount")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                if !a_cons.is_empty() && a_cons == b_amt { s += 0.5; }

                // petit hint SELL > PAY > THANK
                let rank = |cid:&str| -> i32 { match cid { "SELL" | "SELL_TEST" => 3, "PAY" => 2, "THANK" => 1, _ => 0 } };
                let ra = rank(a.get("concept_id").and_then(|v| v.as_str()).unwrap_or(""));
                let rb = rank(b.get("concept_id").and_then(|v| v.as_str()).unwrap_or(""));
                if ra > rb { s += 0.3; }
                s
            };

            let n = insts.len();
            let mut outdeg = vec![0.0; n];
            let mut indeg  = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    if i==j { continue; }
                    let s = link_score(&insts[i], &insts[j]);
                    outdeg[i] += s; indeg[j] += s;
                }
            }
            let mut start = 0usize;
            let mut best = f64::MIN;
            for i in 0..n {
                let val = outdeg[i] - indeg[i];
                if val > best { best = val; start = i; }
            }
            let mut used = vec![false; n];
            let mut order: Vec<usize> = vec![start];
            used[start] = true;
            for _ in 0..(n-1) {
                let last = *order.last().unwrap();
                let mut best_j = None;
                let mut best_s = -1.0;
                for j in 0..n {
                    if used[j] { continue; }
                    let s = link_score(&insts[last], &insts[j]);
                    if s > best_s { best_s = s; best_j = Some(j); }
                }
                if let Some(j) = best_j { used[j] = true; order.push(j); }
            }
            ordered = order.into_iter().map(|idx| insts[idx].clone()).collect();
        }

        // Construit un plan textuel basique
        let mut props: Vec<Value> = Vec::with_capacity(ordered.len());
        for i in ordered {
            let cid = i.get("concept_id").and_then(|v| v.as_str()).unwrap_or("");

            // Rôles (possédés)
            let ag = role(&i, "Agent");
            let rv = role(&i, "Receiver");
            let th = role(&i, "Theme");
            let co = i.get("roles").and_then(|r| r.get("Consideration")).and_then(|v| v.as_str()).unwrap_or("").to_string();

            let text = match cid {
                "SELL" | "SELL_TEST" => {
                    let mut s = String::new();
                    s.push_str(if !ag.is_empty() { &ag } else { "Quelqu'un" });
                    s.push_str(" vend ");
                    s.push_str(if !th.is_empty() { &th } else { "quelque chose" });
                    if !rv.is_empty() { s.push_str(" à "); s.push_str(&rv); }
                    if !co.is_empty() { s.push_str(" pour "); s.push_str(&co); }
                    s.push('.');
                    s
                }
                "PAY" => {
                    let ag = role(&i,"Agent");
                    let rv = role(&i,"Receiver");
                    let am = i.get("roles").and_then(|r| r.get("Amount")).and_then(|v| v.as_str()).unwrap_or("");
                    if am.is_empty() {
                        if rv.is_empty() { format!("{ag} paie.") } else { format!("{ag} paie {rv}.") }
                    } else {
                        if rv.is_empty() { format!("{ag} paie {am}.") } else { format!("{ag} paie {am} à {rv}.") }
                    }
                }
                "THANK" => {
                    let mut s = String::new();
                    s.push_str(if !ag.is_empty() { &ag } else { "Quelqu'un" });
                    s.push_str(" remercie");
                    if !rv.is_empty() { s.push(' '); s.push_str(&rv); }
                    s.push('.');
                    s
                }
                _ => {
                    let mut s = String::new();
                    s.push_str(if !ag.is_empty() { &ag } else { "Quelqu'un" });
                    if !th.is_empty() {
                        s.push_str(" agit sur ");
                        s.push_str(&th);
                        if !rv.is_empty() { s.push_str(" avec "); s.push_str(&rv); }
                    } else {
                        s.push_str(" fait quelque chose");
                        if !rv.is_empty() { s.push_str(" avec "); s.push_str(&rv); }
                    }
                    s.push('.');
                    s
                }
            };

            props.push(json!(text));
        }

        Ok(Capsule { typ: "Plan".into(), val: json!({ "propositions": props }) })
    }
}

// ===== compose.from_activate (robuste) =====
struct ComposeFromActivate;
impl Kernel for ComposeFromActivate {
    fn name(&self) -> &'static str { "compose.from_activate" }

    fn run(&self,
           env:&std::collections::HashMap<String, Capsule>,
           input:&serde_json::Value,
           _:&serde_json::Value) -> anyhow::Result<Capsule>
    {
        use serde_json::{json, Value};
        use regex::Regex;
        use std::collections::{HashMap,HashSet};

        let lang = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");
        let allow: HashSet<String> = input.get("allow").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_else(|| ["APOLOGIZE","EXPLAIN_SHORT","PROPOSE_OPTIONS","ASK_PREFERENCE","REMIND","NEGOTIATE"]
                .into_iter().map(|s| s.to_string()).collect());

        // 1) Contexte monde (1er élément suffit pour les rôles par défaut)
        let ctx_inst: Option<Value> = input.get("context").and_then(|v| v.as_array())
            .and_then(|a| a.first()).map(|v| resolve(env, v)).transpose()?;

        // 2) Récup activations
        let mut calls: Vec<(f64,String)> = vec![];
        let pick_text = |v:&Value| -> Option<String> {
            v.get("text").and_then(|x| x.as_str()).map(|s| s.to_string())
             .or_else(|| v.as_str().map(|s| s.to_string()))
        };
        if let Some(arr) = input.get("activations").and_then(|v| v.as_array()) {
            for a in arr {
                if let Some(s) = pick_text(a) {
                    let score = a.get("score").and_then(|x| x.as_f64()).unwrap_or(1.0);
                    calls.push((score, s));
                }
            }
        } else if let Some(props) = input.pointer("/plan/propositions").and_then(|v| v.as_array()) {
            for p in props {
                if let Some(s) = pick_text(p) {
                    let score = p.get("score").and_then(|x| x.as_f64()).unwrap_or(1.0);
                    calls.push((score, s));
                }
            }
        } else {
            return Err(anyhow::anyhow!("compose.from_activate: 'activations' (ou plan.propositions) requis"));
        }

        // tri par score décroissant + dédup
        calls.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut seen = HashSet::<String>::new();
        calls.retain(|(_,s)| seen.insert(s.clone()));

        // 3) helpers context
        let get_role = |i:&Value, k:&str| -> Option<String> {
            i.get("roles").and_then(|r| r.get(k)).and_then(|v| v.as_str()).map(|s| s.to_string())
        };
        let fallback_for = |role:&str, ctx:&Option<Value>| -> String {
            match role {
                "Agent"    => "Support".to_string(),
                "Receiver" => ctx.as_ref().and_then(|c| get_role(c,"Receiver")).unwrap_or_else(|| "Client".to_string()),
                "Theme"    => ctx.as_ref().and_then(|c| get_role(c,"Theme")).unwrap_or_default(),
                _          => ctx.as_ref().and_then(|c| get_role(c,role)).unwrap_or_default(),
            }
        };

        // 4) parse ACT(args) + k=v support
        let re_call = Regex::new(r"^([A-Z_]+)\((.*)\)$").unwrap();
        let mut texts: Vec<String> = vec![];

        for (_score, then_s) in calls {
            let Some(caps) = re_call.captures(&then_s) else { continue };
            let act = caps.get(1).unwrap().as_str().to_string();
            if !allow.contains(&act) { continue; }

            let args_raw = caps.get(2).unwrap().as_str().trim();
            let mut roles_map = serde_json::Map::new();
            for tok in args_raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                if let Some(eq) = tok.find('=') {
                    let (k,v) = tok.split_at(eq);
                    let (k, v) = (k.trim(), v[1..].trim());
                    roles_map.insert(k.to_string(), json!(v));
                } else {
                    let val = fallback_for(tok, &ctx_inst);
                    if !val.is_empty() { roles_map.insert(tok.to_string(), json!(val)); }
                }
            }

            // **Mapping Receiver->Agent** si Agent absent
            if !roles_map.contains_key("Agent") {
                if let Some(rv) = roles_map.get("Receiver").and_then(|x| x.as_str()) {
                    roles_map.insert("Agent".into(), json!(rv));
                } else if let Some(ctx) = &ctx_inst {
                    if let Some(rv) = get_role(ctx, "Receiver") {
                        roles_map.insert("Agent".into(), json!(rv));
                    }
                }
            }

            // Sous-VM: concept.load -> instance.new -> surface.realize
            let sub = Program { program: vec![
                Node{ id: Some("c".into()), op:"concept.load".into(),
                    r#in: json!({"id": act, "lang": lang}), out: json!({}), params: json!({}) },
                Node{ id: Some("i".into()), op:"instance.new".into(),
                    r#in: json!({"concept":"$c","lang":lang,"roles": roles_map }), out: json!({}), params: json!({}) },
                Node{ id: Some("t".into()), op:"surface.realize".into(),
                    r#in: json!({"concept":"$c","instance":"$i","style":{"lang":lang}}),
                    out: json!({}), params: json!({}) },
            ]};

            match run_with_env(&Vm::new(), &sub) {
                Ok((_outs, subenv)) => {
                    if let Some(t) = subenv.get("t").and_then(|c| c.val.get("text")).and_then(|v| v.as_str()) {
                        texts.push(t.to_string());
                    } else { texts.push(then_s.clone()); }
                }
                Err(_) => { texts.push(then_s.clone()); }
            }
        }

        Ok(Capsule{ typ:"Plan".into(), val: json!({ "propositions": texts }) })
    }
}


// ===== cdl.autocurate =====
struct CdlAutoCurate;
impl Kernel for CdlAutoCurate {
    fn name(&self) -> &'static str { "cdl.autocurate" }
    fn run(&self, env:&HashMap<String,Capsule>, input:&Value, _: &Value) -> Result<Capsule> {
        let concept = resolve(env, input.get("concept").ok_or_else(|| anyhow!("concept required"))?)?;
        let save = input.get("save").and_then(|v| v.as_bool()).unwrap_or(false);

        let mut conc = concept.clone();

        // syn
        if let Some(stats) = input.get("stats") {
            if let Some(syn) = stats.get("syn").and_then(|v| v.as_object()) {
                let mut base: Vec<String> = conc.pointer("/lexemes/fr/syn")
                    .and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
                // garde syn avec freq >= 2
                let mut adds: Vec<String> = syn.iter().filter_map(|(k,v)| if v.as_i64().unwrap_or(0) >= 2 { Some(k.clone()) } else { None }).collect();
                base = merge_sorted_unique(base, &adds);
                conc["lexemes"]["fr"]["syn"] = json!(base);
            }
            // frames
            if let Some(fr) = stats.get("frames").and_then(|v| v.as_object()) {
                let mut base: Vec<String> = conc.pointer("/lexemes/fr/frames")
                    .and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
                let mut adds: Vec<String> = fr.iter().filter_map(|(k,v)| if v.as_i64().unwrap_or(0) >= 2 { Some(k.clone()) } else { None }).collect();
                base = merge_sorted_unique(base, &adds);
                conc["lexemes"]["fr"]["frames"] = json!(base);
            }
        }
        // defaults dédupliqués
        if let Some(defs) = conc.get_mut("defaults").and_then(|v| v.as_array_mut()) {
            use std::collections::HashSet;
            let mut seen: HashSet<(String,String)> = HashSet::new();
            let mut keep: Vec<Value> = vec![];
            for d in defs.iter() {
                let k = (d.get("if").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                         d.get("then").and_then(|v| v.as_str()).unwrap_or("").to_string());
                if seen.insert(k) { keep.push(d.clone()); }
            }
            *defs = keep;
        }

        if save {
            let prog = Program { program: vec![
                Node{ id: Some("s".into()), op:"concept.save".into(), r#in: json!({"concept": conc}), out: json!({}), params: json!({}) },
            ]};
            let _ = run_with_env(&Vm::new(), &prog)?;
        }

        Ok(Capsule{ typ:"Concept".into(), val: conc })
    }
}

struct AnalogyMap;
impl Kernel for AnalogyMap {
    fn name(&self) -> &'static str { "analogy.map" }
    fn run(&self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _params: &serde_json::Value
    ) -> anyhow::Result<Capsule> {
        use serde_json::{json, Value};
        fn roles_of(c: &serde_json::Value) -> Vec<String> {
        if let Some(a) = c.get("roles").and_then(|v| v.as_array()) {
            return a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
        }
        if let Some(m) = c.get("roles").and_then(|v| v.as_object()) {
            return m.keys().cloned().collect();
        }
        vec![]
    }
        let src = resolve(env, input.get("source").ok_or_else(|| anyhow::anyhow!("source required"))?)?;
        let tgt = resolve(env, input.get("target").ok_or_else(|| anyhow::anyhow!("target required"))?)?;
        let src_roles = roles_of(&src);
        let tgt_roles = roles_of(&tgt);

        let mut map = serde_json::Map::new();
        for r in &src_roles {
            if tgt_roles.iter().any(|t| t == r) { map.insert(r.clone(), json!(r)); }
        }
        for (a,b) in [("Patient","Theme"),("Theme","Patient"),
                      ("Amount","Consideration"),("Consideration","Amount"),
                      ("Receiver","Goal"),("Goal","Receiver")] {
            if src_roles.iter().any(|r| r==a) && tgt_roles.iter().any(|r| r==b) {
                map.entry(a.to_string()).or_insert(json!(b));
            }
        }
        let conf = (map.len() as f64) / (src_roles.len().max(1) as f64);
        Ok(Capsule{ typ:"Analogy".into(), val: json!({ "role_map": map, "confidence": conf }) })
    }
}


struct AnalogyRemapInstance;
impl Kernel for AnalogyRemapInstance {
    fn name(&self) -> &'static str { "analogy.remap_instance" }
    fn run(&self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _params: &serde_json::Value
    ) -> anyhow::Result<Capsule> {
        use serde_json::{json, Value};

        let inst_v = resolve(env, input.get("instance").ok_or_else(|| anyhow::anyhow!("instance required"))?)?;
        let tgt_v  = resolve(env, input.get("to_concept").ok_or_else(|| anyhow::anyhow!("to_concept required"))?)?;
        let map_v  = resolve(env, input.get("role_map").ok_or_else(|| anyhow::anyhow!("role_map required"))?)?;
        let map_obj = map_v.get("role_map").and_then(|v| v.as_object())
                        .or_else(|| map_v.as_object())
                        .ok_or_else(|| anyhow::anyhow!("role_map must be object or contain role_map"))?;

        let roles_src = inst_v.get("roles").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let mut roles_new = serde_json::Map::new();
        for (src_role, tgt_role_v) in map_obj {
            if let Some(tgt_role) = tgt_role_v.as_str() {
                if let Some(val) = roles_src.get(src_role).and_then(|v| v.as_str()) {
                    roles_new.insert(tgt_role.to_string(), json!(val));
                }
            }
        }
        let inst_new = json!({
            "concept_id": tgt_v.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            "roles": roles_new,
            "meta": { "lang": input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr") }
        });
        Ok(Capsule{ typ:"Instance".into(), val: inst_new })
    }
}
struct AnalogyRemap;
impl Kernel for AnalogyRemap {
    fn name(&self) -> &'static str { "analogy.remap" }

    fn run(
        &self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _: &serde_json::Value
    ) -> anyhow::Result<Capsule> {
        use serde_json::{json, Value};

        let src_i  = resolve(env, input.get("instance").ok_or_else(|| anyhow::anyhow!("analogy.remap: instance required"))?)?;
        let tgt_c  = resolve(env, input.get("concept").ok_or_else(|| anyhow::anyhow!("analogy.remap: concept required"))?)?;
        let lang   = input.get("lang").and_then(|v| v.as_str()).unwrap_or("fr");

        let src_roles = src_i.get("roles").and_then(|v| v.as_object()).cloned().unwrap_or_default();

        let get = |names: &[&str]| -> String {
            for n in names {
                if let Some(s) = src_roles.get(*n).and_then(|v| v.as_str()) {
                    if !s.trim().is_empty() { return s.trim().to_string(); }
                }
            }
            String::new()
        };

        // Lire depuis l’instance source (EAT)
        let agent = get(&["Agent","Giver","Subject","Actor"]);
        let theme = get(&["Patient","Theme","Object"]);

        // Choisir clé cible en fonction des rôles exposés par le concept cible (DEVOUR)
        let tgt_role_set: std::collections::HashSet<String> = match tgt_c.get("roles") {
            Some(serde_json::Value::Array(a))  => a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(),
            Some(serde_json::Value::Object(m)) => m.keys().cloned().collect(),
            _ => Default::default(),
        };


        let theme_key = if tgt_role_set.contains("Theme") { "Theme" }
                        else if tgt_role_set.contains("Patient") { "Patient" }
                        else { "Theme" };

        let mut roles = serde_json::Map::new();
        if !agent.is_empty() { roles.insert("Agent".into(), json!(agent)); }
        if !theme.is_empty() { roles.insert(theme_key.into(), json!(theme)); }

        let inst = json!({
            "concept_id": tgt_c.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            "roles": roles,
            "meta": { "lang": lang }
        });
        Ok(Capsule{ typ: "Instance".into(), val: inst })
    }
}

struct ConceptSpecialize;
impl Kernel for ConceptSpecialize {
    fn name(&self) -> &'static str { "concept.specialize" }
    fn run(&self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _params: &serde_json::Value
    ) -> anyhow::Result<Capsule> {
        use serde_json::{json, Value};

        let base = resolve(env, input.get("base").ok_or_else(|| anyhow::anyhow!("concept.specialize: base required"))?)?;
        let new_id = input.get("id").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("concept.specialize: id required"))?;

        let mut c = base.as_object().cloned().ok_or_else(|| anyhow::anyhow!("base must be object"))?;
        c.insert("id".into(), json!(new_id));

        // parents += base.id
        if let Some(pid) = base.get("id").and_then(|v| v.as_str()) {
            let mut parents = match c.get("parents") {
                Some(Value::Array(a)) => a.clone(),
                Some(Value::String(s)) => vec![json!(s)],
                _ => vec![],
            };
            if !parents.iter().any(|v| v.as_str() == Some(pid)) {
                parents.push(json!(pid));
            }
            c.insert("parents".into(), Value::Array(parents));
        }

        // roles: on ne touche pas (hérités)
        // add_roles (array => ajoute; object => fusion)
        // ... dans ConceptSpecialize::run, bloc add_roles ...
        if let Some(add_roles) = input.get("add_roles") {
            match (c.get("roles"), add_roles) {
                (Some(Value::Array(orig)), Value::Array(extra)) => {
                    let mut set = std::collections::BTreeSet::<String>::new();
                    for it in orig.iter().chain(extra.iter()) {
                        if let Some(s) = it.as_str() { set.insert(s.to_string()); }
                    }
                    c.insert("roles".into(), Value::Array(set.into_iter().map(Value::String).collect()));
                }
                (Some(Value::Object(orig)), Value::Object(extra)) => {
                    let mut m = orig.clone();
                    for (k, v) in extra { m.insert(k.clone(), v.clone()); }
                    c.insert("roles".into(), Value::Object(m));
                }
                (None, Value::Array(extra)) => {
                    // ⬇️ ICI: wrap en Value::Array
                    c.insert("roles".into(), Value::Array(extra.clone()));
                }
                (None, Value::Object(extra)) => {
                    c.insert("roles".into(), Value::Object(extra.clone()));
                }
                _ => {}
            }
        }
        // defaults += add_defaults
        if let Some(Value::Array(add)) = input.get("add_defaults") {
            let mut defs = c.get("defaults").and_then(|v| v.as_array()).cloned().unwrap_or_default();
            defs.extend(add.clone());
            c.insert("defaults".into(), Value::Array(defs));
        }

        // constraints += add_constraints (si fourni)
        if let Some(Value::Array(add)) = input.get("add_constraints") {
            let mut cons = c.get("constraints").and_then(|v| v.as_array()).cloned().unwrap_or_default();
            cons.extend(add.clone());
            c.insert("constraints".into(), Value::Array(cons));
        }

        // lexemes merge (léger) : on autorise un delta
        if let Some(delta) = input.get("lexeme_delta").and_then(|v| v.as_object()) {
            let mut lex = c.get("lexemes").cloned().unwrap_or(json!({}));
            let lm = lex.as_object_mut().unwrap();
            for (lang, spec) in delta {
                let mut tgt = lm.get(lang).cloned().unwrap_or(json!({}));
                let to = tgt.as_object_mut().unwrap();
                if let Some(frames) = spec.get("frames").and_then(|v| v.as_array()) {
                    let merged = to.get("frames").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                    let mut set = std::collections::BTreeSet::<String>::new();
                    for it in merged.iter().chain(frames.iter()) {
                        if let Some(s) = it.as_str() { set.insert(s.to_string()); }
                    }
                    to.insert("frames".into(), Value::Array(set.into_iter().map(Value::String).collect()));
                }
                if let Some(lemma) = spec.get("verb_lemma").and_then(|v| v.as_str()) {
                    to.insert("verb_lemma".into(), json!(lemma));
                }
                lm.insert(lang.clone(), Value::Object(to.clone()));
            }
            c.insert("lexemes".into(), Value::Object(lm.clone()));
        }

        Ok(Capsule{ typ:"Concept".into(), val: Value::Object(c) })
    }
}


// -- Checks --

struct CheckEquals;
impl Kernel for CheckEquals {
    fn name(&self) -> &'static str { "check.equals" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "lhs": "$a", "rhs": "$b" }
        let lhs = resolve(env, &input["lhs"])?;
        let rhs = resolve(env, &input["rhs"])?;
        if lhs != rhs {
            return Err(anyhow!("check.equals failed"));
        }
        Ok(Capsule { typ: "Check".into(), val: json!({"ok": true}) })
    }
}

struct CheckJsonSchema;
impl Kernel for CheckJsonSchema {
    fn name(&self) -> &'static str { "check.json_schema" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "json": "$caps|inline", "required_keys": ["a","b"] }
        let v = resolve(env, &input["json"])?;
        let req = input.get("required_keys").and_then(|x| x.as_array())
            .ok_or_else(|| anyhow!("required_keys must be array"))?;
        let keys: Vec<&str> = req.iter()
            .map(|k| k.as_str().ok_or_else(|| anyhow!("keys must be strings")))
            .collect::<Result<_>>()?;
        let check_obj = |obj: &serde_json::Map<String, Value>| -> Result<()> {
            for ks in &keys {
                if !obj.contains_key(*ks) { return Err(anyhow!("missing key: {}", ks)); }
            }
            Ok(())
        };
        match v {
            Value::Object(ref obj) => check_obj(obj)?,
            Value::Array(ref arr) => {
                for it in arr {
                    if let Some(obj) = it.as_object() { check_obj(obj)?; }
                }
            },
            _ => return Err(anyhow!("json must be object or array of objects")),
        }
        Ok(Capsule { typ: "Check".into(), val: json!({"ok": true}) })
    }
}

struct CheckMoney;
impl Kernel for CheckMoney {
    fn name(&self) -> &'static str { "check.money" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "money": "$m", "currency": "EUR", "min_cents": 0 }
        let m = resolve(env, &input["money"])?;
        let cents = m.get("amount_cents").and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("money.amount_cents"))?;
        let cur = m.get("currency").and_then(|v| v.as_str()).unwrap_or("EUR");
        if let Some(exp) = input.get("currency").and_then(|v| v.as_str()) {
            if cur != exp { return Err(anyhow!("currency mismatch {} != {}", cur, exp)); }
        }
        if let Some(minc) = input.get("min_cents").and_then(|v| v.as_i64()) {
            if cents < minc { return Err(anyhow!("amount below minimum")); }
        }
        Ok(Capsule { typ: "Check".into(), val: json!({"ok": true}) })
    }
}

struct CheckDateBetween;
impl Kernel for CheckDateBetween {
    fn name(&self) -> &'static str { "check.date_between" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "date": "2025-01-31", "start": "2025-01-01", "end": "2025-12-31", "formats": ["%Y-%m-%d","%d/%m/%Y"] }
        fn parse_date(s: &str, formats: &[&str]) -> Result<NaiveDate> {
            for fmt in formats {
                if let Ok(d) = NaiveDate::parse_from_str(s, fmt) { return Ok(d); }
            }
            Err(anyhow!("invalid date: {}", s))
        }
        let formats: Vec<&str> = input.get("formats").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_else(|| vec!["%Y-%m-%d"]);
        let date_s  = resolve(env, &input["date"])?.as_str().ok_or_else(|| anyhow!("date must be string"))?.to_string();
        let start_s = resolve(env, &input["start"])?.as_str().ok_or_else(|| anyhow!("start must be string"))?.to_string();
        let end_s   = resolve(env, &input["end"])?.as_str().ok_or_else(|| anyhow!("end must be string"))?.to_string();
        let d = parse_date(&date_s, &formats)?;
        let s = parse_date(&start_s, &formats)?;
        let e = parse_date(&end_s, &formats)?;
        if d < s || d > e { return Err(anyhow!("date out of range")); }
        Ok(Capsule { typ: "Check".into(), val: json!({"ok": true}) })
    }
}

// -- Proofs --
// Aplati une Value en string (utile pour contains / not_contains)
fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b)   => b.to_string(),
        Value::Null      => String::new(),
        Value::Array(a)  => a.iter().map(value_to_string).collect::<Vec<_>>().join(" "),
        Value::Object(o) => {
            // cas fréquent: capsules Text {"text":"..."}
            if let Some(t) = o.get("text").and_then(|x| x.as_str()) {
                t.to_string()
            } else {
                o.values().map(value_to_string).collect::<Vec<_>>().join(" ")
            }
        }
    }
}
// -- prove.all --
struct ProveAll;
impl Kernel for ProveAll {
    fn name(&self) -> &'static str { "prove.all" }
    fn run(&self, env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        let tests = input.get("tests").and_then(|v| v.as_array()).ok_or_else(|| anyhow!("tests must be array"))?;
        for (idx, t) in tests.iter().enumerate() {
            let op = t.get("op").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("tests[{}].op missing", idx))?;
            match op {
                "equals" => {
                    let lhs = resolve(env, &t["lhs"])?;
                    let rhs = resolve(env, &t["rhs"])?;
                    if lhs != rhs { return Err(anyhow!("prove.equals failed at {}", idx)); }
                }
                "near" => {
                    let lhs = resolve(env, &t["lhs"])?.as_f64().ok_or_else(|| anyhow!("near lhs must be number"))?;
                    let rhs = resolve(env, &t["rhs"])?.as_f64().ok_or_else(|| anyhow!("near rhs must be number"))?;
                    let atol = t.get("atol").and_then(|v| v.as_f64()).unwrap_or(1e-9);
                    if (lhs - rhs).abs() > atol { return Err(anyhow!("prove.near failed at {}", idx)); }
                }
                "schema" => {
                    let v = resolve(env, &t["json"])?;
                    let req = t.get("required_keys").and_then(|x| x.as_array()).ok_or_else(|| anyhow!("required_keys must be array"))?;
                    let keys: Vec<&str> = req.iter().map(|k| k.as_str().ok_or_else(|| anyhow!("keys must be strings"))).collect::<Result<_>>()?;
                    let check_obj = |obj: &serde_json::Map<String, Value>| -> Result<()> {
                        for ks in &keys { if !obj.contains_key(*ks) { return Err(anyhow!("missing key: {}", ks)); } }
                        Ok(())
                    };
                    match v {
                        Value::Object(ref obj) => check_obj(obj)?,
                        Value::Array(ref arr) => { for it in arr { if let Some(obj) = it.as_object() { check_obj(obj)?; } } },
                        _ => return Err(anyhow!("schema: json must be object or array of objects")),
                    }
                }
                "money" => {
                    let m = resolve(env, &t["money"])?;
                    let cents = m.get("amount_cents").and_then(|v| v.as_i64()).ok_or_else(|| anyhow!("money.amount_cents"))?;
                    if let Some(minc) = t.get("min_cents").and_then(|v| v.as_i64()) { if cents < minc { return Err(anyhow!("money below minimum at {}", idx)); } }
                    if let Some(cur) = t.get("currency").and_then(|v| v.as_str()) {
                        let cur_m = m.get("currency").and_then(|v| v.as_str()).unwrap_or(cur);
                        if cur_m != cur { return Err(anyhow!("money currency mismatch at {}", idx)); }
                    }
                }
                "date_between" => {
                    let parse = |s: &str| -> Result<chrono::NaiveDate> {
                        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"] {
                            if let Ok(d) = chrono::NaiveDate::parse_from_str(s, fmt) { return Ok(d); }
                        }
                        Err(anyhow!("invalid date: {}", s))
                    };
                    let d  = parse(resolve(env, &t["date"])?.as_str().ok_or_else(|| anyhow!("date must be string"))?)?;
                    let s  = parse(resolve(env, &t["start"])?.as_str().ok_or_else(|| anyhow!("start must be string"))?)?;
                    let e  = parse(resolve(env, &t["end"])?.as_str().ok_or_else(|| anyhow!("end must be string"))?)?;
                    if d < s || d > e { return Err(anyhow!("date out of range at {}", idx)); }
                }

                // ✅ nouveaux opérateurs
                "contains" => {
                    let hay = value_to_string(&resolve(env, &t["lhs"])?);
                    let needle = t.get("rhs")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow!("contains: rhs must be string"))?;
                    if !hay.contains(needle) {
                        return Err(anyhow!("prove.contains failed at {}", idx));
                    }
                }
                "not_contains" => {
                    let hay = value_to_string(&resolve(env, &t["lhs"])?);
                    let needle = t.get("rhs")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow!("not_contains: rhs must be string"))?;
                    if hay.contains(needle) {
                        return Err(anyhow!("prove.not_contains failed at {}", idx));
                    }
                }

                other => return Err(anyhow!("unknown prove op: {}", other))
            }
        }
        Ok(Capsule{ typ: "Proof".into(), val: json!({"ok": true, "n": tests.len()}) })
    }
}


// -- Render --

// Interprète "$id.path[/100] [EUR]" en Money {amount_cents, currency}
fn eval_money_expr_like(s: &str, env: &std::collections::HashMap<String, Capsule>) -> Option<Value> {
    use serde_json::Value;
    let mut expr = s.trim().to_string();
    if !expr.starts_with('$') { return None; }

    // devise optionnelle
    let mut currency = "EUR";
    if let Some(stripped) = expr.strip_suffix(" EUR") {
        expr = stripped.trim().to_string();
    }

    // "/100" optionnel (on l'ignore pour l'affichage car on travaille déjà en centimes)
    let _div100 = expr.ends_with("/100");
    if _div100 { expr = expr[..expr.len()-4].trim().to_string(); }

    // Résolution manuelle $id[.path...]
    if let Some(rest) = expr.strip_prefix('$') {
        let mut parts = rest.split('.');
        let id = parts.next().unwrap_or("");
        let cap = env.get(id)?;
        let mut cur = cap.val.clone();

        let mut had_path = false;
        for seg in parts {
            had_path = true;
            if let Ok(idx) = seg.parse::<usize>() {
                cur = cur.as_array()?.get(idx)?.clone();
            } else {
                cur = cur.as_object()?.get(seg)?.clone();
            }
        }

        // Si pas de chemin: accepter directement un Money {amount_cents,...}
        if !had_path {
            if let Some(obj) = cur.as_object() {
                if obj.get("amount_cents").and_then(|v| v.as_i64()).is_some() {
                    let curcode = obj.get("currency").and_then(|v| v.as_str()).unwrap_or(currency);
                    let cents   = obj.get("amount_cents").and_then(|v| v.as_i64()).unwrap();
                    return Some(json!({ "amount_cents": cents, "currency": curcode }));
                }
            }
        }

        // Sinon: on attend un entier/float (centimes)
        let cents = cur.as_i64().or_else(|| cur.as_f64().map(|f| f.round() as i64))?;
        return Some(json!({ "amount_cents": cents, "currency": currency }));
    }
    None
}


struct RenderText;
impl Kernel for RenderText {
    fn name(&self) -> &'static str { "render.text" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &Value, params: &Value) -> Result<Capsule> {
        let style = params.get("style").and_then(|v| v.as_str()).unwrap_or("neutral");

        // Ne JAMAIS appeler resolve() si c'est une chaîne "$..."; on évalue l'expression
        let read_money = |key: &str| -> Result<Value> {
            let raw = &input[key];
            if let Some(s) = raw.as_str() {
                if s.trim().starts_with('$') {
                    if let Some(m) = eval_money_expr_like(s, env) { return Ok(m); }
                    return Err(anyhow!("bad money expr: {}", s));
                }
            }
            // Sinon: valeur déjà résoluble (Money ou ref simple)
            let v = resolve(env, raw)?;
            Ok(v)
        };

        let sub = read_money("subtotal")?;
        let vat = read_money("vat")?;
        let tot = read_money("total")?;

        let to_eur = |m: &Value| -> Result<String> {
            let c = m.get("amount_cents").and_then(|v| v.as_i64()).ok_or_else(|| anyhow!("bad money"))?;
            let cur = m.get("currency").and_then(|v| v.as_str()).unwrap_or("EUR");
            Ok(format!("{:.2} {}", c as f64 / 100.0, cur))
        };

        let text = match style {
            "brief"   => format!("{} + TVA {} = {}.", to_eur(&sub)?, to_eur(&vat)?, to_eur(&tot)?),
            _         => format!("Sous-total: {}, TVA: {}, Total: {}.", to_eur(&sub)?, to_eur(&vat)?, to_eur(&tot)?),
        };

        Ok(Capsule { typ: "Text".into(), val: json!({ "text": text }) })
    }
}


// -- SQLite tool --

struct SqliteCall;
impl Kernel for SqliteCall {
    fn name(&self) -> &'static str { "call.sqlite" }
    fn run(&self, _env: &HashMap<String, Capsule>, input: &Value, _: &Value) -> Result<Capsule> {
        // input: { "db_path": "path/to.db", "sql": "SELECT ..." }
        let db  = input.get("db_path").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("db_path"))?;
        let sql = input.get("sql").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("sql must be string"))?;
        let conn = Connection::open(db)?;
        let mut stmt = conn.prepare(sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::<Value>::new();
        while let Some(row) = rows.next()? {
            let mut obj = serde_json::Map::new();
            let col_count = row.as_ref().column_count();
            for i in 0..col_count {
                let name_owned: String = row.as_ref()
                    .column_name(i)                // Result<&str, Error>
                    .map(|s| s.to_string())        // Result<String, Error>
                    .unwrap_or_else(|_| format!("c{}", i));
                let vref = row.get_ref(i)?;
                let jv = match vref {
                    ValueRef::Null       => Value::Null,
                    ValueRef::Integer(n) => json!(n),
                    ValueRef::Real(f)    => json!(f),
                    ValueRef::Text(t)    => json!(String::from_utf8_lossy(t).to_string()),
                    ValueRef::Blob(b)    => json!(format!("[blob {} bytes]", b.len())),
                };
                obj.insert(name_owned, jv);
            }
            out.push(Value::Object(obj));
        }
        Ok(Capsule { typ: "Table".into(), val: Value::Array(out) })
    }
}

// -- Citations --

// -- Cite require (robuste) --
struct CiteRequire;
impl Kernel for CiteRequire {
    fn name(&self) -> &'static str { "cite.require" }
    fn run(&self, env: &std::collections::HashMap<String, Capsule>, input: &serde_json::Value, _: &serde_json::Value) -> anyhow::Result<Capsule> {
        use serde_json::Value;

        // Essaie d'extraire un Vec<source> depuis plein de formes possibles.
        fn extract_sources(env: &std::collections::HashMap<String, Capsule>, v: &Value) -> anyhow::Result<Vec<Value>> {
            // 1) Si c’est une ref "$...": on essaie de résoudre; si échec, on tente un fallback manuel "$id"
            if let Some(s) = v.as_str() {
                if s.starts_with('$') {
                    // tentative directe via resolve (gère $id.path)
                    if let Ok(res) = resolve(env, v) {
                        return extract_sources(env, &res);
                    }
                    // fallback: $id tout seul
                    let id = s.trim_start_matches('$').split('.').next().unwrap_or("");
                    if let Some(cap) = env.get(id) {
                        return extract_sources(env, &cap.val);
                    }
                }
            }
            // 2) Déjà un array => ok
            if let Some(arr) = v.as_array() {
                return Ok(arr.clone());
            }
            // 3) Objet: sources dans un champ "sources" ou variante proche
            if let Some(obj) = v.as_object() {
                for key in ["sources", "items", "docs", "results"] {
                    if let Some(arr) = obj.get(key).and_then(|x| x.as_array()) {
                        return Ok(arr.clone());
                    }
                }
            }
            Err(anyhow::anyhow!("cite.require: cannot extract sources (expected array or object with 'sources')"))
        }

        let src = input.get("sources").ok_or_else(|| anyhow::anyhow!("cite.require: 'sources' required"))?;
        let min = input.get("min").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        let sources = extract_sources(env, src)?;
        if sources.len() < min {
            return Err(anyhow::anyhow!("cite.require: not enough sources ({} < {})", sources.len(), min));
        }
        Ok(Capsule { typ: "Cite".into(), val: serde_json::json!({ "count": sources.len(), "sources": sources }) })
    }
}
// -- plan.patch: remplace 'in' (et eventuellement 'params') de nœuds par id
// -- plan.patch: modifie les champs "in"/"params" des noeuds par id
struct PlanPatch;

impl Kernel for PlanPatch {
    fn name(&self) -> &'static str { "plan.patch" }

    fn run(
        &self,
        env: &std::collections::HashMap<String, Capsule>,
        input: &serde_json::Value,
        _: &serde_json::Value
    ) -> Result<Capsule> {
        // program: "$compiled" | {"program":[...]} | [steps...]
        let prog_v = resolve(env, input.get("program").ok_or_else(|| anyhow!("program required"))?)?;
        let mut steps = if let Some(a) = prog_v.get("program").and_then(|v| v.as_array()) {
            a.clone()
        } else if let Some(a) = prog_v.as_array() {
            a.clone()
        } else {
            return Err(anyhow!("plan.patch: program must be {{\"program\":[...]}} or an array of steps"));
        };

        let sets = input.get("set").and_then(|v| v.as_array()).ok_or_else(|| anyhow!("set[] required"))?;

        for patch in sets {
            let target_id = patch.get("id").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("patch.id required"))?;
            let new_in     = patch.get("in").cloned();
            let new_params = patch.get("params").cloned();

            for node in &mut steps {
                // d'abord lecture immuable
                let is_target = node
                    .get("id").and_then(|v| v.as_str())
                    .map(|s| s == target_id)
                    .unwrap_or(false);
                if !is_target { continue; }

                // puis mutation
                if let Some(obj) = node.as_object_mut() {
                    if let Some(ni) = new_in.clone() {
                        obj.insert("in".to_string(), ni);
                    }
                    if let Some(np) = new_params.clone() {
                        obj.insert("params".to_string(), np);
                    }
                } else {
                    return Err(anyhow!("plan.patch: step is not an object"));
                }
            }
        }

        Ok(Capsule { typ: "Program".into(), val: json!({ "program": steps }) })
    }
}




