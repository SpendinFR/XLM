use xlm_runtime::{Vm, Program, Node, run_with_env};
use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand};
use std::fs;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};

#[derive(Parser)]
#[command(name="xlm", version, about="XLM-Prime CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

// === remplace ton enum Cmd par ceci ===
#[derive(Subcommand)]
enum Cmd {
    Run { file: String },
    Demo { name: String },
    RagIndex { folder: String },
    RagQuery { q: String, k: Option<usize> },

    // NEW: autogrow complet (induction -> learn/fuse/save -> cues -> edges optionnels)
    Autogrow {
        /// Dossier du corpus (fichiers .txt)
        folder: Option<String>,
        /// Langue (défaut: "fr")
        lang: Option<String>,
        /// Seuil de fréquence pour promouvoir les cues
        #[arg(long, default_value_t = 15.0)]
        freq_min: f64,
        /// Induire au plus N nouveaux concepts depuis le corpus (0 = désactivé)
        #[arg(long, default_value_t = 12)]
        induce_top: usize,
        /// Miner des arêtes causales GO-lite (lent si gros corpus)
        #[arg(long, default_value_t = false)]
        edges: bool,
        /// Activer un sous-graphe de démo après minage (utilise reason.graph.activate si présent)
        #[arg(long, default_value_t = false)]
        activate: bool,
        /// Composer une trame textuelle simple depuis les propositions activées (compose.generic)
        #[arg(long, default_value_t = false)]
        compose: bool,
    },
}



// === remplace ton fn main() par ceci (le reste inchangé) ===
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run{ file } => run_file(&file),
        Cmd::Demo{ name } => run_demo(&name),
        Cmd::RagIndex{ folder } => rag_index(&folder),
        Cmd::RagQuery{ q, k } => rag_query(&q, k.unwrap_or(3)),

        // NEW
        Cmd::Autogrow { folder, lang, freq_min, induce_top, edges, activate, compose } => {
            let folder = folder.unwrap_or_else(|| "./corpus".to_string());
            let lang = lang.unwrap_or_else(|| "fr".to_string());
            autogrow(&folder, &lang, freq_min, induce_top, edges, activate, compose)
        }
    }
}


fn run_file(file: &str) -> Result<()> {
    let s = fs::read_to_string(file)?;
    let prog: Program = serde_json::from_str(&s)?;
    let vm = Vm::new();
    let outs = vm.run(&prog)?;
    for (i, c) in outs.iter().enumerate() {
        println!("#{} {} {}", i, c.typ, c.val);
    }
    Ok(())
}


fn rag_index(folder: &str) -> Result<()> {
    use std::fs;
    use xlm_rag::{Doc, index};
    let mut docs = vec![];
    for entry in fs::read_dir(folder)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("md") {
            let text = fs::read_to_string(&path)?;
            let title = path.file_stem().unwrap().to_string_lossy().to_string();
            docs.push(Doc{ uri: path.to_string_lossy().to_string(), title, text });
        }
    }
    index(docs)?;
    println!("Indexed folder {}", folder);
    Ok(())
}

fn rag_query(q: &str, k: usize) -> Result<()> {
    let docs = xlm_rag::query(q, k)?;
    for d in docs {
        println!("- {} ({})", d.title, d.uri);
    }
    Ok(())
}

// NEW: liste les IDs (et tente de lire l'id depuis le JSON si possible)
fn collect_concept_ids(dir: &str) -> Result<Vec<String>> {
    use std::path::Path;
    let mut ids = Vec::<String>::new();
    if !Path::new(dir).exists() { return Ok(ids); }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") { continue; }

        if let Ok(s) = fs::read_to_string(&path) {
            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                if let Some(id) = v.get("id").and_then(|x| x.as_str()) {
                    ids.push(id.to_string());
                    continue;
                }
            }
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            let id = stem.split('.').next().unwrap_or(stem);
            ids.push(id.to_string());
        }
    }
    ids.sort();
    ids.dedup();
    Ok(ids)
}

fn collect_known_lemmas(dir: &str) -> Result<std::collections::BTreeSet<String>> {
    let mut out = std::collections::BTreeSet::new();
    for entry in fs::read_dir(dir).unwrap_or_else(|_| fs::read_dir(".").unwrap()) {
        if let Ok(ent) = entry {
            let path = ent.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") { continue; }
            if let Ok(s) = fs::read_to_string(&path) {
                if let Ok(v) = serde_json::from_str::<Value>(&s) {
                    if let Some(id) = v.get("id").and_then(|x| x.as_str()) {
                        out.insert(id.to_lowercase());
                    }
                    if let Some(lem) = v.pointer("/lexemes/fr/verb_lemma").and_then(|x| x.as_str()) {
                        out.insert(lem.to_lowercase());
                    }
                }
            }
        }
    }
    Ok(out)
}

fn extract_candidate_lemmas_fr(line: &str) -> Vec<String> {
    let mut v = Vec::new();
    for raw in line.split(|c: char| !c.is_alphabetic() && c != '-' && c != '\'') {
        let t = raw.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
        if t.len() < 4 { continue; }
        if t.ends_with("er") || t.ends_with("ir") || t.ends_with("re") || t.ends_with("oir") {
            match t.as_str() {
                "etre" | "être" | "avoir" | "faire" | "aller" | "dire" | "pouvoir" |
                "devoir" | "falloir" | "voir" | "savoir" | "venir" => {},
                _ => v.push(t),
            }
        }
    }
    v
}
fn today_iso() -> String {
    chrono::Utc::now().naive_utc().date().format("%Y-%m-%d").to_string()
}

fn concept_load_by_id(id: &str, lang: &str) -> Result<Value> {
    let prog = Program { program: vec![
        Node{ id: Some("c".into()), op:"concept.load".into(),
              r#in: json!({"id": id, "lang": lang}), out: json!({}), params: json!({}) },
    ]};
    let (_outs, env) = run_with_env(&Vm::new(), &prog)?;
    env.get("c").map(|c| c.val.clone()).ok_or_else(|| anyhow!("concept.load failed for {id}"))
}

fn tokens_for_concept(c:&Value, lang:&str) -> Vec<String> {
    let mut t = Vec::<String>::new();
    if let Some(lemma) = c.pointer(&format!("/lexemes/{}/verb_lemma", lang)).and_then(|v| v.as_str()) {
        t.push(lemma.to_lowercase());
    }
    if let Some(arr) = c.pointer(&format!("/lexemes/{}/syn", lang)).and_then(|v| v.as_array()) {
        for s in arr.iter().filter_map(|x| x.as_str()) {
            t.push(s.to_lowercase());
        }
    }
    t.sort(); t.dedup(); t
}

fn sample_lines_for_tokens(folder:&str, toks:&[String], cap:usize) -> Result<Vec<String>> {
    if toks.is_empty() || cap==0 { return Ok(vec![]); }
    let mut out = Vec::<String>::new();
    for entry in fs::read_dir(folder)? {
        if out.len() >= cap { break; }
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") { continue; }
        let content = fs::read_to_string(&path)?;
        for line in content.lines().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let low = line.to_lowercase();
            if toks.iter().any(|t| low.contains(t)) {
                out.push(line.to_string());
                if out.len() >= cap { break; }
            }
        }
    }
    Ok(out)
}

fn fuse_defaults_and_save(base:&Value, defaults_delta: Vec<Value>, lang:&str) -> Result<()> {
    if defaults_delta.is_empty() { return Ok(()); }
    let delta = json!({ "id": base.get("id").and_then(|v| v.as_str()).unwrap_or("UNKNOWN"),
                        "defaults": defaults_delta });
    let prog = Program { program: vec![
        Node{ id: Some("f".into()), op:"cdl.fuse".into(),
              r#in: json!({ "base": base, "delta": delta }),
              out: json!({}), params: json!({}) },
    ]};
    let (_o, env) = run_with_env(&Vm::new(), &prog)?;
    let fused = env.get("f").ok_or_else(|| anyhow!("fuse failed"))?.val.clone();

    // Sauvegarde atomique en passant l’objet (pas "$f")
    let _ = Vm::new().run(&Program{ program: vec![
        Node{ id: Some("s".into()), op:"concept.save".into(),
              r#in: json!({"concept": fused, "lang": lang}),
              out: json!({}), params: json!({}) },
    ]});
    Ok(())
}

// --------- cœur: auto-curation loop ----------
fn auto_curate_loop(folder:&str, lang:&str,
                    max_instances_per_concept: usize,
                    max_abduce_per_instance: usize,
                    enable_counterfactual: bool) -> Result<usize>
{
    let today_s = today_iso();
    let ids = collect_concept_ids("./concepts")?;
    let mut total_promoted = 0usize;

    for cid in ids {
        let concept = concept_load_by_id(&cid, lang)?;
        let toks = tokens_for_concept(&concept, lang);
        if toks.is_empty() { continue; }

        let lines = sample_lines_for_tokens(folder, &toks, max_instances_per_concept)?;
        if lines.is_empty() { continue; }

        let mut defaults_delta: Vec<Value> = Vec::new();

        for line in lines {
            // 1) bind_text (allow_partial=true)
            let prog_bind = Program { program: vec![
                Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
                      r#in: json!({"concept": concept, "text": line, "lang": lang, "allow_partial": true}),
                      out: json!({}), params: json!({}) },
            ]};
            let Ok((_o, env_b)) = run_with_env(&Vm::new(), &prog_bind) else { continue; };
            let Some(inst) = env_b.get("inst") else { continue; };
            let inst_v = inst.val.clone();

            // 2) abduce → suggestions(add/remove/then)
            let prog_abd = Program { program: vec![
                Node{ id: Some("abd".into()), op:"reason.abduce".into(),
                      r#in: json!({ "concept": concept, "instance": inst_v }),
                      out: json!({}), params: json!({}) },
            ]};
            let Ok((_o, env_a)) = run_with_env(&Vm::new(), &prog_abd) else { continue; };
            let Some(abd) = env_a.get("abd") else { continue; };

            let suggs = abd.val.get("suggestions").and_then(|v| v.as_array()).cloned().unwrap_or_default();
            for (k, s) in suggs.into_iter().enumerate() {
                if k >= max_abduce_per_instance { break; }
                let then_v = s.get("then").cloned().unwrap_or(json!(null));

                let mut d = json!({
                    "if": concept.get("id").and_then(|v| v.as_str()).unwrap_or(&cid),
                    "then": then_v,
                    "meta": { "support": 1.0, "last_seen": today_s }
                });
                if let Some(add) = s.get("add").and_then(|v| v.as_array()) {
                    let a: Vec<String> = add.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
                    if !a.is_empty() { d["when_has"] = json!(a); }
                }
                if let Some(rem) = s.get("remove").and_then(|v| v.as_array()) {
                    let r: Vec<String> = rem.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
                    if !r.is_empty() { d["unless_has"] = json!(r); }
                }
                defaults_delta.push(d);
                total_promoted += 1;
            }

            // 3) (optionnel) counterfactual pour densifier “THEN” si on ajoute des rôles manquants
            if enable_counterfactual {
                // on construit un "set" minimal sur les rôles à ajouter (placeholder neutre)
                let mut set_obj = serde_json::Map::new();
                if let Some(sugg0) = abd.val.get("suggestions").and_then(|v| v.as_array()).and_then(|a| a.first()) {
                    if let Some(add) = sugg0.get("add").and_then(|v| v.as_array()) {
                        for role in add.iter().filter_map(|x| x.as_str()) {
                            set_obj.insert(role.to_string(), json!("[AUTO]"));
                        }
                    }
                }
                if !set_obj.is_empty() {
                    let prog_cf = Program { program: vec![
                        Node{ id: Some("cf".into()), op:"reason.counterfactual".into(),
                              r#in: json!({ "concept": concept, "instance": inst_v, "set": Value::Object(set_obj.clone()) }),
                              out: json!({}), params: json!({}) },
                    ]};
                    if let Ok((_o, env_cf)) = run_with_env(&Vm::new(), &prog_cf) {
                        if let Some(cf) = env_cf.get("cf") {
                            if let Some(added) = cf.val.get("added").and_then(|v| v.as_array()) {
                                for v in added {
                                    // on promeut chaque proposition ‘added’ en default léger
                                    let then_str = if let Some(s) = v.as_str() { json!(s) } else { v.clone() };
                                    let mut d = json!({
                                        "if": concept.get("id").and_then(|x| x.as_str()).unwrap_or(&cid),
                                        "then": then_str,
                                        "meta": { "support": 1.0, "last_seen": today_s }
                                    });
                                    if !set_obj.is_empty() {
                                        d["when_has"] = json!( set_obj.keys().cloned().collect::<Vec<_>>() );
                                    }
                                    defaults_delta.push(d);
                                    total_promoted += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4) fusion + sauvegarde (dédup par "then" déjà géré dans cdl.fuse::union_defaults)
        fuse_defaults_and_save(&concept, defaults_delta, lang)?;
    }

    Ok(total_promoted)
}
fn discover_and_induce(folder:&str, lang:&str, top_n: usize) -> anyhow::Result<Vec<String>> {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;

    if top_n == 0 { return Ok(vec![]); }

    let known = collect_known_lemmas("./concepts").unwrap_or_default();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut first_sentence: BTreeMap<String, String> = BTreeMap::new();

    for entry in fs::read_dir(folder)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") { continue; }
        let content = fs::read_to_string(&path)?;
        for line in content.lines().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            for lem in extract_candidate_lemmas_fr(line) {
                if known.contains(&lem) { continue; }
                *counts.entry(lem.clone()).or_insert(0) += 1;
                first_sentence.entry(lem).or_insert(line.to_string());
            }
        }
    }

    let mut items: Vec<(String,usize)> = counts.into_iter().collect();
    items.sort_by(|a,b| b.1.cmp(&a.1));
    items.truncate(top_n);

    let mut created = Vec::new();
    for (lem, _cnt) in items {
        let id = lem.to_uppercase().replace(' ', "_");
        let example = first_sentence.get(&lem).cloned().unwrap_or_else(|| format!("{}.", lem));
        println!("==> induce {}", id);

        let prog = Program { program: vec![
            Node{ id: Some("c".into()), op:"cdl.induce".into(),
                  r#in: serde_json::json!({"id": id, "lemma": lem, "sentence": example, "lang": lang}),
                  out: serde_json::json!({}), params: serde_json::json!({}) },
            Node{ id: Some("save".into()), op:"concept.save".into(),
                  r#in: serde_json::json!({"concept":"$c","lang":lang}), out: serde_json::json!({}), params: serde_json::json!({}) },
        ]};
        let outs = Vm::new().run(&prog)?;
        let mut ok = false;
        for c in outs {
            if c.typ == "Saved" { ok = true; }
        }
        if ok {
            created.push(id.clone());

            // NEW: lexicon.update best-effort
            let _ = Vm::new().run(&Program{ program: vec![
                Node{ id: Some("lx".into()), op:"lexicon.update".into(),
                      r#in: serde_json::json!({"lang": lang, "lemma": lem, "forms":[lem], "syn":[], "count": 1.0}),
                      out: serde_json::json!({}), params: serde_json::json!({}) },
            ]});

            // NEW: bootstrap analogique léger (TRANSFER-like) si verbe "long-tail"
            if id.contains("ISER") || id.contains("IFIER") {
                let delta = serde_json::json!({
                    "id": id,
                    "roles": ["Agent","Receiver","Theme"],
                    "lexemes": { "fr": { "frames": ["Agent V Theme à Receiver"] } }
                });
                let prog_f = Program { program: vec![
                    Node{ id: Some("base".into()), op:"concept.load".into(),
                          r#in: serde_json::json!({"id":"TRANSFER","lang":lang}), out: serde_json::json!({}), params: serde_json::json!({}) },
                    Node{ id: Some("f".into()), op:"cdl.fuse".into(),
                          r#in: serde_json::json!({"base":"$base","delta": delta}), out: serde_json::json!({}), params: serde_json::json!({}) },
                    Node{ id: Some("s".into()), op:"concept.save".into(),
                          r#in: serde_json::json!({"concept":"$f","lang":lang}), out: serde_json::json!({}), params: serde_json::json!({}) },
                ]};
                let _ = Vm::new().run(&prog_f);
            }
        }
    }
    Ok(created)
}


fn mine_edges_go_lite(folder:&str, lang:&str) -> anyhow::Result<usize> {
    use std::collections::HashMap;

    let mut lex2cid: HashMap<&'static str, &'static str> = HashMap::new();
    lex2cid.insert("vendre","SELL"); lex2cid.insert("payer","PAY");
    lex2cid.insert("remercier","THANK"); lex2cid.insert("prêter","LEND");
    lex2cid.insert("preter","LEND"); lex2cid.insert("rendre","RETURN");

    #[derive(Clone,Copy)] struct P<'a>{ a:&'a str, b:&'a str, typ:&'a str, guard:&'a str }
    let pairs = [
        P{a:"SELL", b:"PAY",    typ:"CAUSES",  guard:"NONE"},
        P{a:"PAY",  b:"THANK",  typ:"ENABLES", guard:"NONE"},
        P{a:"LEND", b:"RETURN", typ:"CAUSES",  guard:"IF"},
    ];

    let mut lines_all: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(folder)? {
        let p = entry?.path();
        if p.extension().and_then(|s| s.to_str()) != Some("txt") { continue; }
        let s = std::fs::read_to_string(&p)?;
        for l in s.lines().map(|t| t.trim()).filter(|t| !t.is_empty()) {
            lines_all.push(l.to_string());
        }
    }

    // fenêtre {-1,0,+1} avec pénalité distance
    let mut edges: HashMap<(String,String,String,String), f64> = HashMap::new();
    for i in 0..lines_all.len() {
        let l0 = lines_all[i].to_lowercase();
        let mut set0: Vec<&str> = Vec::new();
        for (lem,cid) in lex2cid.iter() { if l0.contains(lem) { set0.push(cid); } }
        if set0.is_empty() { continue; }

        for off in [-1isize, 1isize] {
            let j = i as isize + off;
            if j < 0 || j >= lines_all.len() as isize { continue; }
            let lj = lines_all[j as usize].to_lowercase();
            let mut setj: Vec<&str> = Vec::new();
            for (lem,cid) in lex2cid.iter() { if lj.contains(lem) { setj.push(cid); } }
            if setj.is_empty() { continue; }

            // distance 1 ⇒ w=1.0 (ici off ∈ {-1,+1})
            let w = 1.0;

            for P{a:pa,b:pb,typ,guard} in pairs {
                if set0.iter().any(|&x| x==pa) && setj.iter().any(|&y| y==pb) {
                    *edges.entry((pa.to_string(),pb.to_string(),typ.to_string(),guard.to_string()))
                          .or_insert(0.0) += w;
                }
            }
        }
    }

    // persist via kernel
    let mut promoted = 0usize;
    for ((from,to,typ,guard), score) in edges {
        let cnt = score.max(0.1); // >=0.1
        let prog = Program { program: vec![
            Node{ id: Some("e".into()), op:"graph.edges.update".into(),
                  r#in: json!({
                      "from": from, "to": to, "type": typ, "guard": guard,
                      "delta_support": cnt, "lang": lang
                  }),
                  out: json!({}), params: json!({}) },
        ]};
        for c in Vm::new().run(&prog)? {
            if c.typ == "EdgeUpdated" { promoted += 1; }
        }
    }
    Ok(promoted)
}


fn autogrow(folder: &str, lang: &str, freq_min: f64,
            induce_top: usize, do_edges: bool, do_activate: bool, do_compose: bool) -> anyhow::Result<()> {

    // 0) (optionnel) induction de concepts
    if induce_top > 0 {
        let created = discover_and_induce(folder, lang, induce_top)?;
        if !created.is_empty() {
            println!("+ concepts induits: {:?}", created);
        } else {
            println!("(induction) aucun nouveau concept retenu");
        }
    }

    // 1) learn+fuse+save pour tous les concepts
    let concept_dir = "./concepts";
    let ids = collect_concept_ids(concept_dir)?;
    if ids.is_empty() {
        eprintln!("(autogrow) Aucun concept trouvé dans {}.", concept_dir);
    }

    // Active learning: top fichiers "novateurs"
    let known = collect_known_lemmas("./concepts").unwrap_or_default();
    let top_files = rank_corpus_by_novelty(folder, &known, 128).unwrap_or_default();
    if !top_files.is_empty() {
        println!("(active) sélection de {} fichiers à haute nouveauté", top_files.len());
    }

    for cid in &ids {
        println!("==> learn/fuse/save {}", cid);
        let in_obj = if top_files.is_empty() {
            serde_json::json!({
                "concept": { "id": cid },
                "folder": folder,
                "lang": lang,
                "save": true
            })
        } else {
            serde_json::json!({
                "concept": { "id": cid },
                "folder": folder,
                "lang": lang,
                "save": true,
                "files": top_files     // NEW: on passe la sélection
            })
        };

        let prog = Program { program: vec![
            Node{ id: Some("learn".into()), op:"cdl.learn_from_corpus".into(),
                  r#in: in_obj, out: serde_json::json!({}), params: serde_json::json!({}) },
        ]};
        let outs = Vm::new().run(&prog)?;
        for c in outs {
            if c.typ == "LearnReport" {
                let a_f = c.val.get("added_frames").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                let a_s = c.val.get("added_syn").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                println!("   +frames: {}, +syn: {}", a_f, a_s);
            }
        }
    }

    // 2) cues globales
    println!("==> cues.learn / cues.fuse");
    let prog_cues = Program { program: vec![
        Node{ id: Some("cues1".into()), op:"cdl.cues.learn".into(),
              r#in: serde_json::json!({ "lang": lang, "folder": folder }),
              out: serde_json::json!({}), params: serde_json::json!({}) },
        Node{ id: Some("cues2".into()), op:"cdl.cues.fuse".into(),
              r#in: serde_json::json!({ "lang": lang, "freq_min": freq_min }),
              out: serde_json::json!({}), params: serde_json::json!({}) },
    ]};
    let _ = Vm::new().run(&prog_cues)?;

    // 2.5) ENTITIES + FACTS (global)
    println!("==> nlp.entities.learn / nlp.facts.learn");
    let prog_nf = Program { program: vec![
        Node{ id: Some("ent".into()), op:"nlp.entities.learn".into(),
              r#in: json!({ "lang": lang, "folder": folder }), out: json!({}), params: json!({}) },
        Node{ id: Some("fac".into()), op:"nlp.facts.learn".into(),
              r#in: json!({ "lang": lang, "folder": folder }), out: json!({}), params: json!({}) },
    ]};
    let _ = Vm::new().run(&prog_nf)?;

    // 3.5) DEFAULTS depuis edges (auto-promotion)
    println!("==> cdl.defaults.suggest");
    let prog_def = Program { program: vec![
        Node{ id: Some("dflt".into()), op:"cdl.defaults.suggest".into(),
              r#in: json!({ "lang": lang, "min_support": 4.0, "min_weight": 0.8, "max_promote": 300 }),
              out: json!({}), params: json!({}) },
    ]};
    let _ = Vm::new().run(&prog_def)?;

    // 3) edges (GO-lite)
    if do_edges {
        println!("==> mining edges (GO-lite)");
        let n = mine_edges_go_lite(folder, lang)?;
        println!("   edges updated: {}", n);
    }
    // 5) Auto-curation (abduce + counterfactual) — densifie les defaults sans gros volume
    println!("==> auto-curate (abduce/counterfactual)");
    let promoted = auto_curate_loop(
        folder,
        lang,
        /* max_instances_per_concept */ 48,
        /* max_abduce_per_instance   */ 2,
        /* enable_counterfactual     */ true
    )?;
    println!("   promoted defaults: {}", promoted);

    // 4) (démo) activation + compose
    if do_activate {
        println!("==> reason.graph.activate (démo)");
        let prog = Program { program: vec![
            Node{ id: Some("sell".into()), op:"concept.load".into(),
                  r#in: serde_json::json!({"id":"SELL","lang":lang}), out: serde_json::json!({}), params: serde_json::json!({}) },
            Node{ id: Some("i".into()), op:"concept.bind_text".into(),
                  r#in: serde_json::json!({"concept":"$sell","text":"Paul vend un livre à Marie pour 20 euros","lang":lang,"allow_partial":true}),
                  out: serde_json::json!({}), params: serde_json::json!({}) },
            Node{ id: Some("g".into()), op:"reason.graph.activate".into(),
                  r#in: serde_json::json!({"instances":["$i"], "k": 2, "budget": 64 }),
                  out: serde_json::json!({}), params: serde_json::json!({}) },
        ]};
        let outs = Vm::new().run(&prog)?;
        for c in &outs {
            if c.typ == "Plan" { println!("   activate -> {}", c.val); }
        }
        if do_compose {
            println!("==> compose.generic (démo)");
            let prog2 = Program { program: vec![
                Node{ id: Some("plan".into()), op:"compose.generic".into(),
                      r#in: serde_json::json!({"instances":["$i"]}), out: serde_json::json!({}), params: serde_json::json!({}) },
            ]};
            let outs2 = Vm::new().run(&prog2)?;
            for c in outs2 {
                if c.typ == "Plan" { println!("   compose -> {}", c.val); }
            }
        }
    }

    println!("✓ autogrow terminé.");
    Ok(())
}

// très simple: score = (#lemmes inconnus) + 0.5*(#lemmes rares)
fn score_file_novelty(path:&std::path::Path, known:&BTreeSet<String>) -> anyhow::Result<f64> {
    let s = std::fs::read_to_string(path)?;
    let mut unknown = 0usize;
    let mut rare = 0usize;
    for line in s.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
        for lem in extract_candidate_lemmas_fr(line) {
            if !known.contains(&lem) { unknown += 1; }
            // heuristique "rare" : longueur>6 ou terminaison -iser/-ifier
            if lem.len() >= 7 || lem.ends_with("iser") || lem.ends_with("ifier") { rare += 1; }
        }
    }
    Ok(unknown as f64 + 0.5*(rare as f64))
}

fn rank_corpus_by_novelty(folder:&str, known_lemmas:&std::collections::BTreeSet<String>, cap: usize)
    -> anyhow::Result<Vec<String>>
{
    use regex::Regex;
    use std::fs;
    use std::path::Path;

    // Charge signals connus
    let mut known_signals = std::collections::BTreeSet::new();

    // concepts: frames & syn
    for entry in fs::read_dir("./concepts").unwrap_or_else(|_| fs::read_dir(".").unwrap()) {
        if let Ok(ent) = entry {
            let path = ent.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") { continue; }
            if let Ok(s) = fs::read_to_string(&path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    if let Some(fr) = v.pointer("/lexemes/fr/frames").and_then(|x| x.as_array()) {
                        for f in fr { if let Some(t)=f.as_str() { known_signals.insert(format!("frame:{}", t)); } }
                    }
                    if let Some(sy) = v.pointer("/lexemes/fr/syn").and_then(|x| x.as_array()) {
                        for t in sy { if let Some(tk)=t.as_str() { known_signals.insert(format!("syn:{}", tk)); } }
                    }
                }
            }
        }
    }
    // cues
    for p in ["./stores/cues.fr.json", "./cues.fr.json"] {
        if Path::new(p).exists() {
            if let Ok(s) = fs::read_to_string(p) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    if let Some(arr) = v.get("cues").and_then(|x| x.as_array()).or_else(|| v.as_array()) {
                        for c in arr {
                            if let Some(txt) = c.get("text").and_then(|x| x.as_str()).or_else(|| c.as_str()) {
                                known_signals.insert(format!("cue:{}", txt.to_lowercase()));
                            }
                        }
                    }
                }
            }
        }
    }
    // edges
    for p in ["./stores/graph.edges.json", "./graph.edges.json"] {
        if Path::new(p).exists() {
            if let Ok(s) = fs::read_to_string(p) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    let arr = v.get("edges").and_then(|x| x.as_array()).cloned()
                               .or_else(|| v.as_array().cloned()).unwrap_or_default();
                    for e in arr {
                        let a = e.get("from").and_then(|x| x.as_str()).unwrap_or("");
                        let b = e.get("to").and_then(|x| x.as_str()).unwrap_or("");
                        if !a.is_empty() && !b.is_empty() {
                            known_signals.insert(format!("edge:{}->{}", a,b));
                        }
                    }
                }
            }
        }
    }
    // entités/faits (juste pour petite pénalité si ultra connus)
    let re_amount = Regex::new(r"(?i)\b(\d{1,3}(?:[ \u00A0.,]\d{3})*|\d+)(?:[.,]\d+)?\s?(€|eur|euros|\$|usd)\b").unwrap();
    let re_date_iso = Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap();

    // Score nouveauté
    let mut scored: Vec<(f64, String)> = Vec::new();
    for entry in fs::read_dir(folder)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") { continue; }
        let content = fs::read_to_string(&path).unwrap_or_default();

        // Types explicites
        let mut score: f64 = 0.0_f64;
        let mut penalties: f64 = 0.0_f64;

        for line in content.lines().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let low = line.to_lowercase();

            // Bonus pour nouveaux lemmes verbaux
            for tok in low.split(|c:char| !c.is_alphabetic()) {
                if tok.len() >= 4 && (tok.ends_with("er") || tok.ends_with("ir") || tok.ends_with("re") || tok.ends_with("oir")) {
                    if !known_lemmas.contains(tok) {
                        score += 1.5_f64;
                    }
                }
            }

            // Bigrammes approximativement “nouveaux”
            let words: Vec<&str> = low.split_whitespace().collect();
            for w in words.windows(2) {
                let bi = format!("{} {}", w[0], w[1]);
                if !known_signals.contains(&format!("syn:{}", bi)) && !known_signals.contains(&format!("cue:{}", bi)) {
                    score += 0.05_f64;
                }
            }

            // Pénalités (dates/montants très fréquents)
            let n_amt = re_amount.find_iter(line).count() as f64;
            let n_dt  = re_date_iso.find_iter(line).count() as f64;
            penalties += 0.02_f64 * (n_amt + n_dt);
        }

        let s_final: f64 = f64::max(score - penalties, 0.0_f64);
        scored.push((s_final, path.to_string_lossy().to_string()));
    }

    scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(cap);
    Ok(scored.into_iter().map(|(_,p)| p).collect())
}

fn run_demo(name: &str) -> Result<()> {
    match name {
        "doc_json_tva"   => demo_doc_json_tva(),
        "plan_invoice"   => demo_plan_invoice(),
        "invoice_json"   => demo_invoice_json(),   // <-- nouveau
        "rules_route"   => demo_rules_route(),
        "eval_basic"    => demo_eval_basic(),
        "autoplan_invoice" => demo_autoplan_invoice(),
        "qwen_plan_raw" => demo_qwen_plan_raw(),
        "concept_eat" => demo_concept_eat_fr(),
        "concept_sell" => demo_concept_sell_fr(),
        "concept_abs_sell" => demo_concept_abstract_sell(),
        "concept_text_sell" => demo_concept_bind_text_sell(),
        "concept_transfer" => demo_transfer_give_lend_sell(),
        "concept_classify" => demo_transfer_classify(),
        "why_inherit" => demo_why_inherit(),
        "abduce_lend" => demo_abduce_lend(),
        "counterfactuals" => demo_counterfactuals(),
        "plan_lend" => demo_plan_lend(),
        "plan_sell" => demo_plan_sell_from_give(),
        "scenario_lend_return" => Ok(demo_scenario_lend_return()?),
        "scenario_sell_pay" => Ok(demo_scenario_sell_pay()?),
        "induce_borrow" => Ok(demo_induce_borrow_then_return()?),
        "scenario_sell_pay_thank" => Ok(demo_scenario_sell_pay_thank()?),
        "story_chain" => demo_story_chain(),
        "story_chain_auto" => demo_story_chain_auto(),
        "story_defaults" => demo_story_defaults(),
        "story_defaults_except_return" => demo_story_defaults_except_return(),
        "cdl_fuse_sell" => demo_cdl_fuse_sell(),
        "analogy_eat_devour" => demo_analogy_eat_devour(),
        "why_return_from_lend" => demo_why_return_from_lend(),
        "story_cycle" => demo_story_cycle(),
        "story_cycle_with_exception" => demo_story_cycle_with_exception(),
        "specialize_eat_poisoned" => demo_specialize_eat_poisoned(),


        _ => Err(anyhow!("unknown demo: {}", name)),
    }
}


fn demo_doc_json_tva() -> Result<()> {
    // lines: 15.00 + 35.00 = 50.00 EUR; VAT 20% => 10.00; total = 60.00
    let lines = vec![
        json!({"amount_cents": 1500, "currency":"EUR"}),
        json!({"amount_cents": 3500, "currency":"EUR"}),
    ];
    let expected_total = json!({"amount_cents": 6000, "currency":"EUR"});

    let prog = Program { program: vec![
        // calculs exacts
        Node{ id: Some("subtotal".into()), op: "compute.sum_money".into(), r#in: json!({"table": lines, "currency":"EUR"}), out: json!({}), params: json!({})},
        Node{ id: Some("vat".into()),      op: "compute.vat".into(),       r#in: json!({"money":"$subtotal","rate":0.20}), out: json!({}), params: json!({})},
        Node{ id: Some("total".into()),    op: "compute.add_money".into(), r#in: json!({"items": ["$subtotal","$vat"], "currency":"EUR"}), out: json!({}), params: json!({})},

        // preuves (fail-closed) : total == attendu
        Node{ id: Some("proof".into()), op: "prove.all".into(), r#in: json!({
            "tests": [ {"op":"equals", "lhs":"$total", "rhs": expected_total} ]
        }), out: json!({}), params: json!({})},

        // RAG → citations obligatoires
        Node{ id: Some("sources".into()), op: "rag.query".into(), r#in: json!({
            "q": "taux normal de TVA en France 20%", "k": 2
        }), out: json!({}), params: json!({})},

        Node{ id: Some("cite".into()), op: "cite.require".into(), r#in: json!({
            "sources": "$sources", "min": 1
        }), out: json!({}), params: json!({})},

        // rendu final
        Node{ id: Some("text".into()), op: "render.text".into(), r#in: json!({
            "subtotal":"$subtotal","vat":"$vat","total":"$total"
        }), out: json!({}), params: json!({"style":"neutral"})},
    ]};

    let vm = Vm::new();
    let outs = vm.run(&prog)?;
    for c in outs {
        if c.typ == "Text" {
            println!("{}", c.val["text"].as_str().unwrap_or(""));
        }
        if c.typ == "Cite" {
            println!("Sources: {}", c.val);
        }
    }
    Ok(())
}
fn demo_plan_invoice() -> Result<()> {
    use serde_json::json;
    // On prépare un sous-plan déterministe (les mêmes étapes que la démo TVA)
    let sub_steps = json!([
        { "id":"subtotal", "op":"compute.sum_money", "in":{"table":[
            {"amount_cents":1500,"currency":"EUR"},
            {"amount_cents":3500,"currency":"EUR"}
        ], "currency":"EUR"} },
        { "id":"vat",      "op":"compute.vat",       "in":{"money":"$subtotal","rate":0.20} },
        { "id":"total",    "op":"compute.add_money", "in":{"items":["$subtotal","$vat"], "currency":"EUR"} }
    ]);

    // Programme maître : compile -> run -> preuve -> cite -> rendu (compact)
    let prog = Program { program: vec![
        Node{ id: Some("compiled".into()), op: "plan.compile".into(),
              r#in: json!({ "steps": sub_steps }), out: json!({}), params: json!({}) },

        Node{ id: Some("trace".into()), op: "plan.run".into(),
              r#in: json!({ "program": "$compiled" }), out: json!({}), params: json!({}) },

        // Calcul direct pour rendre (les valeurs du sous-plan sont dans le trace, ici on recalc proprement)
        Node{ id: Some("subtotal".into()), op: "compute.sum_money".into(),
              r#in: json!({"table":[
                {"amount_cents":1500,"currency":"EUR"},
                {"amount_cents":3500,"currency":"EUR"}],"currency":"EUR"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("vat".into()), op: "compute.vat".into(),
              r#in: json!({"money":"$subtotal","rate":0.20}), out: json!({}), params: json!({}) },

        Node{ id: Some("total".into()), op: "compute.add_money".into(),
              r#in: json!({"items":["$subtotal","$vat"],"currency":"EUR"}), out: json!({}), params: json!({}) },

        Node{ id: Some("proof".into()), op: "prove.all".into(),
              r#in: json!({"tests":[{"op":"equals","lhs":"$total","rhs":{"amount_cents":6000,"currency":"EUR"}}]}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("sources".into()), op: "rag.query".into(),
              r#in: json!({"q":"taux normal de TVA 20% France", "k": 1}), out: json!({}), params: json!({}) },

        Node{ id: Some("cite".into()), op: "cite.require".into(),
              r#in: json!({"sources":"$sources", "min":1}), out: json!({}), params: json!({}) },

        Node{ id: Some("text_full".into()), op: "render.text".into(),
              r#in: json!({"subtotal":"$subtotal","vat":"$vat","total":"$total"}), out: json!({}), params: json!({"style":"neutral"}) },

        Node{ id: Some("text_compact".into()), op: "critic.compact".into(),
              r#in: json!({"text":"$text_full", "max_chars": 80}), out: json!({}), params: json!({}) },
    ]};

    let vm = Vm::new();
    let outs = vm.run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Trace" => println!("Trace: {}", c.val),
            "Cite"  => println!("Sources: {}", c.val),
            "Text"  => println!("Sortie: {}", c.val["text"].as_str().unwrap_or("")),
            _ => {}
        }
    }
    Ok(())
}
fn demo_invoice_json() -> Result<()> {
    use serde_json::json;

    // 1) Sous-plan déterministe
    let sub_steps = json!([
        { "id":"subtotal", "op":"compute.sum_money", "in":{"table":[
            {"amount_cents":1500,"currency":"EUR"},
            {"amount_cents":3500,"currency":"EUR"}
        ], "currency":"EUR"} },
        { "id":"vat",      "op":"compute.vat",       "in":{"money":"$subtotal","rate":0.20} },
        { "id":"total",    "op":"compute.add_money", "in":{"items":["$subtotal","$vat"], "currency":"EUR"} }
    ]);

    // 2) Programme maître : compile -> lint -> run (trace)
    //    + RECALCUL local des montants -> rendu JSON -> garde -> sources -> rendu texte -> compact
    let prog = Program { program: vec![
        Node{ id: Some("compiled".into()), op: "plan.compile".into(),
              r#in: json!({ "steps": sub_steps }), out: json!({}), params: json!({}) },

        Node{ id: Some("lint".into()), op: "plan.lint".into(),
              r#in: json!({ "program": "$compiled" }), out: json!({}), params: json!({}) },

        Node{ id: Some("trace".into()), op: "plan.run".into(),
              r#in: json!({ "program": "$compiled" }), out: json!({}), params: json!({}) },

        // 🔥 Recalcule local des mêmes valeurs (sinon $subtotal/$vat/$total n’existent pas ici)
        Node{ id: Some("subtotal".into()), op: "compute.sum_money".into(),
              r#in: json!({"table":[
                {"amount_cents":1500,"currency":"EUR"},
                {"amount_cents":3500,"currency":"EUR"}
              ], "currency":"EUR"}), out: json!({}), params: json!({}) },

        Node{ id: Some("vat".into()), op: "compute.vat".into(),
              r#in: json!({"money":"$subtotal","rate":0.20}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("total".into()), op: "compute.add_money".into(),
              r#in: json!({"items":["$subtotal","$vat"], "currency":"EUR"}),
              out: json!({}), params: json!({}) },

        // Rendu JSON structuré
        Node{ id: Some("json_doc".into()), op: "render.invoice.json".into(),
              r#in: json!({ "subtotal":"$subtotal","vat":"$vat","total":"$total",
                            "currency":"EUR","issued_at":"2025-11-06" }),
              out: json!({}), params: json!({}) },

        // Garde JSON (fail-closed)
        Node{ id: Some("guarded".into()), op: "critic.guard.json".into(),
              r#in: json!({
                "json":"$json_doc",
                "required_keys":["subtotal_cents","vat_cents","total_cents","currency","issued_at"],
                "numeric_keys":["subtotal_cents","vat_cents","total_cents"],
                "non_negative_keys":["subtotal_cents","vat_cents","total_cents"],
                "exact_keys": true
              }), out: json!({}), params: json!({}) },

        // Sources → citations obligatoires
        Node{ id: Some("sources".into()), op: "rag.query".into(),
              r#in: json!({"q":"taux normal de TVA 20% France", "k": 1}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("cite".into()), op: "cite.require".into(),
              r#in: json!({"sources":"$sources", "min":1}),
              out: json!({}), params: json!({}) },

        // Rendu texte + compact
        Node{ id: Some("text_full".into()), op: "render.text".into(),
              r#in: json!({"subtotal":"$subtotal","vat":"$vat","total":"$total"}),
              out: json!({}), params: json!({"style":"neutral"}) },

        Node{ id: Some("text_compact".into()), op: "critic.compact".into(),
              r#in: json!({"text":"$text_full", "max_chars": 84}), out: json!({}), params: json!({}) },
    ]};

    let vm = Vm::new();
    let outs = vm.run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Lint"  => println!("Lint: {}", c.val),
            "Trace" => println!("Trace: {}", c.val),
            "JSON"  => println!("JSON: {}", c.val),
            "Cite"  => println!("Sources: {}", c.val),
            "Text"  => println!("Sortie: {}", c.val["text"].as_str().unwrap_or("")),
            _ => {}
        }
    }
    Ok(())
}
fn demo_rules_route() -> Result<()> {
    use serde_json::json;

    // 1) Persiste une règle "brevity" (1-shot)
    let rule = json!({"max_chars": 48, "replace":[["Sous-total","ST"],["Total","T"]]});
    let prog_apply = Program { program: vec![
        Node{ id: Some("r1".into()), op:"rules.apply".into(),
              r#in: json!({"name":"brevity","rule": rule}), out: json!({}), params: json!({}) },
    ]};
    Vm::new().run(&prog_apply)?;

    // 2) Sous-plan initial (inclut rag + cite pour montrer le routing)
    let sub_steps = json!([
        { "id":"subtotal", "op":"compute.sum_money", "in":{"table":[
            {"amount_cents":1500,"currency":"EUR"},
            {"amount_cents":3500,"currency":"EUR"}
        ], "currency":"EUR"} },
        { "id":"vat",      "op":"compute.vat",       "in":{"money":"$subtotal","rate":0.20} },
        { "id":"total",    "op":"compute.add_money", "in":{"items":["$subtotal","$vat"], "currency":"EUR"} },
        { "id":"sources",  "op":"rag.query",         "in":{"q":"taux normal de TVA 20% France","k":1} },
        { "id":"cite",     "op":"cite.require",      "in":{"sources":"$sources","min":1} },
        { "id":"text_full","op":"render.text",       "in":{"subtotal":"$subtotal","vat":"$vat","total":"$total"}, "params":{"style":"neutral"} }
    ]);

    // 3) Programme maître unique (même env → $compiled/$routed existent)
    let prog = Program { program: vec![
        // compile + lint initial
        Node{ id: Some("compiled".into()), op:"plan.compile".into(),
              r#in: json!({"steps": sub_steps}), out: json!({}), params: json!({}) },
        Node{ id: Some("lint1".into()),    op:"plan.lint".into(),
              r#in: json!({"program":"$compiled"}), out: json!({}), params: json!({}) },

        // routing (budget serré) + lint routed + run routed
        Node{ id: Some("routed".into()),   op:"tools.route".into(),
              r#in: json!({"program":"$compiled","budget_ms": 80}), out: json!({}), params: json!({}) },
        Node{ id: Some("lint2".into()),    op:"plan.lint".into(),
              r#in: json!({"program":"$routed"}), out: json!({}), params: json!({}) },
        Node{ id: Some("trace".into()),    op:"plan.run".into(),
              r#in: json!({"program":"$routed"}), out: json!({}), params: json!({}) },

        // recalcul local (démonstration) + rendu + application des règles persistées
        Node{ id: Some("subtotal".into()), op:"compute.sum_money".into(),
              r#in: json!({"table":[
                {"amount_cents":1500,"currency":"EUR"},
                {"amount_cents":3500,"currency":"EUR"}
              ], "currency":"EUR"}), out: json!({}), params: json!({}) },
        Node{ id: Some("vat".into()),      op:"compute.vat".into(),
              r#in: json!({"money":"$subtotal","rate":0.20}), out: json!({}), params: json!({}) },
        Node{ id: Some("total".into()),    op:"compute.add_money".into(),
              r#in: json!({"items":["$subtotal","$vat"], "currency":"EUR"}), out: json!({}), params: json!({}) },
        Node{ id: Some("text_full2".into()), op:"render.text".into(),
              r#in: json!({"subtotal":"$subtotal","vat":"$vat","total":"$total"}), out: json!({}), params: json!({"style":"neutral"}) },
        Node{ id: Some("text_rules".into()), op:"critic.apply_rules".into(),
              r#in: json!({"text":"$text_full2","rules":["brevity"]}), out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Lint"    => println!("Lint: {}", c.val),
            "Program" => println!("Program: {}", c.val), // routed/compiled si tu veux inspecter
            "Trace"   => println!("Trace: {}", c.val),
            "Text"    => if let Some(t) = c.val.get("text").and_then(|v| v.as_str()) {
                            println!("Texte (rules): {}", t);
                         },
            _ => {}
        }
    }
    Ok(())
}
fn demo_eval_basic() -> Result<()> {
    use serde_json::{json, Value};

    // Programme test: somme -> TVA -> total ; preuve: total == 6000 cents
    let test_prog = json!({
        "program": [
            { "id":"subtotal", "op":"compute.sum_money",
              "in":{"table":[
                 {"amount_cents":1500,"currency":"EUR"},
                 {"amount_cents":3500,"currency":"EUR"}], "currency":"EUR"} },
            { "id":"vat",      "op":"compute.vat",
              "in":{"money":"$subtotal","rate":0.20} },
            { "id":"total",    "op":"compute.add_money",
              "in":{"items":["$subtotal","$vat"], "currency":"EUR"} }
        ]
    });

    let prog = Program { program: vec![
        // ajoute/écrase le cas "invoice_20pct"
        Node{ id: Some("add".into()), op:"eval.add_case".into(),
              r#in: json!({
                "name":"invoice_20pct",
                "program": test_prog,
                "tests":[ {"op":"equals", "lhs":"$total", "rhs":{"amount_cents":6000,"currency":"EUR"}} ]
              }), out: json!({}), params: json!({}) },

        // lance l'évaluation (filtre "invoice_*")
        Node{ id: Some("report".into()), op:"eval.run".into(),
              r#in: json!({"filter":"invoice_*"}), out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        if c.typ == "EvalReport" {
            println!("Eval: {}", c.val);
        }
    }
    Ok(())
}

fn demo_autoplan_invoice() -> Result<()> {
    use serde_json::{json, Value};

    let instruction = r#"
Calcule une facture simple avec deux lignes (15.00 + 35.00 EUR), applique une TVA de 20%,
vérifie que le total fait 60.00 EUR via prove.all, cite une source expliquant que la TVA normale en France est 20%,
et termine par un rendu lisible (render.text).
"#;

    // ---- Phase 1: autogen -> compile -> lint (on imprime le programme) ----
    let prog1 = Program { program: vec![
        Node{ id: Some("draft".into()), op:"plan.autogen".into(),
            r#in: json!({
                "instruction": instruction,
                "model": "qwen2.5:7b-instruct-q4_K_M",
                "allow": [
                  "compute.sum_money","compute.vat","compute.add_money",
                  "prove.all","rag.query","cite.require","render.text"
                ],
                "budget_ms": 160
            }), out: json!({}), params: json!({}) },

        Node{ id: Some("compiled".into()), op:"plan.compile".into(),
            r#in: json!({"steps": "$draft"}), out: json!({}), params: json!({}) },

        Node{ id: Some("lint1".into()), op:"plan.lint".into(),
            r#in: json!({"program":"$compiled"}), out: json!({}), params: json!({}) },
    ]};

    let outs1 = Vm::new().run(&prog1)?;
    let mut compiled_prog: Option<Value> = None;

    println!("--- AUTOGEN / COMPILE ---");
    for c in &outs1 {
        match c.typ.as_str() {
            "Program" => { println!("Program: {}", c.val); compiled_prog = Some(c.val.clone()); }
            "Lint"    => println!("Lint: {}", c.val),
            _ => {}
        }
    }
    let compiled_prog = compiled_prog.ok_or_else(|| anyhow!("no compiled program produced"))?;

    // ---- Phase 1.5: PATCH -> LINT (on force les champs attendus) ----
    let patched_prog_node = Node{
        id: Some("patched".into()), op:"plan.patch".into(),
        r#in: json!({
            "program": compiled_prog,
            "set": [
                // (optionnel) figer le rate
                { "id":"1", "in": {            // <-- "1" = ton noeud VAT
                    "money":"$0",              // <-- "$0" = subtotal
                    "rate": 0.20
                }},
                // **FIX CRITIQUE** : comparer les centimes, pas l'objet Money
                { "id":"3", "in": {            // <-- "3" = ton noeud prove
                    "tests":[ { "op":"equals", "lhs":"$2.amount_cents", "rhs": 6000 } ]
                }}
            ]
        }),
        out: json!({}), params: json!({})
    };

    let prog_patch = Program { program: vec![
        patched_prog_node,
        Node{ id: Some("lint2".into()), op:"plan.lint".into(),
              r#in: json!({"program":"$patched"}), out: json!({}), params: json!({}) },
    ]};


    let outs_patch = Vm::new().run(&prog_patch)?;
    let mut patched_program: Option<Value> = None;
    println!("--- PATCH / LINT ---");
    for c in &outs_patch {
        match c.typ.as_str() {
            "Program" => { println!("Program: {}", c.val); patched_program = Some(c.val.clone()); }
            "Lint"    => println!("Lint: {}", c.val),
            _ => {}
        }
    }
    let patched_program = patched_program.ok_or_else(|| anyhow!("no patched program produced"))?;

    // ---- Phase 2: ROUTE -> LINT -> RUN ----
    let prog2 = Program { program: vec![
        Node{ id: Some("routed".into()), op:"tools.route".into(),
            r#in: json!({ "program": patched_program, "budget_ms": 160 }), out: json!({}), params: json!({}) },
        Node{ id: Some("lint3".into()), op:"plan.lint".into(),
            r#in: json!({"program":"$routed"}), out: json!({}), params: json!({}) },
        Node{ id: Some("trace".into()), op:"plan.run".into(),
            r#in: json!({"program":"$routed"}), out: json!({}), params: json!({}) },
    ]};

    let outs2 = Vm::new().run(&prog2)?;
    println!("--- ROUTE / RUN ---");
    for c in outs2 {
        match c.typ.as_str() {
            "Lint"  => println!("Lint: {}", c.val),
            "Trace" => {
                println!("Trace: {}", c.val);
                if let Some(arr) = c.val.get("outputs").and_then(|v| v.as_array()) {
                    for it in arr {
                        if it.get("typ").and_then(|v| v.as_str()) == Some("Text") {
                            if let Some(t) = it.get("val").and_then(|v| v.get("text")).and_then(|v| v.as_str()) {
                                println!("→ {}", t);
                            }
                        }
                    }
                }
            }
            "Program" => println!("Program: {}", c.val),
            _ => {}
        }
    }
    Ok(())
}


fn demo_qwen_plan_raw() -> Result<()> {
    use serde_json::json;

    let qwen_plan = json!({
      "program": [
        {"id": "0", "op": "compute.sum_money", "in": {"table": [
          {"amount_cents": 1500, "currency": "EUR"},
          {"amount_cents": 3500, "currency": "EUR"}
        ], "currency": "EUR"}, "params": {}},
        {"id": "1", "op": "compute.vat", "in": {"money": "$0", "rate": 0.20}, "params": {}},
        {"id": "2", "op": "compute.add_money", "in": {"items": ["$0", "$1"], "currency": "EUR"}, "params": {}},
        {"id": "3", "op": "prove.all", "in": {"tests": [
          {"op": "equals", "lhs": "$2.amount_cents", "rhs": 6000}
        ]}, "params": {}},
        {"id": "4", "op": "rag.query", "in": {"q": "What is the standard VAT rate in France?", "k": 1}, "params": {}},
        {"id": "5", "op": "cite.require", "in": {"sources": "$4.sources", "min": 1}, "params": {}},
        {"id": "6", "op": "render.text", "in": {
          "subtotal": "$0.amount_cents/100 EUR",
          "vat": "$1.amount_cents/100 EUR",
          "total": "$2.amount_cents/100 EUR"
        }, "params": {"style": "neutral"}}
      ]
    });

    // compile → lint → run
    let prog = Program { program: vec![
        Node{ id: Some("compiled".into()), op:"plan.compile".into(),
              r#in: json!({"steps": qwen_plan}), out: json!({}), params: json!({}) },
        Node{ id: Some("lint".into()), op:"plan.lint".into(),
              r#in: json!({"program":"$compiled"}), out: json!({}), params: json!({}) },
        Node{ id: Some("trace".into()), op:"plan.run".into(),
              r#in: json!({"program":"$compiled"}), out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Lint"  => println!("Lint: {}", c.val),
            "Trace" => println!("Trace: {}", c.val),
            _ => {}
        }
    }
    Ok(())
}
fn demo_concept_eat_fr() -> anyhow::Result<()> {
    let prog = Program { program: vec![
        Node{ id: Some("eat".into()), op:"concept.load".into(),
              r#in: json!({"id":"EAT","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("inst".into()), op:"concept.bind".into(),
              r#in: json!({"concept":"$eat",
                           "form":{"lang":"fr",
                                   "roles":{"Agent":"Le chat","Patient":"la souris","Loc":"la cuisine","Time":"midi"}}}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("who".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHO","concept":"$eat","instance":"$inst"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("why".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHY","concept":"$eat","instance":"$inst"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("text".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$eat","instance":"$inst","style":{"lang":"fr","voice":"active"}}),
              out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Answer" => println!("Q:{} -> {}", c.val.get("op").and_then(|v| v.as_str()).unwrap_or("?"), c.val.get("answer").unwrap_or(&json!(null))),
            "Text"   => println!("Texte: {}", c.val.get("text").and_then(|v| v.as_str()).unwrap_or("")),
            _ => {}
        }
    }
    Ok(())
}
fn demo_concept_sell_fr() -> anyhow::Result<()> {
    let prog = Program { program: vec![
        Node{ id: Some("sell".into()), op:"concept.load".into(),
              r#in: json!({"id":"SELL","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("inst".into()), op:"concept.bind".into(),
              r#in: json!({"concept":"$sell",
                           "form":{"lang":"fr",
                                   "roles":{
                                       "Giver":"Paul",
                                       "Receiver":"Marie",
                                       "Theme":"un livre",
                                       "Consideration":"20 euros"
                                   }}}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("who".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHO","concept":"$sell","instance":"$inst"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("what".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHAT","concept":"$sell","instance":"$inst"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("text".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$sell","instance":"$inst","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Answer" => println!("Q:{} -> {}", c.val.get("op").and_then(|v| v.as_str()).unwrap_or("?"), c.val.get("answer").unwrap_or(&json!(null))),
            "Text"   => println!("Texte: {}", c.val.get("text").and_then(|v| v.as_str()).unwrap_or("")),
            _ => {}
        }
    }
    Ok(())
}
fn demo_concept_abstract_sell() -> anyhow::Result<()> {
    let prog = Program { program: vec![
        Node{ id: Some("sell".into()), op:"concept.load".into(),
              r#in: json!({"id":"SELL","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("abs".into()), op:"concept.abstract".into(),
              r#in: json!({"concept":"$sell"}), out: json!({}), params: json!({}) },
    ]};
    let outs = Vm::new().run(&prog)?;
    for c in outs {
        if c.typ == "Parents" { println!("SELL parents -> {}", c.val); }
    }
    Ok(())
}

fn demo_concept_bind_text_sell() -> anyhow::Result<()> {
    let prog = Program { program: vec![
        Node{ id: Some("sell".into()), op:"concept.load".into(),
              r#in: json!({"id":"SELL","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$sell","text":"Paul vend un livre à Marie pour 20 euros","lang":"fr"}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("who".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHO","concept":"$sell","instance":"$inst"}), out: json!({}), params: json!({}) },
        Node{ id: Some("what".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHAT","concept":"$sell","instance":"$inst"}), out: json!({}), params: json!({}) },
        Node{ id: Some("text".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$sell","instance":"$inst","style":{"lang":"fr","frame_idx":1}}),
              out: json!({}), params: json!({}) },
    ]};
    let outs = Vm::new().run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Answer" => println!("Q:{} -> {}", c.val.get("op").and_then(|v| v.as_str()).unwrap_or("?"), c.val.get("answer").unwrap_or(&json!(null))),
            "Text"   => println!("Texte: {}", c.val.get("text").and_then(|v| v.as_str()).unwrap_or("")),
            "Parents"=> println!("Parents: {}", c.val),
            _ => {}
        }
    }
    Ok(())
}
fn demo_transfer_give_lend_sell() -> anyhow::Result<()> {
    let sentences = [
        ("GIVE",    "Paul donne un livre à Marie"),
        ("SELL",    "Paul vend un livre à Marie pour 20 euros"),
        ("LEND",    "Paul prête un livre à Marie pendant deux semaines")
    ];

    for (target, sent) in sentences {
        println!("--- {}", target);
        let prog = Program { program: vec![
            Node{ id: Some("c".into()), op:"concept.load".into(),
                  r#in: json!({"id": target, "lang":"fr"}), out: json!({}), params: json!({}) },

            Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
                  r#in: json!({"concept":"$c","text": sent, "lang":"fr"}), out: json!({}), params: json!({}) },

            Node{ id: Some("who".into()), op:"concept.query".into(),
                  r#in: json!({"op":"WHO","concept":"$c","instance":"$inst"}), out: json!({}), params: json!({}) },

            Node{ id: Some("what".into()), op:"concept.query".into(),
                  r#in: json!({"op":"WHAT","concept":"$c","instance":"$inst"}), out: json!({}), params: json!({}) },

            Node{ id: Some("text".into()), op:"surface.realize".into(),
                  r#in: json!({"concept":"$c","instance":"$inst","style":{"lang":"fr","frame_idx":0}}), out: json!({}), params: json!({}) },
        ]};

        let outs = Vm::new().run(&prog)?;
        for c in outs {
            match c.typ.as_str() {
                "Answer" => println!("Q:{} -> {}", c.val.get("op").and_then(|v| v.as_str()).unwrap_or("?"), c.val.get("answer").unwrap_or(&json!(null))),
                "Text"   => println!("Texte: {}", c.val.get("text").and_then(|v| v.as_str()).unwrap_or("")),
                _ => {}
            }
        }
    }
    Ok(())
}

fn demo_transfer_classify() -> anyhow::Result<()> {
    // On part d’un TRANSFER “générique” en texte, et on laisse le kernel classifier → SELL/GIVE/LEND
    let sentences = [
        "Paul vend un livre à Marie pour 20 euros",
        "Paul donne un livre à Marie",
        "Paul prête un livre à Marie pendant deux semaines"
    ];

    for sent in sentences {
        println!("--- classify: {}", sent);
        let prog = Program { program: vec![
            Node{ id: Some("t".into()), op:"concept.load".into(),
                  r#in: json!({"id":"TRANSFER","lang":"fr"}), out: json!({}), params: json!({}) },

            Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
                  r#in: json!({"concept":"$t","text": sent, "lang":"fr"}), out: json!({}), params: json!({}) },

            Node{ id: Some("sub".into()), op:"transfer.classify".into(),
                  r#in: json!({"instance":"$inst"}), out: json!({}), params: json!({}) },

            // Réalise avec la frame 0 du bon sous-concept
            Node{ id: Some("c_sub".into()), op:"concept.load".into(),
                  r#in: json!({"id":"$sub","lang":"fr"}), out: json!({}), params: json!({}) },

            Node{ id: Some("text".into()), op:"surface.realize".into(),
                  r#in: json!({"concept":"$c_sub","instance":"$inst","style":{"lang":"fr","frame_idx":0}}),
                  out: json!({}), params: json!({}) },
        ]};

        let outs = Vm::new().run(&prog)?;
        for c in outs {
            match c.typ.as_str() {
                "Subtype" => {
                    if let Some(s) = c.val.as_str() {
                        println!("→ subtype: {}", s);
                    } else {
                        println!("→ subtype: ?");
                    }
                },
                "Text"    => println!("Texte: {}", c.val.get("text").and_then(|v| v.as_str()).unwrap_or("")),
                _ => {}
            }
        }
    }
    Ok(())
}
fn demo_why_inherit() -> anyhow::Result<()> {
    let cases = [
        ("SELL", "Paul vend un livre à Marie pour 20 euros"),
        ("GIVE", "Paul donne un livre à Marie"),
        ("LEND", "Paul prête un livre à Marie pendant deux semaines")
    ];
    for (cid, sent) in cases {
        println!("--- WHY {}: {}", cid, sent);
        let prog = Program { program: vec![
            Node{ id: Some("c".into()), op:"concept.load".into(),
                  r#in: json!({"id": cid, "lang":"fr"}), out: json!({}), params: json!({}) },

            Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
                  r#in: json!({"concept":"$c","text": sent, "lang":"fr"}), out: json!({}), params: json!({}) },

            Node{ id: Some("why".into()), op:"concept.query".into(),
                  r#in: json!({"op":"WHY","concept":"$c","instance":"$inst"}), out: json!({}), params: json!({}) },
        ]};

        let outs = Vm::new().run(&prog)?;
        for c in outs {
            if c.typ == "Answer" && c.val.get("op").and_then(|v| v.as_str()) == Some("WHY") {
                println!("WHY -> {}", c.val.get("answer").unwrap_or(&json!(null)));
            }
        }
    }
    Ok(())
}
fn demo_abduce_lend() -> anyhow::Result<()> {
    // Phrase sans durée → on doit proposer d'ajouter Duration pour activer la règle de LEND
    let sent = "Paul prête un livre à Marie";
    let prog = Program { program: vec![
        Node{ id: Some("lend".into()), op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$lend","text": sent, "lang":"fr"}), out: json!({}), params: json!({}) },

        // Abduction ciblée: sous-type LEND (ou target_contains: "EXPECT(")
        Node{ id: Some("abd".into()), op:"reason.abduce".into(),
              r#in: json!({"concept":"$lend","instance":"$inst","for_subtype":"LEND"}), out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        if c.typ == "Abduce" {
            println!("Abduce -> {}", c.val);
        }
    }
    Ok(())
}
fn demo_counterfactuals() -> anyhow::Result<()> {
    // 1) GIVE sans paiement -> ajoute Consideration => FEEL_OBLIGATION disparaît
    println!("--- CF #1: GIVE, ajout de paiement");
    let prog1 = Program { program: vec![
        Node{ id: Some("c".into()), op:"concept.load".into(),
              r#in: json!({"id":"GIVE","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$c","text":"Paul donne un livre à Marie","lang":"fr"}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("cf".into()), op:"reason.counterfactual".into(),
              r#in: json!({"concept":"$c","instance":"$inst","set":{"Consideration":"20 euros"}}),
              out: json!({}), params: json!({}) },
    ]};
    let outs1 = Vm::new().run(&prog1)?;
    for c in outs1 {
        if c.typ == "Counterfactual" { println!("CF1 -> {}", c.val); }
    }

    // 2) LEND sans durée -> ajoute Duration => EXPECT(RETURN) apparaît
    println!("--- CF #2: LEND, ajout de durée");
    let prog2 = Program { program: vec![
        Node{ id: Some("c".into()), op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$c","text":"Paul prête un livre à Marie","lang":"fr"}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("cf".into()), op:"reason.counterfactual".into(),
              r#in: json!({"concept":"$c","instance":"$inst","set":{"Duration":"deux semaines"}}),
              out: json!({}), params: json!({}) },
    ]};
    let outs2 = Vm::new().run(&prog2)?;
    for c in outs2 {
        if c.typ == "Counterfactual" { println!("CF2 -> {}", c.val); }
    }

    Ok(())
}
fn demo_plan_lend() -> anyhow::Result<()> {
    let prog = Program { program: vec![
        Node{ id: Some("c".into()), op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("inst0".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$c","text":"Paul prête un livre à Marie","lang":"fr"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("why0".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHY","concept":"$c","instance":"$inst0"}), out: json!({}), params: json!({}) },

        Node{ id: Some("plan".into()), op:"reason.plan".into(),
              r#in: json!({"concept":"$c","instance":"$inst0","goal":{"subtype":"LEND","activate_contains":"EXPECT("}}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("inst1".into()), op:"instance.apply_patch".into(),
              r#in: json!({"instance":"$inst0","set":"$plan.set","unset":"$plan.unset"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("why1".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHY","concept":"$c","instance":"$inst1"}), out: json!({}), params: json!({}) },

        Node{ id: Some("text".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$c","instance":"$inst1","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Answer" if c.val.get("op").and_then(|v| v.as_str())==Some("WHY") =>
                println!("WHY -> {}", c.val.get("answer").unwrap_or(&json!(null))),
            "Plan" => println!("PLAN -> {}", c.val),
            "Text" => println!("Texte: {}", c.val.get("text").and_then(|v| v.as_str()).unwrap_or("")),
            _ => {}
        }
    }
    Ok(())
}
fn demo_plan_sell_from_give() -> anyhow::Result<()> {
    let prog = Program { program: vec![
        Node{ id: Some("g".into()), op:"concept.load".into(),
              r#in: json!({"id":"GIVE","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("inst0".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$g","text":"Paul donne un livre à Marie","lang":"fr"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("plan".into()), op:"reason.plan".into(),
              r#in: json!({"concept":"$g","instance":"$inst0","goal":{"subtype":"SELL"}}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("inst1".into()), op:"instance.apply_patch".into(),
              r#in: json!({"instance":"$inst0","set":"$plan.set","unset":"$plan.unset"}),
              out: json!({}), params: json!({}) },

        // classifie après patch pour charger le bon sous-concept
        Node{ id: Some("sub".into()), op:"transfer.classify".into(),
              r#in: json!({"instance":"$inst1"}), out: json!({}), params: json!({}) },

        Node{ id: Some("c_sub".into()), op:"concept.load".into(),
              r#in: json!({"id":"$sub","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("text".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$c_sub","instance":"$inst1","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Plan" => println!("PLAN -> {}", c.val),
            "Subtype" => println!("→ subtype: {}", c.val.as_str().unwrap_or("?")),
            "Text" => println!("Texte: {}", c.val.get("text").and_then(|v| v.as_str()).unwrap_or("")),
            _ => {}
        }
    }
    Ok(())
}
fn demo_scenario_lend_return() -> anyhow::Result<()> {
    use serde_json::json;

    let prog = Program { program: vec![
        // 1) LEND avec durée (matche la frame)
        Node{ id: Some("lend".into()), op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("lend0".into()), op:"concept.bind_text".into(),
              r#in: json!({
                "concept":"$lend",
                "text":"Paul prête un livre à Marie pendant deux semaines",
                "lang":"fr"
              }),
              out: json!({}), params: json!({}) },

        // plan → patch pour activer EXPECT(RETURN)
        Node{ id: Some("plan".into()), op:"reason.plan".into(),
              r#in: json!({"concept":"$lend","instance":"$lend0","goal":{"subtype":"LEND","activate_contains":"EXPECT("}}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("lend1".into()), op:"instance.apply_patch".into(),
              r#in: json!({"instance":"$lend0","set":"$plan.set","unset":"$plan.unset"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("t1".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$lend","instance":"$lend1","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },

        // 2) RETURN utilise les rôles de LEND (Giver/Receiver/Theme)
        Node{ id: Some("ret".into()), op:"concept.load".into(),
              r#in: json!({"id":"RETURN","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("giver".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$lend1","role":"Giver"}), out: json!({}), params: json!({}) },
        Node{ id: Some("recv".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$lend1","role":"Receiver"}), out: json!({}), params: json!({}) },
        Node{ id: Some("theme".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$lend1","role":"Theme"}), out: json!({}), params: json!({}) },

        Node{ id: Some("ret_inst".into()), op:"instance.new".into(),
              r#in: json!({"concept":"$ret","lang":"fr",
                           "roles": { "Agent":"$recv", "Receiver":"$giver", "Theme":"$theme" }}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("t2".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$ret","instance":"$ret_inst","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    let mut texts = Vec::new();
    for c in outs {
        if c.typ == "Text" {
            if let Some(s) = c.val.get("text").and_then(|v| v.as_str()) { texts.push(s.to_string()); }
        }
    }
    for t in texts { println!("{}", t); }
    Ok(())
}

fn demo_scenario_sell_pay() -> anyhow::Result<()> {
    use serde_json::json;

    // 1) On part d'un SELL explicite (avec montant)
    let prog = Program { program: vec![
        // SELL
        Node{ id: Some("sell".into()), op:"concept.load".into(),
              r#in: json!({"id":"SELL","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("sell_inst".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$sell","text":"Paul vend un livre à Marie pour 20 euros","lang":"fr"}),
              out: json!({}), params: json!({}) },

        // Récupère les rôles utiles du SELL
        Node{ id: Some("giver".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_inst","role":"Giver"}), out: json!({}), params: json!({}) },
        Node{ id: Some("recv".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_inst","role":"Receiver"}), out: json!({}), params: json!({}) },
        Node{ id: Some("amt".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_inst","role":"Consideration"}), out: json!({}), params: json!({}) },

        // 2) PAY : Agent = Receiver (acheteur), Receiver = Giver (vendeur), Amount = Consideration
        Node{ id: Some("pay".into()), op:"concept.load".into(),
              r#in: json!({"id":"PAY","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("pay_inst".into()), op:"instance.new".into(),
              r#in: json!({"concept":"$pay","lang":"fr",
                           "roles": { "Agent":"$recv", "Receiver":"$giver", "Amount":"$amt" }}),
              out: json!({}), params: json!({}) },

        // Réalise les deux phrases
        Node{ id: Some("t1".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$sell","instance":"$sell_inst","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("t2".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$pay","instance":"$pay_inst","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) }
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        if c.typ == "Text" {
            if let Some(s) = c.val.get("text").and_then(|v| v.as_str()) {
                println!("{}", s);
            }
        }
    }
    Ok(())
}
fn demo_induce_borrow_then_return() -> anyhow::Result<()> {
    use serde_json::json;

    let sentence = "Marie emprunte un livre à Paul pendant deux semaines";

    let prog = Program { program: vec![
        // 1) Induire BORROW et sauvegarder
        Node{ id: Some("c".into()), op:"cdl.induce".into(),
              r#in: json!({"id":"BORROW","lemma":"emprunter","sentence": sentence,"lang":"fr"}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("save".into()), op:"concept.save".into(),
              r#in: json!({"concept":"$c","lang":"fr"}), out: json!({}), params: json!({}) },

        // 2) Charger BORROW, le binder au texte d’exemple, WHY, réaliser
        Node{ id: Some("b".into()), op:"concept.load".into(),
              r#in: json!({"id":"BORROW","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$b","text": sentence, "lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("why".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHY","concept":"$b","instance":"$inst"}), out: json!({}), params: json!({}) },
        Node{ id: Some("t1".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$b","instance":"$inst","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },

        // 3) RETURN : Agent = Agent(BORROW), Receiver = Giver(BORROW), Theme idem
        Node{ id: Some("ret".into()), op:"concept.load".into(),
              r#in: json!({"id":"RETURN","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("ag".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$inst","role":"Agent"}), out: json!({}), params: json!({}) },
        Node{ id: Some("giv".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$inst","role":"Giver"}), out: json!({}), params: json!({}) },
        Node{ id: Some("th".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$inst","role":"Theme"}), out: json!({}), params: json!({}) },

        Node{ id: Some("ret_i".into()), op:"instance.new".into(),
              r#in: json!({"concept":"$ret","lang":"fr","roles": { "Agent":"$ag", "Receiver":"$giv", "Theme":"$th" }}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("t2".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$ret","instance":"$ret_i","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        match c.typ.as_str() {
            "Saved" => println!("Concept BORROW sauvegardé: {}", c.val),
            "Answer" if c.val.get("op").and_then(|v| v.as_str())==Some("WHY") => {
                println!("WHY(BORROW) -> {}", c.val.get("answer").unwrap_or(&json!(null)));
            }
            "Text" => println!("{}", c.val.get("text").and_then(|v| v.as_str()).unwrap_or("")),
            _ => {}
        }
    }
    Ok(())
}
fn demo_scenario_sell_pay_thank() -> anyhow::Result<()> {
    use serde_json::json;

    let prog = Program { program: vec![
        // SELL (texte → bind)
        Node{ id: Some("sell".into()), op:"concept.load".into(),
              r#in: json!({"id":"SELL","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("sell_i_raw".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$sell","text":"Paul vend un livre à Marie pour 20 euros","lang":"fr"}),
              out: json!({}), params: json!({}) },

        // 🔁 NORMALISATION: Agent→Giver (et on recopie Receiver/Theme/Consideration)
        Node{ id: Some("ag".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_i_raw","role":"Agent"}), out: json!({}), params: json!({}) },
        Node{ id: Some("rcv".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_i_raw","role":"Receiver"}), out: json!({}), params: json!({}) },
        Node{ id: Some("th".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_i_raw","role":"Theme"}), out: json!({}), params: json!({}) },
        Node{ id: Some("amt".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_i_raw","role":"Consideration"}), out: json!({}), params: json!({}) },

        Node{ id: Some("sell_i".into()), op:"instance.new".into(),
              r#in: json!({"concept":"$sell","lang":"fr",
                           "roles":{"Giver":"$ag","Receiver":"$rcv","Theme":"$th","Consideration":"$amt"}}),
              out: json!({}), params: json!({}) },

        // PAY : Agent = Receiver(SELL), Receiver = Giver(SELL), Amount = Consideration(SELL)
        Node{ id: Some("giver".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_i","role":"Giver"}), out: json!({}), params: json!({}) },
        Node{ id: Some("recv".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_i","role":"Receiver"}), out: json!({}), params: json!({}) },
        Node{ id: Some("amt2".into()), op:"instance.get_role".into(),
              r#in: json!({"instance":"$sell_i","role":"Consideration"}), out: json!({}), params: json!({}) },

        Node{ id: Some("pay".into()), op:"concept.load".into(),
              r#in: json!({"id":"PAY","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("pay_i".into()), op:"instance.new".into(),
              r#in: json!({"concept":"$pay","lang":"fr",
                           "roles":{"Agent":"$recv","Receiver":"$giver","Amount":"$amt2"}}),
              out: json!({}), params: json!({}) },

        // THANK : Agent = acheteuse, Receiver = vendeur
        Node{ id: Some("thanks".into()), op:"concept.load".into(),
              r#in: json!({"id":"THANK","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("thanks_i".into()), op:"instance.new".into(),
              r#in: json!({"concept":"$thanks","lang":"fr",
                           "roles":{"Agent":"$recv","Receiver":"$giver"}}),
              out: json!({}), params: json!({}) },

        // Réalisation
        Node{ id: Some("t1".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$sell","instance":"$sell_i","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("t2".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$pay","instance":"$pay_i","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("t3".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$thanks","instance":"$thanks_i","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        if c.typ == "Text" {
            if let Some(s) = c.val.get("text").and_then(|v| v.as_str()) { println!("{}", s); }
        }
    }
    Ok(())
}

fn demo_story_chain() -> anyhow::Result<()> {
    use serde_json::json;
    let prog = Program { program: vec![
        Node{ id: Some("txt".into()), op:"surface.realize".into(),
              r#in: json!({
                "plan": { "propositions": [
                  "Paul prête un livre à Marie pendant deux semaines",
                  "Deux semaines plus tard, Marie rend le livre à Paul",
                  "Paul remercie Marie pour sa ponctualité"
                ]},
                "style": { "lang":"fr" }
              }),
              out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    for c in outs {
        if c.typ == "Text" {
            if let Some(s) = c.val.get("text").and_then(|v| v.as_str()) {
                println!("{}", s);
            }
        }
    }
    Ok(())
}
fn demo_story_chain_auto() -> anyhow::Result<()> {
    use serde_json::json;
    let prog = Program { program: vec![
        Node{ id: Some("lend".into()), op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("lend_i".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$lend","text":"Paul prête un livre à Marie pendant deux semaines","lang":"fr"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("plan".into()), op:"story.compose".into(),
              r#in: json!({"seed":"$lend_i"}), out: json!({}), params: json!({}) },

        Node{ id: Some("t".into()), op:"surface.realize".into(),
              r#in: json!({"plan":"$plan"}), out: json!({}), params: json!({}) },
    ]};

    for c in Vm::new().run(&prog)? {
        if c.typ == "Text" { println!("{}", c.val["text"].as_str().unwrap_or("")); }
    }
    Ok(())
}
// match name { ... 
//    "story_defaults" => Ok(demo_story_defaults()?),
//    "story_defaults_except_return" => Ok(demo_story_defaults_except_return()?),
// ... }

fn demo_story_defaults() -> anyhow::Result<()> {
    use serde_json::json;
    let prog = Program { program: vec![
        // Seed = LEND
        Node{ id: Some("lend".into()),    op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("i".into()),       op:"concept.bind_text".into(),
              r#in: json!({"concept":"$lend","text":"Paul prête un livre à Marie pendant deux semaines","lang":"fr"}), out: json!({}), params: json!({}) },

        // COMPOSE (phrase initiale)
        Node{ id: Some("t0".into()),      op:"surface.realize".into(),
              r#in: json!({"concept":"$lend","instance":"$i","style":{"lang":"fr"}}), out: json!({}), params: json!({}) },
        Node{ id: Some("p0".into()),      op:"plan.wrap_text".into(),
              r#in: json!({"texts":["$t0.text"], "kind":"COMPOSE"}), out: json!({}), params: json!({}) },

        // CAUSALIZE (defaults)
        Node{ id: Some("p_def".into()),   op:"reason.causalize".into(),
              r#in: json!({"instance":"$i"}), out: json!({}), params: json!({}) },

        // MERGE
        Node{ id: Some("p_all".into()),   op:"plan.concat".into(),
              r#in: json!({"plans": ["$p0","$p_def"]}), out: json!({}), params: json!({}) },

        // REALIZE (plan → texte)
        Node{ id: Some("t".into()),       op:"surface.realize".into(),
              r#in: json!({"plan":"$p_all", "style":{"lang":"fr"}}), out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    let final_text = outs.iter()
        .rev()
        .find(|c| c.typ == "Text")
        .and_then(|c| c.val.get("text"))
        .and_then(|v| v.as_str());
    if let Some(s) = final_text {
        println!("{}", s);
    }
    Ok(())
}

fn demo_story_defaults_except_return() -> anyhow::Result<()> {
    use serde_json::json;
    let prog = Program { program: vec![
        Node{ id: Some("lend".into()),  op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("i".into()),     op:"concept.bind_text".into(),
              r#in: json!({"concept":"$lend","text":"Paul prête un livre à Marie pendant deux semaines","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("t0".into()),    op:"surface.realize".into(),
              r#in: json!({"concept":"$lend","instance":"$i","style":{"lang":"fr"}}), out: json!({}), params: json!({}) },
        Node{ id: Some("p0".into()),    op:"plan.wrap_text".into(),
              r#in: json!({"texts":["$t0.text"], "kind":"COMPOSE"}), out: json!({}), params: json!({}) },

        // Defaults MAIS avec exception: on skip RETURN
        Node{ id: Some("p_def".into()), op:"reason.causalize".into(),
              r#in: json!({"instance":"$i","skip":["RETURN"]}), out: json!({}), params: json!({}) },

        Node{ id: Some("p_all".into()), op:"plan.concat".into(),
              r#in: json!({"plans": ["$p0","$p_def"]}), out: json!({}), params: json!({}) },

        Node{ id: Some("t".into()),     op:"surface.realize".into(),
              r#in: json!({"plan":"$p_all","style":{"lang":"fr"}}), out: json!({}), params: json!({}) },
    ]};

    let outs = Vm::new().run(&prog)?;
    let final_text = outs.iter()
        .rev()
        .find(|c| c.typ == "Text")
        .and_then(|c| c.val.get("text"))
        .and_then(|v| v.as_str());
    if let Some(s) = final_text {
        println!("{}", s);
    }
    Ok(())
}
fn demo_cdl_fuse_sell() -> anyhow::Result<()> {
    use serde_json::json;
    let delta = json!({
        "id":"SELL",
        "roles": ["Loc","Time"],                          // tolérant: union
        "lexemes": { "fr": { "frames": [
            "Agent V Theme à Receiver",                   // doublon ok (dédup)
            "Agent V Theme à Receiver pour Consideration" // frame utile si absente
        ]}},
        "defaults": [
            {"if":"SELL","then":"EXPECT(Receiver,THANK(Giver))","strength":0.5}
        ]
    });

    let prog = Program { program: vec![
        Node{ id: Some("sell".into()), op:"concept.load".into(),
              r#in: json!({"id":"SELL","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("fused".into()), op:"cdl.fuse".into(),
              r#in: json!({ "base":"$sell", "delta": delta }), out: json!({}), params: json!({}) },

        Node{ id: Some("save".into()), op:"concept.save".into(),
              r#in: json!({"concept":"$fused","lang":"fr"}), out: json!({}), params: json!({}) },

        // Sanity: bind + realize toujours OK
        Node{ id: Some("inst".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$fused","text":"Paul vend un livre à Marie pour 20 euros","lang":"fr"}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("t".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$fused","instance":"$inst","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
    ]};

    for c in Vm::new().run(&prog)? {
        if c.typ == "Saved" { println!("Fused concept saved: {}", c.val); }
        if c.typ == "Text"  { println!("{}", c.val["text"].as_str().unwrap_or("")); }
    }
    Ok(())
}
fn demo_analogy_eat_devour() -> anyhow::Result<()> {
    use serde_json::json;

    let prog = Program { program: vec![
        // EAT source
        Node{ id: Some("eat".into()), op:"concept.load".into(),
              r#in: json!({"id":"EAT","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("eat_i".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$eat","text":"Le loup mange un mouton","lang":"fr"}),
              out: json!({}), params: json!({}) },

        // DEVOUR cible
        Node{ id: Some("dev".into()), op:"concept.load".into(),
              r#in: json!({"id":"DEVOUR","lang":"fr"}), out: json!({}), params: json!({}) },

        // Remap rôles eat_i -> devour_i
        Node{ id: Some("dev_i".into()), op:"analogy.remap".into(),
              r#in: json!({"instance":"$eat_i","concept":"$dev","lang":"fr"}),
              out: json!({}), params: json!({}) },

        // Réalisation
        Node{ id: Some("t1".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$eat","instance":"$eat_i","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("t2".into()), op:"surface.realize".into(),
              r#in: json!({"concept":"$dev","instance":"$dev_i","style":{"lang":"fr","frame_idx":0}}),
              out: json!({}), params: json!({}) },
    ]};

    for c in Vm::new().run(&prog)? {
        if c.typ == "Text" {
            if let Some(s) = c.val.get("text").and_then(|v| v.as_str()) {
                println!("{}", s);
            }
        }
    }
    Ok(())
}
fn demo_why_return_from_lend() -> anyhow::Result<()> {
    use serde_json::json;

    let prog = Program { program: vec![
        // Concepts
        Node{ id: Some("lend".into()), op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },
        Node{ id: Some("ret".into()),  op:"concept.load".into(),
              r#in: json!({"id":"RETURN","lang":"fr"}), out: json!({}), params: json!({}) },

        // Instances
        Node{ id: Some("lend_i".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$lend","text":"Paul prête un livre à Marie pendant deux semaines","lang":"fr"}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("ret_i".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$ret","text":"Marie rend le livre à Paul","lang":"fr"}),
              out: json!({}), params: json!({}) },

        // WHY avec contexte
        Node{ id: Some("why".into()), op:"concept.query".into(),
              r#in: json!({"op":"WHY","instance":"$ret_i","context":["$lend_i"]}),
              out: json!({}), params: json!({}) },

        // Plan + réalisation
        Node{ id: Some("plan".into()), op:"story.compose".into(),
              r#in: json!({"instances":["$lend_i","$ret_i"]}),
              out: json!({}), params: json!({}) },
        Node{ id: Some("t".into()), op:"surface.realize".into(),
              r#in: json!({"plan":"$plan"}),
              out: json!({}), params: json!({}) },
    ]};

    for c in Vm::new().run(&prog)? {
        match c.typ.as_str() {
            "Answer" => {
                if let Some(h) = c.val["hypotheses"].as_array().and_then(|a| a.first()) {
                    if let Some(s) = h["text"].as_str() { println!("{}", s); }
                }
            }
            "Text" => {
                let s = c.val["text"].as_str().unwrap_or("");
                if !s.is_empty() { println!("{}", s); }
            }
            _ => {}
        }
    }
    Ok(())
}
fn demo_story_cycle() -> anyhow::Result<()> {
    use serde_json::json;

    let prog = Program { program: vec![
        Node{ id: Some("lend".into()), op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("lend_i".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$lend","text":"Paul prête un livre à Marie pendant deux semaines","lang":"fr"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("plan".into()), op:"story.compose".into(),
              r#in: json!({"seed":"$lend_i"}), out: json!({}), params: json!({}) },

        Node{ id: Some("story".into()), op:"surface.realize".into(),
              r#in: json!({"plan":"$plan"}), out: json!({}), params: json!({}) },

        // ⬇⬇⬇  JSON bien fermé ici (pas d’accolade en trop)
        Node{ id: Some("ret".into()), op:"concept.bind_text".into(),
              r#in: json!({
                  "concept": {
                      "id": "RETURN",
                      "roles": ["Agent","Receiver","Theme"],
                      "lexemes": { "fr": { "verb_lemma": "rendre",
                                            "frames": ["Agent V Theme à Receiver"] } }
                  },
                  "text": "Marie rend le livre à Paul",
                  "lang": "fr"
              }),
              out: json!({}), params: json!({}) },

        Node{ id: Some("why".into()), op:"concept.query".into(),
              r#in: json!({ "op":"WHY", "instance":"$ret", "context":["$lend_i"], "exceptions":[] }),
              out: json!({}), params: json!({}) },
    ]};

    for c in Vm::new().run(&prog)? {
        if c.typ == "Text" {
            if let Some(s) = c.val["text"].as_str() { println!("{}", s); }
        } else if c.typ == "Answer" {
            if let Some(s) = c.val.get("hypotheses")
                                  .and_then(|h| h.get(0))
                                  .and_then(|h| h.get("text"))
                                  .and_then(|v| v.as_str()) {
                println!("{}", s);
            }
        }
    }
    Ok(())
}

fn demo_story_cycle_with_exception() -> anyhow::Result<()> {
    use serde_json::json;

    let prog = Program { program: vec![
        Node{ id: Some("lend".into()), op:"concept.load".into(),
              r#in: json!({"id":"LEND","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("lend_i".into()), op:"concept.bind_text".into(),
              r#in: json!({"concept":"$lend","text":"Paul prête un livre à Marie pendant deux semaines","lang":"fr"}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("reason".into()), op:"concept.reason".into(),
              r#in: json!({"instances":["$lend_i"], "ops":["CAUSALIZE"], "params":{"lang":"fr","exceptions":["NO_RETURN_EXPECTED"]}}),
              out: json!({}), params: json!({}) },

        Node{ id: Some("plan".into()), op:"story.compose".into(),
                r#in: json!({"seed":"$lend_i","reason":"$reason"}), out: json!({}), params: json!({}) },

        Node{ id: Some("story".into()), op:"surface.realize".into(),
              r#in: json!({"plan":"$plan"}), out: json!({}), params: json!({}) },

        Node{ id: Some("ret".into()), op:"concept.bind_text".into(),
              r#in: json!({
                  "concept": {
                      "id": "RETURN",
                      "roles": ["Agent","Receiver","Theme"],
                      "lexemes": { "fr": { "verb_lemma": "rendre",
                                            "frames": ["Agent V Theme à Receiver"] } }
                  },
                  "text": "Marie rend le livre à Paul",
                  "lang": "fr"
              }),
              out: json!({}), params: json!({}) },

        Node{ id: Some("why".into()), op:"concept.query".into(),
              r#in: json!({ "op":"WHY", "instance":"$ret", "context":["$lend_i"], "exceptions":["NO_RETURN_EXPECTED"] }),
              out: json!({}), params: json!({}) },
    ]};

    for c in Vm::new().run(&prog)? {
        if c.typ == "Text" {
            if let Some(s) = c.val["text"].as_str() { println!("{}", s); }
        } else if c.typ == "Answer" {
            if let Some(s) = c.val.get("hypotheses")
                                  .and_then(|h| h.get(0))
                                  .and_then(|h| h.get("text"))
                                  .and_then(|v| v.as_str()) {
                println!("{}", s);
            }
        }
    }
    Ok(())
}
fn demo_specialize_eat_poisoned() -> anyhow::Result<()> {
    use serde_json::json;
    let prog = Program { program: vec![
        Node{ id: Some("eat".into()), op:"concept.load".into(),
              r#in: json!({"id":"EAT","lang":"fr"}), out: json!({}), params: json!({}) },

        Node{ id: Some("eat_p".into()), op:"concept.specialize".into(),
              r#in: json!({
                  "base":"$eat",
                  "id":"EAT_POISONED",
                  "add_defaults":[ {"if":"EAT_POISONED","then":"CAUSE(Agent,SICK)","strength":0.7} ],
                  "lexeme_delta": { "fr": { "verb_lemma":"manger", "frames":["Agent V Patient"] } }
              }), out: json!({}), params: json!({}) },

        Node{ id: Some("inst".into()), op:"instance.new".into(),
              r#in: json!({ "concept":"$eat_p", "roles":{ "Agent":"Le chat", "Patient":"une souris empoisonnée" }, "lang":"fr" }),
              out: json!({}), params: json!({}) },

        Node{ id: Some("cause".into()), op:"reason.causalize".into(),
              r#in: json!({ "instance":"$inst" }), out: json!({}), params: json!({}) },

        Node{ id: Some("plan1".into()), op:"story.compose".into(),
              r#in: json!({ "instances":[ "$inst" ] }), out: json!({}), params: json!({}) },

        Node{ id: Some("plan2".into()), op:"plan.concat".into(),
              r#in: json!({ "plans":[ "$plan1", "$cause" ] }), out: json!({}), params: json!({}) },

        Node{ id: Some("t".into()), op:"surface.realize".into(),
              r#in: json!({ "plan":"$plan2" }), out: json!({}), params: json!({}) },
    ]};

    for c in Vm::new().run(&prog)? {
        if c.typ == "Text" { println!("{}", c.val["text"].as_str().unwrap_or("")); }
    }
    Ok(())
}




