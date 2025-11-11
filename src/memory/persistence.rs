use super::graph::{GraphData, KnowledgeGraph};
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub struct KnowledgeGraphLoader;

impl KnowledgeGraphLoader {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<KnowledgeGraph> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(KnowledgeGraph::new());
        }
        let data = fs::read_to_string(path)
            .with_context(|| format!("impossible de lire le graphe depuis {:?}", path))?;
        let graph_data: GraphData = serde_json::from_str(&data)
            .with_context(|| format!("JSON invalide pour le graphe {:?}", path))?;
        Ok(KnowledgeGraph::from_parts(graph_data))
    }
}

pub struct KnowledgeGraphWriter;

impl KnowledgeGraphWriter {
    pub fn save_to_path(path: impl AsRef<Path>, graph: &KnowledgeGraph) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("impossible de créer le dossier {:?}", parent))?;
        }
        let data = serde_json::to_string_pretty(&graph.snapshot())?;
        fs::write(path, data)
            .with_context(|| format!("impossible d'écrire le graphe dans {:?}", path))?;
        Ok(())
    }
}
