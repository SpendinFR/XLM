use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use xlm::ingestion::{IngestionPipeline, OllamaLLMClient, PromptTemplate};
use xlm::memory::{KnowledgeGraphLoader, KnowledgeGraphWriter};
use xlm::reasoning::{PathExplainer, RelationQueryService};
use xlm::resources::{WikidataImportConfig, WikidataImporter};
use xlm::RelationKind;

#[derive(Parser)]
#[command(name = "xlm", version, about = "XLM Concept Graph CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    #[arg(long, default_value = "info", global = true)]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    Ingest {
        #[arg(long)]
        concept: String,
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        graph: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:11434/api/generate")]
        llm_endpoint: String,
        #[arg(long, default_value = "qwen3")]
        llm_model: String,
    },
    Query {
        #[arg(long)]
        concept: String,
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        relation: Option<String>,
    },
    Explain {
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        #[arg(long)]
        graph: PathBuf,
        #[arg(long, default_value_t = 4)]
        max_depth: usize,
    },
    Export {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Bootstrap {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long, default_value = "fr")]
        language: String,
        #[arg(long, default_value_t = 35_000)]
        limit: usize,
        #[arg(long, default_value_t = 500)]
        batch_size: usize,
        #[arg(long, default_value_t = 250)]
        pause_ms: u64,
        #[arg(long, default_value = "https://query.wikidata.org/sparql")]
        endpoint: String,
    },
}

fn init_tracing(level: &str) {
    let filter = EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    match cli.command {
        Commands::Ingest {
            concept,
            file,
            graph,
            llm_endpoint,
            llm_model,
        } => {
            let llm = OllamaLLMClient::new(llm_endpoint, llm_model)?;
            let pipeline = IngestionPipeline::new(Arc::new(llm), PromptTemplate::default());
            let mut knowledge = KnowledgeGraphLoader::load_from_path(&graph)?;
            let outcome = pipeline.ingest_file(&mut knowledge, &concept, &file)?;
            KnowledgeGraphWriter::save_to_path(&graph, &knowledge)?;
            info!("concept" = %outcome.concept_label, relations = outcome.relations_added, "notes" = ?outcome.notes);
        }
        Commands::Query {
            concept,
            graph,
            relation,
        } => {
            let knowledge = KnowledgeGraphLoader::load_from_path(&graph)?;
            let service = RelationQueryService::new(&knowledge);
            if let Some(kind_str) = relation {
                let kind = RelationKind::from_str(&kind_str)?;
                let relations = service.filter_by_kind(&concept, kind);
                println!("{}", serde_json::to_string_pretty(&relations)?);
            } else if let Some(relations) = service.relations_for(&concept) {
                println!("{}", serde_json::to_string_pretty(&relations)?);
            } else {
                error!(%concept, "concept introuvable");
            }
        }
        Commands::Explain {
            from,
            to,
            graph,
            max_depth,
        } => {
            let knowledge = KnowledgeGraphLoader::load_from_path(&graph)?;
            let explainer = PathExplainer::new(&knowledge);
            if let Some(path) = explainer.explain(&from, &to, max_depth) {
                println!("{}", serde_json::to_string_pretty(&path)?);
            } else {
                println!("Aucun chemin trouvé entre {} et {}", from, to);
            }
        }
        Commands::Export { graph, output } => {
            let knowledge = KnowledgeGraphLoader::load_from_path(&graph)?;
            std::fs::write(
                &output,
                serde_json::to_string_pretty(&knowledge.snapshot())?,
            )?;
            info!("export" = %output.display());
        }
        Commands::Bootstrap {
            graph,
            language,
            limit,
            batch_size,
            pause_ms,
            endpoint,
        } => {
            let mut knowledge = KnowledgeGraphLoader::load_from_path(&graph)?;
            let config = WikidataImportConfig::new(language)?
                .with_max_records(limit)
                .with_batch_size(batch_size)
                .with_endpoint(endpoint)
                .with_pause(Duration::from_millis(pause_ms));
            let importer = WikidataImporter::new(config)?;
            let stats = importer.import_into(&mut knowledge)?;
            KnowledgeGraphWriter::save_to_path(&graph, &knowledge)?;
            info!(
                "import_lemmas" = stats.lemmas_added,
                relations = stats.relations_added,
                concepts = stats.concepts_created,
                batches = stats.batches_processed,
                elapsed_ms = stats.elapsed.as_millis() as u64,
                "message" = "Import Wikidata terminé"
            );
        }
    }

    Ok(())
}
