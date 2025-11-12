use crate::domain::RelationKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub system: String,
    pub user: String,
}

impl PromptTemplate {
    pub fn default() -> Self {
        let relation_lines: Vec<String> = RelationKind::all()
            .iter()
            .map(|kind| {
                let meta = kind.metadata();
                format!("- {}: {}", meta.name, meta.description)
            })
            .collect();

        let system = format!(
            "Tu es un extracteur de connaissances conceptuelles.\nTu dois retourner un JSON strict respectant le schéma donné.",
        );

        let user = format!(
            concat!(
                "Analyse le texte suivant et extrait toutes les relations pertinentes.\n",
                "Relations autorisées :\n{}\n\n",
                "Réponds avec un JSON de la forme :\\n",
                "{{",
                "  \"concept_label\": \"...\",",
                "  \"relations\": [",
                "    {{",
                "      \"subject\": \"...\",",
                "      \"relation\": \"...\",",
                "      \"object\": \"...\",",
                "      \"confidence\": 0.0-1.0,",
                "      \"justification\": \"phrase ou justification\"",
                "    }}",
                "  ],",
                "  \"notes\": \"observations supplémentaires\"",
                "}}"
            ),
            relation_lines.join("\n")
        );

        Self { system, user }
    }
}

#[derive(Debug, Clone)]
pub struct PromptBuilder {
    template: PromptTemplate,
}

impl PromptBuilder {
    pub fn new(template: PromptTemplate) -> Self {
        Self { template }
    }

    pub fn build(&self, concept_slug: &str, text: &str) -> (String, String) {
        let system = self.template.system.clone();
        let user = format!(
            "{}\n\nConcept ciblé : {}\n\nTexte :\n{}",
            self.template.user, concept_slug, text
        );
        (system, user)
    }
}
