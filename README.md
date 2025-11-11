# XLM Concept Graph

XLM Concept Graph est une plateforme d'apprentissage symbolique visant à construire une mémoire de connaissances complète en s'appuyant sur 20 relations conceptuelles fondamentales. Chaque concept est décrit dans un fichier texte, analysé par un LLM local, puis intégré dans un graphe persistant permettant un raisonnement multi-sauts.

## Objectifs

- **Apprendre par concepts et relations** : extraire les liens causaux, fonctionnels, sociaux et structurels qui définissent le monde.
- **Compréhension explicite** : toutes les connaissances sont stockées sous forme de noeuds et d'arêtes documentés.
- **Raisonnement** : la mémoire graphique permet d'expliquer, de chaîner et de comparer les concepts.

## Architecture

1. **Domain** — Modèles des concepts, relations et métadonnées.
2. **Ingestion** — Pipeline (fichier → prompt → LLM → validation) pour créer/mettre à jour le graphe.
3. **Mémoire** — Graphe persistant avec sérialisation JSON et index multi-vues.
4. **Raisonnement** — Services de requête et d'inférence sur les relations.
5. **CLI** — Interface `xlm` pour ingérer, interroger et expliquer les connaissances.

## LLM local

L'ingestion s'appuie sur un modèle Qwen 3 exécuté via Ollama. Le client `OllamaLLMClient` dialogue directement avec l'API HTTP locale (`http://127.0.0.1:11434/api/generate`) en envoyant le prompt utilisateur et système puis en extrayant la portion JSON produite par le modèle de type "thinking".

Pendant la validation, seules les relations dont la confiance est supérieure ou égale à `0.8` sont conservées. Les sorties du modèle peuvent donc contenir davantage d'hypothèses, mais seules celles jugées très fiables alimentent le graphe persistant.

## Relations supportées

Les 20 relations canoniques couvrent les interactions physiques, logiques et sociales :

- **Physique** : Cause→Effet, Action→Objet, Agent→Action, Propriété→Entité, Catégorie→Instance, Partie→Tout, Localisation→Entité, Temps→Évènement, Matière→Objet, But→Action.
- **Logique & Société** : Possession, Comparaison, État Initial→Final, Fonction/Utilisation, Condition, Opposition, Similitude, Dépendance, Processus→Résultat, Construction sociale.

Chaque relation est définie dans le code avec un identifiant stable, une description détaillée et un positionnement dans la taxonomie.

## Commandes principales

```bash
# Ingestion d'un concept à partir d'un fichier texte
xlm ingest --concept animal --file concepts/animal.txt --graph data/graph.json --llm-endpoint http://127.0.0.1:11434/api/generate --llm-model qwen3

# Requête sur les relations d'un concept
xlm query --concept animal --relation category-instance --graph data/graph.json

# Explication d'une chaîne relationnelle
xlm explain --from animal --to nourriture --max-depth 4 --graph data/graph.json

# Export du graphe complet
xlm export --graph data/graph.json --output data/graph_export.json
```

## Tests

Les tests utilisent `cargo test` et les snapshots `insta` pour vérifier les prompts et le parsing JSON.

```bash
cargo test
```

## Licence

Projet expérimental — à adapter selon vos besoins internes.
