# Plan de projet : XLM Concept Graph

## Vision globale

Construire un système d'apprentissage symbolique centré sur 20 relations fondamentales permettant de modéliser les concepts du monde et leurs interconnexions. L'objectif est de constituer une mémoire explicite, navigable et exploitable par raisonnement, alimentée par un LLM local chargé d'extraire les relations depuis des descriptions textuelles.

## Piliers

1. **Relations canoniques** — 20 relations (10 physiques, 10 logico-sociales) servant de vocabulaire standard pour indexer toutes les connaissances.
2. **Pipeline d'ingestion** — Chaque fichier texte représentant un concept est transformé en graphes de connaissances via un appel à un LLM local.
3. **Mémoire graphique** — Les concepts et leurs relations sont stockés dans un graphe persistant consultable et extensible.
4. **Raisonnement multi-sauts** — Le graphe supporte des inférences (chaînes causales, buts, dépendances, etc.) pour récupérer et recomposer la connaissance.

## Étapes principales

1. **Définition du modèle de données**
   - Enumération typée des 20 relations avec métadonnées.
   - Structures pour les concepts, attributs, relations et justifications.
   - Normalisation des identifiants (UUID + label humain).

2. **Interface LLM locale**
   - Spécification d'un `LLMClient` générique.
   - Formats d'entrée/sortie stricts (JSON) pour garantir un apprentissage reproductible.
   - Implémentations : `OllamaLLMClient` (HTTP vers `http://127.0.0.1:11434/api/generate`) et `MockLLMClient` (tests).

3. **Pipeline d'ingestion**
   - Lecture des fichiers texte.
   - Génération de prompts standardisés décrivant les 20 relations.
   - Validation et normalisation des sorties du LLM.
   - Création/extension du graphe de connaissances.

4. **Mémoire & Persistance**
   - `KnowledgeGraph` en mémoire utilisant des index efficaces.
   - Sérialisation JSON pour sauvegarde/chargement.
   - Historique des sources (fichiers, prompts, réponses LLM).

5. **Raisonnement**
   - Requêtes relationnelles (par sujet, objet, type).
   - Inférence multi-sauts par parcours de graphe (DFS/BFS) avec filtres de relations.
   - Génération d'explications en retraçant les justifications.

6. **Interface CLI**
   - Commandes principales : `ingest`, `query`, `explain`, `export`.
   - Options pour paramétrer l'endpoint et le modèle Ollama ainsi que les chemins de mémoire.
   - Logs structurés via `tracing`.

7. **Tests & Exemples**
   - Tests unitaires sur la normalisation et l'inférence.
   - Tests snapshot (`insta`) pour la génération de prompts et la validation de parsing.

## Flux complet "fichier texte → connaissances"

1. L'utilisateur fournit `concepts/animal.txt`.
2. Le pipeline crée un prompt rappelant les 20 relations et le contenu du fichier.
3. Le LLM local répond en JSON (`relations`, `concepts`, `assertions`).
4. Le validateur vérifie chaque relation (types reconnus, concepts créés au besoin).
5. Le graphe est mis à jour : création des noeuds, des arêtes, stockage de la source.
6. Le graphe est sauvegardé (`knowledge_graph.json`).
7. L'utilisateur peut interroger (`xlm query --concept animal --relation category`) pour obtenir toutes les instances reliées.

## Résultat attendu

Une base de code modulaire offrant :
- une description exhaustive des relations possibles,
- une ingestion automatique à partir de textes et d'un LLM local,
- une mémoire graphique exploitable pour des inférences et des requêtes riches,
- une base solide pour entraîner un futur LLM propriétaire s'appuyant sur une représentation conceptuelle robuste.
