# Analyse de Sentiment Ballon d'Or (YouTube)

Ce projet propose une application et des scripts Python pour l'analyse de sentiment sur des commentaires YouTube liés au Ballon d'Or, avec relabeling binaire équilibré et visualisations.

## Structure du projet

- `app.py` : Application Streamlit pour explorer les résultats, les figures et les textes.
- `relabel_balanced.py` : Script pour relabeliser tous les commentaires en binaire (positif/négatif) de façon équilibrée, en utilisant les neutres pour équilibrer les classes.
- `archive/2026-01-08/` : Données brutes et scripts d'extraction/labeling.
- `data/processed/` : Données labellisées prêtes à l'emploi.
- `figs/` : Figures générées (répartition des labels, wordclouds, etc).
- `reports/` : Résultats de modèles et prédictions.
- `src/` : Scripts de visualisation et d'entraînement.

## Fonctionnalités principales

- **Relabeling binaire équilibré** :
  - Utilise le modèle `cardiffnlp/twitter-xlm-roberta-base-sentiment` (HuggingFace) pour prédire le sentiment de chaque commentaire.
  - Les classes positives et négatives sont équilibrées en complétant avec des neutres si besoin.
  - Résultat : un CSV prêt pour l'entraînement ou l'analyse, parfaitement équilibré.

- **Application Streamlit** :
  - Affichage des métriques de modèles, figures, exploration de texte, téléchargement de résultats.
  - Gestion robuste des erreurs (fichiers manquants, colonnes absentes, etc).

- **Scripts d'extraction et de visualisation** :
  - Extraction de commentaires YouTube, nettoyage, labeling initial, visualisation des distributions et wordclouds.

## Installation

1. Cloner le repo :
   ```bash
   git clone https://github.com/ghassen-7/Analyse-Sentiment-BO.git
   cd Analyse-Sentiment-BO
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   # ou, si besoin :
   pip install transformers pandas torch emoji unidecode streamlit
   ```

## Utilisation

- **Relabeling équilibré** :
  ```bash
  python relabel_balanced.py
  # Génère comments_labeled_binary_relabel.csv
  ```
- **Lancer l'application Streamlit** :
  ```bash
  streamlit run app.py
  ```

## Données attendues
- `archive/2026-01-08/comments_only.csv` : Fichier source des commentaires (colonne `commentaire`).
- `data/processed/comments_labeled_binary.csv` : Fichier labellisé d'origine.
- `comments_labeled_binary_relabel.csv` : Fichier labellisé équilibré généré par le script.

## Auteurs
- Projet initial, extraction et labeling : ghassen-7
- Génération du script de relabeling équilibré et robustesse : GitHub Copilot (GPT-4.1)

## Licence
MIT

## Contact
Pour toute question, ouvrir une issue sur le repo ou contacter ghassen-7 sur GitHub.
