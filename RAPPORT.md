# Rapport de Projet : Analyse de Sentiment sur les Commentaires YouTube (Ballon d'Or)

**Date :** 16 Janvier 2026  
**Sujet :** NLP / Machine Learning Supervisé  
**Langage :** Python (Scikit-Learn, Streamlit)  

---

## Table des Matières

1. [Introduction](#1-introduction)
2. [Collecte et Labellisation des Données](#2-collecte-et-labellisation-des-données)
3. [Préparation et Équilibrage](#3-préparation-et-équilibrage)
4. [Vectorisation et Feature Engineering](#4-vectorisation-et-feature-engineering)
5. [Modélisation et Algorithmes](#5-modélisation-et-algorithmes)
6. [Analyse des Résultats](#6-analyse-des-résultats)
7. [Industrialisation (Application Streamlit)](#7-industrialisation-application-streamlit)
8. [Conclusion et Perspectives](#8-conclusion-et-perspectives)

---

## 1. Introduction

### Contexte du Projet
Les réseaux sociaux, et en particulier YouTube, sont devenus des vecteurs majeurs de l'opinion publique. Dans le domaine du sport, l'attribution du **Ballon d'Or** suscite chaque année des débats passionnés. Ce projet vise à analyser automatiquement la polarité des commentaires (positif ou négatif) postés sous des vidéos traitant de cet événement.

### Problématique
Contrairement aux datasets académiques classiques (IMDb, Amazon Reviews), les données issues de YouTube sont "sauvages" :
*   **Absence de labels** : Personne n'a noté "Positif" ou "Négatif" à côté du commentaire.
*   **Bruit** : Fautes d'orthographe, argot ("cr7 the goat"), emojis, ironie.
*   **Déséquilibre** : Tendance naturelle à commenter pour critiquer ou aduler sans juste milieu.

L'objectif est de construire un **pipeline de Machine Learning complet**, allant de la constitution d'une Vérité Terrain (Ground Truth) jusqu'au déploiement d'une interface de prédiction, en passant par l'entraînement de modèles supervisés performants.

---

## 2. Collecte et Labellisation des Données

### 2.1 Extraction des Données
Nous avons utilisé l'**API YouTube Data v3** pour extraire les commentaires de vidéos virales sur le Ballon d'Or.
*   **Volume Brut** : 3 242 commentaires.
*   **Champs extraits** : Texte, Auteur, Date, Likes.

### 2.2 Stratégie de Labellisation (AI-Assisted Labeling)
Face à l'absence de labels manuels, nous avons opté pour une approche de **Transfer Learning** pour générer notre Vérité Terrain.
Nous avons utilisé le modèle **`cardiffnlp/twitter-xlm-roberta-base-sentiment`**, un Transformer (BERT-like) pré-entraîné sur ~198 millions de tweets dans 8 langues.

**Pourquoi ce choix ?**
*   Ce modèle comprend les spécificités du langage web (emojis, abréviations, hashtags), très proche du style des commentaires YouTube.
*   Il offre une fiabilité bien supérieure aux lexiques statiques (VADER ou TextBlob) pour capter le contexte.

**Résultat de la labellisation automatique :**
*   Positifs : 1 767 (Classe majoritaire)
*   Négatifs : 969
*   Neutres : 506

---

## 3. Préparation et Équilibrage

### 3.1 Problème du Déséquilibre
Le dataset initial présentait un fort déséquilibre en faveur des commentaires positifs.
Un entraînement direct sur ces données aurait biaisé les modèles : un algorithme "paresseux" prédisant toujours "Positif" obtiendrait une accuracy artificiellement élevée, mais échouerait à détecter les critiques (Recall faible sur la classe négative).

### 3.2 Stratégie Retenue
Nous avons appliqué une double stratégie pour obtenir un dataset binaire (Positif/Négatif) parfaitement équilibré :

1.  **Undersampling de la classe Majoritaire** : Réduction aléatoire du nombre de commentaires positifs pour s'aligner sur la classe minoritaire.
2.  **Récupération de Neutres** : Certains commentaires classés "Neutres" par le modèle (score ~0.5) contenaient en réalité des opinions polarisées. Nous avons converti les neutres pertinents pour enrichir la classe Négative.

### 3.3 Dataset Final (Processed)
*   **Total** : 1 938 observations.
*   **Positifs** : 969 (50%).
*   **Négatifs** : 969 (50%).
*   **Avantage** : Ce parfait équilibre permet d'utiliser l'Accuracy comme métrique fiable et garantit que le modèle accorde autant d'importance aux deux classes.

---

## 4. Vectorisation et Feature Engineering

Avant de nourrir les algorithmes, le texte brut doit être converti en vecteurs numériques.

### 4.1 Nettoyage
*   Suppression des caractères spéciaux non pertinents.
*   Gestion des emojis (conservés ou traduits selon les cas, ici traités comme des tokens par le Vectorizer si pertinents).

### 4.2 TF-IDF (Term Frequency - Inverse Document Frequency)
Nous avons choisi l'approche **TF-IDF** plutôt que le simple comptage (Bag of Words) pour pénaliser les mots très fréquents mais peu informatifs.

**Configuration Optimisée :**
*   **N-grammes (1, 2)** : Nous considérons les mots seuls ("nul") ET les paires de mots ("pas nul"). C'est crucial pour capter la négation.
*   **Min DF = 2** : Élimination des termes qui n'apparaissent qu'une seule fois (bruit/typos).
*   **Max DF = 0.9** : Élimination des mots apparaissant dans plus de 90% des documents (stopwords implicites).
*   **Stopwords** : Utilisation d'une liste personnalisée de mots vides français.

**Résultat** : Un vocabulaire riche de **5 009 features** (termes uniques) décrivant notre corpus.

---

## 5. Modélisation et Algorithmes

Nous avons comparé quatre familles d'algorithmes classiques de Scikit-Learn.
Le dataset a été séparé en **Train Set (80%)** et **Test Set (20%)** de manière stratifiée.

### 5.1 Les Candidats

1.  **Decision Tree (Arbre de Décision)**
    *   *Principe* : Suite de règles Si/Alors basées sur la présence de mots clés.
    *   *Avantage* : Très interprétable.
    *   *Inconvénient* : Tendance au sur-apprentissage (Overfitting) sur du texte.

2.  **Random Forest**
    *   *Principe* : Moyenne de multiples arbres de décision entraînés sur des sous-parties des données.
    *   *Avantage* : Réduit la variance de l'arbre simple, plus robuste.

3.  **Logistic Regression**
    *   *Principe* : Modèle linéaire probabiliste. Cherche une frontière linéaire pour séparer les classes.
    *   *Avantage* : Excellent baseline pour le NLP, performant sur les données éparses (sparse).

4.  **Linear SVC (Support Vector Classifier)**
    *   *Principe* : Cherche l'hyperplan qui maximise la marge entre les points positifs et négatifs.
    *   *Spécificité* : Optimisé pour les espaces de haute dimension (comme le TF-IDF avec 5000 dimensions).

---

## 6. Analyse des Résultats

### 6.1 Tableau Comparatif

| Modèle | Accuracy | Précision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **LinearSVC** | **83.51%** | 84.57% | 81.96% | **0.8325** |
| Logistic Regression | 83.25% | 84.86% | 80.93% | 0.8285 |
| Random Forest | 80.15% | 77.46% | 85.05% | 0.8108 |
| Decision Tree | 73.20% | 88.14% | 53.61% | 0.6667 |

*(Note : Résultats basés sur le Test Set de 388 échantillons)*

### 6.2 Interprétation
*   **Le Champion : LinearSVC**. Avec 83.5% d'accuracy, c'est le modèle le plus performant. Cela confirme la théorie : les données textuelles TF-IDF sont souvent linéairement séparables dans des espaces de haute dimension.
*   **Le Challenger : Logistic Regression**. Très proche du SVM, elle offre l'avantage de fournir des probabilités (degré de certitude), ce que le SVM standard ne fait pas directement.
*   **L'Échec de l'Arbre** : Le Decision Tree a un **Recall catastrophique (53%)**. Il rate près de la moitié des commentaires négatifs. Il a probablement "appris par cœur" des mots clés spécifiques du Train Set qui ne se retrouvent pas dans le Test Set (Overfitting).

---

## 7. Industrialisation : Application Streamlit

Pour rendre ces modèles utilisables par un utilisateur final, nous avons développé une application Web interactive avec **Streamlit**.

### 7.1 Fonctionnalités Clés
1.  **Dashboard des Performances** : Visualisation graphique des métriques (Accuracy, Temps d'entraînement) pour chaque modèle.
2.  **Explorateur de Données** : Tableaux filtrables pour lire les commentaires bruts et leurs labels.
3.  **Analyseur de Texte** : Zone de texte libre où l'utilisateur peut taper une phrase ("Ce joueur est surcoté") et voir en temps réel la prédiction du meilleur modèle (LinearSVC).
4.  **Nuages de Mots (WordClouds)** : Visualisation des termes les plus fréquents pour chaque classe (ex: "légende", "mérité" pour Positif vs "vol", "honte" pour Négatif).

---

## 8. Conclusion et Perspectives

### Bilan
Ce projet a permis de démontrer qu'il est possible d'atteindre une performance solide (**>83% d'accuracy**) sur des données YouTube bruitées, en combinant des techniques modernes de labellisation (RoBERTa) avec des algorithmes classiques robustes (SVM/Logistic Regression). L'étape cruciale a été **l'équilibrage des données**, sans lequel les modèles auraient été biaisés et inutilisables.

### Limites
*   **Ironie** : Les modèles linéaires (SVM) peinent encore à détecter l'ironie subtile ("Bravo pour ce vol !").
*   **Contexte** : Le modèle analyse le commentaire isolément, sans voir la vidéo ni les réponses précédentes.

### Pistes d'Amélioration
1.  **Fine-Tuning BERT** : Au lieu d'utiliser TF-IDF + SVM, nous pourrions ré-entraîner (fine-tune) une couche de classification sur CamemBERT ou FlauBERT spécifique à notre corpus. Cela améliorerait la compréhension du contexte.
2.  **Analyse Temporelle** : Visualiser l'évolution du sentiment jour après jour après la remise du trophée.
3.  **Filtrage du Spam** : Ajouter une étape de pré-traitement pour retirer les commentaires purement publicitaires ou hors-sujet.

---
*Projet réalisé dans le cadre du module Machine Learning / NLP.*
