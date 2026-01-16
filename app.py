"""
Application Streamlit pour l'analyse de sentiment YouTube
Centralise les résultats de modèles, les prédictions et les visualisations
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuration Streamlit
st.set_page_config(
    page_title="Analyse Sentiment YouTube",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTES - Chemins réels du projet
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGS_DIR = PROJECT_ROOT / "figs"

# Fichiers attendus
DATASET_PATH = DATA_DIR / "comments_labeled_binary.csv"
RESULTS_PATH = REPORTS_DIR / "tp_min_results.csv"
EXAMPLES_TREE_PATH = REPORTS_DIR / "tp_test_predictions_tree.csv"

# Images attendues
EXPECTED_IMAGES = {
    "label_counts.png": "Répartition des labels (graphique équilibré)",
    "label_counts_balanced.png": "Répartition des labels - Version équilibrée",
    "label_counts_raw.png": "Répartition des labels - Données brutes",
    "length_distribution.png": "Distribution de la longueur des commentaires",
    "top_terms.png": "Top 20 termes les plus fréquents",
    "top_terms_label_0.png": "Top 20 termes - Commentaires négatifs (classe 0)",
    "top_terms_label_1.png": "Top 20 termes - Commentaires positifs (classe 1)",
    "wordcloud_label_0.png": "Nuage de mots - Commentaires négatifs (classe 0)",
    "wordcloud_label_1.png": "Nuage de mots - Commentaires positifs (classe 1)",
}

# ============================================================================
# FONCTIONS UTILITAIRES DE CHARGEMENT
# ============================================================================

@st.cache_data(ttl=300)
def load_dataset():
    """
    Charge le dataset de commentaires labelisés.
    Retourne un DataFrame ou None en cas d'erreur.
    """
    if not DATASET_PATH.exists():
        return None
    
    try:
        df = pd.read_csv(
            DATASET_PATH,
            sep=";",
            encoding="utf-8-sig"
        )
        # Vérifier les colonnes attendues
        if "commentaire" not in df.columns or "label" not in df.columns:
            st.error(f"❌ Colonnes manquantes. Attendues : 'commentaire', 'label'. Trouvées : {list(df.columns)}")
            return None
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def load_results():
    """
    Charge les résultats des modèles (accuracies).
    Retourne un DataFrame ou None en cas d'erreur.
    """
    if not RESULTS_PATH.exists():
        return None
    
    try:
        df = pd.read_csv(
            RESULTS_PATH,
            sep=";",
            encoding="utf-8-sig"
        )
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def load_examples_tree():
    """
    Charge les exemples et prédictions du modèle Arbre de Décision.
    Retourne un DataFrame ou None en cas d'erreur.
    """
    if not EXAMPLES_TREE_PATH.exists():
        return None
    
    try:
        df = pd.read_csv(
            EXAMPLES_TREE_PATH,
            sep=";",
            encoding="utf-8-sig"
        )
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def load_image(image_name):
    """
    Charge une image depuis le dossier figs/.
    Retourne le chemin complet ou None si absent.
    """
    image_path = FIGS_DIR / image_name
    if image_path.exists():
        return image_path
    return None

# ============================================================================
# FONCTIONS UTILITAIRES - TRAITEMENT
# ============================================================================

def confusion_table(y_true, y_pred):
    """
    Calcule et retourne une matrice de confusion (2×2) formatée.
    Entrée : deux pandas Series avec valeurs 0/1.
    Sortie : DataFrame formaté pour affichage.
    """
    cm = pd.crosstab(
        y_true,
        y_pred,
        rownames=["Réel"],
        colnames=["Prédit"],
        margins=False
    )
    # Renommer les index pour clarté
    cm.index = [f"Classe {i}" for i in cm.index]
    cm.columns = [f"Classe {i}" for i in cm.columns]
    return cm

def filter_df(df, query=None, label_filter=None):
    """
    Filtre le DataFrame selon une requête texte et/ou un label.
    
    Params:
    - df : DataFrame source
    - query : chaîne à chercher dans 'commentaire' (insensible à la casse)
    - label_filter : None, 0 ou 1 pour filtrer par label
    
    Retour : DataFrame filtré (max 500 lignes)
    """
    result = df.copy()
    
    # Filtre texte
    if query and query.strip():
        try:
            result = result[
                result["commentaire"].str.contains(
                    query,
                    case=False,
                    na=False,
                    regex=False
                )
            ]
        except Exception as e:
            st.warning(f"⚠️ Erreur lors du filtrage texte : {e}")
    
    # Filtre label
    if label_filter is not None:
        result = result[result["label"] == label_filter]
    
    # Limiter à 500 lignes pour la performance
    return result.head(500)

# ============================================================================
# SECTION 1 : VUE D'ENSEMBLE
# ============================================================================

def page_overview():
    """Affiche un résumé du dataset et permet le téléchargement."""
    st.title("📊 Vue d'ensemble")
    
    df = load_dataset()
    
    if df is None:
        st.warning(f"⚠️ Impossible de charger le dataset. Fichier attendu : {DATASET_PATH}")
        return
    
    # Statistiques générales
    st.subheader("Statistiques du dataset")
    col1, col2, col3 = st.columns(3)
    
    total_comments = len(df)
    label_counts = df["label"].value_counts().sort_index()
    
    with col1:
        st.metric("Total commentaires", total_comments)
    
    with col2:
        pct_0 = (label_counts.get(0, 0) / total_comments * 100) if total_comments > 0 else 0
        st.metric("Classe 0 (négatif)", f"{label_counts.get(0, 0)} ({pct_0:.1f}%)")
    
    with col3:
        pct_1 = (label_counts.get(1, 0) / total_comments * 100) if total_comments > 0 else 0
        st.metric("Classe 1 (positif)", f"{label_counts.get(1, 0)} ({pct_1:.1f}%)")
    
    st.divider()
    
    # Afficher les 5 premières lignes
    st.subheader("Aperçu des données (5 premières lignes)")
    display_df = df[["commentaire", "label"]].head(5).copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)
    
    st.divider()
    
    # Bouton de téléchargement CSV
    st.subheader("Télécharger le dataset")
    csv_data = df.to_csv(index=False, sep=";", encoding="utf-8-sig")
    st.download_button(
        label="📥 Télécharger en CSV (complet)",
        data=csv_data,
        file_name="comments_labeled_binary.csv",
        mime="text/csv"
    )

# ============================================================================
# SECTION 2 : RÉSULTATS MODÈLES
# ============================================================================

def page_model_results():
    """Affiche les résultats des modèles (accuracies)."""
    st.title("🎯 Résultats des modèles")
    
    results_df = load_results()
    
    if results_df is None:
        st.warning(f"⚠️ Impossible de charger les résultats. Fichier attendu : {RESULTS_PATH}")
        return
    
    # Afficher le tableau des résultats
    st.subheader("Tableau des accuracies")
    st.dataframe(results_df, use_container_width=True)
    
    st.divider()
    
    # Créer et afficher le graphique des accuracies
    st.subheader("Graphique des accuracies")
    
    # Supposer colonnes "model" et "accuracy"
    if "model" in results_df.columns and "accuracy" in results_df.columns:
        try:
            # Afficher les accuracies sous forme de colonnes (colonnes latérales)
            cols = st.columns(len(results_df))
            for idx, (col, row) in enumerate(zip(cols, results_df.itertuples(index=False))):
                with col:
                    st.metric(
                        label=row.model,
                        value=f"{row.accuracy:.4f}",
                        delta=f"{row.accuracy * 100:.2f}%"
                    )
        except Exception as e:
            st.error(f"❌ Erreur lors de la création du graphique : {e}")
    else:
        st.warning("⚠️ Colonnes 'model' et/ou 'accuracy' non trouvées dans le fichier de résultats.")
    
    st.divider()
    
    # Notes d'interprétation
    st.subheader("📝 Notes d'interprétation")
    st.info(
        "Les modèles ont été entraînés et évalués sur un ensemble de test. "
        "L'accuracy indique le pourcentage de prédictions correctes. "
        "Comparer les modèles pour identifier le plus performant."
    )

# ============================================================================
# SECTION 3 : EXEMPLES ARBRE DE DÉCISION
# ============================================================================

def page_tree_examples():
    """Affiche les prédictions du modèle Arbre de Décision avec matrice de confusion."""
    st.title("🌳 Exemples Arbre de Décision (test)")
    
    df = load_examples_tree()
    
    if df is None:
        st.warning(f"⚠️ Impossible de charger les exemples. Fichier attendu : {EXAMPLES_TREE_PATH}")
        return
    
    # Vérifier les colonnes attendues
    expected_cols = ["commentaire", "y_true", "y_pred_tree"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colonnes manquantes : {missing_cols}. Trouvées : {list(df.columns)}")
        return
    
    st.subheader("Échantillon filtrable")
    
    # Slider pour sélectionner le nombre de lignes
    max_samples = len(df)
    n_samples = st.slider(
        "Nombre de lignes à afficher",
        min_value=1,
        max_value=min(max_samples, 100),
        value=min(20, max_samples)
    )
    
    # Afficher l'échantillon
    display_df = df[["commentaire", "y_true", "y_pred_tree"]].head(n_samples).copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)
    
    st.divider()
    
    # Matrice de confusion
    st.subheader("Matrice de confusion (2×2)")
    cm = confusion_table(df["y_true"], df["y_pred_tree"])
    st.dataframe(cm, use_container_width=True)
    
    st.divider()
    
    # Bouton de téléchargement du sous-ensemble
    csv_data = display_df.to_csv(index=False, sep=";", encoding="utf-8-sig")
    st.download_button(
        label="📥 Télécharger l'échantillon en CSV",
        data=csv_data,
        file_name="tree_predictions_sample.csv",
        mime="text/csv"
    )

# ============================================================================
# SECTION 4 : FIGURES
# ============================================================================

def page_figures():
    """Affiche les visualisations générées (wordclouds, feature importance, etc.)."""
    st.title("📈 Figures et visualisations")
    
    st.subheader("Images générées")
    
    found_any = False
    for image_name, description in EXPECTED_IMAGES.items():
        image_path = load_image(image_name)
        
        if image_path:
            st.image(str(image_path), caption=description, use_container_width=True)
            found_any = True
        else:
            st.info(f"ℹ️ Image absente : {image_name}")
    
    if not found_any:
        st.warning(f"⚠️ Aucune image trouvée dans le dossier {FIGS_DIR}")

# ============================================================================
# SECTION 5 : EXPLORATION TEXTE
# ============================================================================

def page_text_exploration():
    """Permet de filtrer les commentaires par texte et label."""
    st.title("🔍 Exploration texte")
    
    df = load_dataset()
    
    if df is None:
        st.warning(f"⚠️ Impossible de charger le dataset. Fichier attendu : {DATASET_PATH}")
        return
    
    st.subheader("Filtres")
    st.info("⚠️ Le filtrage par classe se fait sur les **vrais labels** du dataset (labels manuels), pas sur les prédictions du modèle.")
    
    # Recherche texte
    search_query = st.text_input(
        "Chercher un mot ou expression dans les commentaires",
        placeholder="ex : excellent, terrible, etc."
    )
    
    # Filtre label
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_all = st.checkbox("Tous les labels", value=True)
    with col2:
        filter_positive = st.checkbox("Seulement positifs - classe 1 (vrais labels)")
    with col3:
        filter_negative = st.checkbox("Seulement négatifs - classe 0 (vrais labels)")
    
    # Déterminer le filtre label
    label_filter = None
    if filter_all:
        label_filter = None
    elif filter_positive:
        label_filter = 1
    elif filter_negative:
        label_filter = 0
    
    st.divider()
    
    # Appliquer les filtres
    filtered_df = filter_df(df, query=search_query, label_filter=label_filter)
    
    st.subheader(f"Résultats (max 500 lignes) - {len(filtered_df)} commentaires trouvés")
    
    if len(filtered_df) > 0:
        display_df = filtered_df[["commentaire", "label"]].copy()
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df, use_container_width=True)
        
        st.divider()
        
        # Téléchargement du sous-ensemble filtré
        csv_data = filtered_df.to_csv(index=False, sep=";", encoding="utf-8-sig")
        st.download_button(
            label="📥 Télécharger le sous-ensemble filtré en CSV",
            data=csv_data,
            file_name="filtered_comments.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Aucun commentaire ne correspond aux critères de filtrage.")

# ============================================================================
# PAGE D'ACCUEIL
# ============================================================================

def page_home():
    """Page d'accueil avec instructions."""
    st.title("🎬 Analyse Sentiment YouTube")
    
    st.markdown("""
    Bienvenue ! Cette application centralise l'analyse de sentiment de commentaires YouTube.
    
    ### 📋 Comment utiliser
    
    1. **Vue d'ensemble** : Consultez les statistiques du dataset et téléchargez les données
    2. **Résultats modèles** : Comparez les performances des modèles
    3. **Arbre de décision** : Explorez les prédictions et la matrice de confusion
    4. **Figures** : Visualisez les analyses (wordclouds, feature importance)
    5. **Exploration texte** : Recherchez des commentaires spécifiques
    
    ### 🚀 Lancement local
    
    Si vous n'avez pas encore lancé l'application, exécutez :
    
    ```bash
    # Installation de Streamlit (si non installé)
    pip install streamlit pandas numpy
    
    # Lancement de l'application
    streamlit run app.py
    ```
    
    L'application s'ouvrira dans votre navigateur par défaut.
    
    ### ⚙️ Configuration
    
    - **Environnement** : Python 3.10 avec venv `.venv`
    - **Données** : `data/processed/comments_labeled_binary.csv`
    - **Rapports** : Fichiers CSV dans `reports/`
    - **Figures** : Images PNG dans `figs/`
    
    ### 🛡️ Notes de robustesse
    
    - Si un fichier manque, l'application affiche un message d'alerte clair
    - Les données sont mises en cache (TTL : 300 secondes)
    - Aucun appel réseau externe
    - Design épuré, thème clair/sombre natif Streamlit
    
    ---
    
    Utilisez le menu latéral pour naviguer entre les sections.
    """)

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale : gère la navigation et l'affichage."""
    
    # Barre latérale avec navigation
    with st.sidebar:
        st.title("🎯 Navigation")
        
        page = st.radio(
            "Sélectionnez une section",
            [
                "Accueil",
                "Vue d'ensemble",
                "Résultats modèles",
                "Arbre de décision",
                "Figures",
                "Exploration texte"
            ],
            index=0
        )
        
        st.divider()
        st.markdown("""
        **À propos**
        
        Application Streamlit pour l'analyse de sentiment YouTube.
        Centralize les modèles, prédictions et visualisations.
        """)
    
    # Affichage de la page sélectionnée
    if page == "Accueil":
        page_home()
    elif page == "Vue d'ensemble":
        page_overview()
    elif page == "Résultats modèles":
        page_model_results()
    elif page == "Arbre de décision":
        page_tree_examples()
    elif page == "Figures":
        page_figures()
    elif page == "Exploration texte":
        page_text_exploration()

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    main()
