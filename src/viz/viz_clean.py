# -*- coding: utf-8 -*-
"""
viz_clean.py
- Visualisations simples et reproductibles pour le dataset labellisé
- Lit data/processed/comments_labeled_binary.csv (commentaire;label)
- Sauve des figures dans figs/
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.utils import resample

# Chemins robustes basés sur l'emplacement du script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IN_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'comments_labeled_binary.csv')
OUT_DIR = os.path.join(BASE_DIR, 'figs')

os.makedirs(OUT_DIR, exist_ok=True)

# Lecture
if not os.path.exists(IN_FILE):
    raise SystemExit(f"Fichier introuvable: {IN_FILE}\nChemin courant: {os.path.abspath(os.getcwd())}\nAssurez-vous que 'comments_labeled_binary.csv' est présent dans data/processed/")

df = pd.read_csv(IN_FILE, sep=';', encoding='utf-8-sig')
df.columns = [c.replace('\ufeff','').strip() for c in df.columns]
if not {"commentaire", "label"}.issubset(df.columns):
    raise SystemExit(f"Le fichier doit contenir 'commentaire' et 'label'. Colonnes lues: {list(df.columns)}")

df['commentaire'] = df['commentaire'].astype(str).str.strip()

# Liste minimale de stopwords français — enrichissable
FRENCH_STOPWORDS = {
    'alors','au','aucuns','aussi','autre','avant','avec','avoir','bon','car','ce','cela','ces',
    'ceux','chaque','ci','comme','comment','dans','des','du','dedans','dehors','depuis','devrait',
    'doit','donc','dos','droite','début','elle','elles','en','encore','essai','est','et','eu',
    'fait','faites','fois','font','force','haut','hors','ici','il','ils','je','juste','la','le',
    'les','leur','là','ma','maintenant','mais','mes','mine','moins','mon','mot','même','ni','nom',
    'nos','notre','nous','nouveaux','ou','où','par','parce','parole','pas','personnes','peut','peu',
    'pièce','plupart','pour','pourquoi','quand','que','quel','quelle','quelles','quels','qui','sa',
    'sans','ses','seulement','si','sien','son','sont','sous','sur','ta','tandis','tellement','tes',
    'ton','tous','tout','trop','très','tu','un','une','va','voient','vont','votre','vous','ça','étaient',
    'état','étions','été','être','de'
}

# Répartition des labels (brut)
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title('Répartition des labels (dataset relabelisé équilibré)')
plt.xlabel('Label (0=négatif, 1=positif)')
plt.ylabel('Nombre de commentaires')
plt.savefig(os.path.join(OUT_DIR, 'label_counts.png'), bbox_inches='tight')
plt.close()
print(f"✓ label_counts.png sauvegardé")

# Pas besoin d'équilibrage artificiel puisque le dataset est déjà équilibré
# On crée quand même les autres noms de fichiers pour compatibilité
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title('Répartition des labels (brut)')
plt.xlabel('Label (0=négatif, 1=positif)')
plt.ylabel('Nombre de commentaires')
plt.savefig(os.path.join(OUT_DIR, 'label_counts_raw.png'), bbox_inches='tight')
plt.close()
print(f"✓ label_counts_raw.png sauvegardé")

plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title('Répartition des labels (équilibré)')
plt.xlabel('Label (0=négatif, 1=positif)')
plt.ylabel('Nombre de commentaires')
plt.savefig(os.path.join(OUT_DIR, 'label_counts_balanced.png'), bbox_inches='tight')
plt.close()
print(f"✓ label_counts_balanced.png sauvegardé")

# TF-IDF global (1-2 grammes) avec stopwords
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.9, stop_words=list(FRENCH_STOPWORDS))
X = vectorizer.fit_transform(df['commentaire'].values)
feature_names = np.array(vectorizer.get_feature_names_out())

# Top termes globaux
tfidf_sum = np.array(X.sum(axis=0)).ravel()
top_idx = tfidf_sum.argsort()[::-1][:30]
top_terms = feature_names[top_idx]
top_vals = tfidf_sum[top_idx]

plt.figure(figsize=(8,6))
sns.barplot(x=top_vals[:20], y=top_terms[:20])
plt.title('Top 20 termes TF-IDF (global)')
plt.xlabel('Somme TF-IDF')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'top_terms.png'))
plt.close()
print(f"✓ top_terms.png sauvegardé")

# Wordclouds par label
for label in sorted(df['label'].unique()):
    texts = df.loc[df['label'] == label, 'commentaire'].astype(str).values
    text_blob = "\n".join(texts)
    wc = WordCloud(width=800, height=400, background_color='white', collocations=False, stopwords=FRENCH_STOPWORDS)
    wc.generate(text_blob)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    label_name = 'négatif' if label == 0 else 'positif'
    plt.title(f'Wordcloud - Commentaires {label_name} (label={label})')
    outpath = os.path.join(OUT_DIR, f'wordcloud_label_{label}.png')
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"✓ wordcloud_label_{label}.png sauvegardé")

# Histogramme des longueurs des commentaires
lengths = df['commentaire'].str.split().apply(len)
plt.figure(figsize=(8,5))
sns.histplot(lengths, bins=40, kde=True)
plt.xlabel('Longueur (nombre de tokens)')
plt.ylabel('Fréquence')
plt.title('Distribution des longueurs des commentaires')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'length_distribution.png'))
plt.close()
print(f"✓ length_distribution.png sauvegardé")

# Top n-grammes par label (0 vs 1)
# Vectorisation séparée par classe pour extraire les termes distinctifs
for lab in sorted(df['label'].unique()):
    texts_lab = df.loc[df['label'] == lab, 'commentaire'].values
    vec_lab = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.9, stop_words=list(FRENCH_STOPWORDS))
    X_lab = vec_lab.fit_transform(texts_lab)
    feats_lab = np.array(vec_lab.get_feature_names_out())
    scores_lab = np.array(X_lab.sum(axis=0)).ravel()
    order = scores_lab.argsort()[::-1]
    top_feats = feats_lab[order][:20]
    top_vals = scores_lab[order][:20]

    label_name = 'négatifs' if lab == 0 else 'positifs'
    plt.figure(figsize=(8,6))
    sns.barplot(x=top_vals, y=top_feats)
    plt.title(f'Top 20 termes TF-IDF - Commentaires {label_name} (label={lab})')
    plt.xlabel('Somme TF-IDF')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'top_terms_label_{lab}.png'))
    plt.close()
    print(f"✓ top_terms_label_{lab}.png sauvegardé")

print(f"\n[OK] Toutes les figures enregistrées dans: {os.path.abspath(OUT_DIR)}")
print(f"[INFO] Total de commentaires: {len(df)}, Répartition: {df['label'].value_counts().to_dict()}")
