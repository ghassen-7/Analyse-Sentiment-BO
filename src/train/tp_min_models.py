# -*- coding: utf-8 -*-
"""
tp_min_models.py
- Lecture du dataset comments_labeled_binary.csv (commentaire;label)
- Vectorisation TF-IDF (1-2 grammes)
- Entrainement: DecisionTree, LogisticRegression (L2), LinearSVC
- Sauvegarde: tp_min_results.csv, tp_test_predictions_tree.csv
- Affichage: top-mots pro/anti de la Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

IN_FILE = "data/processed/comments_labeled_binary.csv"   # colonnes: commentaire ; label

# 1) Lecture robuste (BOM) + contrôle colonnes
df = pd.read_csv(IN_FILE, sep=";", encoding="utf-8-sig")
df.columns = [c.replace("\ufeff","").strip() for c in df.columns]
if not {"commentaire","label"}.issubset(df.columns):
    raise SystemExit(f"Le fichier doit contenir 'commentaire' et 'label'. Colonnes lues: {list(df.columns)}")

df["commentaire"] = df["commentaire"].astype(str).str.strip()
df["label"] = df["label"].astype(int)

X_text = df["commentaire"].values
y = df["label"].values

# 2) Vectorisation sac-de-mots (comme dans le TP)
# (Réorganisation) : on split AVANT la vectorisation pour pouvoir équilibrer le train

# Nouveau split 80/20 sur textes (reproductible)
txt_train, txt_test, y_train, y_test = train_test_split(
    df["commentaire"].values, y, test_size=0.2, random_state=42, stratify=y
)

# Équilibrage du train par oversampling de la classe minoritaire
from sklearn.utils import resample
from collections import Counter

train_df = pd.DataFrame({"commentaire": txt_train, "label": y_train})
print("Distribution des labels (train avant équilibrage):", Counter(train_df["label"]))
counts = train_df["label"].value_counts()
min_class = counts.idxmin()
max_count = counts.max()
minor_df = train_df[train_df["label"] == min_class]
major_df = train_df[train_df["label"] != min_class]
minor_upsampled = resample(minor_df, replace=True, n_samples=max_count, random_state=42)
train_balanced = pd.concat([major_df, minor_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Distribution des labels (train après équilibrage):", Counter(train_balanced["label"]))

X_train_texts = train_balanced["commentaire"].values
y_train = train_balanced["label"].values
X_test_texts = txt_test

vectorizer = TfidfVectorizer(analyzer="word",
                             lowercase=True,
                             ngram_range=(1,2),   # uni + bi-grammes
                             min_df=2,
                             max_df=0.9)
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# 4) Trois modèles (avec class_weight='balanced' quand applicable)
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced"),
    "LinearSVC": LinearSVC(C=1.0, max_iter=5000, class_weight="balanced"),
}

results = []
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy = {acc:.4f}")
    print("Confusion matrix:"); print(confusion_matrix(y_test, y_pred))
    print("Classification report:"); print(classification_report(y_test, y_pred, digits=4))
    results.append((name, acc))

# 5) Sauvegarde résultats + 15 exemples pour l'Arbre (comme le TP)
pd.DataFrame(results, columns=["model","accuracy"]).to_csv(
    "reports/tp_min_results.csv", index=False, sep=";", encoding="utf-8-sig"
)

tree = models["DecisionTree"]
y_pred_tree = tree.predict(X_test)
out = pd.DataFrame({"commentaire": txt_test, "y_true": y_test, "y_pred_tree": y_pred_tree})
out.to_csv("reports/tp_test_predictions_tree.csv", index=False, sep=";", encoding="utf-8-sig")

print("\n[OK] Résultats enregistrés: reports/tp_min_results.csv, reports/tp_test_predictions_tree.csv")
