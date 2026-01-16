# -*- coding: utf-8 -*-
"""
train_with_relabel.py
- Entraînement de modèles ML sur le dataset relabelisé équilibré
- Fichier source : comments_labeled_binary_relabel.csv (1938 commentaires 50/50)
- Modèles : DecisionTree, LogisticRegression, RandomForest, LinearSVC
- Sauvegarde : results_relabel.csv, predictions_relabel.csv
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

# 1) Chargement du dataset relabelisé équilibré
IN_FILE = "data/processed/comments_labeled_binary.csv"
if not os.path.exists(IN_FILE):
    raise FileNotFoundError(f"❌ Fichier introuvable : {IN_FILE}")

print(f"📊 Chargement du dataset relabelisé : {IN_FILE}")
df = pd.read_csv(IN_FILE, sep=";", encoding="utf-8-sig")
df.columns = [c.replace("\ufeff","").strip() for c in df.columns]

if not {"commentaire","label"}.issubset(df.columns):
    raise SystemExit(f"❌ Le fichier doit contenir 'commentaire' et 'label'. Colonnes lues: {list(df.columns)}")

df["commentaire"] = df["commentaire"].astype(str).str.strip()
df["label"] = df["label"].astype(int)

print(f"✅ {len(df)} commentaires chargés")
print(f"📈 Répartition des labels : {df['label'].value_counts().to_dict()}")

# 2) Split train/test (80/20) avec stratification
X_text = df["commentaire"].values
y = df["label"].values

X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🔀 Split : {len(X_train_txt)} train, {len(X_test_txt)} test")

# 3) Vectorisation TF-IDF (1-2 grammes, min_df=2, max_df=0.9)
print("🔤 Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(
    analyzer="word",
    lowercase=True,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)
X_train = vectorizer.fit_transform(X_train_txt)
X_test = vectorizer.transform(X_test_txt)
print(f"✅ Vocabulaire : {len(vectorizer.vocabulary_)} termes")

# 4) Entraînement de plusieurs modèles
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    "LogisticRegression": LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=42),
    "LinearSVC": LinearSVC(C=1.0, max_iter=5000, random_state=42),
}

results = []
print("\n🤖 Entraînement des modèles...")
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

# 5) Sauvegarde des résultats
out_results = "reports/results_relabel.csv"
os.makedirs("reports", exist_ok=True)
pd.DataFrame(results).to_csv(out_results, index=False, sep=";", encoding="utf-8-sig")
print(f"\n✅ Résultats sauvegardés : {out_results}")

# 6) Sauvegarde des prédictions du meilleur modèle (ou DecisionTree pour cohérence)
tree = models["DecisionTree"]
y_pred_tree = tree.predict(X_test)
out_preds = pd.DataFrame({
    "commentaire": X_test_txt,
    "y_true": y_test,
    "y_pred": y_pred_tree
})
out_preds_file = "reports/predictions_relabel.csv"
out_preds.to_csv(out_preds_file, index=False, sep=";", encoding="utf-8-sig")
print(f"✅ Prédictions sauvegardées : {out_preds_file}")

# 7) Affichage du résumé
print("\n" + "="*60)
print("RÉSUMÉ DES PERFORMANCES")
print("="*60)
for r in results:
    print(f"{r['model']:20s} | Acc: {r['accuracy']:.4f} | Prec: {r['precision']:.4f} | Rec: {r['recall']:.4f} | F1: {r['f1_score']:.4f}")
print("="*60)
print(f"\n🎉 Entraînement terminé avec succès sur le dataset relabelisé !")
