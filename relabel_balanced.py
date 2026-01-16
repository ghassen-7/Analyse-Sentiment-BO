import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import os

# Charger les commentaires
comments_path = "archive/2026-01-08/comments_only.csv"
if not os.path.exists(comments_path):
    raise FileNotFoundError(f"❌ Fichier introuvable : {comments_path}")
df = pd.read_csv(comments_path, encoding="utf-8")
print(f"📊 Chargement de {len(df)} commentaires depuis {comments_path}")
texts = df["commentaire"].astype(str).tolist()

# Modèle HuggingFace multilingue
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
print(f"🤖 Chargement du modèle {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=False)
pipe = TextClassificationPipeline(
    model=mdl,
    tokenizer=tok,
    truncation=True,
    max_length=256,
    batch_size=64,
)
print("✅ Modèle chargé avec succès")

# Prédiction
print("🔮 Prédiction du sentiment pour tous les commentaires...")
preds = pipe(texts)
labels = [p["label"].lower() for p in preds]
print("✅ Prédictions terminées")

# Séparation
pos_idx = [i for i, l in enumerate(labels) if l == "positive"]
neg_idx = [i for i, l in enumerate(labels) if l == "negative"]
neutre_idx = [i for i, l in enumerate(labels) if l == "neutral"]
print(f"📈 Répartition initiale : {len(pos_idx)} positifs, {len(neg_idx)} négatifs, {len(neutre_idx)} neutres")

# Équilibrage
n_target = min(len(pos_idx), len(neg_idx))
print(f"⚖️  Équilibrage : objectif de {n_target} exemples par classe")
# Si besoin, compléter la classe minoritaire avec des neutres
if len(pos_idx) < len(neg_idx):
    needed = n_target - len(pos_idx)
    pos_idx += neutre_idx[:needed]
    print(f"   → Ajout de {needed} neutres aux positifs")
elif len(neg_idx) < len(pos_idx):
    needed = n_target - len(neg_idx)
    neg_idx += neutre_idx[:needed]
    print(f"   → Ajout de {needed} neutres aux négatifs")

# Limiter à n_target pour chaque classe
pos_idx = pos_idx[:n_target]
neg_idx = neg_idx[:n_target]

# Construction du DataFrame équilibré
sel_idx = pos_idx + neg_idx
labels_bin = [1]*n_target + [0]*n_target
df_bal = df.iloc[sel_idx].copy()
df_bal["label"] = labels_bin

# Sauvegarde
out_path = "comments_labeled_binary_relabel.csv"
df_bal[["commentaire","label"]].to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
print(f"\n✅ {len(df_bal)} commentaires équilibrés (binaire) → {out_path}")
print(f"   → {n_target} positifs (1), {n_target} négatifs (0)")
print(f"   → Équilibre parfait : 50/50")
