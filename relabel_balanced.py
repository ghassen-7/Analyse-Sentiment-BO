import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Charger les commentaires
comments_path = "archive/2026-01-08/comments_only.csv"
df = pd.read_csv(comments_path, encoding="utf-8")
texts = df["commentaire"].astype(str).tolist()

# Modèle HuggingFace multilingue
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=False)
pipe = TextClassificationPipeline(
    model=mdl,
    tokenizer=tok,
    truncation=True,
    max_length=256,
    batch_size=64,
)

# Prédiction
preds = pipe(texts)
labels = [p["label"].lower() for p in preds]

# Séparation
pos_idx = [i for i, l in enumerate(labels) if l == "positive"]
neg_idx = [i for i, l in enumerate(labels) if l == "negative"]
neutre_idx = [i for i, l in enumerate(labels) if l == "neutral"]

# Équilibrage
n_target = min(len(pos_idx), len(neg_idx))
# Si besoin, compléter la classe minoritaire avec des neutres
if len(pos_idx) < len(neg_idx):
    pos_idx += neutre_idx[:n_target - len(pos_idx)]
elif len(neg_idx) < len(pos_idx):
    neg_idx += neutre_idx[:n_target - len(neg_idx)]

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
df_bal[["commentaire","label"]].to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"✅ {len(df_bal)} commentaires équilibrés (binaire) → {out_path}")
print(f"   → {n_target} positifs (1), {n_target} négatifs (0)")
