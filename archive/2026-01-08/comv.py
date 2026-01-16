# hugg_min_binary.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# 1) Lire les commentaires
#   - comments_only.csv avec colonne "commentaire"
#   - sinon comments.csv avec "author","text"
try:
    df = pd.read_csv("comments_only.csv", encoding="utf-8")
    texts = df["commentaire"].astype(str).tolist()
except FileNotFoundError:
    df2 = pd.read_csv("comments.csv", encoding="utf-8")
    df2["author"] = df2["author"].astype(str)
    df2["text"]   = df2["text"].astype(str)
    df2["commentaire"] = df2["author"].fillna("") + " " + df2["text"].fillna("")
    texts = df2["commentaire"].tolist()
    df = df2[["commentaire"]].copy()

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# 2) Charger le modèle (préférer safetensors) et le tokenizer "slow" pour éviter les soucis
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=False)

pipe = TextClassificationPipeline(
    model=mdl,
    tokenizer=tok,
    truncation=True,        # évite l'avertissement
    max_length=256,         # coupe proprement
    batch_size=64,          # ajuste selon ta RAM CPU
                 # renvoie un dict {"label","score"}
)

# 3) Inférence
preds = pipe(texts)
labels_bin = [1 if p["label"].lower() == "positive" else 0 for p in preds]

# 4) Map en binaire
def to_bin(p):
    if isinstance(p, list):
        best = max(p, key=lambda x: x["score"])
        label = best["label"].lower()
    else:
        label = p["label"].lower()
    return 1 if label == "positive" else 0

labels_bin = [to_bin(p) for p in preds]

# 5) Sauvegarde + résumé
out = "comments_labeled_binary_hf.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

n_total = len(df)
n_pos = int(sum(labels_bin))
n_neg = n_total - n_pos
print(f"✅ {n_total} commentaires labellisés → {out}")
print(f"   → {n_pos} positifs (1)")
print(f"   → {n_neg} négatifs/neutres (0)")
