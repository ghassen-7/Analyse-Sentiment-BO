import pandas as pd
from unidecode import unidecode
from emoji import demojize

# Charger ton CSV (avec colonnes "author" et "text")
df = pd.read_csv("comments.csv", encoding="utf-8")

# Concaténer @auteur + texte
df["author"] = df["author"].astype(str)
df["text"]   = df["text"].astype(str)
df["commentaire"] = df["author"].fillna("") + " " + df["text"].fillna("")

# Lexique positif
pos_words = [
    "merite","mérité","bravo","classe","monstre","incroyable","magnifique",
    "felicitation","félicitations","respect","legend","légende","gg","good","fort",
    "excellent","parfait","super","top","trop fort","merci","gentil","pleure","émouvant","exceptionnel","FETE","content","fière","indispensable"
]

# Emojis positifs (via demojize)
pos_emojis = {
    ":red_heart:", ":flexed_biceps:", ":clapping_hands:",
    ":fire:", ":thumbs_up:", ":partying_face:", ":party_popper:","👏🏾"
}

def label_binary(text):
    if not isinstance(text, str):
        return 0
    # Normaliser texte (sans accents)
    t = unidecode(text.lower())
    # Emojis convertis en tokens
    t_emo = demojize(text)

    # Si mot positif OU emoji positif détecté → 1
    if any(w in t for w in pos_words) or any(e in t_emo for e in pos_emojis):
        return 1
    else:
        return 0

# Appliquer
df["label"] = df["commentaire"].apply(label_binary)

# Sauvegarde
df[["commentaire","label"]].to_csv("comments_labeled_binary.csv", index=False, encoding="utf-8-sig")

# Résumé rapide
n_total = len(df)
n_pos = (df["label"] == 1).sum()
n_neg = n_total - n_pos
print(f"✅ {n_total} commentaires labellisés → comments_labeled_binary.csv")
print(f"   → {n_pos} positifs (1)")
print(f"   → {n_neg} négatifs/neutres (0)")
