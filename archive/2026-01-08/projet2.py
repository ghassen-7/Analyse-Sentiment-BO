import json, csv, sys
from pathlib import Path

info = json.load(open("VKancdDIpOU.info.json", "r", encoding="utf-8"))
comments = info.get("comments", [])
with open("comments.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["id","author","text","like_count","timestamp","parent"])
    w.writeheader()
    for c in comments:
        w.writerow({
            "id": c.get("id"),
            "author": c.get("author"),
            "text": c.get("text"),
            "like_count": c.get("like_count"),
            "timestamp": c.get("timestamp"),
            "parent": c.get("parent"),
        })
print("OK → comments.csv")
