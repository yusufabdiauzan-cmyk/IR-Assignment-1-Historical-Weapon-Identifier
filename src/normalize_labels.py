import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "weapons_dataset_expanded.csv"
OUTPUT = ROOT / "data" / "weapons_dataset_normalized.csv"

LABEL_MAP = {
    # European swords
    "longsword": "longsword",
    "long sword": "longsword",

    "arming sword": "arming_sword",

    "rapier": "rapier",

    "estoc": "estoc",

    "claymore": "claymore",

    "falchion": "falchion",

    "zweihander": "zweihander",

    # Japanese
    "katana": "katana",

    "wakizashi": "wakizashi",

    "tanto": "tanto",

    "odachi": "odachi",

    "tachi": "tachi",

    # Chinese
    "jian": "jian",

    "dao": "dao",

    # Polearms
    "halberd": "halberd",

    "glaive": "glaive",

    "spear": "spear",

    "pike": "pike",

    # Axes
    "battle axe": "battle_axe",

    "throwing axe": "throwing_axe",

    # Blunt
    "mace": "mace",

    "war hammer": "war_hammer",

    "flail": "flail",

    # fallback generic
    "sword": "sword",
}

NORMALIZATION_RULES = {
    # European swords
    "longsword": "longsword",
    "arming sword": "arming_sword",
    "rapier": "rapier",
    "estoc": "estoc",
    "falchion": "falchion",
    "claymore": "claymore",

    # Japanese
    "katana": "katana",
    "wakizashi": "wakizashi",
    "tanto": "tanto",
    "odachi": "odachi",
    "tachi": "tachi",

    # Chinese
    "jian": "jian",
    "dao": "dao",

    # Polearms
    "halberd": "halberd",
    "glaive": "glaive",
    "spear": "spear",
    "pike": "pike",

    # Axes
    "battle axe": "battle_axe",
    "throwing axe": "throwing_axe",

    # Blunt
    "mace": "mace",
    "war hammer": "war_hammer",
    "flail": "flail",
}

def normalize_label(text: str) -> str:
    t = text.lower()

    for key, value in LABEL_MAP.items():
        if key in t:
            return value

    return "other"

def main():
    df = pd.read_csv(INPUT)

    df["label"] = df["text"].apply(normalize_label)

    df.to_csv(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()