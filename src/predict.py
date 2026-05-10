from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_METADATA_PATH = ROOT / "data" / "weapon_metadata_expanded.json"
CURATED_METADATA_PATH = ROOT / "data" / "curated_weapon_metadata.json"


FEATURE_GROUPS = {
    "edged": {
        "triggers": {
            "blade",
            "bladed",
            "edge",
            "edged",
            "single edged",
            "single-edged",
            "double edged",
            "double-edged",
            "tip",
            "pointy",
            "curved",
            "hilt",
            "guard",
            "pommel",
            "sword",
            "dagger",
            "knife",
            "sabre",
            "saber",
        },
        "metadata_terms": {"edged and bladed weapons", "swords", "daggers", "knives", "axes"},
    },
    "archery_accessory": {
        "triggers": {"thumb ring", "archer ring", "archery ring"},
        "metadata_terms": {"archery accessories", "thumb draw"},
    },
    "thrown": {
        "triggers": {"throwing", "thrown", "ring", "disc", "disk", "circular", "circle"},
        "metadata_terms": {"thrown", "throwing blades", "throwing balls", "throwing spears", "throwing axes"},
    },
    "bows": {
        "triggers": {"bow", "longbow", "shortbow", "arrow", "arrows", "archery"},
        "metadata_terms": {"bows", "longbows", "short bows", "composite bows"},
    },
    "crossbows": {
        "triggers": {"crossbow", "bolt", "quarrel"},
        "metadata_terms": {"crossbows"},
    },
    "polearms": {
        "triggers": {"polearm", "spear", "lance", "pike", "halberd", "glaive", "shaft", "trident"},
        "metadata_terms": {"polearms", "spears"},
    },
    "blunt": {
        "triggers": {"club", "mace", "hammer", "staff", "flail", "baton", "cudgel"},
        "metadata_terms": {"blunt weapons", "clubs", "maces", "staves"},
    },
}

@dataclass
class WeaponFeatures:
    curvature: str | None = None
    edge_type: str | None = None
    handedness: str | None = None
    carry_orientation: str | None = None
    blade_length_cm: float | None = None

def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _active_feature_groups(text: str) -> set[str]:
    normalized = _normalize_text(text)
    active: set[str] = set()
    if "ring" in normalized and re.search(r"\b(alongside|beside|next to|near|with)\s+(?:the\s+)?bows?\b", normalized):
        active.add("archery_accessory")
    for group, config in FEATURE_GROUPS.items():
        if any(trigger in normalized for trigger in config["triggers"]):
            active.add(group)
    # "Displayed alongside bows" describes museum placement, not the object itself.
    if re.search(r"\b(alongside|beside|next to|near)\s+(?:the\s+)?bows?\b", normalized):
        active.discard("bows")
    return active


def _label_matches_group(meta: dict[str, Any], group: str) -> bool:
    metadata_text = _normalize_text(
        " ".join(
            [
                meta.get("name", ""),
                meta.get("summary", ""),
                " ".join(meta.get("signals", [])),
            ]
        )
    )
    return any(term in metadata_text for term in FEATURE_GROUPS[group]["metadata_terms"])


def _metadata_text(meta: dict[str, Any]) -> str:
    return _normalize_text(
        " ".join(
            [
                meta.get("name", ""),
                meta.get("summary", ""),
                " ".join(meta.get("signals", [])),
            ]
        )
    )


def _extract_length_cm(text: str) -> float | None:
    normalized = text.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*cm\b", normalized)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b", normalized)
    if match:
        return float(match.group(1)) * 100

    return None

def extract_features(text: str) -> WeaponFeatures:
    normalized = _normalize_text(text)

    features = WeaponFeatures()

    if "curved" in normalized:
        features.curvature = "curved"
    elif "straight" in normalized:
        features.curvature = "straight"

    if "single edged" in normalized or "single-edged" in normalized:
        features.edge_type = "single"
    elif "double edged" in normalized or "double-edged" in normalized:
        features.edge_type = "double"

    if "two handed" in normalized:
        features.handedness = "two-handed"
    elif "one handed" in normalized:
        features.handedness = "one-handed"

    if "blade-side down" in normalized or "edge down" in normalized:
        features.carry_orientation = "edge-down"
    elif "blade-side up" in normalized or "edge up" in normalized:
        features.carry_orientation = "edge-up"

    features.blade_length_cm = _extract_length_cm(text)

    return features

def _feature_multiplier(description: str, meta: dict[str, Any]) -> float:
    text = _normalize_text(description)
    name = _normalize_text(meta.get("name", ""))
    metadata = _metadata_text(meta)
    length_cm = _extract_length_cm(description)
    multiplier = 1.0

    if "single edged" in text or "single edge" in text or "one edged" in text:
        if any(term in metadata for term in ["single edged", "single edge", "one sided", "one sharp edge"]):
            multiplier *= 2.0
        if "double edged" in metadata or "double edge" in metadata:
            multiplier *= 0.45
        if "edgeless" in metadata:
            multiplier *= 0.35

    if "double edged" in text or "double edge" in text or "two edged" in text:
        if "double edged" in metadata or "double edge" in metadata:
            multiplier *= 2.0
        if any(term in metadata for term in ["single edged", "single edge", "one sided"]):
            multiplier *= 0.45

    if "curved" in text or "curve" in text:
        if "curved" in metadata or "curve" in metadata:
            multiplier *= 1.8
        if "straight" in metadata and "curved" not in metadata:
            multiplier *= 0.65

    if any(term in text for term in ["forward curved", "forward curving", "forward curve", "recurved"]):
        if any(term in metadata for term in ["yatagan", "forward curved", "forward curving", "recurved"]):
            multiplier *= 2.0
        if "odachi" in metadata or "katana" in metadata or "tachi" in metadata:
            multiplier *= 0.35

    if any(term in text for term in ["ivory", "bone hilt", "bone handle", "ornate spine", "decorated spine"]):
        if any(term in metadata for term in ["yatagan", "ivory hilt", "bone", "ornate spine", "decorated"]):
            multiplier *= 4.0
        if "japanese" in metadata:
            multiplier *= 0.45

    if "single handed" in text or "one handed" in text or "single hand" in text or "one hand" in text:
        if "one handed" in metadata or "single handed" in metadata or "one hand" in metadata:
            multiplier *= 1.7
        if "two handed" in metadata or "two hand" in metadata:
            multiplier *= 0.55

    if "pointy" in text or "pointed" in text or "tip" in text:
        if any(term in metadata for term in ["point", "tip", "thrust", "pierce"]):
            multiplier *= 1.2

    if any(term in text for term in ["ring", "disc", "disk", "circular", "circle"]):
        ring_near_bows = re.search(r"\b(alongside|beside|next to|near|with)\s+(?:the\s+)?bows?\b", text) is not None
        if ring_near_bows and any(term in metadata for term in ["thumb ring", "archery accessories", "thumb draw"]):
            multiplier *= 2.5
        if ring_near_bows and any(term in metadata for term in ["chakram", "sharpened outer edge", "war quoit"]):
            multiplier *= 0.3
        if not ring_near_bows and any(term in metadata for term in ["chakram", "circular", "sharpened outer edge", "war quoit"]):
            multiplier *= 2.0
        if "bow" in metadata or "crossbow" in metadata:
            multiplier *= 0.65

    if any(term in text for term in ["blade side down", "edge down", "facing down", "blade down"]):
        if name in {"tachi", "odachi"} or "edge down carry" in metadata:
            multiplier *= 5.0
        if "katana" in metadata or "edge facing upward" in metadata or "edge up" in metadata:
            multiplier *= 0.45

    if any(term in text for term in ["wrapped handle", "wrapped grip", "wrapped hilt", "long wrapped handle"]):
        if "japanese" in metadata or "wrapped" in metadata or "samurai" in metadata:
            multiplier *= 2.2

    if length_cm is not None and length_cm >= 120:
        if name in {"odachi"} or any(term in metadata for term in ["nodachi", "very long blade", "greatswords", "great sword"]):
            multiplier *= 4.0
        if "katana" in metadata or "longsword" in metadata or "80 to 110 cm" in metadata:
            multiplier *= 0.4
        if "shortswords" in metadata or "dagger" in metadata or "less than 60 cm" in metadata:
            multiplier *= 0.65

    return multiplier


class WeaponIdentifier:
    def __init__(
        self,
        artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> None:
        classifier_path = artifact_dir / "classifier.joblib"
        config_path = artifact_dir / "model_config.json"
        if not classifier_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                "Model artifacts were not found. Run `python -m src.train` first."
            )

        self.classifier = joblib.load(classifier_path)
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if CURATED_METADATA_PATH.exists():
            self.metadata.update(json.loads(CURATED_METADATA_PATH.read_text(encoding="utf-8")))
        self.embedder = SentenceTransformer(self.config["embedding_model"])

    def predict(self, description: str, top_k: int = 3) -> list[dict[str, Any]]:
        text = description.strip()
        if not text:
            return []

        embedding = self.embedder.encode(
            [text],
            normalize_embeddings=self.config.get("normalize_embeddings", True),
            show_progress_bar=False,
        )
        probabilities = self.classifier.predict_proba(embedding)[0]
        adjusted_probabilities = probabilities.copy()
        active_groups = _active_feature_groups(text)
        if active_groups:
            candidate_mask = np.array(
                [
                    any(
                        _label_matches_group(
                            self.metadata.get(str(label), {}),
                            group,
                        )
                        for group in active_groups
                    )
                    for label in self.classifier.classes_
                ],
                dtype=bool,
            )
            if candidate_mask.any():
                adjusted_probabilities = np.where(
                    candidate_mask,
                    adjusted_probabilities,
                    adjusted_probabilities * 0.35,
                )

        multipliers = np.array(
            [
                _feature_multiplier(text, self.metadata.get(str(label), {}))
                for label in self.classifier.classes_
            ]
        )
        #adjusted_probabilities = adjusted_probabilities * multipliers
        total_probability = adjusted_probabilities.sum()
        if total_probability > 0:
            adjusted_probabilities = adjusted_probabilities / total_probability

        order = np.argsort(adjusted_probabilities)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in order:
            label = str(self.classifier.classes_[idx])
            meta = self.metadata.get(
                label,
                {
                    "name": label.replace("_", " ").title(),
                    "summary": "No description is available for this label yet.",
                    "signals": [],
                },
            )
            
            results.append(
                {
                    "label": label,
                    "name": meta["name"],
                    "score": float(adjusted_probabilities[idx]),
                    "raw_score": float(probabilities[idx]),
                    "summary": meta["summary"],
                    "signals": meta.get("signals", []),
                    "source_title": meta.get("source_title", ""),
                    "source_url": meta.get("source_url", ""),
                }
            )
        return results


def format_prediction(results: list[dict[str, Any]]) -> str:
    if not results:
        return "Please enter a description with visible features such as blade shape, length, hilt, guard, or how it was worn."

    best = results[0]
    confidence = best["score"]
    confidence_note = ""
    if confidence < 0.35:
        confidence_note = "\n\nThe confidence is low, so treat this as an initial lead rather than an identification."

    signals = ", ".join(best["signals"]) if best["signals"] else "No signal notes available."
    source = ""
    if best.get("source_url"):
        source = f"\n\nSource: [Wikipedia: {best.get('source_title') or best['name']}]({best['source_url']})"
    
    if best["score"] < 0.5:
        confidence_note = "\n\n⚠️ Low confidence: description may be too vague or too generic."
    elif best["score"] > 0.8:
        confidence_note = "\n\nHigh confidence prediction based on strong feature match."
    else:
        confidence_note = ""
    
    return (
        f"### Likely match: {best['name']}\n\n"
        f"Confidence: **{confidence:.0%}**\n\n"
        f"{best['summary']}\n\n"
        f"Common identifying signals: {signals}."
        f"{source}"
        f"{confidence_note}\n\n"
        "This tool is intended for historical, educational, and museum-style artifact identification."
    )
