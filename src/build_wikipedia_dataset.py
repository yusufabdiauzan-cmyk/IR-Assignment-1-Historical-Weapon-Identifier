from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import OrderedDict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
import hashlib


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_DATASET = DATA_DIR / "weapons_dataset_expanded.csv"
DEFAULT_METADATA = DATA_DIR / "weapon_metadata_expanded.json"
CURATED_METADATA = DATA_DIR / "curated_weapon_metadata.json"
DEFAULT_MANIFEST = DATA_DIR / "wikipedia_dataset_manifest_expanded.json"
API_URL = "https://en.wikipedia.org/w/api.php"
SOURCE_PAGE = "Wikipedia dataset builder (multi-source)"
SOURCE_PAGES = list(dict.fromkeys([
    "List of premodern combat weapons",
    "List of medieval weapons",
    "List of swords",
    "List of historical swords",
    "Types of swords",
    "Classification of swords",
    "Fighting knife",
    "Axe",
    "Polearm",
    "Spear",
    "Sword",
    "Shield",
    "List of types of spears",
    "Viking Age arms and armour",
    "Whip",
    "Mace",
    "Flail (weapon)",
    "Morning star (weapon)",
    "War hammer",
    "Battle axe",
    "Throwing axe",
    "Throwing stick",
    "Spear-thrower",
    "Bow and arrow",
    "Arrow",
    "Crossbow",
    "Japanese sword",
    "Category:Edged and bladed weapons",
    "Category:Axes",
    "Category:Spears",
    "Category:Polearms",
    "Category:Swords",
    "Category:Archery",
    "Category:Bows (archery)",
    "Category:Blunt weapons",
]))
SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_premodern_combat_weapons"
USER_AGENT = "historical-weapon-identifier-lab/0.2 (educational dataset builder)"
CACHE_DIR = DATA_DIR / "http_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_SEEDS = [
    "Category:Melee weapons",
    "Category:Edged and bladed weapons",
    "Category:Swords",
    "Category:Daggers",
    "Category:Knives",
    "Category:Axes",
    "Category:Polearms",
    "Category:Spears",
    "Category:Throwing weapons",
    "Category:Bows (archery)",
    "Category:Blunt weapons",
    "Category:Clubs and maces",
    "Category:Flails",
    "Category:Whips",
    "Category:Staves",
    "Category:Shuriken",
    "Category:Arrows",
    "Category:Bolts (archery)",
    "Category:Crossbows",
    "Category:Shields",
    "Category:Ancient weapons",
    "Category:Medieval weapons",
]

GENERIC_LABELS = {
    "sword",
    "dagger",
    "knife",
    "axe",
    "polearm",
    "spear",
    "bow",
    "crossbow",
    "shield",
    "weapon",
    "blunt_weapon",
}

EXCLUDED_SECTIONS = {"See also",
"References",
}

"""
{
    "Gunpowder-based",
    "Flamethrowers",
    "See also",
    "References",
}
"""

CATEGORY_METADATA = {
    "sword": {
        "name": "Sword",
        "summary": "A bladed melee weapon designed primarily for cutting or thrusting.",
        "signals": ["long blade", "single or double edged", "hilt", "guard"],
    },
    "dagger": {
        "name": "Dagger",
        "summary": "A short stabbing weapon with a pointed blade.",
        "signals": ["short blade", "concealable", "thrusting"],
    },
    "axe": {
        "name": "Axe",
        "summary": "A weapon with a bladed head mounted perpendicular to the handle.",
        "signals": ["heavy blade", "wooden shaft", "chopping edge"],
    },
    "bow": {
        "name": "Bow",
        "summary": "A ranged weapon used to launch arrows.",
        "signals": ["limbs", "string", "arrows", "archery"],
    },
    "crossbow": {
        "name": "Crossbow",
        "summary": "A mechanical ranged weapon firing bolts.",
        "signals": ["stock", "trigger", "bolt", "mechanical release"],
    },
    "polearm": {
        "name": "Polearm",
        "summary": "A long-shafted weapon such as spear or halberd.",
        "signals": ["long shaft", "thrusting", "blade on pole"],
    },
    "blunt": {
        "name": "Blunt Weapon",
        "summary": "A weapon designed to strike with impact rather than cutting.",
        "signals": ["club", "mace", "hammer", "impact"],
    },
    "thrown": {
        "name": "Thrown Weapon",
        "summary": "Weapons designed to be thrown.",
        "signals": ["chakram", "throwing knife", "projectile"],
    },
}

CONTROL_ROWS = [
    "I saw an old blade in a case but cannot remember whether it was straight or curved.",
    "The object had a handle and a metal blade, but the important identifying features were not visible.",
    "It was probably a historical weapon, although the description mixes details from several unrelated traditions.",
    "The label was missing and I only remember that the object was sharp.",
    "A generic old weapon with no reliable details about length, blade shape, hilt, region, or how it was carried.",
    "The description is too vague to choose one cold weapon type confidently.",
]

EXCLUDED_TITLE_TERMS = {
    "firearm",
    "rifle",
    "pistol",
    "revolver",
    "shotgun",
    "machine gun",
    "cannon",
    "artillery",
    "grenade",
    "rocket",
    "missile",
    "torpedo",
    "bomb",
    "gunpowder",
    "explosive",
}

EXCLUDED_CATEGORY_TERMS = {
    "firearms",
    "guns",
    "artillery",
    "ammunition",
    "explosives",
    "grenades",
    "rockets",
    "missiles",
    "military vehicles",
}

RELEVANCE_TERMS = {
    "weapon",
    "sword",
    "blade",
    "bladed",
    "dagger",
    "knife",
    "spear",
    "polearm",
    "axe",
    "mace",
    "club",
    "bow",
    "crossbow",
    "shield",
    "throwing",
    "thrown",
    "martial",
    "combat",
    "warfare",
    "armor",
    "armour",
    "hilt",
    "edge",
    "archery",
    "projectile",
    "sling",
}


def fetch_json(params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    query = urllib.parse.urlencode(sorted(params.items()), doseq=True)
    cache_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
                cache_path.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                return payload

        except HTTPError as error:
            if error.code == 429:
                retry_after = error.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 30
                time.sleep(wait_seconds)
                continue

            if attempt == retries - 1:
                raise

            time.sleep(1.5 * (attempt + 1))

        except Exception:
            if attempt == retries - 1:
                raise

            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError("unreachable")

def fetch_wikitext(page: str) -> str:
    payload = fetch_json(
            {
                "action": "query",
                "prop": "extracts|info",   # Uncommented and replaced "revisions"
                "exintro": "1",            # Uncommented
                "explaintext": "1",        # Uncommented
                "inprop": "url",
                "redirects": "1",
                "titles": "|".join(titles),
                "format": "json",
                "formatversion": "2",
            }
        )
    pages = payload.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        raise RuntimeError(f"Wikipedia page not found: {page}")
    return pages[0]["revisions"][0]["slots"]["main"]["content"]


def fetch_parsed_html(page: str) -> str:
    payload = fetch_json(
        {
            "action": "parse",
            "page": page,
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        }
    )
    return payload["parse"]["text"]


def title_is_excluded(title: str) -> bool:
    normalized = title.lower()
    return any(term in normalized for term in EXCLUDED_TITLE_TERMS)


def category_is_excluded(category: str) -> bool:
    normalized = category.lower()
    return any(term in normalized for term in EXCLUDED_CATEGORY_TERMS)


def looks_like_weapon_article(title: str, extract: str, section: str) -> bool:
    normalized = f"{title} {extract} {section}".lower()
    if title_is_excluded(title) or any(term in normalized for term in EXCLUDED_TITLE_TERMS):
        return False
    #score = sum(term in normalized for term in RELEVANCE_TERMS)
    return True

def fetch_category_entries(
    seeds: list[str] = CATEGORY_SEEDS,
    max_depth: int = 1,
    max_entries: int = 1200,
    pause: float = 0.05,
) -> list[dict[str, str]]:
    entries: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    visited_categories: set[str] = set()
    queue: list[tuple[str, int]] = [(seed, 0) for seed in seeds]

    while queue:
        category, depth = queue.pop(0)
        if category in visited_categories or category_is_excluded(category):
            continue
        visited_categories.add(category)

        continuation: str | None = None
        while True:
            params: dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmtype": "page|subcat",
                "cmlimit": "500",
                "format": "json",
                "formatversion": "2",
            }
            if continuation:
                params["cmcontinue"] = continuation

            payload = fetch_json(params)
            for member in payload.get("query", {}).get("categorymembers", []):
                title = member.get("title", "")
                if not title:
                    continue

                if title.startswith("Category:"):
                    if depth < max_depth and not category_is_excluded(title):
                        queue.append((title, depth + 1))
                    continue

                if ":" in title or title_is_excluded(title):
                    continue

                entries.setdefault(
                    title,
                    {
                        "title": title,
                        "bullet": title,
                        "source_section": f"Wikipedia categories > {category.removeprefix('Category:')}",
                    },
                )
                if len(entries) >= max_entries:
                    return list(entries.values())

            continuation = payload.get("continue", {}).get("cmcontinue")
            if not continuation:
                break
            time.sleep(pause)

        time.sleep(pause)

    return list(entries.values())


def clean_wikitext(text: str) -> str:
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/>", "", text)
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"\[\[([^|\]#]+)(?:#[^\]|]+)?\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^|\]#]+)(?:#[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = text.replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", text).strip(" ;,.")


class WikipediaListParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.section_stack: list[tuple[int, str]] = []
        self.weapons: "OrderedDict[str, dict[str, str]]" = OrderedDict()
        self.heading_level: int | None = None
        self.heading_parts: list[str] = []
        self.li_depth = 0
        self.li_parts: list[str] = []
        self.li_first_title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h2", "h3", "h4", "h5", "h6"}:
            self.heading_level = int(tag[1])
            self.heading_parts = []
            return

        if tag == "li":
            self.li_depth += 1
            if self.li_depth == 1:
                self.li_parts = []
                self.li_first_title = None
            return

        if tag == "a" and self.li_depth >= 1 and self.li_first_title is None:
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if not href.startswith("/wiki/"):
                return
            title = urllib.parse.unquote(href.removeprefix("/wiki/")).replace("_", " ")
            title = title.split("#", 1)[0]
            if ":" not in title:
                self.li_first_title = title

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h2", "h3", "h4", "h5", "h6"} and self.heading_level is not None:
            heading = clean_wikitext(" ".join(self.heading_parts))
            heading = re.sub(r"\s*\[edit\]\s*", "", heading).strip()
            if heading:
                level = self.heading_level
                self.section_stack = [(lvl, name) for lvl, name in self.section_stack if lvl < level]
                self.section_stack.append((level, heading))
            self.heading_level = None
            self.heading_parts = []
            return

        if tag == "li" and self.li_depth >= 1:
            if self.li_depth == 1:
                self._commit_list_item()
                self.li_parts = []
                self.li_first_title = None
            self.li_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.heading_level is not None:
            self.heading_parts.append(data)
        if self.li_depth >= 1:
            self.li_parts.append(data)

    def _commit_list_item(self) -> None:
        if not self.li_first_title:
            return
        sections = [name for _, name in self.section_stack]
        if any(section in EXCLUDED_SECTIONS for section in sections):
            return

        bullet = clean_wikitext(" ".join(self.li_parts))
        if not bullet:
            return

        self.weapons.setdefault(
            self.li_first_title,
            {
                "title": self.li_first_title,
                "bullet": bullet,
                "source_section": " > ".join(sections),
            },
        )


def parse_weapon_links_from_html(html: str) -> list[dict[str, str]]:
    parser = WikipediaListParser()
    parser.feed(html)
    return list(parser.weapons.values())


def parse_weapon_links(wikitext: str) -> list[dict[str, str]]:
    section_stack: list[tuple[int, str]] = []
    weapons: "OrderedDict[str, dict[str, str]]" = OrderedDict()

    for raw_line in wikitext.splitlines():
        heading_match = re.match(r"^(=+)\s*(.*?)\s*\1$", raw_line.strip())
        if heading_match:
            level = len(heading_match.group(1))
            heading = clean_wikitext(heading_match.group(2))
            section_stack = [(lvl, name) for lvl, name in section_stack if lvl < level]
            section_stack.append((level, heading))
            continue

        line = raw_line.strip()
        if not line.startswith("*"):
            continue

        sections = [name for _, name in section_stack]
        if any(section in EXCLUDED_SECTIONS for section in sections):
            continue

        link_match = re.search(r"\[\[([^|\]#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]", line)
        if not link_match:
            continue

        title = link_match.group(1).strip()
        if ":" in title:
            continue

        bullet = clean_wikitext(line.lstrip("* "))
        if not bullet:
            continue

        weapons.setdefault(
            title,
            {
                "title": title,
                "bullet": bullet,
                "source_section": " > ".join(sections),
            },
        )

    return list(weapons.values())


def batched(items: list[dict[str, str]], size: int = 50) -> list[list[dict[str, str]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def fetch_article_extracts(entries: list[dict[str, str]], pause: float = 0.1) -> list[dict[str, str]]:
    pages: list[dict[str, str]] = []
    batches = batched(entries)
    total_batches = len(batches)

    # Combined the loops so the print statement happens DURING the fetching
    for i, batch in enumerate(batches):
        print(f"Fetching batch {i+1} of {total_batches}...")
        
        titles = [entry["title"] for entry in batch]
        payload = fetch_json(
            {
                "action": "query",
                "prop": "extracts|info",
                # REMOVED "exintro": "1" to get content beyond the lead
                "exchars": "2000",         # Request ~2,000 characters of plain text
                "explaintext": "1",
                "inprop": "url",
                "redirects": "1",
                "titles": "|".join(titles),
                "format": "json",
                "formatversion": "2",
            }
        )
        
        by_original = {entry["title"]: entry for entry in batch}
        normalized = {
            item.get("to", item.get("from")): item.get("from")
            for item in payload.get("query", {}).get("normalized", [])
        }
        redirects = {
            item.get("to", item.get("from")): item.get("from")
            for item in payload.get("query", {}).get("redirects", [])
        }

        for page in payload.get("query", {}).get("pages", []):
            if "missing" in page:
                continue
            extract = re.sub(r"\s+", " ", page.get("extract", "")).strip()

            canonical = page["title"]
            original = redirects.get(canonical) or normalized.get(canonical) or canonical
            source = by_original.get(original)
            if not source:
                source = next((entry for entry in batch if entry["title"] == canonical), None)
            if not source:
                continue

            if not looks_like_weapon_article(canonical, extract, source["source_section"]):
                continue

            pages.append(
                {
                    "title": canonical,
                    "bullet": source["bullet"],
                    "source_section": source["source_section"],
                    "extract": extract,
                    "url": page.get(
                        "fullurl",
                        f"https://en.wikipedia.org/wiki/{urllib.parse.quote(canonical.replace(' ', '_'))}",
                    ),
                }
            )
        time.sleep(pause)

    return pages

def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return value or "unknown"


def short_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    truncated = text[:limit].rsplit(" ", 1)[0]
    return f"{truncated}."

def infer_parent_class(label: str, extract: str = "") -> str:
    text = f"{label} {extract}".lower()

    if any(x in text for x in ["bow", "archery"]):
        return "bow"
    if any(x in text for x in ["spear", "pike", "halberd", "polearm"]):
        return "polearm"
    if any(x in text for x in ["axe"]):
        return "axe"
    if any(x in text for x in ["dagger", "knife"]):
        return "dagger"
    if any(x in text for x in ["mace", "club", "hammer", "flail"]):
        return "blunt"
    if any(x in text for x in ["sword", "kilij", "katana", "estoc", "rapier", "yatagan"]):
        return "sword"

    return "other"

def make_rows(pages: list[dict[str, str]], include_control: bool = True) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    metadata: dict[str, Any] = {}
    labels_seen: dict[str, str] = {}

    for page in pages:
        #label = page["title"].lower()
        label = slugify(page["title"].split("(")[0])

        """
        if label in GENERIC_LABELS:
            continue
        """

        if label in labels_seen:
            continue
        labels_seen[label] = page["title"]

        extract = short_text(page["extract"], limit=2000)
        section = page["source_section"] or "Wikipedia cold weapon list"

        # fewer but higher-quality examples
        examples = [
            extract,
            f"This artifact is described as: {extract}",
            f"Historical record: {extract}",
        ]

        for index, example in enumerate(examples):
            rows.append(
                {
                    "text": example,
                    "label": label,
                    "wikipedia_title": page["title"],
                    "wikipedia_url": page["url"],
                    "source_section": section,
                    "source_page": SOURCE_PAGE,
                    "source_example_type": f"wikipedia_clean_{index}",
                    "source_license": "CC BY-SA 4.0",
                }
            )

        metadata[label] = {
            "name": page["title"],
            "summary": extract,
            "signals": [part for part in section.split(" > ") if part],
            "source_title": page["title"],
            "source_url": page["url"],
            "source_page": SOURCE_PAGE,
            "source_license": "CC BY-SA 4.0",
            "parent_class": infer_parent_class(label, extract),
        }

    if include_control:
        for text in CONTROL_ROWS:
            rows.append(
                {
                    "text": text,
                    "label": "unknown_or_ambiguous",
                    "wikipedia_title": "",
                    "wikipedia_url": "",
                    "source_section": "Local ambiguity control examples",
                    "source_page": "local_control_examples",
                    "source_example_type": "ambiguous_control",
                    "source_license": "project-authored",
                }
            )
        metadata["unknown_or_ambiguous"] = {
            "name": "Unknown or Ambiguous",
            "summary": "The description does not contain enough reliable identifying features for a confident historical type.",
            "signals": ["missing blade shape", "missing length", "missing hilt details", "mixed or vague features"],
            "source_title": "",
            "source_url": "",
            "source_page": "local_control_examples",
            "source_license": "project-authored",
        }

    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "text",
        "label",
        "wikipedia_title",
        "wikipedia_url",
        "source_section",
        "source_page",
        "source_example_type",
        "source_license",
    ]
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def build_dataset(
    dataset_path: Path = DEFAULT_DATASET,
    metadata_path: Path = DEFAULT_METADATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    max_pages: int | None = None,
    category_depth: int = 1,
    max_category_entries: int = 5000,
    include_control: bool = True,
) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    list_entries = []
    for page in SOURCE_PAGES:
        html = fetch_parsed_html(page)
        list_entries.extend(parse_weapon_links_from_html(html))

    category_entries = fetch_category_entries(
        max_depth=category_depth,
        max_entries=max_category_entries
    )

    seen = set()
    entries = []
    for e in list_entries + category_entries:
        key = e["title"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append(e)

    if max_pages is not None:
        entries = entries[:max_pages]

    pages = fetch_article_extracts(entries)
    rows, metadata = make_rows(pages, include_control=include_control)

    """
    df = pd.DataFrame(rows)

    MIN_SAMPLES_PER_LABEL = 20

    counts = df["label"].value_counts()
    valid_labels = counts[counts >= MIN_SAMPLES_PER_LABEL].index

    df = df[df["label"].isin(valid_labels)]

    rows = df.to_dict(orient="records")
    """
    
    curated_metadata_count = 0
    metadata.update(CATEGORY_METADATA)
    if CURATED_METADATA.exists():
        curated_metadata = json.loads(CURATED_METADATA.read_text(encoding="utf-8"))
        metadata.update(curated_metadata)
        curated_metadata_count = len(curated_metadata)

    write_csv(dataset_path, rows)
    write_json(metadata_path, metadata)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_page": SOURCE_PAGE,
        "source_url": SOURCE_URL,
        "excluded_sections": sorted(EXCLUDED_SECTIONS),
        "category_seed_count": len(CATEGORY_SEEDS),
        "category_depth": category_depth,
        "max_category_entries": max_category_entries,
        "list_entry_count": len(list_entries),
        "category_entry_count": len(category_entries),
        "candidate_entry_count": len(entries),
        "article_count": len(pages),
        "skipped_or_unavailable_count": len(entries) - len(pages),
        "row_count": len(rows),
        "label_count": len(metadata),
        "curated_metadata_count": curated_metadata_count,
        "includes_ambiguous_control_class": include_control,
        "license_note": "Wikipedia-derived text is licensed under Creative Commons Attribution-ShareAlike 4.0; see each row's wikipedia_url.",
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the weapon dataset from Wikipedia.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--category-depth", type=int, default=1)
    parser.add_argument("--max-category-entries", type=int, default=5000)
    parser.add_argument("--no-control-class", action="store_true")
    args = parser.parse_args()

    manifest = build_dataset(
        dataset_path=args.dataset,
        metadata_path=args.metadata,
        manifest_path=args.manifest,
        max_pages=args.max_pages,
        category_depth=args.category_depth,
        max_category_entries=args.max_category_entries,
        include_control=not args.no_control_class,
    )
    print(f"Wrote {manifest['row_count']} rows across {manifest['label_count']} labels.")
    print(f"Source articles: {manifest['article_count']}")
    print(f"Dataset: {args.dataset}")
    print(f"Metadata: {args.metadata}")


if __name__ == "__main__":
    main()
