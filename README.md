# Historical Weapon Identifier

This is an embeddings-based classifier for identifying likely historical weapons and related artifacts from informal descriptions. It is designed for educational and museum-style artifact identification.

The pipeline is:

```text
free-text description -> sentence embedding -> logistic regression classifier -> top-k artifact labels
```

Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## Run Locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train the classifier:

```powershell
python -m src.train
```

Run the demo:

```powershell
python app.py
```

Open the local URL printed by Gradio, usually:

```text
http://127.0.0.1:7860
```

The app will also train automatically on first launch if `artifacts/classifier.joblib` does not exist yet.

## Try This Input

```text
I saw a sword yesterday. It is mildly curved. There are wrappings on its pommeless handle. It was worn blade-side down. It was also really long.
```

Expected top result: `Tachi`.

## Dataset

The default dataset is in `data/weapons_dataset_expanded.csv`. It is generated from Wikipedia's
`List of premodern combat weapons`, excluding the gunpowder and flamethrower
sections so the project stays focused on cold weapons. Each row keeps the
Wikipedia title, URL, source section, and license field.

Rebuild the dataset from Wikipedia:

```powershell
python -m src.build_wikipedia_dataset
```

For a quick test scrape:

```powershell
python -m src.build_wikipedia_dataset --max-pages 20
```

The script also writes `data/weapon_metadata_expanded.json` for app explanations and
`data/wikipedia_dataset_manifest_expanded.json` with scrape counts. A small
`unknown_or_ambiguous` control class is kept so the app can avoid forcing a
specific identification when the user gives too little detail.

The training script also loads `data/curated_visual_descriptions.csv` when it
exists. These project-authored rows add visitor-style descriptions for labels
that Wikipedia list scraping may miss or under-describe, such as `tachi`,
`odachi`, `wakizashi`, `tanto`, and weapon-adjacent artifacts like
`thumb_ring`.

Wikipedia-derived text is licensed under Creative Commons Attribution-ShareAlike
4.0. Keep attribution fields when redistributing the dataset.

## Hugging Face Space

This project is uploaded to a Hugging Face Space using the Gradio SDK. Include:

```text
app.py: https://huggingface.co/spaces/yusufabdzn/Historical_Weapon_Identifier/resolve/main/app.py
src/ : https://huggingface.co/yusufabdzn/Historical_Weapon_Identifier
data/ : https://huggingface.co/datasets/yusufabdzn/Historical_Weapon_Identifier
```

The Space will download the embedding model and train the lightweight classifier on first startup. If you want the Space to refresh from Wikipedia on startup, run `python -m src.build_wikipedia_dataset` before `python -m src.train`, but the simplest submission path is to upload the already generated `data/` files.

## Notes

This is not a weapon recommendation or procurement tool. It is for historical identification from visual descriptions. Ambiguous inputs should be treated as uncertain leads, not authoritative identifications.
