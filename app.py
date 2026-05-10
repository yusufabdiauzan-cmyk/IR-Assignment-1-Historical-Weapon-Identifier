from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.predict import DEFAULT_ARTIFACT_DIR, WeaponIdentifier, format_prediction
from src.train import train_model


ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    "I saw a sword yesterday. It is mildly curved. There are wrappings on its pommeless handle. It was worn blade-side down. It was also really long.",
    "It was a very narrow straight sword with an ornate hand guard, mostly made for thrusting.",
    "The blade was Chinese, straight, double edged, and had a simple guard with a tassel.",
    "It looked like a short Japanese dagger with a wrapped handle and a small scabbard.",
]


#identifier: WeaponIdentifier | None = None

def get_identifier():
    classifier_path = DEFAULT_ARTIFACT_DIR / "classifier.joblib"

    if not classifier_path.exists():
        train_model()

    return WeaponIdentifier()

"""
def get_identifier() -> WeaponIdentifier:
    global identifier
    if identifier is None:
        classifier_path = DEFAULT_ARTIFACT_DIR / "classifier.joblib"
        if not classifier_path.exists():
            train_model(show_progress=False)
        identifier = WeaponIdentifier()
    return identifier
"""
    
def identify(description: str, top_k: Any) -> tuple[str, pd.DataFrame]:
    try:
        requested_matches = max(1, min(5, int(float(top_k))))
        model = get_identifier()

        results = model.predict(description or "", top_k=requested_matches)

        rows = [
            {
                "Match": r["name"],
                "Confidence": f"{r.get('score', 0.0):.1%}",
                "Short explanation": r["summary"],
            }
            for r in results
        ]

        return (
            format_prediction(results),
            pd.DataFrame(rows),
        )

    except Exception as error:
        return (
            f"### Error\n\n`{type(error).__name__}: {error}`",
            pd.DataFrame(columns=["Match", "Confidence", "Short explanation"]),
        )

"""
def identify(description: str, top_k: Any) -> tuple[str, pd.DataFrame]:
    results = []          # ALWAYS defined
    feature_block = ""   # ALWAYS defined

    try:
        requested_matches = max(1, min(5, int(float(top_k))))
        model = get_identifier()

        results = model.predict(description or "", top_k=requested_matches)

        rows = []
        for r in results:
            rows.append(
                {
                    "Match": r["name"],
                    "Confidence": f"{r.get('score', 0.0):.1%}",
                    "Short explanation": r["summary"],
                }
            )

        # feature extraction (safe)
        if description:
            desc = description.lower()
            notes = []

            
            if "curved" in desc:
                notes.append("Detected feature: curved shape")
            if "wrapped" in desc:
                notes.append("Detected feature: wrapped grip")
            if "blade-side down" in desc:
                notes.append("Detected feature: edge-down orientation")
            
                
            feature_block = "\n".join(notes)

        return (
            format_prediction(results),
            pd.DataFrame(rows),
        )

    except Exception as error:
        return (
            f"### Error\n\n`{type(error).__name__}: {error}`",
            pd.DataFrame(columns=["Match", "Confidence", "Short explanation"]),
        )
"""

with gr.Blocks(title="Historical Weapon Identifier") as demo:
    gr.Markdown(
        "# Historical Weapon Identifier\n"
        "Describe visible features of a historical weapon or related artifact and the classifier will suggest likely artifact types."
    )
    with gr.Row():
        description = gr.Textbox(
            label="Description",
            lines=6,
            placeholder="Example: long mildly curved Japanese sword, wrapped handle, worn blade-side down...",
        )
        with gr.Column():
            top_k = gr.Slider(1, 5, value=3, step=1, label="Number of matches")
            submit = gr.Button("Identify", variant="primary")

    output = gr.Markdown()
    table = gr.Dataframe(
        headers=["Match", "Confidence", "Short explanation"],
        datatype=["str", "str", "str"],
        row_count=(3, "dynamic"),
        column_count=(3, "fixed"),
        interactive=False,
    )

    gr.Examples(examples=EXAMPLES, inputs=description)
    submit.click(identify, inputs=[description, top_k], outputs=[output, table], queue=True)
    description.submit(identify, inputs=[description, top_k], outputs=[output, table], queue=True)
    demo.queue()
    demo.launch(show_error=True)

if __name__ == "__main__":
    demo.launch(show_error=True)
