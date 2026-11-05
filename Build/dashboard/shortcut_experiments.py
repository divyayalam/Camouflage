from functools import lru_cache
import json
from pathlib import Path

import gradio as gr
import keras
import numpy as np
from PIL import Image


RUNS = {
    "Original": "run1_original",
    "Mild": "run2_mild",
    "Stronger": "run3_stronger",
    "Heavy": "run4_heavy",
}
RUNS_DIR = Path(__file__).resolve().parent / "experiment_runs"


def archived_accuracy_rows():
    rows = []
    for name, run in RUNS.items():
        path = RUNS_DIR / run / f"{run}_results.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            scores = [record.get(f"test_{test_run}") for test_run in RUNS.values()]
        else:
            scores = [None] * len(RUNS)
        rows.append([name, *[f"{value:.2%}" if value is not None else "—" for value in scores]])
    return rows


@lru_cache(maxsize=len(RUNS))
def load_run(name):
    if name not in RUNS:
        raise gr.Error("Select a training run.")
    run = RUNS[name]
    path = RUNS_DIR / run / f"{run}_model.keras"
    if not path.is_file():
        raise gr.Error(f"The {name} model is unavailable. Add its training-run files first.")
    return keras.models.load_model(path, compile=False)


def compare_image(image, run, preprocess_image, class_names):
    if image is None:
        raise gr.Error("Upload an image first.")

    # The two inputs differ only by swapping their red and blue channels.
    original = preprocess_image(image)[0]
    swapped = original[..., [2, 1, 0]]
    batch = np.stack((original, swapped))
    scores = load_run(run)(batch, training=False).numpy()

    before_index, after_index = np.argmax(scores, axis=1)
    before_class = class_names[before_index]
    after_class = class_names[after_index]
    changed = "Changed" if before_index != after_index else "Unchanged"
    delta = 100 * float(scores[1, before_index] - scores[0, before_index])
    before_score = float(scores[0, before_index])
    after_score = float(scores[1, before_index])
    summary = (
        f"**{run} · Prediction {changed.lower()}:** {before_class} → {after_class}  \n"
        f"**Score for {before_class}:** {before_score:.1%} → {after_score:.1%} "
        f"({delta:+.1f} percentage points)."
    )

    def as_image(pixels):
        return Image.fromarray(np.rint(pixels * 255).astype(np.uint8))

    def labeled_scores(row):
        return {label: float(score) for label, score in zip(class_names, row)}

    return (
        as_image(original),
        labeled_scores(scores[0]),
        as_image(swapped),
        labeled_scores(scores[1]),
        summary,
    )


def add_experiment_tab(preprocess_image, class_names):
    with gr.Tab("Shortcut Experiments", id="shortcut-experiments"):
        gr.Markdown(
            "### Compare a color change\n"
            "Choose a training run and upload an animal image. The same model classifies "
            "the original and a version with red and blue swapped."
        )
        with gr.Row():
            upload = gr.Image(type="pil", label="Upload one animal image")
            with gr.Column():
                run = gr.Dropdown(
                    choices=list(RUNS), value="Original", label="Training run",
                    info="Mild, Stronger, and Heavy use progressively more augmentation during training.",
                )
                compare = gr.Button("Compare", variant="primary")

        with gr.Row():
            with gr.Column():
                original_image = gr.Image(label="Original model input", interactive=False)
                original_scores = gr.Label(label="Original prediction", num_top_classes=6)
            with gr.Column():
                swapped_image = gr.Image(label="Red and blue swapped", interactive=False)
                swapped_scores = gr.Label(label="Color-swapped prediction", num_top_classes=6)
        summary = gr.Markdown()

        gr.Markdown(
            "### Recorded run accuracy\n"
            "Rows are trained models; columns are archived test conditions. Those test "
            "conditions also vary rotation and other color settings, so the table is "
            "separate from the red/blue comparison above."
        )
        gr.Dataframe(
            headers=["Training run", *(f"{name} test" for name in RUNS)],
            value=archived_accuracy_rows(),
            datatype="str",
            interactive=False,
        )

        def predict_pair(image, selected_run):
            return compare_image(image, selected_run, preprocess_image, class_names)

        compare.click(
            fn=predict_pair,
            inputs=[upload, run],
            outputs=[original_image, original_scores, swapped_image, swapped_scores, summary],
            api_name="compare_shortcut",
        )
