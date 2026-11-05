"""Week 6: compare matched original/grayscale models and inspect channel ablation."""

import csv
from functools import lru_cache
import json

import gradio as gr
import keras
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from channel_ablation import ablation_parts, parse_channels, predict_with_ablation
from shortcut_experiments import RUNS_DIR, load_run


ORIGINAL = "Original (run1_original)"
MITIGATED = "Mitigated (g35_luma)"
MODEL_NAMES = [ORIGINAL, MITIGATED]
UNKNOWN = "Unknown"
GRAY_DIR = RUNS_DIR / "grayscale_local" / "g35_luma_only"
MITIGATED_PATH = GRAY_DIR / "g35_luma" / "g35_luma_model.keras"
VIEWS = ["Color", "Grayscale (luma)"]


@lru_cache(maxsize=1)
def load_mitigated():
    if not MITIGATED_PATH.is_file():
        raise gr.Error("The grayscale-trained checkpoint is missing. See the dashboard setup notes.")
    best = json.loads((GRAY_DIR / "best.json").read_text(encoding="utf-8"))
    if best.get("variant") != "g35_luma" or best.get("base_level") != "run1_original":
        raise gr.Error("The grayscale checkpoint metadata does not match this comparison.")
    return keras.models.load_model(MITIGATED_PATH, compile=False)


def load_comparison_model(name):
    if name == ORIGINAL:
        return load_run("Original")
    if name == MITIGATED:
        return load_mitigated()
    raise gr.Error("Select the original or mitigated model.")


def recorded_metrics():
    """Use the local notebook's final test report, never live image scores."""
    with (GRAY_DIR / "test_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    records = {row[0]: dict(zip(header[1:], row[1:])) for row in rows[1:] if row}
    scores = []
    for key in ("four-run: run1_original", "grayscale: g35_luma"):
        values = [float(records[key][column]) for column in ("test_color", "test_gray", "test_gray_avg")]
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Test accuracy values must be finite probabilities.")
        scores.append(values)
    return np.asarray(scores) * 100


def accuracy_report():
    try:
        values = recorded_metrics()
    except (OSError, ValueError, KeyError, IndexError):
        return [], None, "Add the matching local test_summary.csv to display recorded test metrics."
    rows = [
        [name, *[round(float(v), 2) for v in row], round(float(row[0] - row[1]), 2)]
        for name, row in zip(MODEL_NAMES, values)
    ]
    delta = values[1] - values[0]
    rows.append(["Mitigated minus original (pp)", *[round(float(v), 2) for v in delta],
                 round(float(delta[0] - delta[1]), 2)])
    fig = Figure(figsize=(7.6, 3.6), layout="constrained")
    ax = fig.subplots()
    x = np.arange(3)
    for i, (name, color) in enumerate(zip(MODEL_NAMES, ("#2563eb", "#15803d"))):
        bars = ax.bar(x + (i - 0.5) * 0.36, values[i], 0.36, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=["Color", "Luma grayscale", "Average grayscale"],
           ylabel="Test accuracy (%)", ylim=(0, 100))
    ax.legend(loc="lower center", fontsize=8)
    return rows, fig, (
        "**Recorded test set:** the same 6,000 held-out animal images per view, "
        "from your local `test_summary.csv`. These values stay fixed when you upload an image. "
        "Gap = color accuracy minus luma-grayscale accuracy. The local run trained only g35_luma; "
        "it did not repeat Sid's full variant search."
    )


def score_graph(rows, labels, class_names, title):
    fig = Figure(figsize=(8, 3.5), layout="constrained")
    ax = fig.subplots()
    x = np.arange(len(class_names))
    for i, (scores, label, color) in enumerate(zip(rows, labels, ("#2563eb", "#15803d"))):
        ax.bar(x + (i - 0.5) * 0.36, np.asarray(scores) * 100, 0.36, label=label, color=color)
    ax.set(xticks=x, xticklabels=class_names, ylabel="Class confidence (%)", ylim=(0, 100), title=title)
    ax.legend(fontsize=8)
    return fig


def compare_models(image, view, selected_model, true_label, ablate, channel_text, preprocess_image, class_names):
    if image is None:
        raise gr.Error("Upload an animal image first.")
    if view not in VIEWS:
        raise gr.Error("Choose Color or Grayscale (luma).")
    if selected_model not in MODEL_NAMES:
        raise gr.Error("Select the original or mitigated model.")
    if true_label not in [UNKNOWN, *class_names]:
        raise gr.Error("Choose the true animal class or Unknown.")
    batch = preprocess_image(image)
    if view == VIEWS[1]:
        # Match tf.image.rgb_to_grayscale in Sid's notebook, keeping three input channels.
        gray = np.sum(batch * np.array([0.2989, 0.5870, 0.1140], dtype=np.float32), axis=-1, keepdims=True)
        batch = np.repeat(gray, 3, axis=-1)
    models = [load_comparison_model(name) for name in MODEL_NAMES]
    scores = np.stack([np.asarray(model(batch, training=False))[0] for model in models])
    if scores.shape != (2, len(class_names)) or not np.isfinite(scores).all():
        raise gr.Error("The model outputs do not match the six animal classes.")
    tops = scores.argmax(axis=1)
    ref = class_names.index(true_label) if true_label != UNKNOWN else int(tops[0])
    reference = class_names[ref]
    rows = []
    for name, values, top in zip(MODEL_NAMES, scores, tops):
        rows.append([name, class_names[top], round(float(values[top]) * 100, 2),
                     round(float(values[ref]) * 100, 2),
                     "—" if true_label == UNKNOWN else ("Yes" if class_names[top] == true_label else "No")])
    selected_index = MODEL_NAMES.index(selected_model)
    selected_scores = scores[selected_index]
    delta = float(scores[1, ref] - scores[0, ref]) * 100
    summary = (
        f"**Selected model: {selected_model}** — {class_names[tops[selected_index]]} "
        f"({selected_scores[tops[selected_index]]:.2%}).  \n"
        f"**Compared class: {reference}** "
        f"({'true animal class' if true_label != UNKNOWN else 'original model prediction; true class unknown'}). "
        f"Mitigated minus original confidence: **{delta:+.2f} percentage points**. "
        "Both models received the same input."
    )
    ablated_scores, ablation_plot = None, None
    ablation_summary = "Channel ablation is off. Enable it and compare again to inspect the selected model."
    if ablate:
        try:
            _, _, count, _ = ablation_parts(models[selected_index])
            channels = parse_channels(channel_text, count)
            masked, layer_name = predict_with_ablation(models[selected_index], batch, channels)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        after = masked[0]
        after_top = int(np.argmax(after))
        ablated_scores = dict(zip(class_names, map(float, after)))
        change = float(after[ref] - selected_scores[ref]) * 100
        ablation_summary = (
            f"**{selected_model} · {len(channels)} of {count} feature channels disabled** "
            f"after `{layer_name}`.  \n"
            f"Prediction: {class_names[tops[selected_index]]} ({selected_scores[tops[selected_index]]:.2%}) "
            f"→ {class_names[after_top]} ({after[after_top]:.2%}).  \n"
            f"**{reference} confidence:** {selected_scores[ref]:.2%} → {after[ref]:.2%} "
            f"(**{change:+.2f} pp**, after minus before). Saved weights are unchanged."
        )
        ablation_plot = score_graph([selected_scores, after], ["Channels on", "Selected channels off"],
                                    class_names, "Selected model: channel ablation")
    preview = Image.fromarray(np.rint(np.clip(batch[0], 0, 1) * 255).astype(np.uint8))
    dictionaries = [dict(zip(class_names, map(float, row))) for row in scores]
    return (preview, *dictionaries, summary, rows,
            score_graph(scores, MODEL_NAMES, class_names, "Current image: model comparison"),
            ablated_scores, ablation_summary, ablation_plot)


def add_model_comparison_tab(preprocess_image, class_names):
    with gr.Tab("Model Comparison", id="model-comparison"):
        gr.Markdown(
            "### Original versus grayscale-trained model\n"
            "Compare the same image using the original model and **g35_luma**, trained with "
            "35% grayscale examples. This mitigation targets color dependence. "
            "Its scores do not establish resistance to the planted corner patch."
        )
        with gr.Row():
            upload = gr.Image(type="pil", label="Upload an animal image", height=230)
            with gr.Column():
                view = gr.Radio(choices=VIEWS, value=VIEWS[0], label="Input view for both models")
                selected = gr.Dropdown(choices=MODEL_NAMES, value=MITIGATED, label="Selected model to inspect")
                true_label = gr.Dropdown(choices=[UNKNOWN, *class_names], value=UNKNOWN,
                                         label="True animal class (optional)")
        with gr.Accordion("Optional: turn off internal feature channels", open=False):
            gr.Markdown(
                "Each model has 128 feature channels in its final convolution block. "
                "These are learned features, separate from the image's three RGB channels. "
                "The mask applies only to the selected model during this prediction. "
                "A prediction change shows sensitivity to these channels; it does not identify "
                "which channels encode a shortcut."
            )
            enabled = gr.Checkbox(value=False, label="Enable channel ablation")
            channels = gr.Textbox(value="0", label="Channel IDs to turn off (0–127)",
                                  info="Comma-separated IDs, for example 0, 1, 2. Leave empty for a no-change check.")
        compare = gr.Button("Compare models", variant="primary")
        preview = gr.Image(label="Shared model input (32 × 32)", interactive=False, height=160)
        with gr.Row():
            original_scores = gr.Label(label=ORIGINAL, num_top_classes=len(class_names))
            mitigated_scores = gr.Label(label=MITIGATED, num_top_classes=len(class_names))
        summary = gr.Markdown()
        table = gr.Dataframe(headers=["Model", "Prediction", "Top confidence (%)", "Compared class confidence (%)", "Correct?"],
                             value=[], datatype=["str", "str", "number", "number", "str"], interactive=False,
                             label="Current image results")
        plot = gr.Plot(label="Current image confidence by class")
        with gr.Accordion("Channel ablation results", open=False):
            ablation_summary = gr.Markdown("Enable channel ablation above and compare to see results.")
            ablation_scores = gr.Label(label="Selected model after channel ablation", num_top_classes=len(class_names))
            ablation_plot = gr.Plot(label="Confidence before and after channel ablation")
        gr.Markdown("### Recorded model accuracy")
        metrics, metrics_plot, caption = accuracy_report()
        gr.Markdown(caption)
        gr.Dataframe(headers=["Model", "Color (%)", "Luma grayscale (%)", "Average grayscale (%)", "Color–luma gap (pp)"],
                     value=metrics, datatype=["str", "number", "number", "number", "number"], interactive=False)
        gr.Plot(value=metrics_plot, label="Recorded color and grayscale accuracy")

        def predict(image, input_view, model_name, actual_class, ablate, ids):
            return compare_models(image, input_view, model_name, actual_class, ablate, ids, preprocess_image, class_names)

        compare.click(
            fn=predict, inputs=[upload, view, selected, true_label, enabled, channels],
            outputs=[preview, original_scores, mitigated_scores, summary, table, plot,
                     ablation_scores, ablation_summary, ablation_plot], api_name="compare_models",
            concurrency_id="model-comparison",
        )
