import json
from pathlib import Path

import gradio as gr
from matplotlib.figure import Figure
import numpy as np
from dashboard_ui import INFERENCE_UI


CLASS_NAMES = ["bird", "cat", "deer", "dog", "frog", "horse"]
PAIR_NAMES = {
    "patch_vs_neutral": "Class patch → neutral gray patch",
    "patch_vs_removed": "Class patch → clean image",
    "color_vs_grayscale": "Color → grayscale",
    "color_vs_bgr": "Color → red/blue swapped",
}
MODEL_HEADERS = ["Model", "Images", "With (%)", "Without (%)", "Mean drop (pp)", "Std drop (pp)", "Flips", "Accuracy with (%)", "Accuracy without (%)"]
IMAGE_HEADERS = ["Image index", "True class", "Prediction with", "Prediction without", "With (%)", "Without (%)", "Drop (pp)", "Flipped", "Flip type"]


def model_name(name):
    level, _, kind = name.partition("|")
    if kind == "consistency":
        return f"Consistency ({level})"
    if kind == "grayscale":
        return f"Grayscale augmentation ({level})"
    level_name = {"run1_original": "Original", "run2_mild": "Mild", "run3_stronger": "Stronger", "run4_heavy": "Heavy"}.get(level, level)
    return f'{level_name} ({"patch-trained" if kind == "shortcut" else "baseline"})'


def read_track1(path):
    if not path:
        raise gr.Error("Choose the exported track1_results.json file.")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        meta, models = payload["meta"], payload["models"]
        if meta["delta_definition"].replace(" ", "") != "conf_with-conf_without":
            raise ValueError("The report must define delta as conf_with minus conf_without.")
        if meta["confidence_definition"].lower().strip() != "softmax probability of the true class":
            raise ValueError("The report must contain true-class confidence.")
        if not models:
            raise ValueError("The report has no models.")
        for name, model in models.items():
            if not isinstance(name, str) or not model["pairs"]:
                raise ValueError("A model has no measured pairs.")
            for pair_name, pair in model["pairs"].items():
                images = pair["images"]
                if pair_name not in PAIR_NAMES or not isinstance(images, list) or not images:
                    raise ValueError("The report has an unsupported or empty pair.")
                seen = set()
                for image in images:
                    index = image["image_idx"]
                    if not isinstance(index, int) or index in seen:
                        raise ValueError("Image indices must be unique integers within each pair.")
                    seen.add(index)
                    true = image["true_name"]
                    if true not in CLASS_NAMES or image["true_label"] != CLASS_NAMES.index(true):
                        raise ValueError("A true label does not match the six-animal class mapping.")
                    if image["pred_with"] not in CLASS_NAMES or image["pred_without"] not in CLASS_NAMES:
                        raise ValueError("A prediction has an unknown animal class.")
                    values = np.asarray([image["conf_with"], image["conf_without"], image["delta"]], dtype=float)
                    if not np.isfinite(values).all() or not np.all((values[:2] >= 0) & (values[:2] <= 1)):
                        raise ValueError("Confidence values must be finite probabilities between zero and one.")
                    if not np.isclose(values[0] - values[1], values[2], atol=2.1e-6, rtol=0):
                        raise ValueError("A recorded delta disagrees with conf_with minus conf_without.")
                    if image["flipped"] != (image["pred_with"] != image["pred_without"]):
                        raise ValueError("A flip flag disagrees with its predictions.")
                if int(meta["n_images"]) != len(images) or int(pair["summary"]["n_images"]) != len(images):
                    raise ValueError("The image count does not match the report metadata.")
        return payload
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise gr.Error(f"Could not read this Track 1 export: {error}") from error


def measured_stats(pair):
    images = pair["images"]
    before = np.asarray([image["conf_with"] for image in images], dtype=float)
    after = np.asarray([image["conf_without"] for image in images], dtype=float)
    drops = np.asarray([image["delta"] for image in images], dtype=float) * 100
    return {
        "n": len(images), "before": float(before.mean() * 100), "after": float(after.mean() * 100),
        "drop": float(drops.mean()), "std": float(drops.std(ddof=1)) if len(images) > 1 else 0.,
        "flips": sum(bool(image["flipped"]) for image in images),
        "accuracy_with": 100 * np.mean([image["pred_with"] == image["true_name"] for image in images]),
        "accuracy_without": 100 * np.mean([image["pred_without"] == image["true_name"] for image in images]),
    }


def show_track1(payload, selected_model, selected_pair):
    if not payload or selected_model not in payload["models"]:
        raise gr.Error("Load a Track 1 report first.")
    if selected_pair not in payload["models"][selected_model]["pairs"]:
        raise gr.Error("Select an available comparison pair.")
    pair = payload["models"][selected_model]["pairs"][selected_pair]
    stats = measured_stats(pair)
    model_rows, labels, with_means, without_means = [], [], [], []
    for name, model in payload["models"].items():
        if selected_pair not in model["pairs"]:
            continue
        item = measured_stats(model["pairs"][selected_pair])
        labels.append(model_name(name)); with_means.append(item["before"]); without_means.append(item["after"])
        model_rows.append([
            model_name(name), item["n"], round(item["before"], 2), round(item["after"], 2),
            round(item["drop"], 2), round(item["std"], 2), item["flips"],
            round(float(item["accuracy_with"]), 2), round(float(item["accuracy_without"]), 2),
        ])
    image_rows = [
        [
            image["image_idx"], image["true_name"], image["pred_with"], image["pred_without"],
            round(float(image["conf_with"]) * 100, 2), round(float(image["conf_without"]) * 100, 2),
            round(float(image["delta"]) * 100, 2), "Yes" if image["flipped"] else "No", image.get("flip_type") or "—",
        ]
        for image in pair["images"]
    ]

    comparison = Figure(figsize=(7, 4.5), layout="constrained")
    axis = comparison.subplots()
    y = np.arange(len(labels))
    axis.barh(y - .19, with_means, .38, label="With", color="#2563eb")
    axis.barh(y + .19, without_means, .38, label="Without", color="#ea580c")
    axis.set_yticks(y, labels, fontsize=8); axis.invert_yaxis()
    axis.set_xlim(0, 110); axis.set_xticks(np.arange(0, 101, 20))
    axis.set_xlabel("Mean true-class confidence (%)")
    axis.set_title(PAIR_NAMES[selected_pair]); axis.legend(fontsize=8)
    axis.grid(axis="x", alpha=.18); axis.set_axisbelow(True)

    histogram = Figure(figsize=(7, 4.5), layout="constrained")
    axis = histogram.subplots()
    drops = [float(image["delta"]) * 100 for image in pair["images"]]
    axis.hist(drops, bins=min(20, max(1, len(drops))), color="#2563eb", edgecolor="white")
    axis.axvline(0, color="#334155", linewidth=1)
    axis.set_xlabel("Confidence drop: with − without (percentage points)")
    axis.set_ylabel("Images"); axis.set_title(f"{model_name(selected_model)}: paired confidence drops")
    axis.grid(axis="y", alpha=.18); axis.set_axisbelow(True)

    summary = (
        f'**{model_name(selected_model)} · {PAIR_NAMES[selected_pair]}**  \n'
        f'{stats["n"]} matched test images; {stats["flips"]} predictions flipped. '
        f'Recorded run: {payload["meta"].get("created", "not specified")}. '
        "Values below are calculated from the imported recorded predictions."
    )
    return (
        summary, stats["before"], stats["after"], stats["drop"], stats["std"], stats["flips"],
        comparison, histogram, model_rows, image_rows,
    )


def load_and_show_track1(path):
    payload = read_track1(path)
    model = next(iter(payload["models"]))
    pair = next(iter(payload["models"][model]["pairs"]))
    return (
        payload,
        gr.update(choices=[(model_name(name), name) for name in payload["models"]], value=model),
        gr.update(choices=[(PAIR_NAMES[name], name) for name in payload["models"][model]["pairs"]], value=pair),
        *show_track1(payload, model, pair),
    )


def add_track1_results_panel():
    gr.Markdown(
        "### Open a confidence experiment\n"
        "Choose a saved track1_results.json report. These results follow the true animal class. "
        "A positive drop means its confidence fell without the shortcut."
    )
    file = gr.File(label="Track 1 results JSON", file_types=[".json"], type="filepath")
    load = gr.Button("Load experiment", variant="primary")
    data = gr.State(None)
    with gr.Row():
        model = gr.Dropdown(choices=[], label="Recorded model")
        pair = gr.Dropdown(choices=[], label="Recorded pair")
    summary = gr.Markdown()
    with gr.Row():
        before = gr.Number(label="Mean confidence with (%)", precision=2, interactive=False, placeholder="Load results")
        after = gr.Number(label="Mean confidence without (%)", precision=2, interactive=False, placeholder="Load results")
        drop = gr.Number(label="Mean confidence drop (pp)", precision=2, interactive=False, placeholder="Load results")
        std = gr.Number(label="Std of confidence drops (pp)", precision=2, interactive=False, placeholder="Load results")
        flips = gr.Number(label="Prediction flips", precision=0, interactive=False, placeholder="Load results")
    with gr.Row():
        model_plot = gr.Plot(label="With/without confidence across models", format="png")
        drop_plot = gr.Plot(label="Per-image confidence drop distribution", format="png")
    model_table = gr.Dataframe(headers=MODEL_HEADERS, value=[], interactive=False, label="Recorded results by model")
    image_table = gr.Dataframe(headers=IMAGE_HEADERS, value=[], interactive=False, label="Recorded per-image results")
    outputs = [summary, before, after, drop, std, flips, model_plot, drop_plot, model_table, image_table]
    load.click(
        fn=load_and_show_track1, inputs=file, outputs=[data, model, pair, *outputs],
        api_name="load_track1_results", concurrency_id="track1-results",
        show_progress_on=[summary], **INFERENCE_UI,
    )

    def select_model(payload, selected):
        if not payload or selected not in payload["models"]:
            raise gr.Error("Load a Track 1 report first.")
        choices = payload["models"][selected]["pairs"]
        selected_pair = next(iter(choices))
        return gr.update(choices=[(PAIR_NAMES[name], name) for name in choices], value=selected_pair), *show_track1(payload, selected, selected_pair)

    model.input(
        fn=select_model, inputs=[data, model], outputs=[pair, *outputs],
        api_name="select_track1_model", concurrency_id="track1-results",
    )
    pair.input(
        fn=show_track1, inputs=[data, model, pair], outputs=outputs,
        api_name="select_track1_pair", concurrency_id="track1-results",
    )
