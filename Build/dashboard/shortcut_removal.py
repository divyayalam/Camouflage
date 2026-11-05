import json

import gradio as gr
from matplotlib.figure import Figure
import numpy as np

from shortcut_experiments import RUNS, RUNS_DIR, load_run
from track1_results import add_track1_results_panel


HEADERS = [
    "Comparison", "Training run", "True class", "Class tracked",
    "Before prediction", "After prediction", "Prediction changed",
    "Before confidence (%)", "After confidence (%)", "Confidence drop (pp)",
]


def confidence_graph(before, after, class_names):
    figure = Figure(figsize=(7, 4.2), layout="constrained")
    axis = figure.subplots()
    x = np.arange(len(class_names))
    axis.bar(x - 0.19, np.asarray(before) * 100, 0.38, label="Present", color="#2563eb")
    axis.bar(x + 0.19, np.asarray(after) * 100, 0.38, label="Removed / neutralized", color="#ea580c")
    axis.set_xticks(x, class_names)
    axis.set_ylim(0, 110)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.set_ylabel("Confidence (%)")
    axis.set_title("Current pair: class confidence")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    return figure


def session_graph(records):
    figure = Figure(figsize=(7, 4.2), layout="constrained")
    axis = figure.subplots()
    recent = records[-20:]
    values = [record["change_pp"] for record in recent]
    x = np.arange(len(recent))
    axis.bar(x, values, color=["#2563eb" if value >= 0 else "#ea580c" for value in values])
    axis.axhline(0, color="#334155", linewidth=0.8)
    axis.set_xticks(x, [f'{r["comparison"]}\n{r["run"]}\n{r["reference"]}' for r in recent], fontsize=8)
    axis.set_ylabel("With − without (percentage points)")
    axis.set_title("Session: latest 20 confidence drops")
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    if values and max(abs(value) for value in values) < 0.01:
        axis.set_ylim(-1, 1)
    return figure


def archived_accuracy_graph():
    names = list(RUNS)
    matrix = np.full((len(names), len(names)), np.nan)
    for row, run in enumerate(RUNS.values()):
        path = RUNS_DIR / run / f"{run}_results.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            for column, test in enumerate(RUNS.values()):
                score = record.get(f"test_{test}")
                if score is not None:
                    matrix[row, column] = 100 * float(score)
    figure = Figure(figsize=(7, 4.4), layout="constrained")
    axis = figure.subplots()
    if not np.isfinite(matrix).any():
        axis.text(0.5, 0.5, "Recorded run results are unavailable.", ha="center", va="center")
        axis.set_axis_off()
        return figure
    image = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=100, cmap="Blues")
    axis.set_xticks(np.arange(len(names)), names)
    axis.set_yticks(np.arange(len(names)), names)
    axis.set_xlabel("Archived test condition")
    axis.set_ylabel("Trained model")
    axis.set_title("Recorded accuracy across augmentation conditions")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(
            column, row, f"{value:.1f}%" if np.isfinite(value) else "—",
            ha="center", va="center", color="white" if value >= 60 else "#0f172a",
        )
    figure.colorbar(image, ax=axis, label="Recorded accuracy (%)")
    return figure


def removal_rows(records):
    return [
        [
            r["comparison"], r["run"], r["true_class"] or "—", r["reference"],
            r["before_prediction"], r["after_prediction"], "Yes" if r["flipped"] else "No",
            round(r["before_pct"], 2), round(r["after_pct"], 2), round(r["change_pp"], 2),
        ]
        for r in records
    ]


def analyze_removal(present_image, removed_image, run, true_label, history, preprocess_image, class_names):
    if present_image is None or removed_image is None:
        raise gr.Error("Upload both the shortcut-present image and the removed/neutralized image.")
    if true_label not in class_names:
        raise gr.Error("Select the true animal class to measure true-class confidence.")

    batch = np.concatenate((preprocess_image(present_image), preprocess_image(removed_image)), axis=0)
    before, after = load_run(run)(batch, training=False).numpy()
    before_index, after_index = int(np.argmax(before)), int(np.argmax(after))
    known_label = true_label
    reference_index = class_names.index(true_label)
    reference = class_names[reference_index]
    before_pct, after_pct = float(before[reference_index]) * 100, float(after[reference_index]) * 100
    change = before_pct - after_pct
    record = {
        "comparison": len(history or []) + 1, "run": run, "true_class": known_label,
        "reference": reference, "before_prediction": class_names[before_index],
        "after_prediction": class_names[after_index], "flipped": before_index != after_index,
        "before_pct": before_pct, "after_pct": after_pct, "change_pp": change,
    }
    records = [*(history or []), record]
    summary = (
        f'**Class tracked: {reference}** ('
        f'{"true class" if known_label else "prediction on the present image"}).  \n'
        f'**Prediction {"changed" if record["flipped"] else "unchanged"}:** '
        f'{record["before_prediction"]} → {record["after_prediction"]}.'
    )
    return (
        {name: float(value) for name, value in zip(class_names, before)},
        {name: float(value) for name, value in zip(class_names, after)},
        summary, before_pct, after_pct, change,
        confidence_graph(before, after, class_names), session_graph(records),
        records, removal_rows(records),
    )


def add_removal_tab(preprocess_image, class_names):
    with gr.Tab("Shortcut Removal", id="shortcut-removal"):
        with gr.Tabs():
            with gr.Tab("Image pair", id="removal-pair"):
                gr.Markdown(
                    "### Measure confidence before and after removal\n"
                    "Upload the same animal with the shortcut present and removed or neutralized. "
                    "Keep the animal and other conditions fixed; both images use the same selected model. The installed runs are augmentation baselines; the Track 1 results view can compare patch-trained models."
                )
                with gr.Row():
                    run = gr.Dropdown(choices=list(RUNS), value="Original", label="Baseline training run")
                    true_label = gr.Dropdown(
                        choices=list(class_names), value=None, label="True animal class",
                        info="Required: confidence follows the true class, as in Jay's experiment.",
                    )
                with gr.Row():
                    with gr.Column():
                        present = gr.Image(type="pil", label="Shortcut present")
                        before_scores = gr.Label(label="Before prediction and confidence", num_top_classes=len(class_names))
                    with gr.Column():
                        removed = gr.Image(type="pil", label="Shortcut removed / neutralized")
                        after_scores = gr.Label(label="After prediction and confidence", num_top_classes=len(class_names))
                compare = gr.Button("Analyze removal", variant="primary")
                summary = gr.Markdown()
                with gr.Row():
                    before_confidence = gr.Number(label="Before confidence (%)", precision=2, interactive=False, placeholder="Analyze a pair")
                    after_confidence = gr.Number(label="After confidence (%)", precision=2, interactive=False, placeholder="Analyze a pair")
                    difference = gr.Number(label="Confidence drop: with − without (pp)", precision=2, interactive=False, placeholder="Analyze a pair")
                gr.Markdown("All three numbers follow the true class. A positive drop means its confidence fell without the shortcut.")
                with gr.Row():
                    current_plot = gr.Plot(label="Confidence for each animal", format="png")
                    history_plot = gr.Plot(label="Session confidence drops", format="png")
                gr.Markdown(
                    "### Session measurements\n"
                    "Rows are uploaded-pair measurements for this browser session. They reset on reload. "
                    "Each graph label identifies the run and class tracked."
                )
                history = gr.State([])
                table = gr.Dataframe(headers=HEADERS, value=[], interactive=False, label="Before/after results")
                clear = gr.Button("Clear removal session results")
                gr.Markdown(
                    "### Archived experiment results\n"
                    "This graph uses Track 1's saved accuracy reports. Its test conditions include "
                    "rotation and color augmentation; it is separate from the removal measurements above."
                )
                gr.Plot(value=archived_accuracy_graph(), label="Recorded model accuracy", format="png")

                def compare_images(before, after, selected_run, actual_class, saved_results):
                    return analyze_removal(
                        before, after, selected_run, actual_class, saved_results, preprocess_image, class_names,
                    )

                compare.click(
                    fn=compare_images, inputs=[present, removed, run, true_label, history],
                    outputs=[
                        before_scores, after_scores, summary, before_confidence, after_confidence,
                        difference, current_plot, history_plot, history, table,
                    ],
                    api_name="analyze_shortcut_removal", concurrency_id="removal-results",
                )
                clear.click(
                    fn=lambda: ([], [], None), outputs=[history, table, history_plot],
                    api_name="clear_removal_results", concurrency_id="removal-results",
                )

            with gr.Tab("Track 1 results", id="track1-results"):
                add_track1_results_panel()
