import gradio as gr
import numpy as np

from shortcut_experiments import RUNS, load_run


UNKNOWN_LABEL = "Unknown"
RESULT_HEADERS = [
    "Training run",
    "True class",
    "Normal prediction",
    "Normal confidence",
    "Conflict prediction",
    "Conflict confidence",
    "Prediction changed",
    "Compared class",
    "Score change (pp)",
]


def compare_pair(normal_image, conflict_image, run, true_label, history, preprocess_image, class_names):
    if normal_image is None or conflict_image is None:
        raise gr.Error("Upload both a normal image and its matching conflict image.")
    if true_label not in (None, UNKNOWN_LABEL) and true_label not in class_names:
        raise gr.Error("Select one of the animal classes or Unknown.")

    # Batch both conditions so the selected checkpoint and preprocessing stay fixed.
    batch = np.concatenate((preprocess_image(normal_image), preprocess_image(conflict_image)), axis=0)
    normal, conflict = load_run(run)(batch, training=False).numpy()
    normal_index, conflict_index = int(np.argmax(normal)), int(np.argmax(conflict))
    normal_class, conflict_class = class_names[normal_index], class_names[conflict_index]
    known_label = true_label if true_label in class_names else None
    reference_index = class_names.index(known_label) if known_label else normal_index
    reference_class = class_names[reference_index]

    normal_top, conflict_top = float(normal[normal_index]), float(conflict[conflict_index])
    normal_ref, conflict_ref = float(normal[reference_index]), float(conflict[reference_index])
    score_change = (conflict_ref - normal_ref) * 100
    changed = normal_index != conflict_index
    row = [
        run, known_label or "—", normal_class, f"{normal_top:.1%}",
        conflict_class, f"{conflict_top:.1%}", "Yes" if changed else "No",
        reference_class, f"{score_change:+.1f}",
    ]
    results = [*(history or []), row]
    summary = (
        f"**{run} · Prediction {'changed' if changed else 'unchanged'}:** "
        f"{normal_class} ({normal_top:.1%}) → {conflict_class} ({conflict_top:.1%})  \n"
        f"**{reference_class} score:** {normal_ref:.1%} → {conflict_ref:.1%} "
        f"({score_change:+.1f} percentage points)."
    )
    normal_scores = {label: float(score) for label, score in zip(class_names, normal)}
    conflict_scores = {label: float(score) for label, score in zip(class_names, conflict)}
    return normal_scores, conflict_scores, summary, results, results


def add_normal_conflict_tab(preprocess_image, class_names):
    with gr.Tab("Normal vs. Conflict", id="normal-conflict"):
        gr.Markdown(
            "### Compare two versions of the same image\n"
            "Upload a matched pair. The conflict image should show the same animal with "
            "the visual cue changed. Both images use the same training run."
        )
        with gr.Row():
            run = gr.Dropdown(choices=list(RUNS), value="Original", label="Training run")
            true_label = gr.Dropdown(
                choices=[UNKNOWN_LABEL, *class_names], value=UNKNOWN_LABEL,
                label="True animal class (optional)",
                info="When supplied, the score change follows the true class.",
            )
        with gr.Row():
            with gr.Column():
                normal_image = gr.Image(type="pil", label="Normal image")
                normal_scores = gr.Label(label="Normal prediction and confidence", num_top_classes=len(class_names))
            with gr.Column():
                conflict_image = gr.Image(type="pil", label="Conflict image")
                conflict_scores = gr.Label(label="Conflict prediction and confidence", num_top_classes=len(class_names))

        compare = gr.Button("Compare pair", variant="primary")
        summary = gr.Markdown()
        gr.Markdown(
            "### Session results\n"
            "Each comparison adds a row. If the true class is unknown, the score change "
            "follows the class predicted on the normal image. Results reset on reload."
        )
        history = gr.State([])
        table = gr.Dataframe(
            headers=RESULT_HEADERS, value=[], datatype="str", interactive=False,
            label="Normal versus conflict comparisons",
        )
        clear = gr.Button("Clear session results")

        def predict_pair(normal, conflict, selected_run, actual_class, saved_results):
            return compare_pair(
                normal, conflict, selected_run, actual_class, saved_results, preprocess_image, class_names,
            )

        compare.click(
            fn=predict_pair,
            inputs=[normal_image, conflict_image, run, true_label, history],
            outputs=[normal_scores, conflict_scores, summary, history, table],
            api_name="compare_normal_conflict",
            concurrency_id="normal-conflict-results",
        )
        clear.click(
            fn=lambda: ([], []), outputs=[history, table], api_name="clear_comparison_results",
            concurrency_id="normal-conflict-results",
        )
