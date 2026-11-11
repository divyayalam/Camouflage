# CAMouflage dashboard

An interactive six-class image analysis app: prediction, cue interventions, Shortcut Reliance Scores, Grad-CAM, feature-channel ablation, paired-image comparisons, and recorded final experiment results.

## Run

Use the project's existing Python environment and launch `app.py`. The tested environment is Python 3.13, TensorFlow 2.21.0, Keras 3.15.1, and Gradio 6.30.0. `requirements-dashboard.txt` records the complete tested dashboard package versions for a fresh environment.

The dashboard has four views:

- **Analyze Image:** prediction, color/patch SRS, a four-view Grad-CAM gallery, intervention details, channel masking, and a downloadable JSON image report.
- **Compare Images:** normal/conflict and shortcut-removal measurements, class confidence, and per-session history. Clearing history retains the current-pair graph.
- **Compare Models:** identical inputs across two installed checkpoints, class scores, SRS, and optional correctness against a supplied true class.
- **Experiment Results:** Jay's final saved results, figures, 200-image confidence comparisons, and an optional importer for an exported confidence report.

Models are loaded lazily, cached for inference, and never trained or modified by the app. Long actions disable their relevant controls and restore them after completion or validation errors. Changing inputs clears stale current results.

## Image and metric definitions

Preprocessing follows the final notebook: EXIF orientation, RGB conversion, center square crop, bicubic resize to 32 × 32, float32 pixels divided by 255. The six output classes are bird, cat, deer, dog, frog, horse.

The SRS reference is each model's **own prediction on the unaltered input**, not the true class. A true label is not needed:

- **Color SRS:** 100 × the fraction of grayscale and five non-identity RGB permutations that change that prediction.
- **Patch SRS:** 100 × the fraction of the five patch classes other than that reference prediction which pull the prediction to the patch's class.
- Patches are 4 × 4 in the top-left corner: bird red, cat green, deer blue, dog yellow, frog magenta, horse cyan. A neutral patch is RGB (0.5, 0.5, 0.5) and is a separate occlusion control.
- Confidence drops in the per-image SRS report follow the reference prediction. The imported paired confidence study instead follows the **true class**. Do not mix these definitions.
- Grad-CAM uses the final convolution output, gradients of log class probability, per-image spatial gradient averaging, positive contributions, bilinear upsampling, and epsilon-safe normalization. It follows final notebook cell 49 without mutating model activations or weights.
- Corner attention is the proportion of heatmap mass in the patch region divided by its area fraction (1/64). A ratio of 1 means uniform attention. Maps without positive evidence are explicitly identified and have no interpretable attention ratio.
- Learned-channel ablation is separate from color/patch interventions. It zeroes selected feature maps after the last convolution block's BatchNormalization, without editing weights.

These are sensitivity probes, not a guarantee of correctness or a causal explanation of shortcut learning. Heatmaps are relative-evidence visualizations, not segmentation masks.

## Recorded results and provenance

`assets/notebook_results.json` and `assets/recorded_results.csv` were extracted from the saved outputs of the supplied `cifar10_animal_classification-3.ipynb`, whose SHA-256 is recorded in the JSON. Eight PNGs under `assets/` are the notebook's actual embedded figure outputs, not recreated or inferred plots. Six example PNGs are unmodified held-out CIFAR-10 animal images selected by test-set order.

The recorded final experiment has 6,000 images per accuracy condition, 1,200 in its SRS sample, 300 in its Grad-CAM attention sample, and 200 per paired-confidence comparison. The twelve model summaries include the color and patch consistency winners (`col_mixed_l12`, `patch_l12`). Saved accuracy tables retain their displayed precision; SRS means preserve the final notebook's one-decimal console precision. The bundled confidence tables summarize all thirteen models in the notebook's 200-image study. Raw per-image confidence exports are loaded only when a real `track1_results.json` is supplied.

The live model selector includes only installed model files. The current local grayscale checkpoint (`g35_luma`) is a separate local training run from Jay's recorded grayscale checkpoint (`g35_tone`). Likewise, the archived run1 models on this computer are not assumed to have identical weights to Jay's final run1 checkpoint. Recorded consistency results are never presented as freshly computed live predictions.

## Optional final model assets

Jay's GitHub branch `Jay-track1` was checked at commit `3ab4add74d68a34cc07cde1c454fa674cf07b0e1`: the notebook is committed, while the final checkpoint and JSON/CSV exports are stored in its Drive environment. No GitHub release assets were available.

To enable those checkpoints live, place the exported folders below `experiment_runs`, or set `CAMOUFLAGE_MODELS_DIR` to the exported model root, then restart:

```text
consistency/<variant>/<variant>_model.keras
grayscale/<variant>/<variant>_model.keras
shortcut_experiment/run1_original_shortcut.keras
shortcut_experiment/run4_heavy_shortcut.keras
```

No background training, downloads, Git actions, or third-party sharing occurs when the dashboard runs. Keep model files matched to their experiment metadata. Restart after replacing model files because loaded models are cached.

## Publish

Include `app.py`, `analysis_core.py`, `final_dashboard.py`, `dashboard_ui.py`, `channel_ablation.py`, `track1_results.py`, `assets/`, this README, and `requirements-dashboard.txt`. Preserve the existing `experiment_runs` checkpoint assets. Include the validation report and screenshots if publishing the completed review package. Legacy dashboard modules can remain in the checkout; the final entry point uses the modules listed above.
