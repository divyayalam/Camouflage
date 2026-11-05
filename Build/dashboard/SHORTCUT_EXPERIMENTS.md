# Shortcut Experiments

Start the app in the same Python environment used for the baseline:

```powershell
python app.py
```

The **Prediction** tab runs the original baseline. In **Shortcut Experiments**, select a training run, upload an animal image, and click **Compare**. Both images pass through the same selected model. Red and blue are swapped after resizing so the two model inputs differ only in their color channels. The score change is measured for the class predicted on the original image.

The accuracy table comes from the training archives. The test conditions there include other changes, such as rotation, and are separate from the live red/blue comparison. This is a color-robustness check; it does not measure a class-linked planted shortcut.

Run assets come from [Track 1's archives](https://github.com/divyayalam/Camouflage/tree/562763c5beb568fe7d1ea824821b3089bb4d9873). If the weights are absent in a fresh checkout, extract each archive's run folder into `experiment_runs` beside the app:

```text
experiment_runs/
  run1_original/run1_original_model.keras
  run2_mild/run2_mild_model.keras
  run3_stronger/run3_stronger_model.keras
  run4_heavy/run4_heavy_model.keras
```

Each run folder also contains its matching `*_results.json`. The model weights are ignored by Git within this folder. They are loaded the first time a run is selected.

## Normal versus conflict

Choose a training run, upload two versions of the same image, then select **Compare pair**. Keep the animal and image content fixed; change the cue in the conflict version. Enter the true animal class if known. Each panel reports its predicted class and confidence scores. The summary reports whether the prediction changed, and the score difference (conflict minus normal) for the true class or the normal prediction when the true class is unknown.

The results table records comparisons for the current browser session. **Clear session results** empties the table; it does not change the saved accuracy table in Shortcut Experiments.

To check the interface before the team provides official pairs, use one animal image and its red/blue-swapped version from the Shortcut Experiments tab. Treat this as a color-robustness demonstration.

## Shortcut removal and numerical results

The **Shortcut Removal** tab has two views.

### Image pair

Upload the same animal with the cue present and removed or neutralized. Select the true animal class; it is required to match Jay's definition of confidence as P(true class). Both images pass through the same selected model.

The number panels show true-class confidence with and without the cue. **Confidence drop = with minus without**, in percentage points. Positive means confidence fell after removal; negative means it increased. This is a paired confidence measurement, not a separately agreed Shortcut Reliance Score.

The class-confidence graph shows all six output scores for the current pair. The session graph shows the most recent 20 confidence drops, labeled by comparison, model, and true class. The table retains all comparisons for the browser session. Clearing removal session results empties this table and history graph.

The available live models are the four augmentation baselines already installed. Their results should not be presented as results from the separately patch-trained models. The archive heatmap shows saved augmentation accuracy and is separate from the live removal measurements.

### Track 1 results

Load the exported **track1_results.json** from the final Track 1 section of Jay's updated notebook. The export contains metadata and per-image predictions; model weights are not needed to display it. The notebook records an export at:

```text
/content/drive/MyDrive/cifar10_animal_runs/track1/2026-10-27_153423/track1_results.json
```

The JSON must define confidence as the softmax probability of the true class and delta as conf_with minus conf_without. Select a model and comparison pair to see mean confidence with/without, mean and standard deviation of drops, prediction flips, a cross-model comparison graph, a per-image histogram, and numerical tables.

Supported pairs follow Jay's notebook: class patch versus neutral gray, class patch versus clean image, color versus grayscale, and color versus RGB-to-BGR. Keep these probe types separate when interpreting results.

The notebook's class patch is 4 by 4 pixels in the top-left corner: bird red, cat green, deer blue, dog yellow, frog magenta, horse cyan. The cue is applied to 95% of the shortcut models' training/validation images. The notebook reports that Original and Heavy patch-trained models exist in its Drive execution environment; those weights are separate files from the notebook and the JSON.

Graphs require Matplotlib in the same Python environment as the dashboard.

## Week 6: model comparison and channel ablation

The **Model Comparison** tab compares the saved `run1_original` checkpoint with your locally trained `g35_luma` checkpoint. The original reference is the same run evaluated in the grayscale notebook, rather than the separate `animal_cnn_final.keras` used by the first Prediction tab.

The grayscale model uses the same CNN architecture, trained afresh with 35% of training examples converted to luma grayscale. This targets color dependence. It is not evidence of mitigation of the planted corner-patch shortcut. The local notebook trained only `g35_luma`, not Sid's full eleven-variant search.

Upload an animal image, choose **Color** or **Grayscale (luma)**, select a model to inspect, and optionally supply the true animal class. Click **Compare models**. Both models always receive the same normalized 32 × 32 RGB input. The grayscale option uses the notebook's luma weights `[0.2989, 0.5870, 0.1140]` and repeats the result across three channels. Both predictions, all class scores, and correctness (when the true class is supplied) appear side by side. The confidence difference uses the true class, or the original model's predicted class when the true class is unknown. It is mitigated minus original, in percentage points; one image does not establish improved accuracy.

The recorded table and chart come from `test_summary.csv`: 6,000 held-out animal images per view. They do not change with uploads. Your local run reports color accuracy 83.33% → 81.97%, luma-grayscale accuracy 77.58% → 80.35%, and average-grayscale accuracy 77.13% → 80.33%. The color–luma gap decreases from 5.75 to 1.62 percentage points. These are observed results from one training run, without uncertainty estimates.

Expand **Optional: turn off internal feature channels**, enable ablation, and enter channel IDs such as `0, 1, 2`. This masks those channels after the final convolution block's BatchNormalization, before pooling. The channels are 128 learned feature maps, not the three input color channels. Only the selected model is masked; the main two-model comparison remains unmasked. The separate ablation results compare that model with its own unmasked prediction. Empty IDs provide a no-change check. Channel IDs are zero-based, from 0 to 127. The dashboard never retrains, edits weights, or saves the mask. Changes show sensitivity to the selected channels; no channel ranking or shortcut identity is inferred.

Keep these matching files together beside the dashboard:

```text
channel_ablation.py
model_comparison.py
experiment_runs/grayscale_local/g35_luma_only/
  best.json
  test_summary.csv
  validation_summary.csv
  local_run_provenance.json
  g35_luma/g35_luma_model.keras
  g35_luma/g35_luma_results.json
```

The original `experiment_runs/run1_original` assets are also required. The grayscale checkpoint is about 10 MB. Although `experiment_runs/.gitignore` ignores new `.keras` files, include this particular checkpoint explicitly when publishing so teammates can run the comparison. Retain `best.json` and the matching metrics from the same local run. Start the dashboard with the existing virtual environment's Python and `app.py`; no new packages are required.

Test at least three animal images in both color and grayscale views. Check both model selectors, a known and unknown true class, ablation off, empty IDs, a few valid IDs, and an out-of-range ID. With empty IDs, scores should match the selected model's normal prediction. Turning ablation off must restore the normal scores.

Week 5's imported patch-removal results still require Jay's separate `track1_results.json`; the Week 6 grayscale report does not replace that export.
