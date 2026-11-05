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
