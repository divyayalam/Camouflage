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
