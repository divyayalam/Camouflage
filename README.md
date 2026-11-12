# CAMouflage

CAMouflage is a computer vision project that investigates shortcut learning in image classification models. The goal is to determine whether a CNN is learning the intended object features or relying on contextual shortcuts such as background, color, texture, or other visual cues.

The project uses a six-class subset of CIFAR-10:

- Bird
- Cat
- Deer
- Dog
- Frog
- Horse

## Project Goals

The project focuses on:

- Training and evaluating a baseline CNN
- Investigating prediction failures
- Measuring model robustness across shortcut strengths
- Testing background and color dependence
- Using Grad-CAM to visualize model attention
- Measuring shortcut reliance
- Identifying shortcut-sensitive CNN channels
- Performing feature/channel ablation
- Comparing predictions before and after ablation

## Environment

The project is primarily developed using Google Colab.

Main libraries:

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Setup

Clone the repository:

```bash
git clone https://github.com/divyayalam/Camouflage.git
cd Camouflage
```

Install the required Python libraries:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Google Colab can also be used to run the Track 2 notebook.

## Track 2 Notebook

The main Track 2 analysis notebook is:

```text
track2_predictionAnalysis.ipynb
```

The notebook contains the model explainability, robustness, shortcut-reliance, Grad-CAM, and feature-ablation experiments.

## Baseline Prediction Analysis

The baseline model was evaluated on six CIFAR-10 animal classes.

Track 2 investigated:

- Correct predictions
- Incorrect predictions
- Prediction confidence
- Confusion matrices
- Precision
- Recall
- F1 score

The failure analysis showed that the model could make high-confidence mistakes and appeared sensitive to background, texture, and color information.

## Shortcut Strength Experiments

Models were compared across multiple test conditions:

- Original
- Mild
- Stronger
- Heavy

The original model performed best on clean images but degraded significantly as distortion strength increased.

Models trained with stronger augmentation were more robust under stronger test conditions.

## Color Intervention Experiments

The baseline model achieved:

- Original accuracy: 71.10%
- Grayscale accuracy: 61.22%
- Color-shuffle accuracy: 55.92%

The accuracy decrease suggests that color information contributes significantly to the model's predictions.

## Object and Background Experiments

A center-region proxy mask was used to approximate the object region.

Results:

- Original accuracy: 71.10%
- Object-only accuracy: 39.48%
- Background-only accuracy: 44.83%
- Object-only retention: 0.555

The relatively high background-only accuracy and low object-only retention suggest that the CNN uses contextual background information in addition to the intended object.

## Background-Swap Experiment

The background of each image was replaced with background information from an image belonging to another class.

Results:

- Original accuracy: 71.10%
- Background-swap accuracy: 28.02%
- Conflict drop: 43.08 percentage points

The large conflict drop provides evidence that background information strongly affects the model's predictions.

## Grad-CAM Analysis

Grad-CAM was used to visualize which regions contributed to CNN predictions.

For one analyzed example:

- Original background attention: 67.27%
- Background-swap attention: 71.30%

The increased attention after the background swap is consistent with the model relying on contextual information.

## Shortcut Reliance Score

The project uses a simple Shortcut Reliance Score to summarize multiple indicators of shortcut dependence.

The score combines:

1. Conflict drop
2. Background-only accuracy
3. Loss of object-only retention

Formula:

```text
SRS =
(
    Conflict Drop
    + Background-Only Accuracy
    + (1 - Object-Only Retention)
) / 3
```

Higher values indicate stronger evidence of reliance on non-object features.

This is a project-defined metric rather than a standardized machine-learning metric.

## Feature / Channel Ablation

Activation analysis was used to identify channels that changed strongly between normal and background-swapped images.

The final candidate channels were:

```text
109, 62, 105, 63, 31
```

These channels were ablated by setting their activations to zero before the classifier layers.

### Ablation Results

On a targeted shortcut-sensitive sample:

| Condition | Before Ablation | After Ablation |
|---|---:|---:|
| Normal accuracy | 100% | 96% |
| Background-swap accuracy | 0% | 6% |
| Normal true-class confidence | 76.86% | 76.38% |
| Background-swap true-class confidence | 10.79% | 12.27% |

Six background-swap predictions recovered to the correct class after ablation.

Examples included:

- horse → dog
- frog → cat
- horse → cat
- horse → bird

The 100% normal and 0% background-swap starting accuracies are by construction because this evaluation subset contains images that were correct normally and incorrect after the background swap.

## Interpretation

The experiments provide evidence that the baseline CNN relies on shortcut-related contextual information.

Background modifications caused major accuracy decreases, and Grad-CAM showed substantial attention outside the approximate object region.

Channel ablation produced a partial improvement on shortcut-dependent examples while preserving most normal-image performance.

The limited recovery rate suggests that shortcut reliance is distributed across multiple learned features rather than being controlled by only a few channels.

## Limitations

- CIFAR-10 images are low resolution.
- The object/background experiments use a center-region proxy rather than true segmentation masks.
- Grad-CAM should not be interpreted as a complete explanation of model reasoning.
- Channel ablation only recovered a small percentage of shortcut-dependent examples.
- The Shortcut Reliance Score is a project-defined summary metric.
- Results from the targeted ablation sample should not be interpreted as overall model accuracy.

## Documentation

Additional methodology and ablation analysis can be found in:

```text
docs/track2_ablation_analysis.md
```

## Repository Structure

```text
Camouflage/
├── README.md
├── docs/
│   └── track2_ablation_analysis.md
├── track2_predictionAnalysis.ipynb
└── .gitignore
```

## Track 2 Summary

Track 2 investigated how the CNN makes predictions and whether those predictions depend on shortcut features.

The completed analysis includes:

- baseline prediction investigation
- evaluation metrics
- shortcut-strength comparison
- color interventions
- object/background testing
- background-conflict testing
- Grad-CAM
- Shortcut Reliance Score
- feature/channel ablation
- final ablation verification

These results can now be integrated into the final dashboard and project demonstration.
