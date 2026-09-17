# Camouflage

CAMouflage is a research style tool for detecting and diagnosing "shortcut learning" in image classifiers - cases where a model achieves high accuracy by relying on spurious cues (background color, a watermark, a colored patch) instead of the actual object it's supposed to recognize.

Train a small CNN on a dataset with a deliberately injected shortcut, then use Grad-CAM and feature/channel ablation to visualize and quantify how much the model depends on that shortcut vs. the real signal. Everything is explorable live through an interactive dashboard: pick an image, see where the model is looking, ablate suspect features, and watch the prediction change in real time.

## Overview

Image classifiers can learn to exploit spurious correlations in training data rather than the features we actually care about. 
A model trained on images where, say, cats are usually photographed on red backgrounds may learn "red → cat" instead of "cat → cat."

CAMouflage makes this failure mode visible and measurable by:

1. Training a CNN on a dataset with a controlled, injected shortcut.
2. Evaluating it on both normal and **shortcut-conflict** data (the shortcut and label deliberately mismatched) to expose hidden reliance.
3. Using **Grad-CAM** to visualize where the model is actually looking.
4. Quantifying reliance with a custom **Shortcut Reliance Score (SRS)**.
5. **Ablating** suspect neurons/channels and observing the effect on predictions live.
6. *(Stretch)* Training a shortcut-mitigated model and comparing it against the original.

## Tech Stack

**Modeling**
- [PyTorch](https://pytorch.org/) — model definition & training
- [TorchVision](https://pytorch.org/vision/stable/index.html) — datasets, transforms, pretrained backbones

**Interpretability**
- Grad-CAM (custom / [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam))
- Custom forward-hook–based ablation utilities

**Analysis & Visualization**
- NumPy
- Matplotlib

**Interface**
- [Gradio](https://www.gradio.app/) for the interactive dashboard (React frontend possible as a stretch goal)

**Infra**
- Google Colab for training/experimentation
- Local machine / GPU recommended if pursuing the live webcam stretch goal
