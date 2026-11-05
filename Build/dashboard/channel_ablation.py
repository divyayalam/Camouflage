"""Inference-only masking of learned feature maps; saved weights are never edited."""

from functools import lru_cache
import re

import keras
import numpy as np


def parse_channels(text, count):
    """Accept zero-based channel IDs separated by commas or whitespace."""
    if not text or not text.strip():
        return []
    tokens = re.split(r"[,\s]+", text.strip())
    if len(text) > 2000 or any(not re.fullmatch(r"[0-9]+", token) for token in tokens):
        raise ValueError("Enter channel IDs separated by commas, for example 0, 1, 2.")
    channels = sorted({int(token) for token in tokens})
    if any(channel >= count for channel in channels):
        raise ValueError(f"Channel IDs must be between 0 and {count - 1}.")
    return channels


@lru_cache(maxsize=2)
def ablation_parts(model):
    """Split these functional CNNs after the last convolution and its normalization."""
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("Channel ablation requires a single-input, single-output CNN.")
    indices = [i for i, layer in enumerate(model.layers) if isinstance(layer, keras.layers.Conv2D)]
    if not indices:
        raise ValueError("No convolutional feature channels are available in this model.")
    index = indices[-1]
    # Mask after BatchNormalization so its learned offset cannot restore a zeroed channel.
    if index + 1 < len(model.layers) and isinstance(model.layers[index + 1], keras.layers.BatchNormalization):
        index += 1
    feature_layer = model.layers[index]
    features = feature_layer.output
    if len(features.shape) != 4 or features.shape[-1] is None:
        raise ValueError("This model does not expose supported spatial feature maps.")
    head = keras.Model(model.inputs[0], features)
    tail = keras.Model(features, model.outputs[0])
    return head, tail, int(features.shape[-1]), feature_layer.name


def predict_with_ablation(model, batch, channels):
    head, tail, count, layer_name = ablation_parts(model)
    if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < count for i in channels):
        raise ValueError(f"Channel IDs must be integers between 0 and {count - 1}.")
    features = head(batch, training=False)
    mask = np.ones((1, 1, 1, count), dtype=np.float32)
    mask[..., channels] = 0.0
    # Both parts share the original weights. Only this call's activations are masked.
    result = tail(features * mask, training=False)
    if isinstance(result, (list, tuple)):
        result = result[0]
    return np.asarray(result), layer_name
