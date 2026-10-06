"""Reusable classification evaluation utilities (NumPy only).

Functions
---------
confusion_matrix(y_true, y_pred, labels=None)
    Matrix of counts; rows = true class, columns = predicted class.
per_class_accuracy(y_true, y_pred, labels=None)
    Accuracy within each true class (i.e. per-class recall as a fraction of
    that class's samples).
precision_recall_f1(y_true, y_pred, labels=None, average=None)
    Precision, recall and F1 per class, or macro / micro / weighted averages.

Conventions
-----------
* Inputs are 1-D array-likes of equal length (lists, NumPy arrays, pandas
  Series). Labels may be ints or strings.
* If ``labels`` is omitted, the sorted union of labels seen in ``y_true`` and
  ``y_pred`` is used. Pass ``labels`` explicitly to keep a fixed class order or
  to include classes that never appear in the data.
* Any ratio with a zero denominator (e.g. precision for a class that is never
  predicted) is returned as 0.0 rather than NaN.
"""

from __future__ import annotations

import numpy as np

__all__ = ["confusion_matrix", "per_class_accuracy", "precision_recall_f1"]


def _prepare(y_true, y_pred, labels):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, got "
            f"{y_true.shape[0]} and {y_pred.shape[0]}."
        )
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels).ravel()
        if len(np.unique(labels)) != len(labels):
            raise ValueError("labels must be unique.")
    return y_true, y_pred, labels


def _safe_divide(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.zeros(np.broadcast(num, den).shape, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def confusion_matrix(y_true, y_pred, labels=None):
    """Return an (n_classes, n_classes) integer count matrix.

    Entry ``[i, j]`` is the number of samples whose true class is ``labels[i]``
    and whose predicted class is ``labels[j]``. Samples with a true or predicted
    label not in ``labels`` are ignored.

    Returns
    -------
    cm : np.ndarray of int
    labels : np.ndarray
        The class order used for rows and columns.
    """
    y_true, y_pred, labels = _prepare(y_true, y_pred, labels)
    index = {lab: i for i, lab in enumerate(labels.tolist())}
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm, labels


def per_class_accuracy(y_true, y_pred, labels=None):
    """Fraction of each true class that was predicted correctly.

    Returns
    -------
    acc : np.ndarray of float, shape (n_classes,)
    labels : np.ndarray
    """
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    return _safe_divide(np.diag(cm), cm.sum(axis=1)), labels


def precision_recall_f1(y_true, y_pred, labels=None, average=None):
    """Precision, recall and F1.

    Parameters
    ----------
    average : {None, "macro", "micro", "weighted"}
        * ``None``: per-class arrays.
        * ``"macro"``: unweighted mean over classes.
        * ``"micro"``: pooled over all samples (precision == recall == F1 ==
          accuracy for single-label problems).
        * ``"weighted"``: mean weighted by each class's true support.

    Returns
    -------
    dict with keys ``precision``, ``recall``, ``f1``, ``support`` and
    ``labels``. Values are arrays when ``average`` is None, floats otherwise
    (``support`` stays a per-class array; for averages it is the total).
    """
    if average not in (None, "macro", "micro", "weighted"):
        raise ValueError("average must be None, 'macro', 'micro' or 'weighted'.")

    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    pred_count = cm.sum(axis=0)  # tp + fp
    support = cm.sum(axis=1)     # tp + fn

    if average == "micro":
        p = float(_safe_divide(tp.sum(), pred_count.sum()))
        r = float(_safe_divide(tp.sum(), support.sum()))
        f = float(_safe_divide(2 * p * r, p + r))
        return {"precision": p, "recall": r, "f1": f,
                "support": int(support.sum()), "labels": labels}

    p = _safe_divide(tp, pred_count)
    r = _safe_divide(tp, support)
    f = _safe_divide(2 * p * r, p + r)

    if average is None:
        return {"precision": p, "recall": r, "f1": f,
                "support": support, "labels": labels}

    if average == "macro":
        w = np.ones(len(labels))
    else:  # weighted
        w = support.astype(float)
    total = w.sum()
    if total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "support": int(support.sum()), "labels": labels}
    return {
        "precision": float((p * w).sum() / total),
        "recall": float((r * w).sum() / total),
        "f1": float((f * w).sum() / total),
        "support": int(support.sum()),
        "labels": labels,
    }
