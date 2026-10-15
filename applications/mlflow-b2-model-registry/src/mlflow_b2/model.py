"""Softmax classifier over downsampled grayscale pixels, packaged as an MLflow pyfunc."""
from __future__ import annotations

import io

import mlflow
import numpy as np
from PIL import Image

SIZE = 8


def features(png: bytes) -> np.ndarray:
    """Foreground mask (vs the corner background colour), cropped to its box, downsampled."""
    rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)
    mask = np.abs(rgb - rgb[0, 0]).sum(axis=2) > 48
    ys, xs = np.nonzero(mask)
    if len(ys):
        mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = Image.fromarray(mask.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).ravel() / 255.0


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def probabilities(weights: dict, x: np.ndarray) -> np.ndarray:
    return _softmax(((x - weights["mean"]) / weights["std"]) @ weights["w"] + weights["b"])


def _scores(weights: dict, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = probabilities(weights, x)
    loss = float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
    return loss, float((p.argmax(axis=1) == y).mean())


def fit(x, y, xv, yv, classes, *, epochs, lr, batch_size, seed, on_epoch=None) -> dict:
    """Seeded mini-batch gradient descent; on_epoch(epoch, metrics) is called after each epoch."""
    rng = np.random.default_rng(seed)
    w = {"w": rng.normal(0, 0.01, (x.shape[1], len(classes))).astype(np.float32),
         "b": np.zeros(len(classes), np.float32),
         "mean": x.mean(axis=0), "std": x.std(axis=0) + 1e-3, "classes": np.array(classes)}
    xn = (x - w["mean"]) / w["std"]
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for idx in np.array_split(order, max(1, len(x) // batch_size)):
            g = _softmax(xn[idx] @ w["w"] + w["b"])
            g[np.arange(len(idx)), y[idx]] -= 1
            g /= len(idx)
            w["w"] -= lr * xn[idx].T @ g
            w["b"] -= lr * g.sum(axis=0)
        if on_epoch:
            (tl, ta), (vl, va) = _scores(w, x, y), _scores(w, xv, yv)
            on_epoch(epoch, {"train_loss": tl, "train_accuracy": ta, "val_loss": vl, "val_accuracy": va})
    return w


class ShapesClassifier(mlflow.pyfunc.PythonModel):
    """Input: rows of features() (SIZE*SIZE values in [0, 1]). Output: class names."""

    def load_context(self, context):
        with np.load(context.artifacts["weights"]) as f:
            self.weights = {k: f[k] for k in f.files}

    def predict(self, context, model_input, params=None):
        x = np.asarray(model_input, dtype=np.float32)
        return self.weights["classes"][probabilities(self.weights, x).argmax(axis=1)]
