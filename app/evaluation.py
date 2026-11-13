import numpy as np

# Recorded baseline results from Notebook 03, evaluated on 542 held-out images.
TEST_METRICS = {
    "test_samples": 542,
    "accuracy": 0.9501845018450185,
    "macro_precision": 0.9367,
    "macro_recall": 0.9482,
    "macro_f1": 0.9398,
    "weighted_f1": 0.9490,
}
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]
CONFUSION_MATRIX = np.array([
    [242, 3, 0],
    [7, 105, 0],
    [9, 8, 168],
], dtype=int)
