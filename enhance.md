# Assignment 4 Notebook Enhancement Prompt

Analyze and enhance `assignment_4.ipynb` by doing the following:

1. **Fix all errors** in the notebook:
   - Add `import os`, `import numpy as np`, and `import tensorflow as tf` to the export cell (`df614ecc`) so it is not dependent on prior kernel state.
   - Add `import os` and `import tensorflow as tf` to the sanity-check cell (`a72fd4b8`) for the same reason.

2. **Add early stopping** to the `model.fit()` call:
   - Use `tf.keras.callbacks.EarlyStopping` with `monitor='val_loss'`, `patience=10`, and `restore_best_weights=True`.
   - Pass it as a `callbacks` list to `model.fit()`.

3. **Add a training history plot** after `model.fit()`:
   - Plot accuracy (train and val) and loss (train and val) side by side using `matplotlib`.
   - Use `history.history` to retrieve per-epoch values.
