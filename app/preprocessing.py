"""Keep inference preprocessing identical to the original notebook."""

import numpy as np


def normalize_pixels(pixels):
    """Convert raw 0-255 pixels to float32 values in the 0-1 range.

    The divisor is fixed, so this transformation does not learn from test data.
    A named function allows joblib to reload the pipeline in another process.
    """
    return np.asarray(pixels, dtype=np.float32) / 255.0
