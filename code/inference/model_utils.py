from typing import Any
from tensorflow.keras.models import load_model
import autokeras as ak  # AutoKeras provides its custom objects mapping

def load_bomb_model(model_dir: str) -> Any:
    """
    Load and return a TensorFlow/Keras model from `model_dir`, supplying
    AutoKeras custom layers so they deserialize correctly.
    """
    return load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)


def predict_is_bomb(model: Any, mfcc_window: Any, threshold: float = 0.5) -> bool:
    """Return True if model predicts > threshold on mfcc_window."""
    prob = model.predict(mfcc_window, verbose=0)
    return bool(prob > threshold)
