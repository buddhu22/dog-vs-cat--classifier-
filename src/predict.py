import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Union
from src.config import MODEL_PATH, IMAGE_SIZE, CLASSES
from src.utils import setup_logger, load_and_preprocess_image

logger = setup_logger(__name__)


class Predictor:
    """
    Handles loading the trained CNN model and performing inference.
    
    The model uses sigmoid activation on a single output neuron:
        - output < 0.5 → Cat (label 0)
        - output >= 0.5 → Dog (label 1)
    
    This mapping is consistent with image_dataset_from_directory which assigns:
        cats/ → 0, dogs/ → 1 (alphabetical sort).
    """

    def __init__(self, model_path: Union[str, Path] = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        """Loads the Keras model with comprehensive error handling."""
        if not self.model_path.exists():
            logger.error(f"Model file not found at: {self.model_path}")
            return None

        try:
            model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"  Input shape: {model.input_shape}")
            logger.info(f"  Output shape: {model.output_shape}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def predict_image_array(self, img_array: np.ndarray) -> Tuple[str, float]:
        """
        Predicts the class for a preprocessed image array.

        Args:
            img_array: Shape (1, H, W, 3), normalized to [0, 1].

        Returns:
            (class_label, confidence_percent)
        """
        if self.model is None:
            raise RuntimeError(f"Model is not loaded. Ensure {self.model_path} exists and is a valid .keras file.")

        prediction_prob = self.model.predict(img_array, verbose=0)[0][0]

        # sigmoid output: prob >= 0.5 → Dog (label 1), prob < 0.5 → Cat (label 0)
        if prediction_prob >= 0.5:
            predicted_class = CLASSES[1]  # Dog
            confidence = float(prediction_prob * 100)
        else:
            predicted_class = CLASSES[0]  # Cat
            confidence = float((1 - prediction_prob) * 100)

        logger.info(f"Predicted: {predicted_class} ({confidence:.2f}%) [raw={prediction_prob:.6f}]")
        return predicted_class, confidence

    def predict_single(self, image_path: Union[str, Path]) -> Tuple[str, float]:
        """Loads, preprocesses, and predicts a single image."""
        logger.info(f"Predicting for: {image_path}")
        img_array = load_and_preprocess_image(image_path, target_size=IMAGE_SIZE)
        return self.predict_image_array(img_array)

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Tuple[str, float]]:
        """Predicts classes for a batch of image paths."""
        results = []
        for path in image_paths:
            try:
                results.append(self.predict_single(path))
            except Exception as e:
                logger.error(f"Failed for {path}: {e}")
                results.append(("Error", 0.0))
        return results
