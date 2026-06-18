import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from typing import Tuple, Union, Optional, List
from src.utils import setup_logger

logger = setup_logger(__name__)


class GradCAM:
    """
    Keras 3.x compatible Grad-CAM implementation.
    
    Standard Grad-CAM creates a sub-model via tf.keras.Model([model.inputs], [layer.output, model.output]).
    This FAILS on Keras 3 Sequential models because they don't build their internal
    computation graph until the first forward pass, and even after calling model.predict(),
    the `.output` property remains unavailable.

    Fix: We perform a manual layer-by-layer forward pass inside GradientTape,
    capturing the target conv layer's output tensor and watching it for gradients.
    """

    def __init__(self, model: tf.keras.Model):
        """
        Args:
            model: A trained Keras model (Sequential or Functional).
        """
        self.model = model
        self.last_conv_layer_name, self.last_conv_layer_idx = self._find_last_conv_layer()

    def _find_last_conv_layer(self) -> Tuple[Optional[str], Optional[int]]:
        """
        Automatically detects the last Conv2D layer in the model.
        
        Returns:
            Tuple of (layer_name, layer_index) or (None, None) if not found.
        """
        for i in range(len(self.model.layers) - 1, -1, -1):
            layer = self.model.layers[i]
            if isinstance(layer, tf.keras.layers.Conv2D):
                logger.info(f"Detected last Conv2D layer: '{layer.name}' at index {i}")
                return layer.name, i

        logger.warning("No Conv2D layer found in the model.")
        return None, None

    def generate_heatmap(self, img_array: np.ndarray, class_index: int = 1) -> np.ndarray:
        """
        Generates the Grad-CAM heatmap using a layer-by-layer forward pass.
        
        This approach is compatible with Keras 3.x Sequential models where
        model.output is not available.

        Args:
            img_array: Preprocessed image array of shape (1, H, W, 3).
            class_index: 0 for Cat, 1 for Dog.

        Returns:
            Normalized 2D heatmap array.
        """
        if self.last_conv_layer_idx is None:
            raise ValueError("Cannot generate Grad-CAM: no Conv2D layer found in the model.")

        with tf.GradientTape() as tape:
            x = tf.cast(img_array, tf.float32)

            # Manual layer-by-layer forward pass
            conv_output = None
            for i, layer in enumerate(self.model.layers):
                x = layer(x)
                if i == self.last_conv_layer_idx:
                    conv_output = x
                    tape.watch(conv_output)  # Watch AFTER assignment

            # x is now the final prediction
            # For binary sigmoid: output is P(Dog). 
            # Gradient w.r.t Dog = positive gradient.
            # Gradient w.r.t Cat = negative gradient (P(Cat) = 1 - P(Dog)).
            loss = x[:, 0]

        grads = tape.gradient(loss, conv_output)

        if grads is None:
            raise RuntimeError("Gradient computation returned None. The model graph may be disconnected.")

        # For Cat (class_index=0), negate gradients because P(Cat) = 1 - sigmoid_output
        if class_index == 0:
            grads = -grads

        # Global Average Pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv feature maps by the pooled gradients
        conv_output_val = conv_output[0]
        heatmap = conv_output_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy()

    def overlay_heatmap(
        self,
        img_path: Union[str, Path],
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlays the heatmap on the original image.

        Args:
            img_path: Path to the original image file.
            heatmap: The 2D heatmap array from generate_heatmap().
            alpha: Overlay transparency (0 = fully original, 1 = fully heatmap).
            colormap: OpenCV colormap constant.

        Returns:
            (original_rgb, overlayed_rgb): Both as uint8 numpy arrays.
        """
        img_path_obj = Path(img_path)
        if not img_path_obj.exists():
            raise FileNotFoundError(f"Image not found at {img_path_obj}")

        img = cv2.imread(str(img_path_obj))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend
        overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

        return img, overlayed_img
