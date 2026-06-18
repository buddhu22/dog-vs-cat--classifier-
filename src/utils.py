import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from src.config import IMAGE_SIZE

def setup_logger(name: str) -> logging.Logger:
    """
    Setup a standard professional logger.
    
    Args:
        name (str): The name of the logger (typically __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def load_and_preprocess_image(image_path: Union[str, Path], target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Loads an image from a file path and preprocesses it for the CNN model.
    
    Args:
        image_path (Union[str, Path]): Path to the image file.
        target_size (tuple): Target resize dimensions (width, height). Defaults to IMAGE_SIZE.
        
    Returns:
        np.ndarray: Preprocessed image array of shape (1, height, width, channels)
        
    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If OpenCV fails to read the image file.
    """
    img_path_obj = Path(image_path)
    
    if not img_path_obj.exists():
        raise FileNotFoundError(f"Image not found at path: {img_path_obj}")
        
    # Read using string representation since cv2 doesn't always support Path directly
    img = cv2.imread(str(img_path_obj))
    if img is None:
        raise ValueError(f"Could not read image: {img_path_obj}")
        
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Scale to [0, 1]
    img = img / 255.0
    
    # Expand dimensions for batch inference
    img = np.expand_dims(img, axis=0)
    
    return img
