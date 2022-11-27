import cv2
import numpy as np


def blend(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend to images together with alpha."""
    assert 0 <= alpha <= 1.0
    beta = (1 - alpha)
    return cv2.addWeighted(image1, alpha, image2, beta, gamma=0.0)


def random_horizontal_swap(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Swap random rows of two images.
    """
    width = image1.shape[1]
    random_rows = np.random.choice(width, size=width // 2, replace=False)
    image1[:, random_rows] = image2[:, random_rows]    
    return image1


def random_vertical_swap(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Swap random columns of two images.
    """
    height = image1.shape[0]
    random_columns = np.random.choice(height, size=height // 2, replace=False)
    image1[random_columns] = image2[random_columns]
    return image1


def half_vertical_swap(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Swap images halfs vertically.
    """
    height = image1.shape[0]
    image1_half = image1[: height // 2]
    image2_half = image2[height // 2:]
    return np.vstack((image1_half, image2_half))


def half_horizontal_swap(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Swap images halfs horizontally.
    """
    width = image1.shape[1]
    image1_half = image1[:, :width // 2]
    image2_half = image2[:, width // 2:]
    return np.hstack((image1_half, image2_half))
