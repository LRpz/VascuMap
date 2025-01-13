import numpy as np

import albumentations as albu
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensor
from skimage.filters import apply_hysteresis_threshold

def scale(arr):
    """
    Scales the input array to be in the range [0, 1] by subtracting the minimum value and dividing by the range 
    of values.

    Args:
        arr (numpy.array): The input array to be scaled.

    Returns:
        numpy.array: The scaled array with values in the range [0, 1].
    """

    if np.mean(arr) == 0 or np.mean(arr) == 1:
        return arr

    else:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def contrast(arr, low, top):
    """
    Enhances the contrast of an array by clipping its values based on given percentiles.

    Parameters:
    arr (numpy.ndarray): Input array to enhance the contrast.
    low (float): Lower percentile value for clipping. Values below this percentile will be set to this percentile.
    top (float): Upper percentile value for clipping. Values above this percentile will be set to this percentile.

    Returns:
    numpy.ndarray: Array with enhanced contrast.
    """

    return np.clip(arr, np.percentile(arr, low), np.percentile(arr, top))

def new_axis(arr):
    """
    Adds a new axis to an array.

    Args:
        arr (numpy.array): The input array to which a new axis will be added.

    Returns:
        numpy.array: The array with an added axis.
    """

    return arr[..., np.newaxis]

def hysteresis_thresholding(arr, low=0.15, high=0.5):
    return apply_hysteresis_threshold(arr, low, high).astype(int)

def single_level_thresholding(arr, threshold=0.5):
    return np.array(arr > threshold).astype(int)

def hard_transforms():
    """
    Returns a list of 'hard' augmentations to be applied to images.
    These include flipping and shifting.
    """
    return [
        albu.Flip(),
        albu.ShiftScaleRotate(),
    ]

def post_transforms():
    """
    Returns a list of post-processing transforms such as converting to tensor.
    """
    return [ToTensor()]

def compose(transforms_to_compose):
    """
    Composes multiple lists of transformations together into a single pipeline.

    Args:
        transforms_to_compose (list): A list containing other lists of transformations.

    Returns:
        An Albumentations.Compose object that combines all the transformations.
    """

    # Flatten the list of lists and create a composition
    result = albu.Compose([
    item for sublist in transforms_to_compose for item in sublist
    ])

    return result