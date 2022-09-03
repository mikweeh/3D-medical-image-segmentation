"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    
    intersection = np.sum(np.logical_and(a>0, b>0))
    volumes = np.sum(a>0) + np.sum(b>0)

    if volumes == 0:
        return -1

    return 2 * intersection / volumes

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use
    # the Dice3D function from above to do the computation ;)
    # <YOUR CODE GOES HERE>
    intersection = np.sum(np.logical_and(a>0, b>0))
    volumes = np.sum(a>0) + np.sum(b>0)

    if volumes == 0:
        return -1

    return intersection / (volumes - intersection)


def Sensitivity(pred, gt):
    """
    This will compute the sensitivity of a 3D prediction respect to the ground truth
    in all voxels. Sensitivity = TruePos / (TruePos + FalseNeg)
    """

    tp = np.sum(np.logical_and(gt>0, gt==pred))
    fn = np.sum(np.logical_and(gt>0, pred==0))

    if fn + tp == 0:
        return -1

    return tp / (fn + tp)


    
def Specificity(pred, gt):
    """
    This will compute the specificity of a 3D prediction respect to the ground truth
    in all voxels. Specificity = TrueNeg / (TrueNeg + FalsePos)
    """

    tn = np.sum(np.logical_and(gt==0, pred==0))
    fp = np.sum(np.logical_and(gt==0, pred>0))

    if tn + fp == 0:
        return -1

    return tn / (tn + fp)
