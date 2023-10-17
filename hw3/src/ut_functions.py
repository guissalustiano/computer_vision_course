import numpy as np
import skimage.morphology as skm

def levelx(im, level):
    """Compute the locations where the pixels in a grey level image cross a fixed level

    Parameters
    ----------
    im : ndarray
        2D array containing the grey level image
    level : float
        grey level to cross

    Returns
    -------
    levelx :  ndarray, float64
              ones represent pixels located at a level-crossing. zeros elswwhere.
    """
    im = im - level
    min_im = skm.erosion(im)
    max_im = skm.dilation(im)
    crossings = np.logical_or(np.logical_and(min_im < 0, im > 0), np.logical_and(max_im > 0, im < 0))
    crossings = skm.skeletonize(crossings)
    crossings = np.float64(crossings)
    return crossings
