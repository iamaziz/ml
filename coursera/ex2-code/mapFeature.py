
# from __future__ import division
import numpy as np


def mapFeature(x1, x2):
    '''mapFeature() - Function to generate polynomial features'''

    degree = 6
    out = np.array(
        [x1 ** (i - j) * x2 ** j for i in range(degree + 1) for j in range(i + 1)])
    return out.T
