
from __future__ import division
import math


def sigmoid(z):
    '''sigmoid() - Sigmoid Function'''
    return 1. / (1 + math.e ** -z)
