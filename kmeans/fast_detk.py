""" fast-DetK
Does det-k on the top half (by activation value, not number) of points in a neuron
"""

from typing import List

import numpy as np

from kmeans.detk import DetK
from kmeans.kmeans import input_type


class FastDetK(DetK):
    """
    Throws away some data, and initialises at max and min of kept data
    """

    def __init__(self,
                 X: input_type = None,
                 filename: str = None,
                 mu: List[np.array] = None,
                 discard: float = 1,
                 verbose: bool = False) -> None:
        """
        X the data as a list of Points
        filename - name of file to load from
        mu is the list of centroids
        discard is the fractional range of X to discard
        """
        self.discard = discard
        DetK.__init__(self, X, None, filename, mu, verbose)

    def _init_from_variable(self, X, K):
        """We only take the values of X which are above discard*full range"""
        # Need to compress a (potentially multidimensional array down to a 1D
        # of the size of it
        remaining_dim = X.ndim
        compressed_X = X
        while remaining_dim > 1:
            compressed_X = compressed_X.sum(axis=-1)
            remaining_dim -= 1
        maxX = max(compressed_X)
        minX = min(compressed_X)
        midX = minX + (maxX - minX) * self.discard
        self.maxX = maxX
        self.minX = minX
        self.midX = midX
        self.X = X[compressed_X > midX]
        self.K = K
    # def _convert_to_array(input: input_type):
    #    pass

# egg= FastDetK(X=x_data)
