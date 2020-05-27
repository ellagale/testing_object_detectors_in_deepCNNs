""" KMeansPlusPlus
Implementation of the ++ extensions to the KMeans alorithm
"""
import random

import numpy as np

from kmeans.kmeans import KMeans


class KMeansPlusPlus(KMeans):
    """

    """

    def dist_from_centers(self, mu_count: int):
        """
        Calculates the shortest distance squared between all points and all centroids
        """
        self.D2 = np.array(
            [min([np.linalg.norm(x - mu) ** 2 for mu in self.mu[0:mu_count]]) for x in self.X])

    def _radius(self):
        self.dist_from_centers(self.K)
        self.r = max(self.D2) ** 0.5

    def _choose_next_center(self):
        self.probs = self.D2 / self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        # random.random returns [0-1), we want (0-1]
        r = 1.0 - random.random()
        try:
            idx = np.where(self.cumprobs >= r)[0][0]
        except IndexError:
            print(self.X)
            print(self.mu)
            print(self.probs)
            print(self.cumprobs)
            print(r)
            print(np.where(self.cumprobs >= r))
            raise
        return self.X[idx]

    def init_centers(self):
        self.mu = np.zeros([self.K, self.dimensions])
        mu_count = 1
        self.mu[0, ] = self.X[np.random.choice(self.N, 1, replace=False)]
        # pick a point, any point
        while mu_count < self.K:
            self.dist_from_centers(mu_count)
            self.mu[mu_count, ] = self._choose_next_center()
            mu_count += 1
