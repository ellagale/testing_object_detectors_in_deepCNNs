""" DetK
Code to iterate over KMeans searching for the best matching K value for a dataset.
"""

import random
from typing import List

import numpy as np

from kmeans.kmeans import input_type
from kmeans.kmeansplusplus import KMeansPlusPlus


class DetK(KMeansPlusPlus):
    """
    run KMeansPlusPlus for a range of K values
    """

    def __init__(self,
                 X: input_type = None,
                 K: int = 5,
                 filename: str = None,
                 mu: List[np.array] = None,
                 verbose: bool = False) -> None:
        """
        X the data as a list of Points
        filename - name of file to load from
        mu is the list of centroids
        """
        KMeansPlusPlus.__init__(self, X, K, filename, mu, verbose)
        self.a_mem = {}

    def _zero(self):
        """ wipe this object
        """
        super(DetK, self)._zero()
        self.fs = None
        self.fCentroids = []

    def _build_save_dictionary(self):
        """ extend _build_save_dictionary to include the additional stuff needed to save.
        """
        super(DetK, self)._build_save_dictionary()
        self.savedic['fs'] = self.fs
        self.savedic['fCentroids'] = self.fCentroids

    def _load(self, filehandle):
        """ extend _load to include the new attributes
        """
        super(DetK, self)._load(filehandle)
        self.fs = filehandle['fs']
        self.fCentroids = filehandle['fCentroids']

    def fK(self, Skm1=0):
        self.find_centers()
        mu, clusters = self.mu, self.clusters
        # Sk = sum([np.linalg.norm(m-c)**2 for m in clusters for c in clusters[m]])
        Sk = sum([np.linalg.norm(mu[i] - c) ** 2
                  for i in range(self.K) for c in clusters[i]])

        if self.K == 1 or Skm1 == 0:
            fs = 1.0
        else:
            fs = Sk / (self.a(self.K, self.dimensions) * Skm1)
        return fs, Sk, mu

    def _bounding_box(self):
        return np.amin(self.X, axis=0), np.amax(self.X, axis=0)

    def gap(self):
        dataMin, dataMax = self._bounding_box()
        self.init_centers()
        self.find_centers()
        mu, clusters = self.mu, self.clusters
        Wk = np.log(sum([np.linalg.norm(mu[i] - c) ** 2 / (2 * len(c))
                         for i in range(self.K) for c in clusters[i]]))
        # why 10?
        B = 10
        ms = None
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in self.X:
                Xb.append(random.uniform(dataMin, dataMax))
            Xb = np.array(Xb)
            kb = DetK(K=self.K, X=Xb)
            kb.init_centers()
            kb.find_centers()
            ms, cs = kb.mu, kb.clusters
            BWkbs[i] = np.log(sum([np.linalg.norm(ms[j] - c) ** 2 / (2 * len(c))
                                   for j in range(self.K) for c in cs[j]]))
        Wkb = sum(BWkbs) / B
        sk = np.sqrt(sum((BWkbs - Wkb) ** 2 / float(B)) * np.sqrt(1 + 1 / B))
        return Wk, Wkb, sk, ms

    def runFK(self, maxK, minK=1):
        """ Run fK for values in the range 1..maxK,
        picking the best one in the range minK..MaxK
        """
        ks = range(1, maxK + 1)
        fs = np.zeros(len(ks))
        fCentroidList = []
        Sk = 0
        for k in ks:
            # if self.verbose:
            #    print('k={}'.format(k))
            self.K = k
            self.init_centers()
            fs[k - 1], Sk, centroids = self.fK(Skm1=Sk)
            fCentroidList.append(np.array(centroids))
        self.fs = fs
        self.fCentroids = fCentroidList
        # Now assign the best K centroids, starting from minK
        error = 0.15
        bestF = np.argmin(fs[minK - 1:])
        if fs[bestF] > (1 - error):
            bestF = minK - 1
        self.K = bestF + 1
        self.mu = fCentroidList[bestF]
        self.cluster_points()

    def runGap(self, maxK):
        ks = range(1, maxK)
        gCentroidList = []
        Wks, Wkbs, sks = np.zeros(
            len(ks) + 1), np.zeros(len(ks) + 1), np.zeros(len(ks) + 1)
        for k in ks:
            print('k={}'.format(k))
            self.K = k
            self.init_centers()
            Wks[k - 1], Wkbs[k - 1], sks[k - 1], centroids = self.gap()
            gCentroidList.append(np.array(centroids))
        G = []
        for i in range(len(ks)):
            G.append((Wkbs - Wks)[i] - ((Wkbs - Wks)[i + 1] - sks[i + 1]))
        self.G = np.array(G)
        self.gCentroids = gCentroidList

    def run(self, maxK, which='both'):
        doF = which is 'f' or which is 'both'
        doGap = which is 'gap' or which is 'both'
        if doF:
            self.runFK(maxK)
        if doGap:
            self.runGap(maxK)

    def a(self, k: int, dimensions: int):
        try:
            return self.a_mem[(k, dimensions)]
        except KeyError:
            # Not yet calculated this value
            result = None
            if k == 2:
                result = 1 - 3.0 / (4.0 * dimensions)
            else:
                previous = self.a(k - 1, dimensions)
                result = previous + (1 - previous) / 6.0
            self.a_mem[(k, dimensions)] = result
            return result
