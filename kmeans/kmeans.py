""" KMeans
Implementation of the K-Means algorithm
"""

from typing import List, Union

import numpy as np

from kmeans.activation import Activation

input_type = Union[np.array, Activation, List[Activation]]


class KMeans(object):
    """ Basic KMeans implementation

    Actually quite boring, but is the code that all other variations build upon.

    attributes:
        verbose (bool): whether to be verbose in printing progress
        X (array): the dataset
        K (int): the number of clusters to find
        N (int): the number of items in the dataset
        dimensions (int): the dimensionality of each item in the dataset
        mu ([array]): the current set of midpoints
    """

    def __init__(self,
                 X: input_type = None,
                 K: int = None,
                 filename: str = None,
                 mu: List[np.array] = None,
                 verbose: bool = False) -> None:
        """
        X the data as a list of Points
        K the number of centroids to find
        filename - name of file to load from
        mu is the list of centroids
        """
        self.verbose = verbose
        self._zero()

        if X is not None:
            self._init_from_variable(self._convert_to_array(X), K)
        elif filename is not None:
            self._init_from_file(filename)
        else:
            raise ValueError('must specify X or filename')

        if mu is not None:
            self._init_mu_from_variable(mu)

        if len(self.X) == 0:
            raise UserWarning("No Data Found")
        print("KMeans with {} points".format(len(self.X)))
        if (self.X.ndim == 1):
            # X is a 1D array
            self.N = len(self.X)
            self.dimensions = 1
        else:
            self.N, self.dimensions = self.X.shape

    @staticmethod
    def _convert_to_array(input: input_type) -> np.array:
        """
        Take any of the input style options and create an appropriate numpy
        array from them
        :param input: the passed inputs
        :return: a numpy array
        """
        if input is None or isinstance(input, (np.ndarray, np.generic)):
            return input
        if isinstance(input, Activation):
            return input.vector
        if isinstance(input, list) and \
                np.all([isinstance(x, Activation) for x in input]):
            return np.stack([a.vector for a in input], 0)

        raise TypeError("Invalid type {} passed to Kmeans".format(type(input)))

    def _zero(self):
        self.mu = None
        self.clusters = None
        self.oldmu = None
        self.oldermu = None
        self.method = None

    def _init_from_variable(self, X, K):
        self.X = X
        self.K = K

    def _init_from_file(self, filename):
        """
        filename the name of a previously saved KMeans object
        """
        # TODO: error handlnig
        # TODO: Update to work with new class
        with np.load(filename) as handle:
            self._load(handle['data'].tolist())

    def _init_mu_from_variable(self, mu):
        self.mu = mu

    def _load(self, filehandle):
        self.X = filehandle['X']
        self.K = filehandle['K']
        self.mu = filehandle['mu']
        self.clusters = filehandle['clusters']

    def _get_save_dictionary(self):
        self._build_save_dictionary()
        return self.savedic

    def _build_save_dictionary(self):
        self.savedic = {}
        self.savedic['X'] = self.X
        self.savedic['K'] = self.K
        self.savedic['mu'] = self.mu
        self.savedic['clusters'] = self.clusters

    def save(self, filename):
        self._build_save_dictionary()
        np.savez(filename, data=self.savedic)

    def cluster_points(self):
        # assign all the points of X to their nearest centroid.
        clusters = {}
        for i, _ in enumerate(self.mu):
            clusters[i] = []

        count = 0
        cluster_to_point_mapping = {}
        for x in self.X:
            keys = [(idx, np.linalg.norm(x - mu))
                    for idx, mu in enumerate(self.mu)]
            bestmukey = min(keys, key=lambda t: t[1])[0]
            cluster_to_point_mapping.setdefault(bestmukey, []).append(count)
            if self.verbose:
                print('Count {0}: mu {1}'.format(count, bestmukey))
            count = count + 1
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        self.clusters = clusters
        self.cluster_to_point_mapping = cluster_to_point_mapping

    def _compute_new_centroids(self):
        self.mu = [np.mean(x, axis=0) for x in self.clusters.values()]
        return

    def _has_converged(self):
        if self.oldmu is None:
            # catch first run
            return False

        self.mu = np.unique(self.mu, axis=0)
        # Catch issues with individual MUs converging.
        if len(self.mu) < self.K:
            print("Two centroids converged! Ending")
            # this set of MU is useless. use the previous.
            self.mu = self.oldmu
            self.cluster_points()
            self._compute_new_centroids()
            return True

        # make numpy treat the rows as items
        dtype = {'names': ['f{}'.format(i) for i in range(self.dimensions)],
                 'formats': self.N * [self.mu.dtype]}
        return np.all(np.in1d(self.oldmu, self.mu, True))

    def init_centers(self):
        # pick a set of random points to use as the initial K values
        self.mu = self.X[np.random.choice(self.N, self.K, replace=False)]

    def find_centers(self):
        self.init_centers()
        while not self._has_converged():
            self.oldmu = self.mu
            self.cluster_points()
            self._compute_new_centroids()
