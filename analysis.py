import numpy as np
import matplotlib.pyplot as plt
import itertools
import threading
import imp

"""This module is for developing the machinery required to make neural nets and analyse local and global codes

This module does stuff.
"""

__version__ = '0.1'
__author__ = 'Ella Gale'
__date__ = 'Jan 2017'


class ThreadedRunner(object):
    """ run a task across multiple processors, taking care not to overload them """

    def __init__(self, tasks, maxparallel=8):
        """
        tasks: an array of tuples of the form (function,arguments) to call
        maxparallel: the maximum number of threads to be running at once
        """
        self.threads = [threading.Thread(target=f, kwargs=k) for (f, k) in tasks]
        # TODO: spin up seperate thread managers to maximise throughput
        self.maxparallel = 8
        self.next_thread = 0

    def run(self, threadrunlimit=None):
        """
        threadrunlimit: only run this many threads at most total,
                        if None (default) then run all threads
        """
        runcount = len(self.threads[self.next_thread:])
        if threadrunlimit is not None:
            runcount = min(runcount, threadrunlimit)

        next_thread = 0
        while runcount > 0:
            batch = self.threads[next_thread:next_thread + self.maxparallel]

            # cannot start threads while imp lock is held.
            toLock = imp.lock_held()
            if toLock:
                imp.release_lock()

            # Start all threads in this batch
            for thread in batch:
                thread.start()

            # Wait for them all to finish
            for thread in batch:
                thread.join

            # rest lock state
            if toLock:
                imp.acquire_lock()

            runcount = runcount - len(batch)
            next_thread = next_thread + len(batch)


def fk_plotter(dks, noOfK, lRange=None, error=0.15, xaxis=1, title=None, xlabel=None, ylabel=None, showPlots=1,
               savePlots=0):
    """Produces F(k) plots for each layer of neurons"""
    "lRange = range of layers to plot"
    "error = error below 1 which we consider significant"
    "xaxis = where to draw the xaxis line"
    if lRange == None:
        lRange = range(len(dks))
    for l in lRange:
        # l is the number of layers -- send a smaller dks if you don't want them all!
        fig = plt.figure(l)
        x_data = np.array(range(noOfK)) + 1
        marker = itertools.cycle(['o', '>', '<', 'v', '8', 'd', 's', 'p', '*'])
        for n in range(len(dks[l])):
            # n is the number neurons in a layer
            y_data = dks[l][n].fs
            plt.plot(x_data, y_data, label=str(n), marker=marker.next(), alpha=1)
        if not xaxis == None:
            # note, if you don't want an xaxis, set xaxis='off'
            plt.axhline(xaxis)
        else:
            plt.axhline(0)
        plt.xlim([min(x_data) - 0.25, max(x_data) + 1])
        # plt.legend()
        plt.legend(bbox_to_anchor=(0.9, 1.1), loc='best', ncol=2, framealpha=0.5).draggable()
        # ax.legend().draggable()
        plt.plot([0., noOfK], [1 - error, 1 - error])
        if title == None:
            plt.title('Layer ' + str(l + 1))
        else:
            plt.title(title)
        if xlabel == None:
            plt.xlabel('K')
        else:
            plt.xlabel(xlabel)
        if ylabel == None:
            plt.ylabel('f(K)')
        else:
            plt.ylabel(ylabel)
        if showPlots == 1:
            plt.show()
        if savePlots == 1:
            fig.savefig('Fk' + str(l) + '.png', dpi=fig.dpi)


def jitterer(out, z):
    """This function jitters the x axis
    1: matrix of layer activations of the form:
    2. which layer number to do
    outputs a transposed matrix of no of neurons rows and no of data columns"""
    Jx = np.ones(out[z].T.shape)

    for i in range(out[z].T.shape[0]):
        'this is the number of neurons'
        for j in range(out[z].T.shape[1]):
            'this is the number of data'
            Jx[i, j] = i + 1 + np.random.uniform(-0.25, 0.25)
    return Jx


def normalise_to_zero_one_interval(y, ymin, ymax):
    """Because I always forget the formula"""
    if ymin > ymax: raise TypeError('min and max values the wrong way round!')
    return (y - ymin) / (ymax - ymin)


def plotter(x, y, labels=['x', 'y'], legend=None, linestyle=['o-', '+-', '*.-'], xaxis=None, showPlots=1, savePlots=0):
    """Make nice plots automatically"""
    fig = plt.figure(1)
    xrange = max(x) - min(x)
    yrange = max(y.flatten()) - min(y.flatten())
    if not legend == None:
        for i in range(len(y)):
            plt.plot(x, y[i], linestyle[i / 3], label=legend[i])
    else:
        for i in range(len(y)):
            plt.plot(x, y[i], linestyle[i / 3])
    if not xaxis == None:
        # note, if you don't want an xaxis, set xaxis='off'
        plt.axhline(xaxis)
    else:
        plt.axhline(0)

    plt.axis([min(x.flatten()) - 0.1 * xrange, max(x.flatten()) + 0.1 * xrange,
              min(y.flatten()) - 0.1 * yrange, max(y.flatten()) + 0.1 * yrange])
    plt.ylabel(labels[1])
    plt.xlabel(labels[0])
    if not legend == None:
        plt.legend(framealpha=0.5)
    if showPlots == 1:
        plt.show()
    if savePlots == 1:
        fig.savefig('Hk' + str(x[0]) + '.png', dpi=fig.dpi)


####################################################
## downloaded code
"""
Demo of a function to create Hinton diagrams.

Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
a weight matrix): Positive and negative values are represented by white and
black squares, respectively, and the size of each square represents the
magnitude of each value.

Initial idea from David Warde-Farley on the SciPy Cookbook
"""


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    if __name__ == '__main__':
        hinton(np.random.rand(20, 20) - 0.5)
        plt.show()

## writing model to file and reading it back in test
