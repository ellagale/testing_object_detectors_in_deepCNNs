#!/usr/bin/python

import matplotlib

from precision_calculator3 import no_of_neurons

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
import analysis as a
import operator
import sys
import set_up_caffe_net as s
import Make_activation as m
import itertools
from collections import Counter
from PIL import Image, ImageDraw
import numpy as np
import random

do_check = False
from kmeans.fast_detk import FastDetK
from collections import OrderedDict
import csv

########################################################################################################################
## Analysis script!
## This does not run kmeans or a neural network, instead it expects that you've done that!
## this just does the jitter plots as I did in nov/dec 2017
## edited to do faster jitter plots and search for category selectivity
########################################################################################################################

# Initial values
HLN = 500
noName = 1
name = 'Random'
outSize = 50
tf_data = False
doLabels = False
h5_filename = None
do_check = False
use_loaded_files = True  # False # True # if the data was input as one (or 2) h5 file per class then use this
no_of_k = 0
do_plots = 1  # turns off all plots!
show_plots = 0  # shows plots on screen for manual saving -- use for debugging
save_plots = 1  # saves out the plots n.b. you can both save and show plots
# no_of_layers = len(dks)
doKAnalysis = 0  # turns on or off the Fk plotter
verbose = True
plotFlags = ''  # 'Sp' #Ss'
input_flag = 'K'
doHistogram = False
## sets up empty arrays
label_dict = {}
acts = []
found_labels = []
class_labels = []
local_list = []
class_dict = []
current_neuron_index = 0
do_all_points = False  # whether to check for class selectivity against all points or not
name_leader = 'L1_fc6_layer_neuron'
foundSelectivityList = []
foundClassList = []
foundNeuronList = []

min_overlaps = []
# settings
cluster_by_class = True  # True#True #False #True

do_true_picture = True  # only fires if cluster_by_class, but will actually plot the whole neurons output

do_check = False
Test = False
do_pictures = False  # True
## settings for kmeans analysos
cluster_by_kmeans = True  # True
do_ccma_selectivity = False  # True #True # setting to calculate this stupid thing, it adds time and does a lot
do_heirarchical_k_means = True

num_of_points_for_initial_k_means = 1000  # max number of points to do the initial k-menas iwth
allowed_error = 0.1  # allowed error between the 'best' k and the actual K
max_K_For_half = 20  # max K for half the range
max_K_For_Structured = 10  # the max K allowed for the structured region


########################################################################################################################
#       functions
########################################################################################################################

def filename_to_label(filename):
    """ dewisott """
    if filename.endswith(".h5"):
        filename = filename[:-3]
    return filename


def all_same(items):
    """Gives true if all items in an array are the same"""
    return all(x == items[0] for x in items)


def build_label_dict_from_filenames(acts):
    """ Build a dictionary of labels to points in local format
    Assumes the filename is the correct label.
    """
    label_dict = {}
    found_labels = []
    no_files_in_label = {}
    for activation_filename in acts.activation_files:
        label = activation_filename.split('_')[0]
        found_labels.append(label)
        no_files_in_label[label] = 0
        label_dict[label] = []
    for filename, idx in acts.get_all_point_indices():
        label = filename.split('_')[0]
        label_dict[label].append((filename, idx))
        no_files_in_label[label] += 1
    return label_dict, found_labels, no_files_in_label


def build_label_dict(acts, use_loaded_files=True, verbose=True, doMean=False):
    """
    Builds a dictionary of labels to points in local format
    Gives out a dictionary and a list of found labels
    acts: activation table object
    use_loaded_files: whether to assume one file per label (current default)
    """
    files = acts.get_loaded_files()
    big_list = []
    no_of_files = len(files)
    found_labels = []
    label_dict = {}
    no_files_in_label = {}
    for file_name in files:
        big_list.append([])
        label = filename_to_label(file_name.split('_')[0])
        found_labels.append(label)
    if verbose:
        print('Found {} files in activation table object'.format(no_of_files))
        # print('Be patient, I found {} points'.format(len(acts.get_all_activation_indices())))
    for current_point in acts.get_all_point_indices():
        # TODO:: Make this work with multiple labels
        if isinstance(acts.get_activation(current_point).labels, (bytes, bytearray, str)):
            # old style, the labels are a numpy byte string
            assigned_label = acts.get_activation(current_point).labels.decode('UTF-8')
        else:
            # new style, labels are a list
            assigned_label = acts.get_activation(current_point).labels[0].decode('UTF-8')
        assigned_label = filename_to_label(assigned_label)
        # except AttributeError:
        #     assigned_label = acts.get_activation(current_point).labels.decode('UTF-8')
        # except ValueError:
        #     import pdb; pdb.set_trace()
        for f_no, file_name in enumerate(files):
            label = filename_to_label(file_name.split('_')[0])
            if assigned_label == label:
                big_list[f_no].append(current_point)
                break
    if not len(found_labels) == len(files):
        print('The number of found labels does not match the number of files in activation table')
    if verbose:
        print('Found label: \t No. of points')
        for i in range(len(found_labels)):
            print('{}: \t {}'.format(found_labels[i], len(big_list[i])))
    for i in range(len(found_labels)):
        # print(i, found_labels[i])
        label_dict[found_labels[i]] = big_list[i]
        no_files_in_label[found_labels[i]] = len(big_list[i])
    return label_dict, found_labels, no_files_in_label


def build_cluster_from_class_label(acts, current_neuron_index, label_dict, found_labels, current_neuron=[],
                                   do_check=do_check, no_of_points_to_check=''):
    """Builds a cluster list from the class labels and activation table
    This could be expanded to build a list of centroids as well and a detK object or similar
    acts: activation table object
    current_neuron_index: which neuron to do this for
    label_dict: look up table that matches the found_labels to the local coordinates in the activation table object
    found_labels: list of labels which are keys to label_dict
    returns a list of lists of points which are members of each category
    """
    cluster_list = []
    min_list = []
    max_list = []
    # moved this up here as it costs time
    if current_neuron == []:
        current_neuron = acts.get_activations_for_neuron(current_neuron_index)
    else:
        pass
    for class_index in range(len(label_dict)):
        if verbose:
            print('building cluster based on class labels for label {}'.format(found_labels[class_index]))
        # this is the list of points in our current class in the 2-d tuple format
        list_of_points_in_current_class = label_dict[found_labels[class_index]]
        # if we decide to only compare a subset...
        if not no_of_points_to_check == '':
            list_of_points_in_current_class = random.sample(list_of_points_in_current_class, no_of_points_to_check)
        # this is the list of points in our current class in the 1-d global number format (which corresponds to the vector's dimentions)
        global_list_of_points_in_current_class = acts.from_local_indices(
            list_of_points_in_current_class)  # this gives all the points as a list
        # thus our cluster of points for this class is...
        # current_cluster = acts.get_activations_for_neuron(current_neuron_index)[0][0][global_list_of_points_in_current_class]
        current_cluster = current_neuron.vector[global_list_of_points_in_current_class]
        # and cos that was so fucking belaboured, lets check that shit
        # current_point = list_of_points_in_current_class[point_index]
        if do_check:
            # this checks that acts was built correctly
            for point_index in range(len(list_of_points_in_current_class)):
                current_point_local = list_of_points_in_current_class[point_index]
                current_point_global = acts.from_local_indices([list_of_points_in_current_class[point_index]])[0]
                # the activation for the current neuron in the current point should equal the activation for the current point in the current neuron
                assert acts.get_activation(current_point_local).vector[current_neuron_index] == \
                       acts.get_activations_for_neuron(current_neuron_index)[0][0][current_point_global]
                # and this checks we've clustered it correctly
                assert acts.get_activation(current_point_local).vector[current_neuron_index] == current_cluster[
                    point_index]
                # phew that was effortful
            print('Coordinate transform from local to global check passed :)')
        cluster_list.append(current_cluster)
        min_list.append(min(current_cluster))
        max_list.append(max(current_cluster))
        # note this assertion is assuming a 1d vectro
    return cluster_list, min_list, max_list


def jitterer_list(x_data, colour_flag='mono', title='', save_label='fig', show_plots=False, save_plots=True,
                  do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict={}, outLayerNeuron=False,
                  current_neuron_indices=[]):
    """Lightweight plotter to plot jittered plots of x axis data with various options
    x_data: the x axis data
    colour_flag: flag for colour style: mono; cluster; random; highlight
    title: title for the plot
    save_label: filename if saving
    show_plots: whether to show plots
    save_plots: whether to save plots
    do_x_axis: whether to show the x axis
    do_y_axis: whether to show the y axis
    x_range: range to plot x over
    y_range: range to plot y over
    label_dict: the dictionary of label names for the classes
    outLayerNeuron: whether we're operating on 1-hot encoded output layer (it will label the graphs)
    current_neuron_index: only used with outLayerNeuron to select the current data, alternatively, use ths to select
                        : a cluster to be coloured black
    """
    fig = plt.figure()
    y_data = [1 + np.random.uniform(-0.25, 0.25) for x in x_data]
    if colour_flag == 'mono':
        print('doing mono')
        colour_option = 'k'
        plt.scatter(x_data, y_data, label=title,
                    color=colour_option, marker="o", alpha=0.25, s=1.75)
    # if colour_flag == 'random':
    #     z = np.random.rand(len(x_data))
    #     colour_option = z
    multi_colour_list = itertools.cycle(['blue', 'firebrick', 'darkgreen', 'm',
                                         'black', 'red', 'gold', 'darkcyan',
                                         'olivedrab', 'dodgerblue'])
    greyscale_list = itertools.cycle(['0.5', '0.6', '0.7', '0.8', '0.9'])
    black_list = itertools.cycle(['black'])
    marker_list = itertools.cycle(['s', 'p', '*', '8', '2', 'x', 'D', '+'])
    colour_list = multi_colour_list
    highlight_list = black_list
    if colour_flag == "multi":
        colour_list = greyscale_list
        highlight_list = multi_colour_list
    if colour_flag == 'cluster' or colour_flag == "multi":
        # we expect that x_data is actually a list of lists of points, not just a list of points
        # import code
        # code.interact(local=locals())
        min_of_x, max_of_x = x_data[0][0], x_data[0][0]
        min_of_y, max_of_y = 1, 1
        for cn in range(len(x_data)):
            if (outLayerNeuron and cn not in current_neuron_indices) or outLayerNeuron == False:
                # if not outLayerNeuron or cn != current_neuron_index:
                # If not our selected data, plot it with all teh colours
                x_data_subset = x_data[cn]  # [x[0] for x in cf[cn]]
                y_data_subset = [1 + np.random.uniform(-0.25, 0.25) for x in range(len(x_data_subset))]
                # we need this stuff to plot the full range nicely
                min_of_x = min(min(x_data_subset), min_of_x)
                min_of_y = min(min(y_data_subset), min_of_y)
                max_of_x = max(max(x_data_subset), max_of_x)
                max_of_y = max(max(y_data_subset), max_of_y)
                plt.scatter(x_data_subset, y_data_subset, label='neuron ' + str(0), marker="o", alpha=0.25,
                            color=next(colour_list))
            else:
                pass
                # cos we want to plot the last class last so its on the top
        if outLayerNeuron:
            # now we do the last ones
            for cn in current_neuron_indices:
                if verbose:
                    print('Output layer neuron for class no {}'.format(cn))
                    # If not our selected data, plot it with balck
                x_data_subset = x_data[cn]  # [x[0] for x in cf[cn]]
                y_data_subset = [1 + np.random.uniform(-0.25, 0.25) for x in range(len(x_data_subset))]
                # we need this stuff to plot the full range nicely
                min_of_x = min(min(x_data_subset), min_of_x)
                min_of_y = min(min(y_data_subset), min_of_y)
                max_of_x = max(max(x_data_subset), max_of_x)
                max_of_y = max(max(y_data_subset), max_of_y)
                plt.scatter(x_data_subset, y_data_subset,
                            label='neuron ' + str(0),
                            marker=next(marker_list),
                            alpha=0.75,
                            color=next(highlight_list))
        x_range = [min_of_x, max_of_x + 0.1 * max_of_x]
        y_range = [min_of_y, max_of_y]
    else:
        colour_option = 'k'
        plt.scatter(x_data, y_data, label=title,
                    color=colour_option, marker="o", alpha=0.25, s=1.75)
    ## now we have a scatter plot, lets fuck with it
    if x_range is None:
        # figure it out yourself
        plt.xlim([min(x_data), max(x_data)])
    else:
        plt.xlim(x_range)
    if y_range is None:
        # figure it out yourself
        plt.ylim([min(y_data), max(y_data)])
    else:
        plt.ylim(y_range)
    cur_axes = plt.gca()
    if not do_x_axis:
        cur_axes.axes.get_xaxis().set_visible(False)
    if not do_y_axis:
        cur_axes.axes.get_yaxis().set_visible(False)
    if show_plots:
        plt.show()
        plt.close()
    if save_plots:
        print('saving figure {}.'.format(save_label))
        fig.savefig(save_label, dpi=400)
        plt.close()
    plt.close()
    return


def compute_selectivity_neuron(max_list, min_list, found_labels, verbose=verbose):
    """
    computes the selectivity between classes
    this works on ONE neuron at a time
    :return:
    """
    if verbose:
        print("-- compute_selectivity --")
    # def brute_force_selectivity
    all_max = max_list
    all_min = min_list
    isSelective = False
    selectivity = 0.0
    found_class = ''
    for class_no in range(len(all_max)):
        min_match = all_min[class_no]
        max_match = all_max[class_no]
        min_notmatch = min([all_min[x] for x in range(0, 1000) if not x == class_no])
        max_notmatch = max([all_max[x] for x in range(0, 1000) if not x == class_no])
        if max_match > min_notmatch and max_notmatch > min_match:
            ## this is an overlap
            ## dont think we need to know what the overlap is do we?
            # overlaps[label] = min((max_match - min_notmatch), (max_notmatch - min_match))
            continue
        if min_notmatch < max_match:
            # positive selectivity -- On neuron
            selectivity = min_match - max_notmatch
            isSelective = True
        else:
            # negative selectivity
            selectivity = min_notmatch - max_match
            isSelective = True
        if max_match == 0.0 and selectivity == 0.0:
            # special case - neuron ignores this and others
            if verbose:
                print('ignoring 0.0 case')
            isSelective = False
            continue
        if verbose:
            print('{}: {}-{}\nothers: {}-{}'.format(class_no, min_match, max_match,
                                                    min_notmatch, max_notmatch))
        if isSelective:
            print('Selecitivy of {} found!'.format(selectivity))
            print('current_label is: {}'.format(found_labels[class_no]))
            print('class_no is: {}'.format(class_no))
            found_class = found_labels[class_no]
    return isSelective, selectivity, found_class


def class_code_to_name(class_name, class_dict, class_labels):
    """teeny function to get the name of any class
    uses class_dict nd class_ables from m
    class_name = the imagent code"""
    return class_labels[class_dict[class_name]]


def class_lineno_to_name(line_no=0, class_labels=class_labels):
    """teeny function to get the name of any class
    uses class_dict nd class_ables from m
    class_name = the imagent code"""
    entry = class_labels[line_no]
    meh = entry.split(' ')
    code = meh[0]
    name = meh[1:]
    return entry, code, name


def find_max_per_cluster(cluster_list, setting='max', verbose=verbose):
    """teeny function, loops over a list of arrays and grabs the max or mean"""
    max_per_class = []
    for c in cluster_list:
        if setting == 'max':
            max_per_class.append(max(c))
        elif setting == 'mean':
            max_per_class.append(np.mean(c))
        else:
            print('setting should be max or mean')
    return max_per_class


def compute_ccma_selectivity_neuron(cluster_list, found_labels='', class_dict=class_dict, class_labels=class_labels,
                                    top_class='', verbose=verbose):
    """
    computes the class conditional mean activity based selectivity measure
    this works on ONE neuron at a time
    cluster_list = values sorted into a cluster
    found_labels = labels for each cluster
    top_class = class to do comparison for, if known, if not, I'll calc the top, can be index or label
    :return:
    """
    print("-- compute_ccma_selectivity --")
    if top_class == '':
        # we figure out which is hte top class
        max_list = find_max_per_cluster(cluster_list=cluster_list, setting='mean', verbose=verbose)
        max_index, max_activation = max(enumerate(max_list), key=operator.itemgetter(1))
        entry, code, name = class_lineno_to_name(line_no=max_index, class_labels=class_labels)
        if verbose:
            print(entry)
    elif type(top_class) == int:
        # someones given us an index into cluster_list
        max_index = top_class
        if verbose:
            entry, code, name = class_lineno_to_name(line_no=max_index, class_labels=class_labels)
            print(entry)
    elif type(top_class) == str:
        code = top_class
        max_index = class_dict[code]
    else:
        print('top_class needs to be an index or a class code')
    # now we geht class conditional mean activity for the top_class
    mu_max = np.mean(cluster_list[max_index])
    not_cluster_list = [cluster_list[i] for i in range(len(cluster_list)) if not i == max_index]
    flattened_list = [y for x in not_cluster_list for y in x]
    mu_not_max = np.mean(flattened_list)
    ccma_selectivity = (mu_max - mu_not_max) / (mu_max + mu_not_max)
    return ccma_selectivity, mu_max, mu_not_max, max_index


def old_build_cluster_from_class_label(acts, current_neuron_index, label_dict, found_labels, do_check=do_check):
    """Builds a cluster list from the class labels and activation table
    This could be expanded to build a list of centroids as well and a detK object or similar
    acts: activation table object
    current_neuron_index: which neuron to do this for
    label_dict: look up table that matches the found_labels to the local coordinates in the activation table object
    found_labels: list of labels which are keys to label_dict
    returns a list of lists of points which are members of each category
    """
    cluster_list = []
    for class_index in range(len(label_dict)):
        if verbose:
            print('building cluster based on class labels for label {}'.format(found_labels[class_index]))
        # this is the list of points in our current class in the 2-d tuple format
        list_of_points_in_current_class = label_dict[found_labels[class_index]]
        # this is the list of points in our current class in the 1-d global number format (which corresponds to the vector's dimentions)
        global_list_of_points_in_current_class = acts.from_local_indices(
            list_of_points_in_current_class)  # this gives all the points as a list
        # thus our cluster of points for this class is...
        current_cluster = acts.get_activations_for_neuron(current_neuron_index)[0][0][
            global_list_of_points_in_current_class]
        # and cos that was so fucking belaboured, lets check that shit
        # current_point = list_of_points_in_current_class[point_index]
        if do_check:
            for point_index in range(len(list_of_points_in_current_class)):
                current_point_local = list_of_points_in_current_class[point_index]
                current_point_global = acts.from_local_indices([list_of_points_in_current_class[point_index]])[0]
                # the activation for the current neuron in the current point should equal the activation for the current point in the current neuron
                assert acts.get_activation(current_point_local).vector[current_neuron_index] == \
                       acts.get_activations_for_neuron(current_neuron_index)[0][0][current_point_global]
                # and this checks we've clustered it correctly
                assert acts.get_activation(current_point_local).vector[current_neuron_index] == current_cluster[
                    point_index]
                # phew that was effortful
            print('Coordinate transform from local to global check passed :)')
        cluster_list.append(current_cluster)
        # note this assertion is assuming a 1d vectro
    # 9.2300749
    return cluster_list


def tiger_shark_or_toilet_roll(label_no, label_dict=label_dict, point_no=None, acts=acts,
                               found_labels=found_labels,
                               print_value=True, class_labels={}):
    """Tells you the class of a activation picture by selecting the max activation
    N.B. this expects that you are using fc8 activations
    if not, it will return the neuron wit the highest activation
    point_no: if None, will do all points, else give a list or an integer
    label_no: which found_label do you want to ivestigate
    print_value: whether to print out the value
    """
    tiger_sharks = []

    def find_position_of_max_act_in_vector(vector):
        """lets find that max"""
        out = np.where(vector == vector.max())
        return out, vector.max()

    def _inner(print_value=print_value, class_labels=class_labels, tiger_sharks=tiger_sharks):
        for current_shark_label in tiger_sharks:
            current_shark = current_shark_label.vector
            i, j = find_position_of_max_act_in_vector(current_shark)
            if (class_labels == {}).any() == False:
                label = class_labels[i[0][0]]
            else:
                label = ''
            if print_value:
                print('Position {}, value {}: {}'.format(i, j, label))
            else:
                print('Position {}: {}'.format(i, label))
        return

    if point_no == None:
        # assume you want all points
        tiger_sharks = acts.get_points(label_dict[found_labels[label_no]])
        _inner(print_value=print_value, class_labels=class_labels, tiger_sharks=tiger_sharks)
    elif type(point_no) is int:
        tiger_sharks = [acts.get_points(label_dict[found_labels[label_no]])[point_no]]
        _inner(print_value=print_value, class_labels=class_labels, tiger_sharks=tiger_sharks)
    elif type(point_no) is list:
        # # assuming a sublist of points
        tiger_sharks = [acts.get_points(label_dict[found_labels[label_no]])[i] for i in point_no]
        _inner(print_value=print_value, class_labels=class_labels, tiger_sharks=tiger_sharks)
    return


# if colour == 'random':
#     z = np.random.rand(576)
#     x_data = [x[0] for x in dks[l][n - 1].X]
#     y_data = [1 + np.random.uniform(-0.25, 0.25) for x in dks[l][n - 1].X]
#     plt.scatter(x_data, y_data, label='neuron ' + str(j), c=z, marker="o", alpha=0.5)

def jitterer(x_data, colour_flag='mono', title='', save_label='fig', show_plots=False, save_plots=True,
             do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict={}, outLayerNeuron=False,
             current_neuron_index=0):
    """Lightweight plotter to plot jittered plots of x axis data with various options
    x_data: the x axis data
    colour_flag: flag for colour style: mono; cluster; random;
    title: title for the plot
    save_label: filename if saving
    show_plots: whether to show plots
    save_plots: whether to save plots
    do_x_axis: whether to show the x axis
    do_y_axis: whether to show the y axis
    x_range: range to plot x over
    y_range: range to plot y over
    label_dict: the dictionary of label names for the classes
    outLayerNeuron: whether we're operating on 1-hot encoded output layer (it will label the graphs)
    current_neuron_index: only used with outLayerNeuron to select the current data, alternatively, use ths to select
                        : a cluster to be coloured black
    """
    fig = plt.figure()
    y_data = [1 + np.random.uniform(-0.25, 0.25) for x in x_data]
    if colour_flag == 'mono':
        print('doing mono')
        colour_option = 'k'
        plt.scatter(x_data, y_data, label=title,
                    color=colour_option, marker="o", alpha=0.25, s=1.75)
    # if colour_flag == 'random':
    #     z = np.random.rand(len(x_data))
    #     colour_option = z
    if colour_flag == 'cluster':
        # we expect that x_data is actually a list of lists of points, not just a list of points
        colourList = itertools.cycle(['blue', 'firebrick', 'gray', 'darkgreen', 'm',
                                      'darkorange', 'red', 'gold', 'darkcyan',
                                      'olivedrab', 'dodgerblue'])
        min_of_x, max_of_x = x_data[0][0], x_data[0][0]
        min_of_y, max_of_y = 1, 1
        for cn in range(len(x_data)):
            if ((outLayerNeuron == True) and (not cn == current_neuron_index)) or outLayerNeuron == False:
                # if not outLayerNeuron or cn != current_neuron_index:
                # If not our selected data, plot it with all teh colours
                x_data_subset = x_data[cn]  # [x[0] for x in cf[cn]]
                y_data_subset = [1 + np.random.uniform(-0.25, 0.25) for x in range(len(x_data_subset))]
                # we need this stuff to plot the full range nicely
                min_of_x = min(min(x_data_subset), min_of_x)
                min_of_y = min(min(y_data_subset), min_of_y)
                max_of_x = max(max(x_data_subset), max_of_x)
                max_of_y = max(max(y_data_subset), max_of_y)
                plt.scatter(x_data_subset, y_data_subset, label='neuron ' + str(0), marker="o", alpha=0.5,
                            color=next(colourList))
            else:
                pass
                # cos we want to plot the last class last so its on the top
        if outLayerNeuron == True:
            # now we do that last one
            cn = current_neuron_index
            if verbose:
                print('Output layer neuron for class no {}'.format(cn))
                # If not our selected data, plot it with balck
            x_data_subset = x_data[cn]  # [x[0] for x in cf[cn]]
            y_data_subset = [1 + np.random.uniform(-0.25, 0.25) for x in range(len(x_data_subset))]
            # we need this stuff to plot the full range nicely
            min_of_x = min(min(x_data_subset), min_of_x)
            min_of_y = min(min(y_data_subset), min_of_y)
            max_of_x = max(max(x_data_subset), max_of_x)
            max_of_y = max(max(y_data_subset), max_of_y)
            plt.scatter(x_data_subset, y_data_subset, label='neuron ' + str(0), marker="s", alpha=0.75,
                        color='k')
        x_range = [min_of_x, max_of_x + 0.1 * max_of_x]
        y_range = [min_of_y, max_of_y]
    else:
        colour_option = 'k'
        plt.scatter(x_data, y_data, label=title,
                    color=colour_option, marker="o", alpha=0.25, s=1.75)
    ## now we have a scatter plot, lets fuck with it
    if x_range == None:
        # figure it out yourself
        plt.xlim([min(x_data), max(x_data)])
    else:
        plt.xlim(x_range)
    if y_range == None:
        # figure it out yourself
        plt.ylim([min(y_data), max(y_data)])
    else:
        plt.ylim(y_range)
    cur_axes = plt.gca()
    if do_x_axis == False:
        cur_axes.axes.get_xaxis().set_visible(False)
    if do_y_axis == False:
        cur_axes.axes.get_yaxis().set_visible(False)
    if show_plots:
        plt.show()
        plt.close()
    if save_plots:
        print('saving figure {}.'.format(save_label))
        fig.savefig(save_label, dpi=400)
        plt.close()
    plt.close()
    return


# class_dict is a dictionary of the class_labels to their row number
def make_class_to_line_number_look_up_table(class_labels=class_labels, verbose=False):
    """Function to link the class code with the line number as used in imagenet
    class_labels: output from reading in the class label file in Make_activation
    verbose: set to true if you want a print out on-screen of line number ot class"""
    class_dict = {}
    for line_no in range(len(class_labels)):
        line = class_labels[line_no]
        class_code = line.split(' ')[0]
        class_name = line.split(' ')[1:]
        if verbose:
            print('class no. {}: class: {}: {}'.format(line_no, class_code, ' '.join(class_name)))
        class_dict[class_code] = line_no
    return class_dict


def append_images(images, direction='horizontal',
                  bg_color=(255, 255, 255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.
    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'
    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))
    if direction == 'horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)
    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)
    offset = 0
    for im in images:
        if direction == 'horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1]) / 2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0]) / 2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]
    return new_im


def crop_to_square(image_address):
    """function to crop to open and crop an image to square
    cos rolleiflex was correct"""
    img = Image.open(image_address)
    w, h = img.size
    half_the_width = w / 2
    half_the_height = h / 2
    # get square
    if w < h:
        img = img.crop(
            (
                half_the_width - half_the_width,
                half_the_height - half_the_width,
                half_the_width + half_the_width,
                half_the_height + half_the_width
            )
        )
    elif h < w:
        img = img.crop(
            (
                half_the_width - half_the_height,
                half_the_height - half_the_height,
                half_the_width + half_the_height,
                half_the_height + half_the_height
            )
        )
    return img


def make_collage(out_file='temp.jpg', local_list=local_list, shrink=True, do_square=True, no_of_cols=5,
                 acts=acts, class_dict=class_dict, class_labels=class_labels,
                 verbose=verbose, imagenet_root='/storage/data/imagenet_2012/'):
    """Function to get the selected images and collage them
    local_list: list of points in act to find pictures for
    shrink: whether to shrink images
    do_square: whether to squarify the images
    no_of_cols: how many colums in the collage
    returns an image object"""
    selected_image_list = []
    found_classes = []
    for selected_point in local_list:
        # grab filename
        selected_file = acts.get_file_name(selected_point).decode('UTF-8')
        if verbose:
            pass
            # print(selected_file)
        # we've assumed files are in folders labelled by class!
        class_dir_label = filename_to_label(selected_file.split('_')[0])
        selected_image_list.append(imagenet_root + class_dir_label + '/' + selected_file)
        class_no = class_dict[class_dir_label]
        if not class_no in found_classes:
            found_classes.append(class_no)
    if shrink and do_square:
        images = [crop_to_square(x).resize((277, 277)) for x in selected_image_list]
    elif shrink:
        # this option may not work so do not use it!
        images = [Image.open for x in selected_image_list]
        images = [x.resize((277, 277)) for x in images]
    elif do_square:
        images = [crop_to_square(x) for x in selected_image_list]
    else:
        images = map(Image.open, selected_image_list)
    rows = []
    # make row images
    if len(images) < 10:
        no_of_cols = len(images)
    for row_no in range(int(len(images) / no_of_cols)):
        rows.append(
            append_images(images[0 + no_of_cols * row_no: no_of_cols + no_of_cols * row_no], direction='horizontal'))
    # stack row images
    combined_image = append_images(rows, direction='vertical')
    try:
        combined_image.save(out_file)
    except IOError:
        print("Failed to write {}, trying as png".format(out_file))
        try:
            out_file_png = "{}.png".format(out_file)
            combined_image.save(out_file_png)
        except IOError:
            print("Failed to write {} as png".format(out_file_png))
    if verbose:
        print('Found the following classes:')
        for class_no in found_classes:
            print('{}'.format(class_labels[class_no]))
    # if you want to see it do combined_image.show()
    return combined_image, found_classes


def get_class_clusters_for_neuron(current_neuron_index, x_data=[], acts=acts, label_dict=label_dict,
                                  found_labels=found_labels):
    """function to generate useful info for a single neuron from acts
    current_neuron_index: which neuron to investigate"""
    # ! TODO: perhaps add in kmeans here to get kmeans generated clusters for a neuron, when it works
    if x_data == []:
        # this is slow, so only do it if you have to
        current_neuron = acts.get_activations_for_neuron(current_neuron_index)
        x_data = current_neuron.vector
    else:
        current_neuron = []
        pass
    cluster_list = build_cluster_from_class_label(acts=acts, current_neuron_index=current_neuron_index,
                                                  current_neuron=current_neuron,
                                                  label_dict=label_dict,
                                                  found_labels=found_labels, do_check=do_check,
                                                  verbose=False)
    return x_data, cluster_list


def grab_points_for_a_cluster(current_neuron_index,
                              min_selected_x_data,
                              max_selected_x_data,
                              x_data=[],
                              acts=acts,
                              verbose=verbose):
    """Grabs image names and addresses for images that fall within an activation range
    current_neuron_index: which neuron to use
    min_selected_x_data: minimum activation to grab (leq)
    max_selected_x_data: maximum activation to grab (meq)
    acts: activations
    returns the list sorted from min to max
    """
    if x_data == []:
        current_neuron = acts.get_activations_for_neuron(current_neuron_index)
        x_data = current_neuron.vector
    else:
        pass
    # we find the activations
    selected_activation_indices = \
        np.where(np.logical_and(x_data >= min_selected_x_data, x_data <= max_selected_x_data))[0]
    # we grab the activations
    selected_activations = x_data[selected_activation_indices]
    # we sort them based on value of activation and gt a list
    sorted_indices = np.argsort(selected_activations)
    selected_activations = x_data[selected_activation_indices[sorted_indices]]
    selected_activation_indices = selected_activation_indices[sorted_indices]
    no_of_selected_activations = len(selected_activation_indices)
    if verbose:
        print('No. of selected activations is {}'.format(no_of_selected_activations))
    # selected_activations = x_data[selected_activation_indices]
    if verbose:
        print('Selected activations are {} on average with a standard deviation of {}'.format(
            np.mean(selected_activations), np.std(selected_activations)))
    local_list = acts.to_local_indices(selected_activation_indices)
    return local_list, selected_activations


# def calculate_average_precision_incl_zeros(test_class, local_list='', x_data='', selected_activations='',
#                                            current_neuron_index=current_neuron_index, acts=acts, verbose=verbose):
#     """Wrapper function to calculate the average precision when we have lots of zeros"""
#     if local_list == '':
#         # if you don't pass it in, get the data
#         local_list, selected_activations, x_data = get_local_list_for_neuron(current_neuron_index=current_neuron_index,
#                                                                              minx=0,
#                                                                              acts=acts)
#     total_no_of_points = len(x_data)
#     no_of_points_selected = len(local_list)
#     # grabs all points above 0 (special case)
#     if verbose:
#         print('Taking all points above 0.0, minimum is {}'.format(min(selected_activations)))
#     Ave_precs_x, precs_x, recall_x = calculate_ave_precs_general(
#         test_class=test_class,
#         local_list=local_list,
#         Q_stop='',
#         no_files_in_label=no_files_in_label,
#         verbose=verbose)
#     # this gives the values just above zero
#     precs_all = no_files_in_label[test_class] / total_no_of_points  # a largely pointless measure
#     recall_all = 1.0
#     delta_recall_all = recall_all - recall_x  # difference in recall between this point nd the next
#     weight_precs_all = precs_all * delta_recall_all  # weighted precsion at point x (we do average via weighted sum)
#     Ave_precs_all = Ave_precs_x + weight_precs_all
#     if verbose:
#         print('Class {}, evaluated at Q={}:, found {}'.format(test_class, total_no_of_points,
#                                                               no_files_in_label[test_class]))
#         print('Average precision={}; precision={}; recall={}'.format(Ave_precs_all, precs_all, recall_all))
#     return Ave_precs_all, precs_all, recall_all, Ave_precs_x, precs_x, recall_x


def single_cluster_analysis(current_neuron_index=current_neuron_index,
                            min_selected_x_data=0,
                            max_selected_x_data=100,
                            acts=acts,
                            x_data=[],
                            name_stem='collage',
                            class_dict=class_dict,
                            class_labels=class_labels,
                            verbose=verbose,
                            do_pictures=do_pictures
                            ):
    """wrapper function to analyse a single cluster on a single neuron, this gives out the number of selected activations
    mean and std, their classes and produces a picture
    """
    # Todo could tidy this up
    # # # lets grab some points (currently I bin the activations)
    local_list, selected_activations = grab_points_for_a_cluster(current_neuron_index=current_neuron_index,
                                                                 min_selected_x_data=min_selected_x_data,
                                                                 max_selected_x_data=max_selected_x_data,
                                                                 x_data=x_data,
                                                                 acts=acts,
                                                                 verbose=True)
    if do_pictures == True:
        egg = make_collage(out_file=name_stem + str(current_neuron_index) + '.jpg',
                           local_list=local_list,
                           shrink=True,
                           do_square=True,
                           no_of_cols=10,
                           acts=acts,
                           class_dict=class_dict,
                           class_labels=class_labels,
                           verbose=verbose,
                           imagenet_root='/storage/data/imagenet_2012/')
    else:
        egg = ''
    # egg = make_collage(out_file=name_stem + str(current_neuron_index) + '.jpg', local_list=local_list, shrink=True, do_square=True, no_of_cols=10,
    #             acts=acts, verbose=verbose, imagenet_root='/storage/data/imagenet_2012/')
    # to see picutre, do egg.show()
    return local_list, selected_activations, egg


def spotty_plotter(dks, input_flag='K', doHistogram=False, colour='random', doMu=True, show_plots=0, save_plots=1,
                   cols=2, label='', no_of_layers=1):
    """Make things that look like neuronal plots"""
    "doMu is whether to plot the centers of K-means centroids (mu)"
    "colour= 'random' or 'centroid', or 'black' or ''"
    for l in range(no_of_layers):
        # figsize is in inches, default is 8,6
        fig = plt.figure(l, figsize=(24, 18))
        # t = a.jitterer(out, l)
        # yrange=max(out[l])-min(out[l])
        r = len(dks[l]) / cols
        c = cols
        n = 1
        layer_overlaps = min_overlaps[l]
        if input_flag == 'K':
            # new style using dks
            if doHistogram == True:
                pass
            else:
                for i in range(r):
                    for j in range(c):
                        plt.subplot(r, c, n)
                        if colour == 'centroid':
                            # z = np.random.rand(576)
                            cf = dks[l][n - 1].clusters
                            colourList = itertools.cycle(['blue', 'firebrick', 'gray', 'darkgreen', 'm',
                                                          'darkorange', 'black', 'red', 'gold', 'darkcyan',
                                                          'olivedrab', 'dodgerblue'])
                            for cn in range(len(cf)):
                                x_data = [x[0] for x in cf[cn]]
                                y_data = [1 + np.random.uniform(-0.25, 0.25) for x in cf[cn]]
                                plt.scatter(x_data, y_data, label='neuron ' + str(j), marker="o", alpha=0.5,
                                            color=next(colourList))
                        else:
                            if colour == 'random':
                                z = np.random.rand(576)
                                x_data = [x[0] for x in dks[l][n - 1].X]
                                y_data = [1 + np.random.uniform(-0.25, 0.25) for x in dks[l][n - 1].X]
                                plt.scatter(x_data, y_data, label='neuron ' + str(j), c=z, marker="o", alpha=0.5)
                            if colour == 'black':
                                full_x_data = np.array([x[0] for x in dks[l][n - 1].X])
                                full_y_data = np.array([1 + np.random.uniform(-0.25, 0.25) for x in full_x_data])
                                plt.scatter(full_x_data, full_y_data, label='neuron ' + str(j), color='k', marker="o",
                                            alpha=0.25, s=0.75)
                                das_over_label, das_under_label = layer_overlaps[n - 1]
                                # import pdb
                                # pdb.set_trace()
                                plt.scatter(full_x_data[label_dict[das_over_label]],
                                            full_y_data[label_dict[das_over_label]],
                                            label='neuron {}({})'.format(j, das_over_label), color='green', marker="o",
                                            alpha=0.75, s=1)
                                plt.scatter(full_x_data[label_dict[das_under_label]],
                                            full_y_data[label_dict[das_under_label]],
                                            label='neuron {}({})'.format(j, das_under_label), color='red', marker="o",
                                            alpha=0.75, s=1)
                        if doMu == 1:
                            mu_data = [x[0] for x in dks[l][n - 1].mu]
                            plt.scatter(mu_data, np.ones(len(mu_data)),
                                        marker="^", facecolors='none', edgecolors='k', s=100, alpha=1.0)
                        n = n + 1
                        plt.xlim([-0.05, 1.05])
                        plt.ylim([0.7, 1.3])
                        cur_axes = plt.gca()
                        cur_axes.axes.get_yaxis().set_visible(False)
                        cur_axes.axes.get_xaxis().set_visible(False)
        if show_plots == 1:
            plt.show()
            plt.close()
        if save_plots == 1:
            print('saving figure {}.'.format(l))
            fig.savefig('spotty' + str(l) + label + '.png', dpi=fig.dpi)
            plt.close()


def fs_plotter(fs, K_list=[], layer_name='', current_neuron_index=current_neuron_index, do_horizontal=True,
               horizon=1.0):
    """Function to plot out fs versus K
    fs is input
    K_list is the values of K chosen, we assume usually it goes from 1 to the length of fs"""
    no_of_points = len(fs)
    if do_horizontal == True:
        horz = [1.0 for x in range(len(fs) + 1)]
    if K_list == []:
        # assume a 1-indexed values
        K_list = [1 + x for x in range(len(fs))]
    fig = plt.figure()
    plt.plot(K_list, fs.tolist(), 'o-')
    plt.plot(horz)
    plt.xlabel('K')
    plt.ylabel('fs')
    plt.title(layer_name + str(current_neuron_index))
    print('saving figure {}.'.format(layer_name + str(current_neuron_index)))
    fig.savefig('fs_' + layer_name + '_' + str(current_neuron_index) + '.png', bbox_inches='tight')
    plt.close()
    return


def selectivity_grid(dks, input_flag='K', doHistogram=False, colour='random', doMu=True, show_plots=0, save_plots=1, no_of_layers=1):
    """Makes selectivity measures and a grid"""
    "doMu is whether to plot the centers of K-means centroids (mu)"
    "colour= 'random' or 'centroid', or 'black' or ''"
    for l in range(no_of_layers):
        print
        l
        fig = plt.figure(l)
        # t = a.jitterer(out, l)
        # yrange=max(out[l])-min(out[l])
        r = len(dks[l]) / 2
        c = 2
        n = 1
        if input_flag == 'K':
            # new style using dks
            if doHistogram == True:
                pass
            else:
                for i in range(r):
                    for j in range(c):
                        plt.subplot(r, c, n)
                        cf = dks[l][n - 1].clusters
                        noOfClusters = len(cf)
                        for cn in range(noOfClusters - 1):
                            # NTS this is hacky and only works for a list of 2 things
                            if dks[l][n - 1].mu[cn] > dks[l][n - 1].mu[cn + 1]:
                                old_selectivity = min(cf[cn]) - max(cf[cn + 1])
                            else:
                                old_selectivity = min(cf[cn + 1]) - max(cf[cn])
                        if noOfClusters == 2:
                            # old school selectivity IS DEFINED, so print it
                            print('layer {0}, neuron {1}, selectivity = {2}'.format(l, n, old_selectivity))
                            z = float(old_selectivity)
                            plt.text(0.05, 0.95, 'sel =' + str(old_selectivity), fontsize=14,
                                     verticalalignment='top')
                            if z < 0.5:
                                plt.text(0.05, 0.95, 'sel =' + str(old_selectivity), fontsize=14,
                                         verticalalignment='top', color='white')
                            if z < 0:
                                z = 0
                            plt.text(0.01, 1.97, str(l) + str(n), fontsize=12,
                                     verticalalignment='top')
                        else:
                            z = 0
                        if z < 0.5:
                            plt.text(0.2, 1.2, 'k = ' + str(dks[l][n - 1].K), fontsize=14,
                                     verticalalignment='top', color='white')
                        else:
                            plt.text(0.2, 1.2, 'k = ' + str(dks[l][n - 1].K), fontsize=14,
                                     verticalalignment='top')
                        n = n + 1
                        plt.xlim([-0.05, 1.05])
                        plt.ylim([0.7, 1.3])
                        cur_axes = plt.gca()
                        cur_axes.axes.get_yaxis().set_visible(False)
                        cur_axes.set_axis_bgcolor((z, z, z))
        if show_plots == 1:
            plt.show()
            plt.close()
        if save_plots == 1:
            fig.savefig('spotty' + str(l) + '.png', dpi=fig.dpi)
            plt.close()


# @profile


def new_spotty_plotter(dks, input_flag='K', doHistogram=False, colour='random', doMu=True, show_plots=0, save_plots=1,
                       cols=2, label='', no_of_layers=1):
    """Make things that look like neuronal plots"""
    "doMu is whether to plot the centers of K-means centroids (mu)"
    "colour= 'random' or 'centroid', or 'black' or ''"
    for l in range(no_of_layers):
        # figsize is in inches, default is 8,6
        fig = plt.figure(l, figsize=(24, 18))
        # t = a.jitterer(out, l)
        # yrange=max(out[l])-min(out[l])
        r = len(dks[l]) / cols
        c = cols
        n = 1
        layer_overlaps = min_overlaps[l]
        if input_flag == 'K':
            # new style using dks
            if doHistogram == True:
                pass
            else:
                for i in range(r):
                    for j in range(c):
                        plt.subplot(r, c, n)
                        if colour == 'centroid':
                            # z = np.random.rand(576)
                            cf = dks[l][n - 1].clusters
                            colourList = itertools.cycle(['blue', 'firebrick', 'gray', 'darkgreen', 'm',
                                                          'darkorange', 'black', 'red', 'gold', 'darkcyan',
                                                          'olivedrab', 'dodgerblue'])
                            for cn in range(len(cf)):
                                x_data = [x[0] for x in cf[cn]]
                                y_data = [1 + np.random.uniform(-0.25, 0.25) for x in cf[cn]]
                                plt.scatter(x_data, y_data, label='neuron ' + str(j), marker="o", alpha=0.5,
                                            color=next(colourList))
                        else:
                            if colour == 'random':
                                z = np.random.rand(576)
                                x_data = [x[0] for x in dks[l][n - 1].X]
                                y_data = [1 + np.random.uniform(-0.25, 0.25) for x in dks[l][n - 1].X]
                                plt.scatter(x_data, y_data, label='neuron ' + str(j), c=z, marker="o", alpha=0.5)
                            if colour == 'black':
                                full_x_data = np.array([x[0] for x in dks[l][n - 1].X])
                                full_y_data = np.array([1 + np.random.uniform(-0.25, 0.25) for x in full_x_data])
                                plt.scatter(full_x_data, full_y_data, label='neuron ' + str(j), color='k', marker="o",
                                            alpha=0.25, s=0.75)
                                das_over_label, das_under_label = layer_overlaps[n - 1]
                                # import pdb
                                # pdb.set_trace()
                                plt.scatter(full_x_data[label_dict[das_over_label]],
                                            full_y_data[label_dict[das_over_label]],
                                            label='neuron {}({})'.format(j, das_over_label), color='green', marker="o",
                                            alpha=0.75, s=1)
                                plt.scatter(full_x_data[label_dict[das_under_label]],
                                            full_y_data[label_dict[das_under_label]],
                                            label='neuron {}({})'.format(j, das_under_label), color='red', marker="o",
                                            alpha=0.75, s=1)
                        if doMu == 1:
                            mu_data = [x[0] for x in dks[l][n - 1].mu]
                            plt.scatter(mu_data, np.ones(len(mu_data)),
                                        marker="^", facecolors='none', edgecolors='k', s=100, alpha=1.0)
                        n = n + 1
                        plt.xlim([-0.05, 1.05])
                        plt.ylim([0.7, 1.3])
                        cur_axes = plt.gca()
                        cur_axes.axes.get_yaxis().set_visible(False)
                        cur_axes.axes.get_xaxis().set_visible(False)
        if show_plots == 1:
            plt.show()
            plt.close()
        if save_plots == 1:
            print('saving figure {}.'.format(l))
            fig.savefig('spotty' + str(l) + label + '.png', dpi=fig.dpi)
            plt.close()


def calculate_mean_average_precision(class_name='', current_neuron_index=current_neuron_index, acts=acts,
                                     verbose=verbose, minx=0.000000001):
    """Counts down using -1 to min list length+1 indexing and calculates the mean average precision
    given by: M.A.P. = sum over j of no. of A so far found at position j divided by the position in the list
     Note that we are counting backwards, the formula expects a 1-indexed list and j counts position over a 1-indexed list
     Note, we assume if no class name is given you want the class for the highest activation
     minx = the minimum activation to consider, this must be above 0.0 as these all have the same value"""
    #
    current_neuron = acts.get_activations_for_neuron(current_neuron_index)  # get the neuron's data
    x_data = current_neuron.vector  # get the activations without classes
    # grab your list of points
    local_list, selected_activations = grab_points_for_a_cluster(current_neuron_index,
                                                                 min_selected_x_data=minx,
                                                                 max_selected_x_data=max(x_data),
                                                                 acts=acts,
                                                                 x_data=x_data,
                                                                 verbose=verbose)
    Q = len(local_list)  # total length of list
    # get the test class (this is the correct class or 'A')
    if class_name == '':
        test_class = local_list[-1][0]
    else:
        test_class = class_name
    # set up counters
    MAP = 0  # mean average precision
    count_of_test_class = 0
    # loop backwards through the list, abs j is the position in a 1-indexed list
    for i in range(Q + 1):
        j = -(i + 1)  # 1 indexed
        current_class = local_list[j][0]  # current class
        if j == -Q:
            # if the whole of local_list is the same class (this accounts for zero indexing)
            if verbose:
                print(current_class)
                print('{}/{}'.format(count_of_test_class, abs(j)))
            j = j - 1  # really this is here so we can check j
            break
        if (current_class == test_class):
            count_of_test_class = count_of_test_class + 1
            MAP = MAP + count_of_test_class / (abs(j))  # N.b. this is the sum, we divide by j on the output
    return MAP / Q


def find_zhou_precision(number_of_points=100, local_list=local_list):
    """Finds the maximally occuring class in the top number_of_points and counts it"""
    classes_in_top_100 = [local_list[x][0] for x in range(-number_of_points, 0)]
    zhou = Counter(classes_in_top_100)
    zhou_precs_class = zhou.most_common(1)[0][0]  # the class name
    zhou_precs = zhou.most_common(1)[0][1] / number_of_points  # the precision
    zhou_no_of_classes = len(zhou)
    return zhou_precs_class, zhou_precs, zhou_no_of_classes, zhou




def test_for_repetition_in_activations(x_data, verbose=verbose):
    """Little function to keep an eye out for some wierd shit"""
    if verbose:
        test_1 = Counter(x_data)
    print(test_1)
    if all_same(test_1.values()):
        print('{} values, each repeated {} times'.format(len(test_1.values()), test_1.values()[0]))
    else:
        print('{} values'.format(len(test_1.values())))
    return


def find_position_of_max_act_in_vector(vector):
    """lets find that max"""
    out = np.where(vector == vector.max())
    return out, vector.max()


def find_gaps_between_clusters(cluster_list, dict_keys=[], invert=True):
    """Finds the gaps between clusters
    Takes in cluster_list (from cloud.clusters) and not a detK object so we can use this on anything
    cluster_list: dictionary of clusters
    dict_key: list of dictionary keys to loop over - currently assumes numbers in increasing order
    invert - whether to code the gaps from the highest activations or not"""
    K = len(cluster_list)  # no of clusters
    gap_list = []
    if (K == 0):
        print("WARN:find_gaps_between_clusters called with empty gap list")
        return

    def code_the_gaps(gap_list, invariant_gap_list=[], K=K, invert=invert):
        """mini function as we call this twice"""
        do_invariant = True
        if invariant_gap_list == []:
            invariant_gap_list = gap_list
            do_invariant = False
        max_gap_code, max_gap = find_position_of_max_act_in_vector(np.array(gap_list))
        if do_invariant:
            max_gap_code = np.where(np.array(invariant_gap_list) == max_gap)
        if invert == True:
            # There are K - 1 gaps, if invert, we start counting from the highest activation cluster
            max_gap_code = K - 1 - max_gap_code[0].tolist()[0]
        else:
            max_gap_code = max_gap_code[0].tolist()[0]
        return max_gap_code, max_gap

    if dict_keys == []:
        # assume we know what these keys are, sigh, they are what FAstDetK made
        dict_keys = [i for i in range(K)]
    for c in range(K - 1):
        higher = dict_keys[c + 1]
        lower = dict_keys[c]
        # print('{} - {}'.format(c + 1, c))
        gap = min(cluster_list[higher]) - max(cluster_list[lower])
        assert min(cluster_list[higher]) > max(cluster_list[lower])
        # print(gap)
        gap_list.append(gap)
    # now we analyse this stuff
    max_gap_code, max_gap = code_the_gaps(gap_list=gap_list)
    # now to get the 2nd biggest gap
    new_gap_list = [x for x in gap_list if not x == max_gap]
    assert (len(new_gap_list) + 1) == len(gap_list)
    # now do it again
    if len(new_gap_list) > 1:
        max_2_gap_code, max_2_gap = code_the_gaps(gap_list=new_gap_list, invariant_gap_list=gap_list)
    else:
        # we found a neuron which only has two clusters in the top half
        print('highly selective neuron found')
        max_2_gap_code = 0
        max_2_gap = 0
    return gap_list, max_gap, max_gap_code, max_2_gap, max_2_gap_code


def find_extent_of_top_class(local_list=local_list):
    """starts at the highest index and counts down until the classes no longer match"""
    for i in range(len(local_list) + 1):
        j = -(i + 1)  # 1 indexed
        if j == -1:
            test_class = local_list[j][0]
        current_class = local_list[j][0]
        if j == -len(local_list):
            # if the whole of local_list is the same class 9(this accounts for zero indexing)
            j = j - 1
            break
        if not (current_class == test_class):
            # print(j)
            break
    no_of_members_of_test_class = -j - 1
    return no_of_members_of_test_class


########################################################################################################################
#  main
########################################################################################################################
# m = make activations, which reads in the hdf5 stuff and creates the activation table

def main():
    global acts, class_labels, labels, h5_list, caffe_settings
    isClassSelective = False
    # class_labels is the imagenet labels for 2012, both human readable and n909402394
    # image_directory = '/storage/data/imagenet_2012'
    # set up caffe default
    # image_directory = '/storage/data/top_1_imagenet_2012/'

    # this is hte bit that sets up the caffe networks ------------------------------------------------------------------
    caffe_settings = s.main()
    # caffe_root = s.caffe_root
    image_directory = caffe_settings.image_directory
    labels_file = caffe_settings.labels_file
    model_def = caffe_settings.model_def
    model_weights = caffe_settings.model_weights
    dir_list = caffe_settings.dir_list
    labels = caffe_settings.labels
    net = caffe_settings.net
    transformer = caffe_settings.transformer
    short_labels = caffe_settings.short_labels
    model_file = caffe_settings.model_file
    this_one_file_name = caffe_settings.this_one_file_name
    blob = caffe_settings.blob
    file_root = caffe_settings.file_root
    class_labels = short_labels
    # end of the bit tha sets up the caffe netwroks --------------------------------------------------------------------
    print('I am using the following merged h5 file: {}'.format(this_one_file_name))
    print('Which I expect to be located at: {}'.format(file_root))
    do_check = False
    m.main()
    acts = m.acts
    class_labels = m.class_labels

    class_dict = make_class_to_line_number_look_up_table(class_labels=class_labels, verbose=False)
    # this builds the look-up table between points and the class they are in
    ## This bit is sslow, it loads the label data for all acts
    sys.stdout.write('About to build the label dict (slow)')
    sys.stdout.flush()
    label_dict, found_labels, no_files_in_label = build_label_dict(acts)
    sys.stdout.write('Built the label dict')
    sys.stdout.flush()

    # class_to_found_label_dict={}
    # for

    no_of_images = acts.get_image_count()
    print('Found {} images'.format(no_of_images))

    # whether  to sanity check the created activation table (acts)
    if do_check == True:
        m.check_labels(acts, class_labels)

    ## This chunk checks that hte no_of_neurons is correct
    temp = []  # this is hte no of neurosn each image has data for, these should all be the same!
    if do_check:
        for current_point in acts.get_all_activation_indices():
            temp.append(len(acts.get_activation(current_point).vector))
        assert all_same(temp) == True
        no_of_neurons = temp[0]
    else:
        # simply grab the no of neurons from the first batch point
        # no_of_neurons = len(acts.get_activation(acts.get_batch_indices(2)[0]).vector)
        no_of_neurons = len(acts.get_activation(0).vector)
        print('{} neurons found'.format(no_of_neurons))

    ########################################################################################################################
    #       Das loop
    ########################################################################################################################

    ## code to get useful info for a specific neuron


    if do_check == True:
        for i in range(len(label_dict)):
            print('i{}: row no {}'.format(i, class_dict[found_labels[i]]))

    if Test:
        no_of_neurons = 1
    else:
        # no_of_neurons = 1000
        pass

    out_filename = 'dataRerun.csv'

    out_vector = []
    # for i in range(len(cluster_list)):
    #     print('{}: {}'.format(i, max(cluster_list[i])))

    fieldnames = ['Neuron no.',  # neuron index
                  'top_class_name',
                  'all_K',  # no of K for 'All': whole of (midX to maxX) range
                  'all_No_images',  # No of images over All
                  'biggest_gap',  # Size of biggest gap: this defines the start of 'Struct' range
                  'big_gap_code',  # Coded position of gap: 0 is top cluster, counting down
                  'second_biggest_gap',  # Second biggest gap size --> could be used as struct start
                  '2_Big_gap_code',  # Gap position code
                  'top_class',  # Class with highest activation- could be a list
                  'c_0_no',  # No. images in cluster 0 (top cluster)
                  'c_0_no_class',  # No. of classes in top cluster
                  'struct_no',  # No. of images in structured region
                  'struct_K',  # No. of clusters in struct range --> this may be after a 2nd kmeans
                  'struct_no_class',  # No of classes in structured region
                  'No_top_in_cluster_0',  # No. of top class in top cluster
                  'No_top_class_in_struct',  # No. of top class in structure
                  'No_top_class_in_half_range',  # No. of top class in half range
                  'No_top_class',  # No in the top class overall
                  'pc_top_class_in_top_100',  # pc of top class in top 100
                  'is_class_selective',
                  'ccma_selectivity_top',  # ccma_selectivity to top activating class
                  'mu_max_top',  # average activation of top activating class
                  'ccma_selectivity',  # ccma_selectivity of highest mean activation class
                  'mu_max',  # mean of highest mean activatinging class
                  'mean_act_class_name',  # name of highest mean class
                  'ccma_selectivity_2',  # ccma_selectivity of 2nd highest mean activation class
                  'mu_max_2',  # mean of second highest mean activatinging class
                  'mean_act_class_name_2',  # name of highest mean class
                  'range_top',  # range of top activating class
                  'range_mean',  # range of class with highest mean activation
                  'range_2_mean',  # range of class with second highest mean activation
                  'gap_selectivity',  # sub-group selectivity on anything above the largest gap
                  'extent_of_top_class'  # number of top activations before the class changes
                  ]

    ccma_selectivity_top = 0  # ccma_selectivity to top activating class
    mu_max_top = 0
    ccma_selectivity = 0  # ccma_selectivity of highest mean activation class
    mu_max = 0  # mean of highest mean activatinging class
    top_mean_class = 0  # name of highest mean class
    ccma_selectivity_2 = 0  # ccma_selectivity of 2nd highest mean activation class
    mu_max_2 = 0  # mean of second highest mean activatinging class
    top_2_mean_class = 0  # name of highest mean class
    range_top = 0  # range of top activating class
    range_mean = 0  # range of class with highest mean activation
    range_2_mean = 0  # range of class with second highest mean activation
    gap_selectivity = 0  # sub-group selectivity on anything above the largest gap
    extent_of_top_class = 0
    normal_range = range(0, 40, 2)
    current_range = [14]  # normal_range #range(0, int(no_of_neurons/2))
    # current_range = [0, 1, 2, 4, 5, 6, 7, 8, 13, 49]
    test_range = [0,
                  11]
    normal_range_for_pictures = range(0, no_of_neurons, 100)
    range_for_pictures = [14]  # test_range + [x for x in normal_range_for_pictures]
    looking_at_output_layer = False
    do_second_Botvinick = False  # True
    do_pictures = False  # True
    do_all_points = False  # True
    # !! DO NOT MOVE THESE VARIABLES UP THE TOP IT BREAKS EVERYTHING!!
    no_of_weak_grandmas = 0  # counter for the number of neurons with a single cluster at the top of the range above the biggest gap
    no_of_sparse_neurons = 0  # counter for the number of neurosn that have less possible K points for half than the setting for k (i.e. sparse and unclustered)
    no_of_single_grandma = 0  # counter for the number of neurons that have only 1 image in top cluster
    # current_range = [0]
    sys.stdout.write('About to start with csvfile')
    sys.stdout.flush()
    with open(out_filename, 'w') as csvfile:
        # fieldnames=out_list
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        # row = ['Spam'] * 5 + ['Baked Beans']
        # writer.writerow(row)
        for current_neuron_index in current_range:
            if verbose:
                print('working on neuron {}'.format(current_neuron_index))
            # this grabs the activations as a multidimensional array
            current_neuron = acts.get_activations_for_neuron(current_neuron_index)
            # this takes the last dimension (the only non-singleton for 1-D vectors)
            # x_data = current_neuron[0][0]jitterer
            x_data = current_neuron.vector
            if cluster_by_class:
                print('hello')
                # this builds a lists of lists of points by class label
                if do_all_points:
                    # this will do a comparison of all points
                    cluster_list, min_list, max_list = build_cluster_from_class_label(acts=acts,
                                                                                      current_neuron_index=current_neuron_index,
                                                                                      label_dict=label_dict,
                                                                                      current_neuron=current_neuron,
                                                                                      found_labels=found_labels,
                                                                                      do_check=do_check)
                    # if cluster_by_kmeans == False and do_ccma_selectivity:
                    #     ccma_selectivity, mu_max, mu_not_max, max_index = \
                    #         compute_ccma_selectivity_neuron(cluster_list=cluster_list, found_labels='',
                    #                                     top_class=top_class,
                    #                                     verbose=verbose)
                    isSelective, selectivity, found_class = compute_selectivity_neuron(max_list, min_list, found_labels)
                    if looking_at_output_layer == True:
                        # as cluster_list is build from label_dict, and label_dict is in a shuffled order
                        # we must find out which class we are really on!
                        if do_true_picture:
                            # do it anyway
                            actual_class = found_labels.index(class_labels[current_neuron_index].split(' ')[0])
                            print('actual class {}'.format(actual_class))
                            if do_pictures:
                                jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                         save_label=name_leader + str(current_neuron_index) + 'cbyc.png',
                                         show_plots=False,
                                         save_plots=True,
                                         do_x_axis=True, do_y_axis=False, x_range=None, y_range=None,
                                         label_dict=label_dict,
                                         outLayerNeuron=True,
                                         current_neuron_index=actual_class)
                    else:
                        # not looking at output layer, lets take the max activation as the output class and plot that!
                        if do_true_picture:
                            # do it anyway
                            local_list, selected_activations = grab_points_for_a_cluster(current_neuron_index,
                                                                                         min_selected_x_data=np.mean(
                                                                                             current_neuron.vector),
                                                                                         max_selected_x_data=max(
                                                                                             current_neuron.vector),
                                                                                         acts=acts,
                                                                                         x_data=x_data,
                                                                                         verbose=verbose)
                            top_class_code = local_list[-1][0]
                            top_class = class_dict[local_list[-1][0]]
                            top_class_name = class_code_to_name(class_name=top_class_code, class_dict=class_dict,
                                                                class_labels=class_labels)
                            # actual_class = found_labels.index(class_labels[current_neuron_index].split(' ')[0])
                            print('maximally activated class is {}'.format(top_class_name))
                            if do_pictures:
                                jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                         save_label=name_leader + str(current_neuron_index) + 'cbycMAX.png',
                                         show_plots=False,
                                         save_plots=True,
                                         do_x_axis=True, do_y_axis=False, x_range=None, y_range=None,
                                         label_dict=label_dict,
                                         outLayerNeuron=True,
                                         current_neuron_index=top_class)
                    if isSelective:
                        # and if it is selective with all points, plot the graph
                        foundSelectivityList.append(selectivity)
                        foundClassList.append(found_class)
                        foundNeuronList.append(current_neuron_index)
                        if looking_at_output_layer == True:
                            # as cluster_list is build from label_dict, and label_dict is in a shuffled order
                            # we must find out which class we are really on!
                            actual_class = found_labels.index(class_labels[current_neuron_index].split(' ')[0])
                            print('actual class {}'.format(actual_class))
                            if do_pictures:
                                jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                         save_label=name_leader + str(current_neuron_index) + 'cbyc.png',
                                         show_plots=False,
                                         save_plots=True,
                                         do_x_axis=True, do_y_axis=False, x_range=None, y_range=None,
                                         label_dict=label_dict,
                                         outLayerNeuron=True,
                                         current_neuron_index=actual_class)
                        else:
                            if do_pictures:
                                # name_leader = 'fc6_layer_neuron'
                                jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                         save_label=name_leader + str(current_neuron_index) + '.png', show_plots=False,
                                         save_plots=True,
                                         do_x_axis=True, do_y_axis=False, x_range=None, y_range=None,
                                         label_dict=label_dict,
                                         outLayerNeuron=False,
                                         current_neuron_index=0)
                else:
                    cluster_list, min_list, max_list = build_cluster_from_class_label(acts=acts,
                                                                                      current_neuron_index=current_neuron_index,
                                                                                      label_dict=label_dict,
                                                                                      found_labels=found_labels,
                                                                                      current_neuron=current_neuron,
                                                                                      do_check=do_check,
                                                                                      no_of_points_to_check=10)
                    isSelective, selectivity, found_class = compute_selectivity_neuron(max_list, min_list, found_labels)
                    if isSelective:
                        print('Found selectivity with 10,000 points, now checking the whole thing')
                        # it is selective on 10,000 points, probably worth trying the whole thing
                        cluster_list, min_list, max_list = build_cluster_from_class_label(acts=acts,
                                                                                          current_neuron_index=current_neuron_index,
                                                                                          label_dict=label_dict,
                                                                                          found_labels=found_labels,
                                                                                          current_neuron=current_neuron,
                                                                                          do_check=do_check)
                        isSelective, selectivity, found_class = compute_selectivity_neuron(max_list, min_list,
                                                                                           found_labels)
                        if isSelective:
                            # and if it is selective with all points, plot the graph
                            foundSelectivityList.append(selectivity)
                            foundClassList.append(found_class)
                            foundNeuronList.append(current_neuron_index)
                            if looking_at_output_layer == True:
                                # as cluster_list is build from label_dict, and label_dict is in a shuffled order
                                # we must find out which class we are really on!
                                actual_class = found_labels.index(class_labels[current_neuron_index].split(' ')[0])
                                print('actual class {}'.format(actual_class))
                                if do_pictures:
                                    jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                             save_label=name_leader + str(current_neuron_index) + '.png',
                                             show_plots=False,
                                             save_plots=True,
                                             do_x_axis=True, do_y_axis=False, x_range=None, y_range=None,
                                             label_dict=label_dict,
                                             outLayerNeuron=True,
                                             current_neuron_index=actual_class)
                            else:
                                if do_pictures:
                                    # name_leader = 'fc6_layer_neuron'
                                    jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                             save_label=name_leader + str(current_neuron_index) + '.png',
                                             show_plots=False,
                                             save_plots=True,
                                             do_x_axis=True, do_y_axis=False, x_range=None, y_range=None,
                                             label_dict=label_dict,
                                             outLayerNeuron=False,
                                             current_neuron_index=0)
                    else:
                        print('Neuron {} is not selectivie with 10,000 points'.format(current_neuron_index))
            if cluster_by_kmeans:
                # this sets up a k_means object
                rangemax = max(current_neuron.vector) - min(current_neuron.vector)  # actually now the correct range!
                sorted_x_data = np.sort(x_data)
                numx = len(current_neuron.vector[current_neuron.vector > 0.5 * rangemax + min(current_neuron.vector)])
                if numx > num_of_points_for_initial_k_means:
                    # sometimes we get far too many points, let pick 1500 as a max allowed
                    print('Too many points ({}) in half-range, drop X or up K...'.format(numx))
                    # argh, some neurons go negative!!!!!
                    discard_ratio = (sorted_x_data[-num_of_points_for_initial_k_means] - min(current_neuron.vector)) \
                                    / rangemax  # max(x_data)
                    cloud = FastDetK(X=current_neuron, discard=discard_ratio)
                    print('Neuron {}, discarding {}% of data'.format(current_neuron_index, discard_ratio))
                else:
                    cloud = FastDetK(X=current_neuron, discard=0.5)
                    print('Neuron {}, discarding 50% of data range'.format(current_neuron_index))
                max_possible_K = len(set(cloud.X))
                max_activation = max(cloud.X)
                K_for_cloud = max_K_For_half
                if max_possible_K < max_K_For_half:
                    no_of_sparse_neurons = no_of_sparse_neurons + 1
                    print('sparse neuron detected! No. so far:{}'.format(no_of_sparse_neurons))
                    K_for_cloud = max_possible_K
                cloud.runFK(K_for_cloud, 1)
                current_best_K = cloud.K
                print('neuron {}: k={}'.format(current_neuron_index, current_best_K))
                cluster_list = cloud.clusters
                if do_pictures or current_neuron_index in range_for_pictures:
                    jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                             save_label=name_leader + str(current_neuron_index) + '_kmeans' + '.png', show_plots=False,
                             save_plots=True,
                             do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
                             outLayerNeuron=False,
                             current_neuron_index=current_neuron_index)
                top_cluster = cluster_list[current_best_K - 1]
                min_top_cluster = min(top_cluster)
                max_top_cluster = max(top_cluster)
                # this is just hte top cluster
                local_list, selected_activations = grab_points_for_a_cluster(current_neuron_index,
                                                                             min_selected_x_data=min_top_cluster,
                                                                             max_selected_x_data=max_top_cluster,
                                                                             acts=acts,
                                                                             x_data=x_data,
                                                                             verbose=verbose)
                if do_pictures or current_neuron_index in range_for_pictures:
                    picture = single_cluster_analysis(current_neuron_index=current_neuron_index,
                                                      min_selected_x_data=min_top_cluster,
                                                      max_selected_x_data=max_top_cluster,
                                                      acts=acts,
                                                      x_data=x_data,
                                                      name_stem='collage',
                                                      class_dict=class_dict,
                                                      class_labels=class_labels,
                                                      verbose=verbose)
                    fs_plotter(fs=cloud.fs, layer_name='prob_', current_neuron_index=current_neuron_index)
                # data for output
                c_0_no = len(local_list)
                top_class = local_list[-1][0]
                top_class_name = class_code_to_name(class_name=top_class, class_dict=class_dict,
                                                    class_labels=class_labels)
                unique_classes_0 = set([x[0] for x in local_list])
                c_0_no_classes = len(unique_classes_0)
                # now we do structural
                # this finds and analyses the gaps
                if len(cluster_list) == 0:
                    print("ERROR: we have an empty cluster list for {}?".format(current_neuron_index))
                    continue
                if current_best_K == 1:
                    # uniform distrubtion or a higher value of k needed!
                    print('Neuron {} has a K of 1, may require further investigation!'.format(current_neuron_index))
                    jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                             save_label=name_leader + str(current_neuron_index) + '_kmeans' + '.png', show_plots=False,
                             save_plots=True,
                             do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
                             outLayerNeuron=False,
                             current_neuron_index=current_neuron_index)
                    fs_plotter(fs=cloud.fs, layer_name='prob_', current_neuron_index=current_neuron_index)
                    gap_list, max_gap, max_gap_code, max_2_gap, max_2_gap_code \
                        = [], 0, 'Null', 0, 'Null'
                else:
                    gap_list, max_gap, max_gap_code, max_2_gap, max_2_gap_code \
                        = find_gaps_between_clusters(cluster_list, dict_keys=[], invert=True)
                    if len(cluster_list[current_best_K - 1]) == 1:
                        # we found a grandma cell for one picture
                        no_of_single_grandma = no_of_single_grandma + 1
                # we define the structural range as anything above the biggest gap
                if max_gap_code == 1:
                    # we've found a grandmother cell alike!
                    no_of_weak_grandmas = no_of_weak_grandmas + 1
                    print('{} weak grandmas'.format(no_of_weak_grandmas))
                    No_classes_struct = len(unique_classes_0)
                    no_of_top_class_in_struct = sum([1 for x in local_list if x[0] == top_class])
                    struct_no = c_0_no
                    struct_K = 1
                else:
                    # we got a structured layer at the top
                    # lets do another k-means
                    struct_K = max_gap_code
                    # middle of the gap between structured and unstructured
                    if not current_best_K == 1:
                        mid_struct_gap = max(cluster_list[current_best_K - max_gap_code - 1]) \
                                         + 0.5 * (min(cluster_list[current_best_K - max_gap_code]) - max(
                            cluster_list[current_best_K - max_gap_code - 1]))
                        total_range = cloud.maxX - cloud.minX
                        as_pc = mid_struct_gap / total_range
                        # numx = len(current_neuron.vector[current_neuron.vector > 0.5 * rangemax])
                        try:
                            structured = FastDetK(X=current_neuron, discard=as_pc)
                        except UserWarning as e:
                            print(
                                'That weird error where it spits on struct region and fails to find any points, sob :(')
                            continue
                        max_possible_K = len(set(structured.X))  # to catch when there are repeated values
                        chosen_max_K = min(max_K_For_Structured, max_possible_K)
                        structured.runFK(chosen_max_K)
                    else:
                        print('Trying a further k-means on Neuron {}, discarding 75% of data'.format(
                            current_neuron_index))
                        try:
                            structured = FastDetK(X=current_neuron, discard=75)
                        except UserWarning as e:
                            print(
                                'That weird error where it spits on struct region and fails to find any points, sob :(')
                            continue
                        max_possible_K = len(set(structured.X))  # to catch when there are repeated values
                        chosen_max_K = min(max_K_For_Structured, max_possible_K)
                        structured.runFK(chosen_max_K)
                        print('Updated K of {} for neuron {}'.format(structured.K, current_neuron_index))
                        gap_list, max_gap, max_gap_code, max_2_gap, max_2_gap_code \
                            = find_gaps_between_clusters(cluster_list, dict_keys=[], invert=True)
                    if do_pictures or current_neuron_index in range_for_pictures:
                        fs_plotter(fs=structured.fs, layer_name='prob_struct_',
                                   current_neuron_index=current_neuron_index)
                    if (structured.K == max_possible_K) and (not max_possible_K == 1):
                        if (structured.fs[0:max_possible_K - 1] < allowed_error).any():
                            # one of the smaller numbers of K gives us clusters to within our accuracy
                            # RE RUN K MEANS!
                            print('K found that is within error range but not optimal')
                            structured.runFK(max_possible_K - 1)
                    updated_cluster_list = {}
                    print('K of structured layer is {}'.format(structured.K))
                    K_below_struct = current_best_K - max_gap_code
                    for old_key in cluster_list.keys():
                        if old_key < K_below_struct:
                            updated_cluster_list[old_key] = cluster_list[old_key]
                    for new_key in structured.clusters.keys():
                        updated_cluster_list[new_key + K_below_struct] = structured.clusters[new_key]
                    if do_pictures or current_neuron_index in range_for_pictures:
                        jitterer(x_data=updated_cluster_list, colour_flag='cluster', title='Yo',
                                 save_label=name_leader + str(current_neuron_index) + '_heir_kmeans' + '.png',
                                 show_plots=False,
                                 save_plots=True,
                                 do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
                                 outLayerNeuron=False,
                                 current_neuron_index=current_neuron_index)
                    local_list_struct, selected_activations_struct = grab_points_for_a_cluster(current_neuron_index,
                                                                                               min_selected_x_data=min(
                                                                                                   structured.X),
                                                                                               max_selected_x_data=max_top_cluster,
                                                                                               acts=acts,
                                                                                               x_data=x_data,
                                                                                               verbose=verbose)
                    unique_classes_struct = set([x[0] for x in local_list_struct])
                    No_classes_struct = len(unique_classes_struct)
                    no_of_top_class_in_struct = sum([1 for x in local_list_struct if x[0] == top_class])
                    struct_no = structured.N
                    struct_K = structured.K
                    # TODO these stats should be a function, here you're doing the same thing 3 times
                    # did not have time ot write it properly so please fix
                # To do!
                # now we do whole (half!) range
                local_list_half, selected_activations_half = grab_points_for_a_cluster(current_neuron_index,
                                                                                       min_selected_x_data=cloud.midX,
                                                                                       max_selected_x_data=max_top_cluster,
                                                                                       acts=acts,
                                                                                       x_data=x_data,
                                                                                       verbose=verbose)
                # c_0_no = len(local_list)
                # top_class = local_list[0][0]
                unique_classes_half = set([x[0] for x in local_list_half])
                No_classes_in = len(unique_classes_half)
                no_of_top_class_in_all = sum([1 for x in local_list_half if x[0] == top_class])
                extent_of_top_class = find_extent_of_top_class(local_list=local_list_half)
                if looking_at_output_layer:
                    local_list, selected_activations = grab_points_for_a_cluster(current_neuron_index,
                                                                                 min_selected_x_data=0,
                                                                                 max_selected_x_data=max_top_cluster,
                                                                                 acts=acts,
                                                                                 x_data=x_data,
                                                                                 verbose=verbose)
                    extent_of_top_class = find_extent_of_top_class(local_list=local_list)
                local_list = local_list_half[-101:-1]
                selected_activations = selected_activations_half[-101:-1]
                no_of_top_class_in_top_100 = sum([1 for x in local_list if x[0] == top_class])
                no_of_top_class = no_files_in_label[top_class]
                if do_ccma_selectivity == True:
                    # this should be bunged up top as well, but for now...
                    # irritiatingly to compute class conditional selectivity, we need to cluster by classes sigh
                    class_cluster_list, min_list, max_list = build_cluster_from_class_label(acts=acts,
                                                                                            current_neuron_index=current_neuron_index,
                                                                                            label_dict=label_dict,
                                                                                            current_neuron=current_neuron,
                                                                                            found_labels=found_labels,
                                                                                            do_check=do_check)
                    # this computes the CCMAS for the highest mean activating class
                    ccma_selectivity, mu_max, mu_not_max, max_index = \
                        compute_ccma_selectivity_neuron(class_cluster_list, found_labels='', class_dict=class_dict,
                                                        class_labels=class_labels, top_class='', verbose=verbose)
                    [top_mean_class, top_mean_code, top_mean_label] = class_lineno_to_name(line_no=max_index,
                                                                                           class_labels=class_labels)
                    if not top_mean_code == top_class:
                        print('Class with top activations is not the class with the highest mean activation')
                        print('Top class: {}; top mean class: {}'.format(top_class_name, top_mean_class))
                        # the class with the highest activation values is not the same as the class with the highest mean activation value!
                        # so d oteh computation for the top-most class
                        ccma_selectivity_top, mu_max_top, mu_not_max_top, max_index_top = \
                            compute_ccma_selectivity_neuron(class_cluster_list, found_labels='', class_dict=class_dict,
                                                            class_labels=class_labels, top_class=top_class,
                                                            verbose=verbose)
                        if do_true_picture:
                            # do it anyway
                            egg = class_code_to_name(class_name=top_mean_class, class_dict=class_dict,
                                                     class_labels=class_labels)
                            # actual_class = found_labels.index(class_labels[current_neuron_index].split(' ')[0])
                            print('maximally activated class is {}'.format(top_class_name))
                            if do_pictures:
                                jitterer(x_data=class_cluster_list, colour_flag='cluster', title='Yo',
                                         save_label=name_leader + 'class_' + str(current_neuron_index) + 'cbycMEAN.png',
                                         show_plots=False,
                                         save_plots=True,
                                         do_x_axis=True, do_y_axis=False, x_range=None, y_range=None,
                                         label_dict=label_dict,
                                         outLayerNeuron=True,
                                         current_neuron_index=max_index)
                    else:
                        ccma_selectivity_top, mu_max_top, mu_not_max_top, max_index_top = ccma_selectivity, mu_max, mu_not_max, max_index
                    # sigh, now lets check out the second most mean activating class...
                    if do_second_Botvinick:
                        # compare the selectivity of the second most mean activating class
                        not_class_cluster_list = [class_cluster_list[i] for i in range(len(class_cluster_list)) if
                                                  not i == max_index]
                        ccma_selectivity_2, mu_max_2, mu_not_max_2, max_index_2 = \
                            compute_ccma_selectivity_neuron(not_class_cluster_list, found_labels='',
                                                            class_dict=class_dict,
                                                            class_labels=class_labels, top_class='', verbose=verbose)
                        [top_2_mean_class, _, _] = class_lineno_to_name(line_no=max_index_2, class_labels=class_labels)
                    else:
                        ccma_selectivity_2, mu_max_2, mu_not_max_2, max_index_2 = 0.0, 0.0, 0.0, max_index
                        top_2_mean_class = ''
                    # do ranges
                    range_top = max(class_cluster_list[max_index_top]) - min(class_cluster_list[max_index_top])
                    range_mean = max(class_cluster_list[max_index]) - min(class_cluster_list[max_index])
                    range_2_mean = max(class_cluster_list[max_index_2]) - min(class_cluster_list[max_index_2])
                    isClassSelective, selectivity, found_class = compute_selectivity_neuron(max_list, min_list,
                                                                                            found_labels, verbose=False)
                    # if the region above the struct were taken as a code, what would the selectivity be?
                    gap_selectivity = max_gap / max_activation
                    if isClassSelective:
                        # and if it is selective with all points, plot the graph
                        foundSelectivityList.append(selectivity)
                        foundClassList.append(found_class)
                        foundNeuronList.append(current_neuron_index)
                        if looking_at_output_layer == True:
                            # as cluster_list is build from label_dict, and label_dict is in a shuffled order
                            # we must find out which class we are really on!
                            actual_class = found_labels.index(class_labels[current_neuron_index].split(' ')[0])
                            print('actual class {}'.format(actual_class))
                            if do_pictures:
                                jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                         save_label=name_leader + str(current_neuron_index) + '.png', show_plots=False,
                                         save_plots=True,
                                         do_x_axis=True, do_y_axis=False, x_range=None, y_range=None,
                                         label_dict=label_dict,
                                         outLayerNeuron=True,
                                         current_neuron_index=actual_class)
                        else:
                            if do_pictures or current_neuron_index in range_for_pictures:
                                # name_leader = 'fc6_layer_neuron'
                                jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                         save_label=name_leader + str(current_neuron_index) + '.png', show_plots=False,
                                         save_plots=True,
                                         do_x_axis=True, do_y_axis=False, x_range=None, y_range=None,
                                         label_dict=label_dict,
                                         outLayerNeuron=False,
                                         current_neuron_index=0)
                row = {'Neuron no.': str(current_neuron_index),  # neuron index
                       'top_class_name': str(top_class_name),
                       'all_K': str(cloud.K),  # no of K for 'All': whole of (midX to maxX) range
                       'all_No_images': str(cloud.N),  # No of images over All
                       'biggest_gap': str(max_gap),  # Size of biggest gap: this defines the start of 'Struct' range
                       'big_gap_code': str(max_gap_code),  # Coded position of gap: 0 is top cluster, counting down
                       'second_biggest_gap': str(max_2_gap),
                       # Second biggest gap size --> could be used as struct start
                       '2_Big_gap_code': str(max_2_gap_code),  # Gap position code
                       'top_class': str(top_class),  # Class with highest activation- could be a list
                       'c_0_no': str(c_0_no),  # No. images in cluster 0 (top cluster)
                       'c_0_no_class': str(c_0_no_classes),  # No. of classes in top cluster
                       'struct_no': str(struct_no),  # No. of images in structured region
                       'struct_K': str(struct_K),  # No. of clusters in struct range --> this may be after a 2nd kmeans
                       'struct_no_class': str(No_classes_struct),  # No of classes in structured region
                       'No_top_in_cluster_0': str(c_0_no_classes),  # No. of top class in top cluster
                       'No_top_class_in_struct': str(no_of_top_class_in_struct),  # No. of top class in structure
                       'No_top_class_in_half_range': str(no_of_top_class_in_all),  # No. of top class in half range
                       'No_top_class': str(no_of_top_class),  # No in the top class overall
                       'pc_top_class_in_top_100': str(no_of_top_class_in_top_100),  # pc of top class in top 100
                       'is_class_selective': isClassSelective,
                       'ccma_selectivity_top': str(ccma_selectivity_top),  # ccma_selectivity to top activating class
                       'mu_max_top': str(mu_max_top),  # average activation of top activating class
                       'ccma_selectivity': str(ccma_selectivity),  # ccma_selectivity of highest mean activation class
                       'mu_max': str(mu_max),  # mean of highest mean activatinging class
                       'mean_act_class_name': str(top_mean_class),  # name of highest mean class
                       'ccma_selectivity_2': str(ccma_selectivity_2),
                       # ccma_selectivity of 2nd highest mean activation class
                       'mu_max_2': str(mu_max_2),  # mean of second highest mean activatinging class
                       'mean_act_class_name_2': str(top_2_mean_class),  # name of highest mean class
                       'range_top': str(range_top),  # range of top activating class
                       'range_mean': str(range_mean),  # range of class with highest mean activation
                       'range_2_mean': str(range_2_mean),  # range of class with second highest mean activation
                       'gap_selectivity': str(gap_selectivity),
                       # sub-group selectivity on anything above the largest gap
                       'extent_of_top_class': str(extent_of_top_class)
                       # number of top activations before the class changes
                       }
                sorted_row = OrderedDict(sorted(row.items(), key=lambda item: fieldnames.index(item[0])))
                writer.writerow(sorted_row)

    print('Neuron, Selectivity, Class')
    for i in range(len(foundSelectivityList)):
        print('{0},{1},{2}'.format(foundNeuronList[i], foundSelectivityList[i], foundClassList[i]))

    # tiger_shark_or_toilet_roll(label_no=1, label_dict = label_dict, point_no = [1,2,7], acts = acts, found_labels = found_labels,
    #                             print_value=True, class_labels=class_labels)
    if do_check:
        for class_key in range(len(label_dict.keys())):
            # check some stuff
            # This gives hte classification for the first member of each class
            tiger_shark_or_toilet_roll(label_no=class_key, label_dict=label_dict, point_no=0, acts=acts,
                                       found_labels=found_labels,
                                       print_value=True, class_labels=class_labels)




if __name__ == '__main__':
    main()
