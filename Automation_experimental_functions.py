######
# file of functions used by the experiments!
from h5_analysis_jitterer import filename_to_label
from h5_analysis_jitterer import build_cluster_from_class_label, jitterer
import sys
from collections import Counter

import numpy as np

from Make_activation import combine_h5_files_in_activation_table
import kmeans


def print_test_acts(acts):
    """
    :param acts:
    prints out example data
    :return:
    """
    print('{} files in table'.format(len(acts.get_all_point_indices())))
    egg = acts.get_all_point_indices()[0]
    point = acts.get_activation(egg)
    print('Example file: {}, vectors are {}-dimensional'.format(point, len(point.vector)))
    print('Example labels: {}'.format(point.labels))
    return point


def make_synset_files(index_into_index_file, index_file, category_file, out_file_name):
    """Function to get Broden into the correct formats
    index_into_index_file is which column of index file to take
    index_file is the files with the images and associated categories
    category file is the file c_oject,txt etc with the human readable labels"""""
    ## Get a list of images which have associated objects and we'll take those as classes
    image_list = []
    correct_class_list = []
    file = open(index_file, 'r')
    for line in file:
        line_list = line.strip().split(',')
        print(line_list)
        if not line_list[index_into_index_file] == '':
            image_list.append([line_list[0], line_list[index_into_index_file]])
            # this the original image name and an image which masks the object
    # gets the human readable lbels from the file
    labels_list = []
    file2 = open(category_file, 'r')
    for line in file2:
        line_list = line.strip().split(',')
        print(line_list)
        if not line_list[1] == 'number':
            labels_list.append([line_list[1], line_list[2]])
    # write it out in the correct format
    with open(out_file_name, "w") as fh:
        for line in labels_list:
            fh.writelines(' '.join(line) + '\n')
    fh.close()
    return image_list, labels_list


def build_label_dict(acts, use_loaded_files=True, verbose=True, doMean=False):
    """
    Builds a dictionary of labels to points in local format
    Gives out a dictionary and a list of found labels
    acts: activation table object
    use_loaded_files: whether to assume one file per label (current default)
    This version can deal with the filenames and labels being different
    """
    sys.stdout.write('About to build the label dict (slow)')
    sys.stdout.flush()
    if use_loaded_files == True:
        # we use the filenames as the labels
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
    else:
        # we assume acts already has the labels
        files = acts.get_loaded_files()
        big_list = []
        no_of_files = len(files)
        found_labels = []
        big_dict = {}
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
            # assigned_label = filename_to_label(assigned_label)
            # except AttributeError:
            #     assigned_label = acts.get_activation(current_point).labels.decode('UTF-8')
            # except ValueError:
            #     import pdb; pdb.set_trace()
            big_dict[current_point] = acts.get_activation(current_point).labels[0].decode('UTF-8')
        # we've got all the points
        if verbose:
            print('Found label: \t No. of points')
        unique_values = set(val for dic in big_dict for val in big_dict.values())
        found_labels = [x for x in unique_values]
        for f_label in found_labels:
            list_of_tuples = [x[0] for x in big_dict.items() if x[1] == f_label]
            label_dict[f_label] = list_of_tuples
            # for f_no, file_name in enumerate(files):
            #     label=filename_to_label(file_name.split('_')[0])
            #     if assigned_label == label:
            #         big_list[f_no].append(current_point)
            #         break
            if verbose:
                print('{}: \t {}'.format(f_label, len(list_of_tuples)))
        sys.stdout.write('Built the label dict')
        sys.stdout.flush()
    return label_dict, found_labels, no_files_in_label


def do_jitter_plot_test(acts, test_class_index=0, current_neuron_index=0, label_dict='', found_labels='',
                        no_files_in_label='', name_leader=''):
    """Little function to plot a jitterplot and return assigned labesl
    If your labels are correct, this is useful to doing the selectivity,
    if not, do one of these to double check there are no odd patterns in the activataions! """
    if label_dict == '':
        # build it
        sys.stdout.write('About to build the label dict (slow)')
        sys.stdout.flush()
        label_dict, found_labels, no_files_in_label = build_label_dict(acts, use_loaded_files=False, verbose=True,
                                                                       doMean=False)
        sys.stdout.write('Built the label dict')
        sys.stdout.flush()
    current_neuron = acts.get_activations_for_neuron(current_neuron_index)
    cluster_list, min_list, max_list = build_cluster_from_class_label(acts=acts,
                                                                      current_neuron_index=current_neuron_index,
                                                                      label_dict=label_dict,
                                                                      found_labels=found_labels,
                                                                      current_neuron=current_neuron,
                                                                      do_check='')
    jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
             save_label=name_leader + 'test_neuron' + str(current_neuron_index) + 'class' + str(
                 test_class_index) + 'cbyc.png', show_plots=False,
             save_plots=True,
             do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
             outLayerNeuron=True,
             current_neuron_index=test_class_index)
    return label_dict, found_labels, no_files_in_label


def get_class_details_broden(classname, labels, short_labels):
    """grabs cat number for a unit
    classname is the word name of the class
    labels is our list"""
    class_name = [x for x in labels if x.split(' ')[1] == classname][0]
    class_name = class_name.split(' ')[0]
    class_pos = short_labels.index(class_name)
    return class_name, class_pos


# def guess_labels():

def guess_n_check_labels(acts, activation_indice_list, correct_class_list, verbose=True):
    """Function to guess labels for known data and compare it to output
    Bascially this reads the guesses from your activation table
    acts is activation table of activations should be test activation type
    activation_indice_list is a list of activations
    correct_class_list is a list of the correct classes for those points
    make sure ccl is decoded byte strings
    do a list of len 1 if they are all the same
    do a empty list of the label is attached to the point"""
    # Test, check that it can correctly assign daisies
    print("Guessing labels")
    # acts.guess_all()
    if correct_class_list == '':
        # assume correct class list is label
        pass  # we deal with it below
    top_1_count = 0
    top_5_count = 0

    for act_index in activation_indice_list:
        # now use the new code to attempt to guess the labels
        # fisrt make it guess
        # But we want the inner details
        test_point = acts.get_test_activation(act_index, True)
        if correct_class_list == '':
            correct_class_list = [test_point.label]  # check this!!!!
        elif len(correct_class_list) == 1:
            # assume all points are the same class
            true_class = correct_class_list[0]
        else:
            # assume its a list of classes for each point
            true_class = correct_class_list[act_index]
        if verbose:
            print("{}: {}".format(test_point.index, test_point.guesses))
        if len(test_point.guesses) > 0 and test_point.guesses[0][0].decode() == true_class:
            top_1_count += 1
        if true_class in [x[0].decode() for x in test_point.guesses]:
            top_5_count += 1

    print('Top 1 correct = {}'.format(top_1_count))
    print('Top 5 correct = {}'.format(top_5_count))
    pc1 = 100 * top_1_count / len(activation_indice_list)
    pc5 = 100 * top_5_count / len(activation_indice_list)
    print('Top 1 correct % = {}'.format(pc1))
    print('Top 5 correct % = {}'.format(pc5))
    return top_1_count, top_5_count, pc1, pc5


def get_image_list_from_file(correct_class_filename):
    """Little function to read in images and correct classes fromr a file
    correct_class_filename = the file to read"""
    image_list = []
    correct_class_list = []
    file = open(correct_class_filename, 'r')
    for line in file:
        line_list = line.strip().split(' ')
        image_list.append(line_list[0])
        if len(line_list) == 2:
            # we have been given classes
            correct_class_list.append(int(line_list[1]))
            check_classes = True
        else:
            # no classes :(
            check_classes = False
    return image_list, correct_class_list, check_classes


def simple_stats_on_vector(input_vector, verbose=True):
    """Simple stats on cluster of activations"""
    egg = Counter(input_vector)
    modal_activation = egg.most_common()
    modal_activation, count = modal_activation[0][0], modal_activation[0][1]
    mean_activation = np.mean(input_vector)
    median_activation = np.median(input_vector)
    ste = np.std(input_vector) / np.sqrt(len(input_vector))
    if verbose:
        print('Mode: {}, count {}, Mean {} +/- {}, Median {}'.format(
            modal_activation, count, mean_activation, ste, median_activation))
    return modal_activation, count, mean_activation, ste, median_activation


def make_synset_files(index_into_index_file, index_file, category_file, out_file_name):
    """Function to get Broden into the correct formats
    index_into_index_file is which column of index file to take
    index_file is the files with the images and associated categories
    category file is the file c_oject,txt etc with the human readable labels"""""
    ## Get a list of images which have associated objects and we'll take those as classes
    image_list = []
    correct_class_list = []
    file = open(index_file, 'r')
    for line in file:
        line_list = line.strip().split(',')
        print(line_list)
        if not line_list[index_into_index_file] == '':
            image_list.append([line_list[0], line_list[index_into_index_file]])
            # this the original image name and an image which masks the object
    # gets the human readable lbels from the file
    labels_list = []
    file2 = open(category_file, 'r')
    for line in file2:
        line_list = line.strip().split(',')
        print(line_list)
        if not line_list[1] == 'number':
            labels_list.append([line_list[1], line_list[2]])
    # write it out in the correct format
    with open(out_file_name, "w") as fh:
        for line in labels_list:
            fh.writelines(' '.join(line) + '\n')
    fh.close()
    return image_list, labels_list


def build_h5_files(r, name_leader='', suffix='', input_filenames='*.h5'):
    """this will make your .h5 files and merge them
    r is a caffe_settings object"""
    import Caffe_AlexNet2 as C
    C.main(r)  # this makes the h5 files
    from merger import merge_layer
    merge_layer(directory=r.image_directory,
                layer_name=r.blob,
                name_leader=name_leader,
                suffix=suffix,
                input_filenames=input_filenames)
    return


def make_acts(r,
              use_merged_h5_file=True,
              use_normalised_h5_file=False,
              h5_files_directory='',
              h5_list_filename='',
              h5_list=[]):
    """Makes an activation table using ActivationTable"""
    if use_merged_h5_file == True:
        acts = kmeans.test_activation_table.ActivationTable(mean=False)
        acts.add_merged_file(r.file_root + r.this_one_file_name)
    elif use_normalised_h5_file:
        acts = kmeans.test_activation_table.ActivationTable(mean=False)
        acts.add_normalised_file(r.file_root + r.this_one_file_name)
    else:
        acts, h5_list = combine_h5_files_in_activation_table(
            h5_file_location=h5_files_directory,
            h5_list_filename=h5_list_filename,
            h5_list=h5_list,
            useFile=True,
            verbose=True)
    print_test_acts(acts)
    return acts


def make_test_acts(r, use_merged_h5_file=True,
                   use_normalised_h5_file=False,
                   h5_files_directory='', h5_list_filename='', h5_list=[]):
    """Makes an activation table using TestActivationTable"""
    if use_merged_h5_file == True:
        acts = kmeans.test_activation_table.TestActivationTable(mean=False)
        acts.add_merged_file(r.file_root + r.this_one_file_name)
    elif use_normalised_h5_file:
        acts = kmeans.test_activation_table.TestActivationTable(mean=False)
        acts.add_normalised_file(r.file_root + r.this_one_file_name)
    else:
        acts, h5_list = combine_h5_files_in_activation_table(
            h5_file_location=h5_files_directory,
            h5_list_filename=h5_list_filename,
            h5_list=h5_list,
            useFile=True,
            verbose=True)
    print_test_acts(acts)
    return acts


def set_up_acts(r, h5_files_directory, h5_file_name, use_merged_h5_file=True):
    """Warpper function to set up the activation table for you"""
    # make acts & test it
    if use_merged_h5_file:
        # this uses the merged file
        acts = make_acts(
            r=r,
            use_merged_h5_file=True,
            h5_files_directory='',
            h5_list_filename='',
            h5_list=[])
    else:
        acts = make_acts(
            r='',
            use_merged_h5_file=False,
            h5_files_directory=h5_files_directory,
            h5_list_filename=h5_file_name,
            h5_list=[])
    return acts


def IM_class_indices(chosen_classes, acts, h5_tail='_fc8_max.h5', use_merged_h5_file=True, ):
    """Wrapper function to grab sets of classes from acts
    chosen classes is a list of hte imagenet classes in n0090909 format
    h5_tail is the end of hte .h5 files for non merged files"""
    out = []
    if use_merged_h5_file:
        for chosen_class in chosen_classes:
            out.append(
                [(f, i) for (f, i) in acts.get_all_point_indices() if f == chosen_class])
    else:
        for chosen_class in chosen_classes:
            out.append(
                [(f, i) for (f, i) in acts.get_all_point_indices() if f == [chosen_class + h5_tail]])
    return out
