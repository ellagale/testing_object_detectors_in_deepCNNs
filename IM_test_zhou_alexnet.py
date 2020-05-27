## script to get data from Zhou's version of AlexNet using imagenet input data
## thingys we need
import csv
import os
import sys

from collections import OrderedDict

import Automation_experimental_functions as A
from Make_activation import combine_h5_files_in_activation_table
from h5_analysis_jitterer import class_code_to_name, filename_to_label, fs_plotter, find_gaps_between_clusters, \
    allowed_error, class_lineno_to_name
from kmeans.fast_detk import FastDetK
from set_up_caffe_net import set_up_caffe, CaffeSettings, Caffe_NN_setup
from h5_analysis_jitterer import build_cluster_from_class_label, jitterer_list
from h5_analysis_jitterer import make_class_to_line_number_look_up_table
from h5_analysis_jitterer import build_label_dict_from_filenames
from precision_calulator import calculate_many_precs_recall_stats,\
    get_local_list_for_neuron, find_zhou_precision
verbose=True

########################################################################################################################
# internal settings
########################################################################################################################
outfilename = 'results.csv'
# this sets up our default settings, I'm doing this in place of changing set_up_caffe_net
r = CaffeSettings()
#dir_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/' # root file with image data in it
#data_set_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/' #
#r.image_directory = data_set_root + 'images/' # directory with test images
###### use this if on kraken
dir_root = '/storage/data/imagenet_2012_227/'
###### use this on your computer
#dir_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/'
data_set_root = os.path.join(dir_root)
r.image_directory = os.path.join(data_set_root) # directory with test images
do_by_hand= True # do not change this
r.model_file ='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/zoo/caffe_reference_places205.caffemodel'
r.deploy_file = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/zoo/caffe_reference_places205.prototxt'
#r.model_file = os.path.join(dir_root, 'zoo/caffe_reference_places365.caffemodel')
# model_file points to the model we want to use
#r.file_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1' # file root uh...? same as dir root?
r.file_root = dir_root
r.labels_file = os.path.join('/storage/data/ilsvrc12/', 'synset_words.txt') # file with the class labels in it  should be ''class code' 'name'\n'
r.dir_list = os.path.join('/storage/data/imagenet_2012_class_list.txt') # the list of directoires which contain our images
#r.labels_file = os.path.join(data_set_root, 'synset_objects227.txt')
#remake_synset_files = True # thsi makes your labels file
r.blob = 'conv5'
r.blob_list =['conv5']
r.output_directory = os.getcwd()
h5_files_directory = os.path.join('/storage/data/Unmerged', 'Merged100ActsOnly_conv5all')
r.do_limited = False
do_h5_files = False
max_K_For_Structured = 50

########################################################################################################################
# functions
########################################################################################################################

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


def build_label_dict(acts, use_loaded_files = True, verbose=True, doMean=False):
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
        files=acts.get_loaded_files()
        big_list=[]
        no_of_files = len(files)
        found_labels=[]
        label_dict = {}
        no_files_in_label = {}
        for file_name in files:
            big_list.append([])
            label = filename_to_label(file_name.split('_')[0])
            found_labels.append(label)
        if verbose:
            print('Found {} files in activation table object'.format(no_of_files))
            #print('Be patient, I found {} points'.format(len(acts.get_all_activation_indices())))
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
                label=filename_to_label(file_name.split('_')[0])
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
            #print(i, found_labels[i])
            label_dict[found_labels[i]] = big_list[i]
            no_files_in_label[found_labels[i]] = len(big_list[i])
    else:
        # we assume acts already has the labels
        files=acts.get_loaded_files()
        big_list=[]
        no_of_files = len(files)
        found_labels=[]
        big_dict={}
        label_dict = {}
        no_files_in_label = {}
        for file_name in files:
            big_list.append([])
            label = filename_to_label(file_name.split('_')[0])
            found_labels.append(label)
        if verbose:
            print('Found {} files in activation table object'.format(no_of_files))
            #print('Be patient, I found {} points'.format(len(acts.get_all_activation_indices())))
        for current_point in acts.get_all_point_indices():
            # TODO:: Make this work with multiple labels
            if isinstance(acts.get_activation(current_point).labels, (bytes, bytearray, str)):
                    # old style, the labels are a numpy byte string
                assigned_label = acts.get_activation(current_point).labels.decode('UTF-8')
            else:
                # new style, labels are a list
                assigned_label = acts.get_activation(current_point).labels[0].decode('UTF-8')
            #assigned_label = filename_to_label(assigned_label)
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





# here we set up where things should be found by caffe
r.caffe_root, r.image_directory, r.labels_file, r.model_def, r.model_weights, r.dir_list, r.labels = \
            set_up_caffe(image_directory=r.image_directory,
                         model_file=r.model_file,
                         #label_file_address='/storage/data/fooling_images_2015/synset_words.txt',
                         label_file_address=r.labels_file,
                         dir_file=r.dir_list,
                         root_dir=os.environ['CAFFE_ROOT'], verbose=True,
                         deploy_file=r.deploy_file)



#here we build our neural net
# NTS i am not entrely sure that the mean image is correct, I may have to make a new one
r.net, r.transformer = Caffe_NN_setup(
    imangenet_mean_image=os.path.join(os.environ['PYCAFFE_ROOT'], 'caffe/imagenet/ilsvrc_2012_mean.npy'),
    batch_size=50, model_def=r.model_def, model_weights=r.model_weights,
    verbose=True, root_dir=r.caffe_root)



r.short_labels = [label.split(' ')[0] for label in r.labels]

print('Make_activation')
global acts, class_labels, h5_list, caffe_settings, dir_list

# this creates a caffe_seetings object and sets the values, normally this is done in set_up_caffe_net
caffe_settings = r
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
output_directory = caffe_settings.output_directory



if do_h5_files:
    import Caffe_AlexNet as C
    C.main(r) # this makes the h5 files
    from merger import merge_layer
    merge_layer(os.getcwd(), 'conv5', '100ActsOnly', 'all')


# mergind doesn't seem to work...

# import Make_activation as m
this_one_file_name = 'Merged100ActsOnly_conv5all.h5'
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# print('!!Over-riding set_up_caffenet settings, am assuming clusterflow is an output layer!!')
# print('I am using the following merged h5 file: {}'.format(this_one_file_name))
# print('Which I expect to be located at: {}'.format(file_root))
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# option = 'merged' #'doAllClasses'#'merged'#doAllClasses'#'merged' # 'merged'#''doFewClasses'#'doTwoClasses'#'doFewClasses'#'doAllClasses'


acts, h5_list = combine_h5_files_in_activation_table(h5_file_location=h5_files_directory,
                                                             h5_list_filename='h5_conv5_list.txt', h5_list=[], useFile=True,
                                                             verbose=True)


# silly little test

print_test_acts(acts)





class_dict = make_class_to_line_number_look_up_table(class_labels=class_labels, verbose=False)


#######

# this builds the look-up table between points and the class they are in
## This bit is sslow, it loads the label data for all acts


sys.stdout.write('About to build the label dict (slow)')
sys.stdout.flush()
assign_labels_from_filename=True
assign_labels_from_NN = False
if assign_labels_from_NN:
    label_dict, found_labels, no_files_in_label = build_label_dict(acts, use_loaded_files = False, verbose=True, doMean=False)
elif assign_labels_from_filename:
    label_dict, found_labels, no_files_in_label = build_label_dict_from_filenames(acts)

sys.stdout.write('Built the label dict')
sys.stdout.flush()

no_of_images = acts.get_image_count()
print('Found {} images'.format(no_of_images))

#


current_neuron_index = 87
current_neuron = acts.get_activations_for_neuron(current_neuron_index)
        # this takes the last dimension (the only non-singleton for 1-D vectors)
        # x_data = current_neuron[0][0]
x_data = current_neuron.vector
cluster_list, min_list, max_list = build_cluster_from_class_label(acts=acts,
                                        current_neuron_index=current_neuron_index,
                                        label_dict=label_dict,
                                        found_labels=found_labels,
                                        current_neuron=current_neuron,
                                        do_check='')



# car_list = n02930766
car_indices =[found_labels.index(x) for x in ['n02930766',
                                              'n04037443',
                                              'n03594945',
                                              'n03770679',
                                              'n02814533',
                                              'n03670208',
                                              'n03777568',
                                              'n03100240',
                                              'n03459775',
                                              'n04285008',
                                              'n04461696',
                                              'n02965783',
                                              'n02974003']]




bus_indices = [found_labels.index(x) for x in ['n04146614','n04487081','n03769881']]



jitterer_list(x_data=cluster_list,
              colour_flag='multi',
              title='',
              save_label='AlexNetPlaces205uv2',
              show_plots=False,
              save_plots=True,
              do_x_axis=True,
              do_y_axis=False,
              x_range=None,
              y_range=None,
              label_dict={},
              outLayerNeuron=True,
              current_neuron_indices=car_indices)

print('Car classes are:')
for x in car_indices:
    print('{}'.format(r.labels[x]))




exit(1)
#out_filename='data.csv'

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
current_range = [79]  # normal_range #range(0, int(no_of_neurons/2))
# current_range = [0, 1, 2, 4, 5, 6, 7, 8, 13, 49]
test_range = [79]  # ,12,3,4,5,6,7,8,9,10,1,2, 31, 68, 69, 682, 1505, 1915, 2025, 2269, 2345, 2474, 2635, 2854, 3329, 3366, 4075, 3092,
# 60, 86, 126, 131, 319, 403, 479, 527, 660, 696, 757, 830, 967, 1123, 1239, 1241, 1359,
# 1381, 1546, 1677, 1730, 1791, 1802, 1843, 1876, 1986, 2001, 2241, 2327, 2397, 2508, 2848,
# 2963, 3012, 3068, 3109, 3297, 3312, 3409, 323,393,3439, 3461, 3496, 3591, 3753, 3845, 3888, 3898, 3903,94,603,656,78,365]
# current_range = range(0,n) #test_range
#normal_range_for_pictures = range(0, no_of_neurons, 100)
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

from h5_analysis_jitterer import build_cluster_from_class_label, jitterer


class_name, class_pos = A.get_class_details_broden('car', labels, short_labels)

current_neuron_index = 87
current_neuron = acts.get_activations_for_neuron(current_neuron_index)
        # this takes the last dimension (the only non-singleton for 1-D vectors)
        # x_data = current_neuron[0][0]
x_data = current_neuron.vector
cluster_list, min_list, max_list = build_cluster_from_class_label(acts=acts,
                                        current_neuron_index=current_neuron_index,
                                        label_dict=label_dict,
                                        found_labels=found_labels,
                                        current_neuron=current_neuron,
                                        do_check='')


jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                     save_label='test' + str(current_neuron_index) + 'cbyc.png', show_plots=False,
                                     save_plots=True,
                                     do_x_axis=True, do_y_axis=False,
                                     x_range=None, y_range=None, label_dict=label_dict,
                                     outLayerNeuron=True,
                                     current_neuron_index=661)

pos_in_cluster_list = found_labels.index(class_name)
test_class_activations = cluster_list[pos_in_cluster_list]


exit(1)
A.simple_stats_on_vector(test_class_activations, verbose=True)



verbose = True
from h5_analysis_jitterer import compute_ccma_selectivity_neuron
import numpy as np

ccma_selectivity, mu_max, mu_not_max, max_index = \
                    compute_ccma_selectivity_neuron(cluster_list,
                                                    found_labels=found_labels,
                                                    class_dict=class_dict,
                                                    class_labels=class_labels,
                                                    top_class='38', verbose=verbose)

ccma_selectivity_top, mu_max, mu_not_max, max_index = \
                    compute_ccma_selectivity_neuron(cluster_list,
                                                    found_labels=found_labels,
                                                    class_dict=class_dict,
                                                    class_labels=class_labels,
                                                    top_class='', verbose=verbose)

local_list, selected_activations, x_data = \
    get_local_list_for_neuron(current_neuron_index=current_neuron_index,
                                    minx=90,
                                    maxx=140,
                                    acts=acts)

from h5_analysis_jitterer import make_collage

make_collage(out_file='Anconv5u79Top74acts.jpg', local_list=local_list,
             shrink=False, do_square=False, no_of_cols=5,
             acts=acts, class_dict=class_dict, class_labels=class_labels,
             verbose=verbose, imagenet_root=r.image_directory)

######################


local_list, selected_activations, x_data = \
    get_local_list_for_neuron(current_neuron_index=current_neuron_index,
                                    minx='',
                                    maxx='',
                                    acts=acts)
zhou_precs_class60, zhou_precs60, zhou_no_of_classes60, zhou60 = find_zhou_precision(
                            number_of_points=60, local_list=local_list)

calculate_many_precs_recall_stats(test_class='38',
                                local_list=local_list,
                                Q_stop='',
                                no_files_in_label=no_files_in_label,
                                no_of_images=no_of_images,
                                verbose=verbose)


############# it works to here

do_true_picture=False
name_leader='ANPlaces365'
cluster_by_kmeans=False
do_ccma_selectivity = True
do_second_Botvinick = False
out_filename = 'ANPlaces365.csv'
verbose=True
do_check = False
do_all_points = True
isSelective = False
num_of_points_for_initial_k_means=5000
max_K_For_half = 500



from h5_analysis_jitterer import cluster_by_class, compute_selectivity_neuron, grab_points_for_a_cluster, \
    single_cluster_analysis, find_extent_of_top_class, compute_ccma_selectivity_neuron

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
        # x_data = current_neuron[0][0]
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
                #isSelective, selectivity, found_class = compute_selectivity_neuron(max_list, min_list, found_labels)
                if looking_at_output_layer == True:
                    # as cluster_list is build from label_dict, and label_dict is in a shuffled order
                    # we must find out which class we are really on!
                    if do_true_picture:
                        # do it anyway
                        actual_class = found_labels.index(class_labels[current_neuron_index].split(' ')[0])
                        print('actual class {}'.format(actual_class))
                        if do_pictures:
                            jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                     save_label=name_leader + str(current_neuron_index) + 'cbyc.png', show_plots=False,
                                     save_plots=True,
                                     do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
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
                # if isSelective:
                #     # and if it is selective with all points, plot the graph
                #     #foundSelectivityList.append(selectivity)
                #     #foundClassList.append(found_class)
                #     #foundNeuronList.append(current_neuron_index)
                #     if looking_at_output_layer == True:
                #         # as cluster_list is build from label_dict, and label_dict is in a shuffled order
                #         # we must find out which class we are really on!
                #         actual_class = found_labels.index(class_labels[current_neuron_index].split(' ')[0])
                #         print('actual class {}'.format(actual_class))
                #         if do_pictures:
                #             jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                #                      save_label=name_leader + str(current_neuron_index) + 'cbyc.png', show_plots=False,
                #                      save_plots=True,
                #                      do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
                #                      outLayerNeuron=True,
                #                      current_neuron_index=actual_class)
                #     else:
                #         if do_pictures:
                #             # name_leader = 'fc6_layer_neuron'
                #             jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                #                      save_label=name_leader + str(current_neuron_index) + '.png', show_plots=False,
                #                      save_plots=True,
                #                      do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
                #                      outLayerNeuron=False,
                #                      current_neuron_index=0)
            else:
                #cluster_list, min_list, max_list = build_cluster_from_class_label(acts=acts,
                #                                                                  current_neuron_index=current_neuron_index,
                #                                                                  label_dict=label_dict,
                #                                                                  found_labels=found_labels,
                #                                                                  current_neuron=current_neuron,
                #                                                                  do_check=do_check,
                #                                                                  no_of_points_to_check=10)
                #isSelective, selectivity, found_class = compute_selectivity_neuron(max_list, min_list, found_labels)
                if isSelective:
                    print('Found selectivity with 10,000 points, now checking the whole thing')
                    # it is selective on 10,000 points, probably worth trying the whole thing
                    cluster_list, min_list, max_list = build_cluster_from_class_label(acts=acts,
                                                                                      current_neuron_index=current_neuron_index,
                                                                                      label_dict=label_dict,
                                                                                      found_labels=found_labels,
                                                                                      current_neuron=current_neuron,
                                                                                      do_check=do_check)
                    isSelective, selectivity, found_class = compute_selectivity_neuron(max_list, min_list, found_labels)
                    if isSelective:
                        # and if it is selective with all points, plot the graph
                        #foundSelectivityList.append(selectivity)
                        #foundClassList.append(found_class)
                        #foundNeuronList.append(current_neuron_index)
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
            top_class_name = class_code_to_name(class_name=top_class, class_dict=class_dict, class_labels=class_labels)
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
                        print('That weird error where it spits on struct region and fails to find any points, sob :(')
                        continue
                    max_possible_K = len(set(structured.X))  # to catch when there are repeated values
                    chosen_max_K = min(max_K_For_Structured, max_possible_K)
                    structured.runFK(chosen_max_K)
                else:
                    print('Trying a further k-means on Neuron {}, discarding 75% of data'.format(current_neuron_index))
                    try:
                        structured = FastDetK(X=current_neuron, discard=75)
                    except UserWarning as e:
                        print('That weird error where it spits on struct region and fails to find any points, sob :(')
                        continue
                    max_possible_K = len(set(structured.X))  # to catch when there are repeated values
                    chosen_max_K = min(max_K_For_Structured, max_possible_K)
                    structured.runFK(chosen_max_K)
                    print('Updated K of {} for neuron {}'.format(structured.K, current_neuron_index))
                    gap_list, max_gap, max_gap_code, max_2_gap, max_2_gap_code \
                        = find_gaps_between_clusters(cluster_list, dict_keys=[], invert=True)
                if do_pictures or current_neuron_index in range_for_pictures:
                    fs_plotter(fs=structured.fs, layer_name='prob_struct_', current_neuron_index=current_neuron_index)
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
                                                        class_labels=class_labels, top_class=top_class, verbose=verbose)
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
                        compute_ccma_selectivity_neuron(not_class_cluster_list, found_labels='', class_dict=class_dict,
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
                    #foundSelectivityList.append(selectivity)
                    #foundClassList.append(found_class)
                    #foundNeuronList.append(current_neuron_index)
                    if looking_at_output_layer == True:
                        # as cluster_list is build from label_dict, and label_dict is in a shuffled order
                        # we must find out which class we are really on!
                        actual_class = found_labels.index(class_labels[current_neuron_index].split(' ')[0])
                        print('actual class {}'.format(actual_class))
                        if do_pictures:
                            jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                     save_label=name_leader + str(current_neuron_index) + '.png', show_plots=False,
                                     save_plots=True,
                                     do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
                                     outLayerNeuron=True,
                                     current_neuron_index=actual_class)
                    else:
                        if do_pictures or current_neuron_index in range_for_pictures:
                            # name_leader = 'fc6_layer_neuron'
                            jitterer(x_data=cluster_list, colour_flag='cluster', title='Yo',
                                     save_label=name_leader + str(current_neuron_index) + '.png', show_plots=False,
                                     save_plots=True,
                                     do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
                                     outLayerNeuron=False,
                                     current_neuron_index=0)
            row = {'Neuron no.': str(current_neuron_index),  # neuron index
                   'top_class_name': str(top_class_name),
                   'all_K': str(cloud.K),  # no of K for 'All': whole of (midX to maxX) range
                   'all_No_images': str(cloud.N),  # No of images over All
                   'biggest_gap': str(max_gap),  # Size of biggest gap: this defines the start of 'Struct' range
                   'big_gap_code': str(max_gap_code),  # Coded position of gap: 0 is top cluster, counting down
                   'second_biggest_gap': str(max_2_gap),  # Second biggest gap size --> could be used as struct start
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
                   'gap_selectivity': str(gap_selectivity),  # sub-group selectivity on anything above the largest gap
                   'extent_of_top_class': str(extent_of_top_class)  # number of top activations before the class changes
                   }
            sorted_row = OrderedDict(sorted(row.items(), key=lambda item: fieldnames.index(item[0])))
            writer.writerow(sorted_row)



