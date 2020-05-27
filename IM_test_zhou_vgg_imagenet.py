## script to get data from Zhou's version of AlexNet using imagenet input data
## thingys we need
import csv
import os
import sys
import itertools
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

import kmeans
from Automation_experimental_functions import print_test_acts, build_label_dict
from Make_activation import combine_h5_files_in_activation_table
from h5_analysis_jitterer import class_code_to_name, make_class_to_line_number_look_up_table
from set_up_caffe_net import set_up_caffe, CaffeSettings, Caffe_NN_setup
from h5_analysis_jitterer import build_cluster_from_class_label, jitterer_list
from h5_analysis_jitterer import make_class_to_line_number_look_up_table
from h5_analysis_jitterer import build_label_dict_from_filenames
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
dir_root = '/storage/data/imagenet_2012_224/'
###### use this on your computer
#dir_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/'
data_set_root = os.path.join(dir_root)
r.image_directory = os.path.join(data_set_root) # directory with test images
do_by_hand= True # do not change this
r.model_file ='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/zoo/vgg16_imagenet.caffemodel'
r.deploy_file = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/zoo/vgg16_imagenet.prototxt'
#r.model_file = os.path.join(dir_root, 'zoo/caffe_reference_places365.caffemodel')
# model_file points to the model we want to use
#r.file_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1' # file root uh...? same as dir root?
r.file_root = dir_root
r.labels_file = os.path.join('/storage/data/ilsvrc12/', 'synset_words.txt') # file with the class labels in it  should be ''class code' 'name'\n'
r.dir_list = os.path.join('/storage/data/imagenet_2012_class_list.txt') # the list of directoires which contain our images
#r.labels_file = os.path.join(data_set_root, 'synset_objects227.txt')
#remake_synset_files = True # thsi makes your labels file
r.img_size = 224
r.blob = 'conv5_3'
r.blob_list=['conv5_3']
r.output_directory = os.getcwd()
h5_files_directory = os.path.join('/storage/data/Unmerged','VggImagenet')
r.do_limited = True
first_run = False # NTS run this once to make .h5 files, then put them in the h5_directory, amke h5_list and rerun
if first_run:
    do_h5_files = True
else:
    do_h5_files = False
# detaiuls for specific graph
current_neuron_index = 185
do_bus = False
do_car = True
figure_out_name = 'VggImagenetLconv5u185'


########################################################################################################################
# functions
########################################################################################################################



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
    verbose=True, root_dir=r.caffe_root, img_size=r.img_size)



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
    #from merger import merge_layer
    #merge_layer(os.getcwd(), 'inception4e', '100ActsOnly', 'all')

if first_run:
    exit(1)

acts, h5_list = combine_h5_files_in_activation_table(h5_file_location=h5_files_directory,
                                                             h5_list_filename='h5_list.txt', h5_list=[], useFile=True,
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

#### test to do ###
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

if do_car:
    jitterer_list(x_data=cluster_list,
              colour_flag='multi',
              title='',
              save_label=figure_out_name,
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

if do_bus:
    jitterer_list(x_data=cluster_list,
              colour_flag='multi',
              title='',
              save_label=figure_out_name,
              show_plots=False,
              save_plots=True,
              do_x_axis=True,
              do_y_axis=False,
              x_range=None,
              y_range=None,
              label_dict={},
              outLayerNeuron=True,
              current_neuron_indices=bus_indices)
    print('Bus classes are:')
    for x in bus_indices:
        print('{}'.format(r.labels[x]))






exit(1)
