
from Make_activation import combine_h5_files_in_activation_table

import os

from h5_analysis_jitterer import class_code_to_name, make_class_to_line_number_look_up_table, build_label_dict

from set_up_caffe_net import set_up_caffe, CaffeSettings, Caffe_NN_setup
import sys
from Automation_experimental_functions import print_test_acts, build_label_dict, make_synset_files


from h5_analysis_jitterer import build_cluster_from_class_label, jitterer


########################################################################################################################
# internal settings
########################################################################################################################
do_by_hand= True # do not change this
do_h5_files = False
verbose=True
h5_list_filename='h5_list.txt'
outfilename = 'results.csv'
# this sets up our default settings, I'm doing this in place of changing set_up_caffe_net
r = CaffeSettings()
#dir_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/' # root file with image data in it
#data_set_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/' #
#r.image_directory = data_set_root + 'images/' # directory with test images
###### use this if on kraken
dir_root = '/storage/data/NetDissect/NetDissect-release1/'
###### use this on your computer
dir_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/'
data_set_root = os.path.join(dir_root, 'dataset/broden1_224')
r.image_directory = os.path.join(data_set_root, 'images') # directory with test images

#r.model_file ='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/zoo/caffe_reference_places365.caffemodel'
r.model_file = os.path.join(dir_root, 'zoo/googlenet_places365.caffemodel')
# model_file points to the model we want to use
#r.file_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1' # file root uh...? same as dir root?
r.file_root = dir_root
r.labels_file = os.path.join(dir_root, 'synset_words.txt') # file with the class labels in it  should be ''class code' 'name'\n'
r.dir_list = os.path.join(dir_root, 'dir_file.txt') # the list of directoires which contain our images
r.labels_file = os.path.join(data_set_root, 'synset_objects224.txt')
remake_synset_files = True # thsi makes your labels file
r.blob = 'inception_5b/output'
r.blob_list =['inception_5b/output']
r.output_directory = os.getcwd()
h5_files_directory = os.path.join(dir_root, 'dataset/broden1_224/googlenet_h5/')
num_of_points_for_initial_k_means = 5000
max_K_For_half=500
max_K_For_Structured = 100
allowed_error = 1

########################################################################################################################
# functions
########################################################################################################################
# does the setting up
r.caffe_root, r.image_directory, r.labels_file, r.model_def, r.model_weights, r.dir_list, r.labels = \
            set_up_caffe(image_directory=r.image_directory,
                         model_file=r.model_file,
                         #label_file_address='/storage/data/fooling_images_2015/synset_words.txt',
                         label_file_address=r.labels_file,
                         dir_file=r.dir_list,
                         root_dir=os.environ['CAFFE_ROOT'], verbose=True)

# builds a version of the net here for you to use
r.net, r.transformer = Caffe_NN_setup(
    imangenet_mean_image=os.path.join(os.environ['PYCAFFE_ROOT'], 'caffe/imagenet/ilsvrc_2012_mean.npy'),
    batch_size=50, model_def=r.model_def, model_weights=r.model_weights,
    verbose=True, root_dir=r.caffe_root)

r.short_labels = [label.split(' ')[0] for label in r.labels]

print('Make_activation')
global acts, class_labels, h5_list, caffe_settings, dir_list

### does local settings
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

## make the h5 files if not already made
if do_h5_files:
    import Caffe_AlexNet as C
    C.main(r) # this makes the h5 files
    # N.B. you may need to make a h5_list.txt of the output files if it fails just after this


acts, h5_list = combine_h5_files_in_activation_table(h5_file_location=h5_files_directory,
                                                    h5_list_filename=h5_list_filename, h5_list=[], useFile=True,
                                                    verbose=True)
# silly little test - test to try and get
print_test_acts(acts)
# makes a look up table between class labels and their line in the labels file
class_dict = make_class_to_line_number_look_up_table(class_labels=class_labels, verbose=False)


sys.stdout.write('About to build the label dict (slow)')
sys.stdout.flush()
label_dict, found_labels, no_files_in_label = build_label_dict(acts, use_loaded_files = False, verbose=True, doMean=False)
sys.stdout.write('Built the label dict')
sys.stdout.flush()
no_of_images = acts.get_image_count()
print('Found {} images'.format(no_of_images))

# our experiment

[x for x in labels if x.split(' ')[1] == 'bus'] # ater code is 203
import Automation_experimental_functions as A
class_name, class_pos = A.get_class_details_broden('bus', labels, short_labels)

current_neuron_index = 603
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
                                     do_x_axis=True, do_y_axis=False, x_range=[0.01,16], y_range=None, label_dict=label_dict,
                                     outLayerNeuron=True,
                                     current_neuron_index=class_pos)

pos_in_cluster_list = found_labels.index(class_name)
test_class_activations = cluster_list[pos_in_cluster_list]

A.simple_stats_on_vector(test_class_activations, verbose=True)



verbose = True
from h5_analysis_jitterer import compute_ccma_selectivity_neuron
import numpy as np

ccma_selectivity, mu_max, mu_not_max, max_index = \
                    compute_ccma_selectivity_neuron(cluster_list,
                                                    found_labels=found_labels,
                                                    class_dict=class_dict,
                                                    class_labels=class_labels,
                                                    top_class='603', verbose=verbose)

ccma_selectivity_top, mu_max, mu_not_max, max_index = \
                    compute_ccma_selectivity_neuron(cluster_list,
                                                    found_labels=found_labels,
                                                    class_dict=class_dict,
                                                    class_labels=class_labels,
                                                    top_class='', verbose=verbose)





