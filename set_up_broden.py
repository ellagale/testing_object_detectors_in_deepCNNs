## script to get data from Zhou's version of AlexNet
## thingys we need

from Make_activation import combine_h5_files_in_activation_table

import kmeans

from set_up_caffe_net import set_up_caffe, CaffeSettings, Caffe_NN_setup

from Automation_experimental_functions import print_test_acts, make_synset_files


########################################################################################################################
# internal settings
########################################################################################################################
outfilename = 'results.csv'
# this sets up our default settings, I'm doing this in place of changing set_up_caffe_net
r = CaffeSettings()
dir_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/' # root file with image data in it
data_set_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/' #
r.image_directory = data_set_root + 'images/' # directory with test images
do_by_hand= True # do not change this
r.model_file ='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/zoo/caffe_reference_places365.caffemodel'
# model_file points to the model we want to use
r.file_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1' # file root uh...? same as dir root?
r.labels_file = dir_root + 'synset_words.txt' # file with the class labels in it  should be ''class code' 'name'\n'
r.dir_list = dir_root + 'dir_file.txt' # the list of directoires which contain our images
r.labels_file = data_set_root + 'synset_objects227.txt'
remake_synset_files = True # thsi makes your labels file

########################################################################################################################
# functions
########################################################################################################################



if remake_synset_files == True:
    # this makes the batch of files for 227x227 pictures used by AlexNet.
    # this sets up the object synset
    image_list, labels_list = make_synset_files(index_into_index_file=7,
                      index_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/index.csv',
                      category_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/c_object.csv',
                      out_file_name="synset_objects227.txt")
    # this sets up the part subset
    image_list_colors, labels_list_colors = make_synset_files(index_into_index_file=6,
                      index_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/index.csv',
                      category_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/c_color.csv',
                      out_file_name="synset_colors227.txt")
    # this sets up the part subset - these are also objects
    image_list_parts, labels_list_parts = make_synset_files(index_into_index_file=8,
                      index_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/index.csv',
                      category_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/c_part.csv',
                      out_file_name="synset_part227.txt")
    # this sets up the material
    image_list_materials, labels_list_scenes = make_synset_files(index_into_index_file=9,
                      index_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/index.csv',
                      category_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/c_material.csv',
                      out_file_name="synset_material227.txt")
    # this sets up the material
    image_list_scenes, labels_list_scenes = make_synset_files(index_into_index_file=10,
                      index_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/index.csv',
                      category_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/c_scene.csv',
                      out_file_name="synset_scene227.txt")
    # this sets up the material
    image_list_textures, labels_list_textures = make_synset_files(index_into_index_file=11,
                      index_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/index.csv',
                      category_file='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_227/c_texture.csv',
                      out_file_name="synset_texture227.txt")




## Get a list of images which have associated objects and we'll take those as classes




# here we set up where things should be found by caffe
r.caffe_root, r.image_directory, r.labels_file, r.model_def, r.model_weights, r.dir_list, r.labels = \
            set_up_caffe(image_directory=r.image_directory,
                         model_file=r.model_file,
                         #label_file_address='/storage/data/fooling_images_2015/synset_words.txt',
                         label_file_address=r.labels_file,
                         dir_file=r.dir_list,
                         root_dir='/home/eg16993/src/caffe', verbose=True)



#here we build our neural net
# NTS i am not entrely sure that the mean image is correct, I may have to make a new one
r.net, r.transformer = Caffe_NN_setup(
    imangenet_mean_image='/home/eg16993/src/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy',
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

h5_list_filename = 'dir_file.txt'

import Caffe_AlexNet2 as C
C.main(r) # this makes the h5 files

acts = kmeans.activation_table.ActivationTable(mean=False) # this sets up the activation table object

acts, h5_list = combine_h5_files_in_activation_table(h5_file_location=file_root + '/dataset/broden1_227/images/',
                                                             h5_list_filename='h5_list.txt', h5_list=[], useFile=True,
                                                             verbose=True)
# silly little test

print_test_acts(acts)

import Automation_experimental_functions as A


# check by just plotting a single unit, any unit, and this also buildsthe label_dict!
label_dict, found_labels, no_files_in_label = A.do_jitter_plot_test(acts,
                                            test_class_index=0,
                                            current_neuron_index=0,
                                            label_dict='',
                                            found_labels='',
                                            no_files_in_label='',
                                            name_leader='alexnet_places')

# Now do our tests:
units=[107, 79]
classes = [] # road, car

_ = A.do_jitter_plot_test(acts,
                          test_class_index=0,
                          current_neuron_index=0,
                          label_dict=label_dict,
                          found_labels=found_labels,
                          no_files_in_label=no_files_in_label,
                          name_leader ='')