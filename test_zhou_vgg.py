## script to get data from Zhou's version of vgg
## thingys we need

from Make_activation import combine_h5_files_in_activation_table

import os

from h5_analysis_jitterer import class_code_to_name, make_class_to_line_number_look_up_table, build_label_dict

from set_up_caffe_net import set_up_caffe, CaffeSettings, Caffe_NN_setup


########################################################################################################################
# internal settings
########################################################################################################################
outfilename = 'results.csv'
# this sets up our default settings, I'm doing this in place of changing set_up_caffe_net
r = CaffeSettings()
#dir_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/' # root file with image data in it
#data_set_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/dataset/broden1_224/' #
dir_root = '/storage/data/NetDissect/NetDissect-release1/'
dir_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/'
data_set_root = os.path.join(dir_root, 'dataset/broden1_224')
r.image_directory = os.path.join(data_set_root, 'images') # directory with test images
do_by_hand= True # do not change this
#r.model_file ='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/zoo/vgg16_places365.caffemodel'
r.model_file = os.path.join(dir_root, 'zoo/vgg16_places365.caffemodel')
# model_file points to the model we want to use
#r.file_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1' # file root uh...? same as dir root?
r.file_root = dir_root
r.labels_file = os.path.join(dir_root, 'synset_words.txt') # file with the class labels in it  should be ''class code' 'name'\n'
r.dir_list = os.path.join(dir_root, 'dir_file.txt') # the list of directoires which contain our images
r.labels_file = os.path.join(data_set_root, 'synset_objects224.txt')
r.deploy_file = 'vgg16_places365.prototxt'
r.img_size = 224
remake_synset_files = True # thsi makes your labels file
r.blob = 'conv5_3'
r.bloblist=[r.blob]
r.output_directory = os.getcwd()
h5_files_directory = dir_root + 'dataset/broden1_224/vgg_h5/'
num_of_points_for_initial_k_means = 5000
max_K_For_half=500
max_K_For_Structured = 100
allowed_error = 1
########################################################################################################################
# functions
########################################################################################################################


def make_synset_files(index_into_index_file, index_file, category_file, out_file_name):
    """Function to get Broden into the correct formats
    index_into_index_file is which column of index file to take
    index_file is the files with the images and associated categories
    category file is the file c_oject,txt etc with the human readable labels"""""
    ## Get a list of images which have associated objects and we'll take those as classes
    image_list = []
    correct_class_list = []
    file = open(index_file,'r')
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

remake_synset_files = False
if remake_synset_files == True:
    # this makes the batch of files for 227x227 pictures used by AlexNet.
    # this sets up the object synset
    image_list, labels_list = make_synset_files(index_into_index_file=7,
                      index_file=os.path.join(data_set_root, "index.csv"),
                      category_file=os.path.join(data_set_root, "c_object.csv"),
                      out_file_name="synset_objects224.txt")
    # this sets up the part subset
    image_list_colors, labels_list_colors = make_synset_files(index_into_index_file=6,
                      index_file=os.path.join(data_set_root, "index.csv"),
                      category_file=os.path.join(data_set_root, "c_color.csv"),
                      out_file_name="synset_colors224.txt")
    # this sets up the part subset - these are also objects
    image_list_parts, labels_list_parts = make_synset_files(index_into_index_file=8,
                      index_file=os.path.join(data_set_root, "index.csv"),
                      category_file=os.path.join(data_set_root, "c_part.csv"),
                      out_file_name="synset_part224.txt")
    # this sets up the material
    image_list_materials, labels_list_scenes = make_synset_files(index_into_index_file=9,
                      index_file=os.path.join(data_set_root, "index.csv"),
                      category_file=os.path.join(data_set_root, "c_material.csv"),
                      out_file_name="synset_material224.txt")
    # this sets up the material
    image_list_scenes, labels_list_scenes = make_synset_files(index_into_index_file=10,
                      index_file=os.path.join(data_set_root, "index.csv"),
                      category_file=os.path.join(data_set_root, "c_scene.csv"),
                      out_file_name="synset_scene224.txt")
    # this sets up the material
    image_list_textures, labels_list_textures = make_synset_files(index_into_index_file=11,
                      index_file=os.path.join(data_set_root, "index.csv"),
                      category_file=os.path.join(data_set_root, "c_texture.csv"),
                      out_file_name="synset_texture224.txt")




## Get a list of images which have associated objects and we'll take those as classes




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
deploy_file = caffe_settings.deploy_file
output_directory = caffe_settings.output_directory


class_dict = make_class_to_line_number_look_up_table(class_labels=r.labels)

h5_list_filename = 'dir_file.txt'

do_h5_files = False
if do_h5_files:
    import Caffe_AlexNet as C
    C.main(r) # this makes the h5 files

acts, h5_list = combine_h5_files_in_activation_table(h5_file_location=h5_files_directory, #os.path.join(file_root, 'dataset/broden1_224/images/vgg_h5'),
                                                             h5_list_filename='h5_conv5_list.txt', h5_list=[], useFile=True,
                                                             verbose=True)
# silly little test
print('{} files in table'.format(len(acts.get_all_point_indices())))
egg=acts.get_all_point_indices()[0]
point=acts.get_activation(egg)
print('Example file: {}, vectors are {}-dimensional'.format(point, len(point.vector)))
print('Example labels: {}'.format(point.labels))

import Automation_experimental_functions as A



# check by just plotting a single unit, any unit, and this also buildsthe label_dict!
label_dict, found_labels, no_files_in_label = A.do_jitter_plot_test(acts,
                                            test_class_index=0,
                                            current_neuron_index=0,
                                            label_dict='',
                                            found_labels='',
                                            no_files_in_label='',
                                            name_leader='vgg_places')

# Now do our tests: #### doesb;t work argh!

def get_class_details_broden(classname, labels, short_labels):
    """grabs cat number for a unit
    classname is the word name of the class
    labels is our list"""
    class_name = [x for x in labels if x.split(' ')[1] == classname][0]
    class_name = class_name.split(' ')[0]
    class_pos = short_labels.index(class_name)
    return class_name, class_pos


unit = 405
current_neuron_index = unit
cat = 38 #car
class_name = [x for x in labels if x.split(' ')[1] == 'car'][0]
class_name = class_name.split(' ')[0]
class_pos = short_labels.index(class_name)


_ = A.do_jitter_plot_test(acts,
                          test_class_index=class_pos,
                          current_neuron_index=unit,
                          label_dict=label_dict,
                          found_labels=found_labels,
                          no_files_in_label=no_files_in_label,
                          name_leader ='vgg_places')

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
                                     save_label='testb' + str(current_neuron_index) + 'cbyc.png', show_plots=False,
                                     save_plots=True,
                                     do_x_axis=True, do_y_axis=False, x_range=None, y_range=None, label_dict=label_dict,
                                     outLayerNeuron=True,
                                     current_neuron_index=93)

pos_in_cluster_list = found_labels.index(class_name)
test_class_activations = cluster_list[pos_in_cluster_list]



A.simple_stats_on_vector(test_class_activations, verbose=True)

from PIL import Image, ImageDraw

verbose = True
from h5_analysis_jitterer import compute_ccma_selectivity_neuron, \
    build_cluster_from_class_label, jitterer
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
    get_local_list_for_neuron(
        current_neuron_index=current_neuron_index,\
        minx='',\
        maxx='',\
        acts=acts)


def make_collage_new(out_file='temp.jpg', local_list=local_list, shrink=True, do_square=True, no_of_cols=5,
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
            #print(selected_file)
        # we've assumed files are in folders labelled by class!
        #class_dir_label = filename_to_label(selected_file.split('_')[0])
        selected_image_list.append(imagenet_root + 'ade20K' + '/' + selected_file)
        #class_no = class_dict[class_dir_label]
        #if not class_no in found_classes:
        #    found_classes.append(class_no)
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
    for row_no in range(int(len(images) /no_of_cols)):
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

filenames=[acts.get_file_name(x).decode() for x in local_list]

from h5_analysis_jitterer import make_collage

make_collage_new(out_file='VGGacts.jpg', local_list=local_list,
             shrink=False, do_square=False, no_of_cols=5,
             acts=acts, class_dict=class_dict, class_labels=class_labels,
             verbose=verbose, imagenet_root=r.image_directory)