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
data_set_root = os.path.join(dir_root, 'dataset/broden1_224')
r.image_directory = os.path.join(data_set_root, 'images') # directory with test images
do_by_hand= True # do not change this
#r.model_file ='/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1/zoo/googlenet_places365.caffemodel'
r.model_file = os.path.join(dir_root, 'zoo/googlenet_places365.caffemodel')
# model_file points to the model we want to use
#r.file_root = '/home/eg16993/neuralNetworks/experiments/NetDissect/NetDissect-release1' # file root uh...? same as dir root?
r.file_root = dir_root
r.labels_file = os.path.join(dir_root, 'synset_words.txt') # file with the class labels in it  should be ''class code' 'name'\n'
r.dir_list = os.path.join(dir_root, 'dir_file.txt') # the list of directoires which contain our images
r.labels_file = os.path.join(data_set_root, 'synset_objects224.txt')
r.deploy_file = 'googlenet_places365.prototxt'
r.img_size = 224
r.blob_list=['inception_5b/output']
remake_synset_files = True # thsi makes your labels file
r.output_directory = os.getcwd()
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

import Caffe_AlexNet as C
C.main(r) # this makes the h5 files

acts, h5_list = combine_h5_files_in_activation_table(h5_file_location=output_directory,
                                                             h5_list_filename='h5_list.txt', h5_list=[], useFile=True,
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

# Now do our tests:

_ = A.do_jitter_plot_test(acts,
                          test_class_index=0,
                          current_neuron_index=0,
                          label_dict=label_dict,
                          found_labels=found_labels,
                          no_files_in_label=no_files_in_label,
                          name_leader ='')
