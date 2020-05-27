import Caffe_AlexNet2 as C
import Make_activation as m
import h5_analysis_jitterer as h
import os
import shutil
import numpy as np

import set_up_caffe_net
from Test_AlexNet_on_directory import what_am_I_from_image

# ---------------------------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!
# IMPORTANT! YOU NEED TO SET Make_activation to load 'prob'ability data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# ---------------------------------------------------------------------------------------------------------------------
local_list=[]
acts = None
verbose = True
class_dict={}
net = None
transformer = ""
# -------------------------------------------------------------------------------------------------------------------- #
#   FUNCTIONS!
# -------------------------------------------------------------------------------------------------------------------- #

def grab_files(local_list=local_list,
            acts=acts,
            class_dict=class_dict,
            verbose=verbose,
            imagenet_root='/storage/data/imagenet_2012/',
            in_class_sub_dirs=True):
    """Function to get the selected images and return their addresses
    in_class_sub_dirs: True if like imagenet, false if like imagenet test set"""
    selected_image_list = []
    found_classes = []
    for selected_point in local_list:
        # grab filename
        selected_file = acts.get_file_name(selected_point).decode('UTF-8')
        if verbose:
            pass
            #print(selected_file)
        class_dir_label = selected_file.split('_')[0]
        if in_class_sub_dirs:
            # we've assumed files are in folders labelled by class!
            selected_image_list.append(imagenet_root + class_dir_label + '/' + selected_file)
        else:
            selected_image_list.append(imagenet_root + selected_file)
        class_no = class_dict[selected_file.split('_')[0]]
        if not class_no in found_classes:
            found_classes.append(class_no)
    return selected_image_list

def check_image_correct(true_class='',
                        local_list=local_list,
                        acts=acts,
                        class_dict=class_dict,
                        verbose=verbose,
                        imagenet_root='/storage/data/imagenet_2012/',
                        net=net,
                        transformer=transformer,
                        in_class_sub_dirs=True):
    """wrapper function to check that a given image is correct in a fresh instantiation of ALexNet
    ture_class needs to be input"""
    selected_image_list = grab_files(local_list=local_list,
        acts=acts, class_dict=class_dict,
        verbose=verbose, imagenet_root=imagenet_root, in_class_sub_dirs=in_class_sub_dirs)
    image_list = selected_image_list
    image_directory=''
    mistake_list_name = []
    mistake_list_no = []
    correct_list_name = []
    correct_list_no = []
    corrected_local_list=[]
    for image_no in range(len(image_list)):
        image_name = image_list[image_no]
        try:
            image = C.caffe.io.load_image(image_directory + image_name)
            good_to_go=True
        except:
            good_to_go=False
        if good_to_go:
            out_list, is_correct = what_am_I_from_image(
                image=image,
                net=net,
                transformer=transformer,
                verbose=verbose,
                found_labels=found_labels,
                class_labels=class_labels,
                true_class=true_class
                )
            if is_correct == False:
                if verbose:
                    print('Error: {} is incorrect'.format(image_name))
                mistake_list_name.append(image_name)
                mistake_list_no.append(image_no)
            else:
                # if its true or the functions doesnot know
                correct_list_name.append(image_name)
                correct_list_no.append(image_no)
                corrected_local_list.append(local_list[image_no])
        #else:
            #mistake_list_name.append(image_name)
            #mistake_list_no.append(image_no)
    return corrected_local_list, correct_list_name, correct_list_no, mistake_list_name, mistake_list_no

do_check=False
usingDocker=0
no_of_guesses=1

# code!!
CaffeSettings = set_up_caffe_net.main()
net = CaffeSettings.net
transformer = CaffeSettings.transformer
m.main()
acts = m.acts
class_labels = m.class_labels
# why did i changethe names? argh
#class_labels = labels # this is the readin text file
found_labels = m.s.short_labels #[label.split(' ')[0] for label in labels] # this is the list of the class codes

class_dict = h.make_class_to_line_number_look_up_table(class_labels=class_labels, verbose=False)
    # this builds the look-up table between points and the class they are in

## This bit is slow, it loads the label data for all acts
label_dict, found_labels, no_files_in_label = h.build_label_dict(acts)

# from Caffe_AlexNet import Get_Model_File
# model_file=Get_Model_File('no_reg_AlexNet')
# if usingDocker:
#     # new set-up with safer deployment for use on all machines
#     caffe_root, image_directory, labels_file, model_def, model_weights, dir_list, labels = set_up_caffe(model_file=model_file)
#     net, transformer = C.Caffe_NN_setup(imangenet_mean_image='python/caffe/imagenet/ilsvrc_2012_mean.npy',
#                                       batch_size=50, model_def=model_def, model_weights=model_weights,
#                                       verbose=True, root_dir=caffe_root)
# else:
#     # old set-up with hardcoded links and old-style unsafe deployment
#     caffe_root, image_directory, labels_file, model_def, model_weights, dir_list, labels = \
#         C.set_up_caffe(image_directory='/storage/data/imagenet_2012',
#                      model_file=model_file,
#                      label_file_address='data/ilsvrc12/synset_words.txt',
#                      dir_file='/storage/data/imagenet_2012_class_list.txt',
#                      root_dir='/home/eg16993/src/caffe', verbose=True)
#     net, transformer = C.Caffe_NN_setup(
#         imangenet_mean_image='/home/eg16993/src/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy',
#         batch_size=50, model_def=model_def, model_weights=model_weights,
#         verbose=True, root_dir=caffe_root)


def simple_move_files(selected_image_list, out_dir='/storage/data/top_images_test_set/'):
    """Function to grab files and move them"""
    for file_no in range(len(selected_image_list)):
        shutil.move(selected_image_list[file_no], out_dir + selected_image_list[file_no].split('/')[-1])
    return

## DO NOT RUN THESE LINES WITHOUT FIRST SORTING OUT THE MAIN BIT OF GRANTOPMOSTIMAGES!
#from Grab_top_most_images import grab_files
#from Grab_top_most_images import check_image_correct


make_collage_pictures = False # this makes a collage of the found pictures
do_move_files= False# this moves the original pictures over -- you probably dont want this
make_square_originals = True # this copies photos over and makes a canonical set with the correct square crop
do_file_write=True # write out a file with the filenames and correct classes
top_certainty = {}
no_top_certainty = {}
do_double_check = True # check one a new instantiaion of AlexNet that the image is correctly classified
do_triple_check = True # Yes I am that paranoid
top_dir='/storage/data/0602_L1_reg_top_1_imagenet_2012/'
no_correct=[]
no_mistake=[]
do_pictures=True



with open('correct_classes.txt', 'w') as file:
    for class_label in found_labels[990:1000]:
        print('yo!{}'.format(class_label))
        assert class_label == label_dict[class_label][0][0]
        certainty_list = []
        correct_list=[]
        correct_point_list = []
        current_directory = top_dir + class_label
        true_class=class_label
        if not os.path.exists(current_directory):
            os.makedirs(current_directory)
        ## 1. sqaurify and shrink files and copy em over
        # local list is the file tuples in acts
        local_list=label_dict[class_label]
        local_list=local_list
        # selected_image_list is the file names and addresses
        selected_image_list = grab_files(local_list=local_list,
                                         acts=acts,
                                         class_dict=class_dict,
                                         verbose=verbose,
                                         imagenet_root='/storage/data/imagenet_2012/',
                                         in_class_sub_dirs=True)
        if make_square_originals:
            for index in range(len(local_list)):
                # not that image_no is an index into local_list
                h.make_collage(out_file=selected_image_list[index].split('/')[-1],
                               local_list=[local_list[index]],
                               shrink=True, do_square=True, no_of_cols=1,
                               acts=acts, class_dict=class_dict, class_labels=class_labels,
                               verbose=False, imagenet_root='/storage/data/imagenet_2012/')
        # now grab the local squarified copies
        new_selected_image_list=grab_files(local_list=local_list,
                                         acts=acts,
                                         class_dict=class_dict,
                                         verbose=verbose,
                                         imagenet_root=top_dir,
                                         in_class_sub_dirs=False)
        # TEST 'em!
        corrected_local_list, correct_list_name, correct_list_no, mistake_list_name, mistake_list_no = check_image_correct(
            true_class=true_class,
            local_list=local_list,
            acts=acts,
            class_dict=class_dict,
            verbose=verbose,
            imagenet_root=top_dir,
            net=net,
            transformer=transformer,
            in_class_sub_dirs=False)
        # get some stats
        no_correct.append(len(correct_list_name))
        no_mistake.append(len(mistake_list_name))
        # remove the mistakes!
        for image in mistake_list_name:
            if not 'data/imagenet_2012/' in image:  # seriously, now we fucking check -- do not delete the originals!
                # kill kill kill
                os.remove(image)
        # move the pictures to where they should be!
        simple_move_files(selected_image_list=correct_list_name, out_dir=current_directory + '/')
        if do_file_write:
            for point in corrected_local_list:
                file.write(acts.get_file_name(point).decode() + ' ' + str(class_dict[class_label]) + '\n')
        # write some stats

csv_file = open('correct_stats.csv', 'w')
csv_file.write('Class\t, No. correct \t, No mistake\n')
for class_no in range(len(found_labels)):
    csv_file.write('{},\t {},\t {}\n'.format(found_labels[class_no], no_correct[class_no], no_mistake[class_no]))
#csv_file.close()
