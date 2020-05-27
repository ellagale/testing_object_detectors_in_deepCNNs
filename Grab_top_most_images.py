import Caffe_AlexNet2 as C
import Make_activation as m
import h5_analysis_jitterer as h
import os
import shutil
import numpy as np
from Test_AlexNet_on_directory import what_am_I_from_image
import set_up_caffe_net as s

# ---------------------------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!
# IMPORTANT! YOU NEED TO SET Make_activation to load 'prob'ability data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# ---------------------------------------------------------------------------------------------------------------------
local_list = []
acts = None
verbose = True
class_dict = {}
net = None


# -------------------------------------------------------------------------------------------------------------------- #
#   FUNCTIONS!
# -------------------------------------------------------------------------------------------------------------------- #

def what_am_I_from_prob(probabilities, no_of_guesses=5, true_class='', verbose=True):
    """Function to classify based on 'prob'ability layer inputs
    probabilities: a vector of input probabilities
    no_of_guesses: how many of the top probabilities do you want?
    true_class: the real class name if known
    outputs are in order: probability, label, human readable name
    """
    is_correct = 2  # lets use trinary, where 2 means indeterminate! :)
    if verbose:
        print('predicted class is:', probabilities.argmax())
        print('output label:{}'.format(found_labels[probabilities.argmax()]))
    top_inds = probabilities.argsort()[::-1][:no_of_guesses]  # reverse sort and take five largest items
    sorted_out_list = [(probabilities[x], found_labels[x]) for x in top_inds]
    out_list = []
    for guess in range(no_of_guesses):
        a_label = ' '.join(h.class_lineno_to_name(line_no=top_inds[guess], class_labels=class_labels)[2])
        out_list.append((sorted_out_list[guess][0] * 100, sorted_out_list[guess][1], a_label))
        if verbose:
            print('{:.2f}%: {}: {} '.format(sorted_out_list[guess][0] * 100, sorted_out_list[guess][1], a_label))
    if not true_class == '':
        # we can test if it is correct
        if type(true_class) == type(''):
            ground_truth = true_class
        else:
            # assume byte array
            ground_truth = true_class.decode()
        if found_labels[top_inds[0]] == ground_truth:
            # is correct
            is_correct = True
            if verbose:
                print('Image is correctly classified')
        else:
            is_correct = False
            if verbose:
                print('Image is incorrectly classified')
    return out_list, is_correct


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
            # print(selected_file)
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


def move_files(local_list=local_list,
               acts=acts, class_dict=class_dict,
               verbose=verbose, imagenet_root='/storage/data/imagenet_2012/',
               out_dir='/storage/data/top_images_test_set/'):
    """Function to grab imagenet files and move them"""
    selected_image_list = grab_files(local_list=local_list,
                                     acts=acts, class_dict=class_dict,
                                     verbose=verbose, imagenet_root='/storage/data/imagenet_2012/',
                                     in_class_sub_dirs=True)
    for file_no in range(len(selected_image_list)):
        shutil.copyfile(selected_image_list[file_no], out_dir + selected_image_list[file_no].split('/')[-1])
    return


def move_and_squarify_files(local_list=local_list,
                            acts=acts, class_dict=class_dict,
                            verbose=verbose, imagenet_root='/storage/data/imagenet_2012/',
                            out_dir='/storage/data/top_images_test_set/'):
    """Function to grab imagenet files and move them"""
    selected_image_list = grab_files(local_list=local_list,
                                     acts=acts, class_dict=class_dict,
                                     verbose=verbose, imagenet_root='/storage/data/imagenet_2012/',
                                     in_class_sub_dirs=True)
    for file_no in range(len(selected_image_list)):
        shutil.copyfile(selected_image_list[file_no], out_dir + selected_image_list[file_no].split('/')[-1])
    return


# -------------------------------------------------------------------------------------------------------------------- #
# Settings
# -------------------------------------------------------------------------------------------------------------------- #

do_check = False
usingDocker = 0
no_of_guesses = 1

# code!!
from set_up_caffe_net import Get_Model_File

m.main()
acts = m.acts
class_labels = m.class_labels

class_dict = h.make_class_to_line_number_look_up_table(class_labels=class_labels, verbose=False)
# this builds the look-up table between points and the class they are in

## This bit is slow, it loads the label data for all acts
label_dict, found_labels, no_files_in_label = h.build_label_dict(acts)
model_file = Get_Model_File('no_reg_AlexNet')
if usingDocker:
    # new set-up with safer deployment for use on all machines
    caffe_root, image_directory, labels_file, model_def, model_weights, dir_list, labels = s.set_up_caffe()
    net, transformer = s.Caffe_NN_setup(imangenet_mean_image='python/caffe/imagenet/ilsvrc_2012_mean.npy',
                                        batch_size=50, model_def=model_def, model_weights=model_weights,
                                        verbose=True, root_dir=caffe_root)
else:
    # old set-up with hardcoded links and old-style unsafe deployment
    caffe_root, image_directory, labels_file, model_def, model_weights, dir_list, labels = \
        s.set_up_caffe(image_directory='/storage/data/imagenet_2012',
                       model_file=model_file,
                       label_file_address='data/ilsvrc12/synset_words.txt',
                       dir_file='/storage/data/imagenet_2012_class_list.txt',
                       root_dir='/home/eg16993/src/caffe', verbose=True)
    net, transformer = s.Caffe_NN_setup(
        imangenet_mean_image='/home/eg16993/src/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy',
        batch_size=50, model_def=model_def, model_weights=model_weights,
        verbose=True, root_dir=caffe_root)


# out_list, is_correct=what_am_I_from_prob(probabilities, no_of_guesses=5, true_class='', verbose=True)
# out_list, is_correct=what_am_I_from_prob(probabilities, no_of_guesses=5, true_class=class_label, verbose=True)
# out_list, is_correct=what_am_I_from_prob(probabilities, no_of_guesses=5, true_class=true_class, verbose=True)

def find_most_active_in_prob(class_label='', no_of_guesses=1, acts=acts, label_dict=label_dict):
    """wrpper function to grab most certain images in prob layer
    class_label is the label of the class you are doing at thsi point
    no_of_guesses: whether to accept top 1 or top 5 acc (top 1 chosen)"""
    point_indices = label_dict[class_label]
    for point in point_indices:
        current_activation = acts.get_activation(point)
        probabilities = current_activation.vector
        true_class = current_activation.label
        out_list, is_correct = what_am_I_from_prob(probabilities, no_of_guesses=no_of_guesses, true_class=true_class,
                                                   verbose=True)
        if is_correct:
            certainty_list.append(out_list[0][0])
            correct_list.append(is_correct)
            correct_point_list.append(point)
    indices, certainty = h.find_position_of_max_act_in_vector(np.array(certainty_list))
    certainty_list.append(certainty)
    local_list = [correct_point_list[x] for x in list(indices[0])]
    return local_list, true_class


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
    image_directory = ''
    mistake_list_name = []
    mistake_list_no = []
    correct_list_name = []
    correct_list_no = []
    corrected_local_list = []
    for image_no in range(len(image_list)):
        image_name = image_list[image_no]
        try:
            image = C.caffe.io.load_image(image_directory + image_name)
            good_to_go = True
        except:
            good_to_go = False
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
        # else:
        # mistake_list_name.append(image_name)
        # mistake_list_no.append(image_no)
    return corrected_local_list, correct_list_name, correct_list_no, mistake_list_name, mistake_list_no


make_collage_pictures = False  # this makes a collage of the found pictures
do_move_files = False  # this moves the original pictures over -- you probably dont want this
make_square_originals = True  # this copies photos over and makes a canonical set with the correct square crop
do_file_write = True  # write out a file with the filenames and correct classes
top_certainty = {}
no_top_certainty = {}
do_double_check = True  # check one a new instantiaion of AlexNet that the image is correctly classified
do_triple_check = True  # Yes I am that paranoid
with open('correct_classes.txt', 'w') as file:
    for class_label in found_labels:
        print('yo!{}'.format(class_label))
        assert class_label == label_dict[class_label][0][0]
        certainty_list = []
        correct_list = []
        correct_point_list = []
        ## 1. this loop finds the  top-most 'sure' images from the saved run through AlexNet that I had previously done
        local_list, true_class = find_most_active_in_prob(class_label=class_label, no_of_guesses=1, acts=acts,
                                                          label_dict=label_dict)
        ## 2. this loop tests the image on a fresh instantiation of imagenet
        if do_double_check:
            # corrected local list is the file indices into acts
            corrected_local_list, correct_list_name, correct_list_no, mistake_list_name, mistake_list_no = check_image_correct(
                true_class=true_class,
                local_list=local_list,
                acts=acts,
                class_dict=class_dict,
                verbose=verbose,
                imagenet_root='/storage/data/imagenet_2012/',
                net=net,
                transformer=transformer,
                in_class_sub_dirs=True)
            # selected_image_list is the file names and addresses
            selected_image_list = grab_files(local_list=corrected_local_list,
                                             acts=acts,
                                             class_dict=class_dict,
                                             verbose=verbose,
                                             imagenet_root='/storage/data/imagenet_2012/',
                                             in_class_sub_dirs=True)
        else:
            selected_image_list = grab_files(local_list=local_list,
                                             acts=acts,
                                             class_dict=class_dict,
                                             verbose=verbose,
                                             imagenet_root='/storage/data/imagenet_2012/',
                                             in_class_sub_dirs=True)
            corrected_local_list = local_list
        ## 3. Squarify the images and copy em over
        if make_square_originals:
            for index in range(len(corrected_local_list)):
                # not that image_no is an index into local_list
                h.make_collage(out_file=selected_image_list[index].split('/')[-1],
                               local_list=[corrected_local_list[index]],
                               shrink=True, do_square=True, no_of_cols=1,
                               acts=acts, class_dict=class_dict, class_labels=class_labels,
                               verbose=False, imagenet_root='/storage/data/imagenet_2012/')
        if do_triple_check:
            # now we test the 277*277 mini pictures and make sure they are correct
            corrected_local_list, correct_list_name, correct_list_no, mistake_list_name, mistake_list_no = check_image_correct(
                true_class=true_class,
                local_list=corrected_local_list,
                acts=acts,
                class_dict=class_dict,
                verbose=verbose,
                imagenet_root='/storage/data/top_images_test_set/',
                net=net,
                transformer=transformer,
                in_class_sub_dirs=False)
            new_selected_image_list = correct_list_name
            # now we remove the bad choices
            for image in mistake_list_name:
                if not '/imagenet_2012/' in image:  # seriously, now we fucking check -- do not delete the originals!
                    # kill kill kill
                    os.remove(image)
        if make_collage_pictures:
            h.make_collage(out_file='collage_' + class_label + '.jpg', local_list=selected_image_list, shrink=True,
                           do_square=True, no_of_cols=5,
                           acts=acts, class_dict=class_dict, class_labels=class_labels,
                           verbose=False, imagenet_root='/storage/data/imagenet_2012/')
        if do_move_files:
            move_files(local_list=local_list,
                       acts=acts, class_dict=class_dict,
                       verbose=verbose, imagenet_root='/storage/data/imagenet_2012/',
                       out_dir='/storage/data/top_images_test_set/')
        if do_file_write:
            for point in corrected_local_list:
                file.write(acts.get_file_name(point).decode() + ' ' + str(class_dict[class_label]) + '\n')
        # top_certainty[class_label] = certainty
        # no_top_certainty[class_label] = len(local_list)
    # h.class_lineno_to_name(line_no=top_inds, class_labels=class_labels)
file.close()

ave_certainty_list = [top_certainty[x] for x in top_certainty.keys()]
list_no_top_certainty = [no_top_certainty[x] for x in no_top_certainty.keys()]
keys_no_top_certainty = [no_top_certainty.keys() for x in no_top_certainty.keys()]

egg = {}
for class_label in found_labels:
    egg[top_certainty[class_label]] = class_label

# egg[min(ave_certainty_list)]
# h.class_lineno_to_name(class_dict[egg[min(ave_certainty_list)]], class_labels=class_labels)


with open('certainty_Stats.csv', 'w') as file:
    for meh in range(1000):
        file.write(str(ave_certainty_list[meh]) + ', ' + str(list_no_top_certainty[meh]) + '\n')
