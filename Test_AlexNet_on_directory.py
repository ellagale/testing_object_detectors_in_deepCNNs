#!/usr/bin/python3
import Caffe_AlexNet as C

import argparse
import os
import h5_analysis_jitterer as h
import csv
from collections import OrderedDict

import set_up_caffe_net as s

FLAGS = None
image = None
net = None
transformer = None
image_directory = '/storage/data/imagenet_2012'
correct_class_filename = ''
caffe_settings = None

def handle_args():
    """ Parse cli arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default=os.getcwd(),
        help='where the image directory is'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='',
        help='image file name'
    )
    parser.add_argument(
        '--no_of_guesses',
        type=int,
        default=5,
        help='number of probabilities to return'
    )
    parser.add_argument(
        '--verbose',
        default=True,
        action='store_true',
        help='more verbose logging.'
    )
    parser.add_argument(
        '--class_list',
        type=str,
        default='correct_classes.txt',
        help='file with list of file names and maybe also the correct classes in'
    )
    #parser.add_argument(
    #    '--model',
    #    type=str,
    #    default='',
    #    help='model file flag--see settings in Caffe_AlexNet.py'
    #)

    flags, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognised flags: {}'.format(unparsed))
        quit(1)
    return flags

usingDocker = False
image_directory = os.getcwd() #'/storage/data/TestImages/freqed_images'
label_file_address='/storage/data/ilsvrc12/synset_words.txt'
no_of_guesses=5

#----------------------------------------------------------------------------------------------------------------------#
#       Functions
#----------------------------------------------------------------------------------------------------------------------#

def what_am_I_from_image(image=image, net=net, transformer=transformer, verbose=True, true_class = '', found_labels=[], class_labels=[]):
    """Function to classify based on 'prob'ability layer inputs
    probabilities: a vector of input probabilities
    no_of_guesses: how many of the top probabilities do you want?
    true_class: the real class name if known
    outputs are in order: probability, label, human readable name
    """
    #TODO: This has not yet been properly tested!!!!!!
    transformed_image = transformer.preprocess('data', image)
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image
    ### perform classification
    output = net.forward()
    probabilities= output['prob'][0]  # the output probability vector for the first image in the batch
    is_correct = 2 # lets use trinary, where 2 means indeterminate! :)
    if verbose:
        print('predicted class is:', probabilities.argmax())
        print('output label:{}'.format(found_labels[probabilities.argmax()]))
    top_inds = probabilities.argsort()[::-1][:no_of_guesses]  # reverse sort and take five largest items
    sorted_out_list = [(probabilities[x], found_labels[x]) for x in top_inds ]
    out_list=[]
    for guess in range(no_of_guesses):
        a_label = ' '.join(h.class_lineno_to_name(line_no=top_inds[guess], class_labels=class_labels)[2])
        if guess==0:
            best_guess = a_label
        out_list.append((sorted_out_list[guess][0]*100, sorted_out_list[guess][1], a_label))
        if verbose:
            print('{:.2f}%: {}: {} '.format(sorted_out_list[guess][0]*100, sorted_out_list[guess][1], a_label))
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

#----------------------------------------------------------------------------------------------------------------------#
#       Program
#----------------------------------------------------------------------------------------------------------------------#

def main():
    global acts, class_labels, h5_list, caffe_settings

    image_directory = FLAGS.image_dir
    image_filename = FLAGS.image
    verbose = FLAGS.verbose
    correct_class_filename = FLAGS.class_list
    #model_file_flag = FLAGS.model_file
    #model_file = Get_Model_File(model_file_flag)
    #model_file = Get_Model_File('AlexNet_standard')
    image_directory = FLAGS.image_dir
    caffe_settings = s.main()
    # caffe_root = s.caffe_root
    #image_directory = caffe_settings.image_directory
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
    #model_file = Get_Model_File('L1_AlexNet')
    #model_file = Get_Model_File('no_reg_AlexNet')
    # first we grab the network
    # if usingDocker:
    #     # new set-up with safer deployment for use on all machines
    #     caffe_root, image_directory, labels_file, model_def, model_weights, dir_list, labels = set_up_caffe(
    #         image_directory=image_directory,
    #         model_file=model_file)
    #     net, transformer = Caffe_NN_setup(imangenet_mean_image='python/caffe/imagenet/ilsvrc_2012_mean.npy',
    #                                       batch_size=50, model_def=model_def, model_weights=model_weights,
    #                                       verbose=True,root_dir=caffe_root)
    # else:
    #     # old set-up with hardcoded links and old-style unsafe deployment
    #     caffe_root, image_directory, labels_file, model_def, model_weights, dir_list, labels = \
    #         C.set_up_caffe(image_directory=image_directory,
    #                      model_file=model_file,
    #                      label_file_address='data/ilsvrc12/synset_words.txt',
    #                      dir_file='/storage/data/imagenet_2012_class_list.txt',
    #                      root_dir='/home/eg16993/src/caffe', verbose=True)
    #     net, transformer = C.Caffe_NN_setup(imangenet_mean_image='/home/eg16993/src/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy',
    #                                   batch_size=1, model_def=model_def, model_weights=model_weights,
    #                                   verbose=True,root_dir=caffe_root)
    # why did i changethe names? argh
    class_labels = labels # this is the readin text file
    found_labels = [label.split(' ')[0] for label in labels] # this is the list of the class codes

    # this chunk grabs the filenames and
    image_list = []
    correct_class_list = []
    file = open(correct_class_filename, 'r')
    for line in file:
        line_list=line.strip().split(' ')
        image_list.append(line_list[0])
        if len(line_list) == 2:
            # we have been given classes
            correct_class_list.append(line_list[1])
            check_classes = True
        else:
            # no classes :(
            check_classes = False

    fieldnames = ['image_name',  # image name
               'true_class',  # true class if known
               'true_class_name',  # true class name if known
               'is_correct',  # did AlexNet get it correct?
               'top_1_prob',  # probabilty (certainty) of the top class (top 1)
               'top_1_prob_class',  # top_class code
               'top_1_prob_name',  # top class human readable name
               'top_2_prob',
               'top_2_prob_class',
               'top_2_prob_name',
               'top_3_prob',
               'top_3_prob_class',
               'top_3_prob_name',
               'top_4_prob',
               'top_4_prob_class',
               'top_4_prob_name',
               'top_5_prob',
               'top_5_prob_class',
               'top_5_prob_name'
               ]

    correct_top_1 = []
    out_filename = 'results.csv'
    with open(out_filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        # now we process all the images
        # TODO: I am sure there is a batch way to do this that would be faster...
        for image_no in range(len(image_list)):
            image_name = image_list[image_no]
            image = C.caffe.io.load_image(image_directory + '/' + image_name)
            if check_classes:
                true_class = found_labels[int(correct_class_list[image_no])]
                true_class_name = ' '.join(class_labels[int(correct_class_list[image_no])].split(' ')[1:])
                out_list, is_correct =what_am_I_from_image(
                    image=image,
                    net=net,
                    transformer=transformer,
                    verbose=verbose,
                    found_labels=found_labels,
                    class_labels=class_labels,
                    true_class=true_class
                )
            else:
                out_list, is_correct =what_am_I_from_image(
                    image=image,
                    net=net,
                    transformer=transformer,
                    verbose=verbose,
                    found_labels=found_labels,
                    class_labels=class_labels,
                    true_class=''
                )
                true_class = 'null'
                true_class_name = 'unknown'
            if check_classes and is_correct:
                correct_top_1.append(image_name)
            row = {'image_name': image_name,            # image name
                    'true_class': true_class,               # true class if known
                    'true_class_name': true_class_name,     # true class name if known
                    'is_correct': str(is_correct),          # did AlexNet get it correct?
                    'top_1_prob': str(out_list[0][0]),      # probabilty (certainty) of the top class (top 1)
                    'top_1_prob_class': out_list[0][1],     # top_class code
                    'top_1_prob_name': out_list[0][2],      # top class human readable name
                    'top_2_prob': str(out_list[1][0]),
                    'top_2_prob_class': out_list[1][1],
                    'top_2_prob_name': out_list[1][2],
                    'top_3_prob': str(out_list[2][0]),
                    'top_3_prob_class': out_list[2][1],
                    'top_3_prob_name': out_list[2][2],
                    'top_4_prob': str(out_list[3][0]),
                    'top_4_prob_class': out_list[3][1],
                    'top_4_prob_name': out_list[3][2],
                    'top_5_prob': str(out_list[4][0]),
                    'top_5_prob_class': out_list[4][1],
                    'top_5_prob_name': out_list[4][2]
                               }
            sorted_row = OrderedDict(sorted(row.items(), key=lambda item: fieldnames.index(item[0])))
            writer.writerow(sorted_row)
    print('No. of correct (top 1): {}'.format(len(correct_top_1)))

### perform classification
# output = net.forward()
# output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
#
# print('predicted class is:', output_prob.argmax())
# labels = np.loadtxt(label_file_address, str, delimiter='\t')
# print('output label:{}', labels[output_prob.argmax()])
# top_inds = output_prob.argsort()[::-1][:no_of_guesses]  # reverse sort and take five largest items
# sorted_out_list = list(zip(output_prob[top_inds], labels[top_inds]))
# for guess in range(no_of_guesses):
#     print('{}'.format(sorted_out_list[guess]))
#
#
# # sort top five predictions from softmax output
# top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
#
# # load ImageNet labels
# labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
# if not os.path.exists(labels_file):
#     !../data/ilsvrc12/get_ilsvrc_aux.sh
#
# labels = np.loadtxt(labels_file, str, delimiter='\t')
#
# print 'output label:', labels[output_prob.argmax()]

if __name__ == '__main__':
    FLAGS = handle_args()
    main()
