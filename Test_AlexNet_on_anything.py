#!/usr/bin/python3
import Caffe_AlexNet as C

import argparse
import os
import h5_analysis_jitterer as h
from set_up_caffe_net import Caffe_NN_setup

FLAGS = None
image = None
net = None
transformer = None
image_directory = '/storage/data/imagenet_2012'

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

    flags, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognised flags: {}'.format(unparsed))
        quit(1)
    return flags

usingDocker = False
image_directory = '/storage/data/TestImages/freqed_images'
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
    if not true_class=='':
        # we can test if it is correct
        if found_labels[top_inds[0]] == true_class.decode():
            # is correct
            is_correct = True
        else:
            is_correct =False
    return out_list, is_correct

#----------------------------------------------------------------------------------------------------------------------#
#       Program
#----------------------------------------------------------------------------------------------------------------------#

def main():
    image_directory = FLAGS.image_dir
    image_filename = FLAGS.image
    verbose = FLAGS.verbose
    if usingDocker:
        # new set-up with safer deployment for use on all machines
        caffe_root, image_directory, labels_file, model_def, model_weights, dir_list, labels = set_up_caffe()
        net, transformer = Caffe_NN_setup(imangenet_mean_image='python/caffe/imagenet/ilsvrc_2012_mean.npy',
                                          batch_size=50, model_def=model_def, model_weights=model_weights,
                                          verbose=True,root_dir=caffe_root)
    else:
        # old set-up with hardcoded links and old-style unsafe deployment
        caffe_root, image_directory, labels_file, model_def, model_weights, dir_list, labels = \
            C.set_up_caffe(image_directory=image_directory,
                         model_file='models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                         label_file_address='data/ilsvrc12/synset_words.txt',
                         dir_file='/storage/data/imagenet_2012_class_list.txt',
                         root_dir='/home/eg16993/src/caffe', verbose=True)
        net, transformer = C.Caffe_NN_setup(imangenet_mean_image='/home/eg16993/src/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy',
                                      batch_size=50, model_def=model_def, model_weights=model_weights,
                                      verbose=True,root_dir=caffe_root)

    class_labels = labels
    found_labels = [label.split(' ')[0] for label in labels]

    image = C.caffe.io.load_image(image_directory + '/' + image_filename)
    #image = C.caffe.io.load_image(image_directory + '/shark2.png')
    #image = C.caffe.io.load_image(image_directory + '/n01484850_76.JPEG')
    #image = C.caffe.io.load_image(image_directory + '/205.jpg')

    what_am_I_from_image(image=image, net=net, transformer=transformer, verbose=verbose, found_labels=found_labels, class_labels=class_labels)
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
#
# batches_of_activations = c.parse_directory(net=net,
#                                             transformer=transformer,
#                                             image_directory=current_image_directory,
#                                             no_of_images=None,
#                                             batch_size=50,
#                                             verbose=False,
#                                             blob_list=blob_list,
#                                             label=true_class)
#
#
# c.classify_directory(batches, labels, check_classification=False,
#                        true_class=None, no_of_guesses=1,
#                        assignIndices=True, assignLabels=False, verbose=True)

if __name__ == '__main__':
    FLAGS = handle_args()
    main()
