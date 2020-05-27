#!/usr/bin/env python3
## This sets up caffe and builds the h5 files

#####################################################
##  1   Set-up
#####################################################
from __future__ import print_function

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import os
import getopt
import sys
import caffe
import kmeans.activation_table
from kmeans.activation_table import ActivationTable
import set_up_caffe_net as s

caffe_settings = None
# import develop as d
# display plots in this notebook
# %matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)  # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import h5py
# this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
import pdb


def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)  # more "modern"


sys.excepthook = info


class Batch(object):
    ''' a batch of images and their classifications
    '''

    def __init__(self, size, images, blobs, probabilities, label):
        ''' create a base batch object.
            size: the numeber of images in this batch
            images: [] of images in the batch
            blobs: {layer -> data}
            probabilties: label guesses.
            label: the actual label of the batch
        '''
        self.size = size
        self.images = images
        ## THIS IS VERY IMPORTANT< THIS COPYS THE VALueS AND NOT A POINTER! AVOIdS THE AWFuL ERROR OF DOOM
        self.blobs = {key: np.copy(value) for key, value in blobs.items()}
        self.probabilities = np.copy(probabilities)
        self.label = label


try:
    opts, args = getopt.getopt(sys.argv[1:], "i:v:", ["imagename=", "verbose"])
except getopt.GetoptError:
    print("{0}: [-i|--imagename=<name>] -v|--verbose]".format(sys.argv[0]))
    sys.exit(1)
for opt, arg in opts:
    if opt in ('-i', '--imagename'):
        noImageName = 0
        image_name = str(arg)
        print("{}".format(image_name))
    elif opt in ('-v', '--verbose'):
        verbose = True

# includeTop = 0 DO NOT USE THIS IS TOO SLOW!
# dir_list = [x[0] for x in os.walk(image_directory)]
# print(dir_list)
# if includeTop != 1:
#     dir_list = dir_list[1:]

# image_directory= dir_list[3]

########################################################
#   0.  Command line parameters and settings
########################################################
no_of_guesses = 1  # sort top five predictions from softmax output
verbose = 1
noImageName = 1
# image_directory = os.getcwd()
image_name = 'image.jpg'
blob_list = ['fc6']  # ['prob','fc8', 'fc7', 'fc6']# ['prob'] #['fc8','fc6']
labels = []
short_labels = []
net = None


# usingDocker=0


# blob_list=['fc6', 'fc8', 'conv5']


##########################################################
#   3   Classification FUnctions
##########################################################

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)

## classigy a direcptyur

# image_directory = caffe_root + 'data/flickr_style/images/'
# image_directory = '/home/eg16993/neuralNetworks/experiments/09112017_CaffeTest/'


def parse_directory(net, transformer, image_directory=os.getcwd(), no_of_images=None, batch_size=50, verbose=True,
                    blob_list=blob_list, label=None):
    """Function to parse images from a directory into AlexNet and generate a list of classifications
    image_directory: where to get the images from
    no_of_images: no. of images to to parse in the directory, default is all
    batch_size: no. to feed in at a time
    verbose: whether to spew up meaningless data
    blob_list: which layers do you want the activations for?
    """
    image_list = os.listdir(image_directory)  # get a list of images to classify
    image_list = [x for x in image_list if ('JPEG' or 'jpg' or 'PNG' or 'png') in x]  # nts png functionality not tested
    if verbose:
        print('Using images from {}'.format(image_directory))
    if len(image_list) == 0:
        print('No images found in {}'.format(image_directory))
        return
    if no_of_images == None:
        no_of_images = len(image_list)  # per diretory
    # assert no_of_images >= batch_size
    print(no_of_images)
    count = 0
    batches_of_activations = []
    # setup loop invariants
    # how many images are left to process
    images_to_process = no_of_images
    # index of first image in this batch
    batch_start = 0
    while images_to_process > 0:
        # feed in and transform the images
        # deal with last case
        if images_to_process > batch_size:
            images_to_process -= batch_size
        else:
            # handle final batch (may be < batch_size)
            batch_size = images_to_process
            images_to_process = 0
        sys.stdout.write('{}-{} '.format(batch_start, batch_start + batch_size))
        batch_images = image_list[batch_start:batch_start + batch_size]

        sys.stdout.flush()
        for batch_image_index, batch_image in enumerate(batch_images):
            # this feeds in a batches worth of picture data
            net.blobs['data'].data[batch_image_index, ...] = transformer.preprocess('data',
                                                                                    caffe.io.load_image(
                                                                                        os.path.join(image_directory,
                                                                                                     batch_image)))
            count = count + 1
            # if i==0 and verbose:
            #    # spew out a picture if you want
            #    image = caffe.io.load_image(os.path.join(image_directory, image_list[i]))
            #    plt.imshow(image)
            #    plt.savefig("example_image.png")
            ### perform classification on a batch
        probabilities = net.forward()['prob']
        temp = {}
        for blob_name in blob_list:
            temp[blob_name] = net.blobs[blob_name].data
        batches_of_activations.append(Batch(batch_size, batch_images, temp, probabilities, label))
        # reestablish invariants
        batch_start += batch_size
        # if verbose:
        #    output_prob = output[0]['prob'][0]  # the output probability vector for the first image in the batch
    if verbose:
        print('{} images processed'.format(count))
    return batches_of_activations


def classify_directory(batches, labels, check_classification=False,
                       true_class=None, no_of_guesses=1,
                       assignIndices=True, assignLabels=False, verbose=True):
    """Function to generate a list of classifications from output
    batches: array of batch objects created by parse_directory
    check_classification: whether to compare the classification to the known result
    true_class: what the actual class is (assumes a batch of the same class)
    no_of_guesses: how many of the top probabilities you want to check for the true_class, default is 1
    assignIndices: whether to return the assigned indices
    assignLabels: whether to return the assinged labels
    """
    # no_of_batches = len(batches)
    if check_classification == True:
        # verify we are using the correct imagenet
        assert true_class in [label.split(' ')[0] for label in labels]
    if check_classification == True:
        assert true_class is not None
    if true_class is not None:
        assert check_classification == True
    # correct_indices = []
    # incorrect_indices =[]
    print('classify')
    for batch_no, batch in enumerate(batches):
        # if batch_no==26:
        #     import pdb;
        #     pdb.set_trace()
        print('{}:{}'.format(batch_no, batch))
        # assign labels
        index_accuracies = []
        assigned_label_indices = []
        assigned_labels = []
        for index in range(batch.size):
            print('{}'.format(index))
            output_prob = batch.probabilities[index]
            assigned_label_index = int(output_prob.argmax())  # this is the int line number in labels
            if verbose:
                print('predicted class is: {}: {}, {}'
                      .format(assigned_label_index,
                              labels[assigned_label_index].split(' ')[0],
                              labels[assigned_label_index].split(' ')[1]))
            # sort top five predictions from softmax output
            # top_inds = output_prob.argsort()[::-1][:no_of_guesses]  # reverse sort and take five largest items
            top_inds = output_prob.argsort()[::-1][:no_of_guesses]  # reverse sort and take five largest items
            top_labels = [labels[x] for x in top_inds]
            sorted_out_list = [(output_prob[x], labels[x]) for x in top_inds]
            # sorted_out_list = zip(output_prob[top_inds], labels[top_inds])
            for guess in range(no_of_guesses):
                print('{}'.format(sorted_out_list[guess]))
            if check_classification:
                if true_class in [label.split(' ')[0] for label in top_labels]:
                    # ! can change this later so loop over the whole list to see if it is in the top five or not
                    # currently only checks the first position
                    if verbose:
                        print('{} in {}'.format(true_class, sorted_out_list))
                        print('correct!')
                    index_accuracies.append(True)
                else:
                    if verbose:
                        print('incorrect')
                    index_accuracies.append(False)
                if assignIndices == True:
                    assigned_label_indices.append(top_inds[0])
                if assignLabels == True:
                    assigned_labels.append(labels[top_inds][0])
            batch.index_accuracies = index_accuracies
            batch.assigned_labels = assigned_labels
            batch.assigned_label_indices = assigned_label_indices
    # if verbose and check_classification:
    #     print('{} correct for this class out of {}, {}%'.format(len(correct_indices), count,
    #           100*float(len(correct_indices))/count))
    return


def convert_alexnet_to_h5_max(batches, blob_name, h5_out_filename, net, labels):
    ''' Parse the alexnet results and output the maximum activation for each correct match
    '''
    if verbose:
        print('Writing out {}'.format(h5_out_filename))
    image_count = sum([sum(batch.index_accuracies) for batch in batches])
    layer = net.blobs[blob_name]
    if layer.width == 1 and layer.height == 1:
        print("Asking to take maximum of single value?")
        # Switch to single neuron code path
        return convert_alexnet_to_h5_new(batches, blob_name, h5_out_filename, net, labels)

    # makea new activation table
    activation_table = ActivationTable()
    # add in all the activations
    activation_handle = activation_table.add_direct(identifier=blob_name,
                                                    image_count=image_count,
                                                    neuron_count=layer.channels,
                                                    labels=labels,
                                                    neuron_x_count=1,
                                                    neuron_y_count=1)
    # now add in all the activations
    for batch in batches:
        activations = batch.blobs[blob_name]
        for index in range(batch.size):
            if not batch.index_accuracies[index]:
                # It got it wrong :(
                continue
            activation_label = caffe_settings.short_labels[batch.assigned_label_indices[index]]
            if activation_label != batch.label:
                # IF WE HIT HERE, it is broken
                import pdb
                pdb.set_trace()
            assert (activation_label == batch.label)
            activation_values = np.amax(activations[index], (1, 2))
            activation_values = np.resize(activation_values, (activation_values.size, 1, 1))
            activation_handle.add_activation(activation_values,
                                             batch.images[index],
                                             activation_label)
    # write out a nice h5 file
    activation_handle.save_to_hdf5(h5_out_filename, regenerate_labels=False)


def convert_alexnet_to_h5_new(batches, blob_name, h5_out_filename, net, labels):
    ''' Currently only adds correct ones '''
    if verbose:
        print('Writing out {}'.format(h5_out_filename))
    image_count = sum([sum(batch.index_accuracies) for batch in batches])
    layer = net.blobs[blob_name]
    # makea new activation table
    activation_table = ActivationTable()
    # add in all the activations
    activation_handle = activation_table.add_direct(identifier=blob_name,
                                                    image_count=image_count,
                                                    neuron_count=layer.channels,
                                                    labels=labels,
                                                    neuron_x_count=layer.width,
                                                    neuron_y_count=layer.height)
    # now add in all the activations
    for batch in batches:
        activations = batch.blobs[blob_name]
        for index in range(batch.size):
            if not batch.index_accuracies[index]:
                # It got it wrong :(
                continue
            activation_label = caffe_settings.short_labels[batch.assigned_label_indices[index]]
            if activation_label != batch.label:
                # IF WE HIT HERE, it is broken
                import pdb
                pdb.set_trace()
            assert (activation_label == batch.label)
            activation_handle.add_activation(activations[index],
                                             batch.images[index],
                                             activation_label)
    # write out a nice h5 file
    activation_handle.save_to_hdf5(h5_out_filename, regenerate_labels=False)


def convert_alexnet_to_h5_all(batches, blob_name, h5_out_filename, net, labels):
    ''' adds all images!'''
    if verbose:
        print('Writing out {}'.format(h5_out_filename))
    image_count = sum([sum(batch.index_accuracies) for batch in batches])
    layer = net.blobs[blob_name]
    # makea new activation table
    activation_table = kmeans.activation_table.ActivationTable()
    # add in all the activations
    activation_handle = activation_table.add_direct(identifier=blob_name,
                                                    image_count=image_count,
                                                    neuron_count=layer.channels,
                                                    labels=labels,
                                                    neuron_x_count=layer.width,
                                                    neuron_y_count=layer.height)
    # now add in all the activations
    for batch in batches:
        activations = batch.blobs[blob_name]
        for index in range(batch.size):
            # if not batch.index_accuracies[index]:
            # It got it wrong :(
            #    continue
            activation_label = labels[batch.assigned_label_indices[index]]
            # if activation_label != batch.label:
            #    import pdb
            #    pdb.set_trace()
            # assert(activation_label==batch.label)
            activation_handle.add_activation(activations[index],
                                             batch.images[index],
                                             activation_label)
    # write out a nice h5 file
    activation_handle.save_to_hdf5(h5_out_filename, regenerate_labels=False)


def convert_alexnet_to_h5(image_list, blob_name, assigned_labels, indices_list, activations, h5_out_filename, net):
    """Takes in alexnet and picture directories
    image_list: list of image names
    assigned_labels: labels to label each point with
    h5_out_filename:  filename to write out to
    blob_name: name of layer to write activations for
    indices_list: list of indicies to write to file - should relate to imagenet
    """
    if verbose:
        print('Writing out {}'.format(h5_out_filename))
    image_count = len(image_list)
    layer = net.blobs[blob_name]
    batch_size = len(activations[0][blob_name])
    # makea new activation table
    activation_table = kmeans.ActivationTable()
    # add in all the activations
    activation_handle = activation_table.add_direct(identifier=blob_name,
                                                    image_count=image_count,
                                                    neuron_count=layer.channels,
                                                    labels=assigned_labels,
                                                    neuron_x_count=layer.width,
                                                    neuron_y_count=layer.height)
    count = 0
    # now add in all the activations
    for index in indices_list:
        if index >= image_count:
            # caused by the fact we've got an incomplete block at the end which will have odd values.
            continue
        # this is the index of a picture we've analysed
        batch_increment, remainder = divmod(index, batch_size)  # modulo aritmatic FTW!

        activation_handle.add_activation(activations[batch_increment][blob_name][remainder],
                                         image_list[index], assigned_labels[index])
    # activation_handle.add_activation(batch[i]
    # net.blobs[blob_name].data[i], h5_out_filename, assigned_labels[i])
    # write out a nice h5 file
    activation_handle.save_to_hdf5(h5_out_filename)
    # h5_out_filename.close()
    return


# j = 0
# blob_name = blob_list[j]
# h5_out_filename = h5_filename_list[j]


# !!NTS this fails if you only give is one batch!


# good_image_list = [image_list[i] for i in correct_indices]  # good images as a list

# if not os.file.exists(dirname + '/' + image_name.split('.')[0] + '.txt')
#    os.make

# train_filename = os.path.join(dirname, 'train.h5')
# test_filename = os.path.join(dirname, 'test.h5')

# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
# To show this off, we'll list the same data file twice.
# with h5py.File(train_filename, 'w') as f:
#     f['data'] = feat_out
#     f['label'] = 'egg'

# with open(os.path.join(bottleneck_filename), 'w') as f:
#    numpy.savetxt(".txt", a, delimiter=",")
#    f.write(feat_out + '\n')
#    f.write(train_filename + '\n')


# egg=kmeans.ActivationTable
# # egg.add_direct(identifier, image_count, neuron_count, labels)
# # egg.add_direct(0, feat_out, 0, 'meh')
# # egg=kmeans.activation_table.ActivationDirect
# image_count = batch_size # change this later
# activation_table = kmeans.ActivationTable()
# activation_handle = activation_table.add_direct(identifier='test',
#                                                     image_count=image_count,
#                                                     neuron_count=4096,
#                                                     labels=assigned_labels,
#                                                     neuron_x_count=4096,
#                                                 neuron_y_count=1)


def main():
    ####################################################################################################################
    #       Set-up netowkr iwth defaukts from imagent 2012

    # this is hte bit that sets up the caffe networks ------------------------------------------------------------------
    global caffe_settings
    caffe_settings = s.main()
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
    # end of the bit tha sets up the caffe netwroks ---------
    # ###### Change this to select either the correct answers (good only) and take the maximum only
    do_good_only = True
    take_maximum_only = True
    # build the net

    ####################################################################################################################
    #       The main loop that does evreything -- this now just runs the stuff to generate the n***.h5 files!
    ####################################################################################################################
    # dir_list = dir_list[27:]

    import pdb
    for current_index, current_class in enumerate(dir_list):
        current_image_directory = os.path.join(image_directory, current_class)
        print('Running in {} ({}/{})'.format(current_image_directory, current_index, len(dir_list)))
        true_class = current_class
        h5_filename_list = [os.path.join(image_directory, '{}_{}_max.h5'.format(current_class, blob)) for blob in
                            blob_list]
        h5_filename_list = [os.path.join(image_directory, '{}.h5'.format(current_class, blob)) for blob in
                            blob_list]
        if os.path.exists(h5_filename_list[0]):
            print("WARNING: {} already exists, skipping {}".format(h5_filename_list[0], current_class))
            continue
        try:
            batches_of_activations = parse_directory(net=net,
                                                     transformer=transformer,
                                                     image_directory=current_image_directory,
                                                     no_of_images=None,
                                                     batch_size=50,
                                                     verbose=False,
                                                     blob_list=blob_list,
                                                     label=true_class)
        except Exception as exception:
            print('true class = {}'.format(true_class))
            print("ERROR: unable to parse directory {}: {}".format(current_class, exception))
            import traceback, pdb
            # we are NOT in interactive mode, print the exception...
            # traceback.print_exception(type, value, tb)
            # ...then start the debugger in post-mortem mode.
            # pdb.pm() # deprecated
            # pdb.post_mortem(tb) # more "modern"
            continue

        classify_directory(batches=batches_of_activations,
                           labels=labels,
                           check_classification=True,
                           true_class=current_class,
                           no_of_guesses=caffe_settings.no_of_guesses,
                           assignIndices=True,
                           assignLabels=False,
                           verbose=False)

        # image_list = good_image_lis
        ### !!! N.B. only correct images are written out!

        for blob_no in range(len(blob_list)):  # t
            if do_good_only:
                if take_maximum_only:
                    convert_alexnet_to_h5_max(batches=batches_of_activations,
                                              blob_name=blob_list[blob_no],
                                              h5_out_filename=h5_filename_list[blob_no],
                                              net=net,
                                              labels=labels)
                else:
                    convert_alexnet_to_h5_new(batches=batches_of_activations,
                                              blob_name=blob_list[blob_no],
                                              h5_out_filename=h5_filename_list[blob_no],
                                              net=net,
                                              labels=labels)
            if not do_good_only:
                if take_maximum_only:
                    raise NotImplemented("Not written this yet!")
                convert_alexnet_to_h5_all(batches=batches_of_activations,
                                          blob_name=blob_list[blob_no],
                                          h5_out_filename=h5_filename_list[blob_no],
                                          net=net,
                                          labels=short_labels)
            # convert_alexnet_to_h5(image_list=image_list, blob_name=blob_list[blob_no], assigned_labels=assigned_label_indices,
            #             indices_list=correct_indices, activations=batches_of_activations,
            #                     h5_out_filename=h5_filename_list[blob_no], net=net)

    # # the parameters are a list of [weights, biases]
    # filters = net.params['conv1'][0].data
    # d.vis_square(filters.transpose(0, 2, 3, 1))
    #
    # # the parameters are a list of [weights, biases]
    # feat = net.blobs['conv1'].data[0, :36]
    # d.vis_square(feat)
    # plt.savefig('conv1Pic')
    #
    # feat = net.blobs['pool5'].data[0]
    # d.vis_square(feat)
    # plt.savefig('pool5Pic')
    #
    # # output fc6 weights and output bals
    # feat_out = net.blobs['fc6'].data[0]
    # plt.subplot(2, 1, 1)
    # plt.plot(feat_out.flat)
    # plt.subplot(2, 1, 2)
    # _ = plt.hist(feat_out.flat[feat_out.flat > 0], bins=100)
    # plt.savefig('fc6_vals')
    #
    # feat = net.blobs['prob'].data[0]
    # plt.figure(figsize=(15, 3))
    # plt.plot(feat.flat)
    # plt.savefig('output_prob')
    #
    # # Write out the data to HDF5 files in a temp directory.
    # # This file is assumed to be caffe_root/examples/hdf5_classification.ipynb
    # dirname = os.path.abspath(image_directory + '/')
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    #


if __name__ == '__main__':
    main()

# convert_alexnet_to_h5(image_list=good_image_list, blob_name=blob_list[0], assigned_labels=assigned_label_indices, h5_out_filename='out.h5')


# !! nts - this is for only one batch
# filename_list = image_list[0:batch_size]
# for i in range(batch_size):
#     # NTS labels are not currently the n0000 they shoudl be
#     activation_handle.add_activation(net.blobs['fc6'].data[i], filename_list[i], assigned_labels[i])

# activation_handle.get_activation(8).vector

# this saves out the activations
# activation_handle.save_to_hdf5(train_filename)

## ! check

# egg = kmeans.ActivationTable()
# egg_handle = egg.add_file(train_filename)
#
# for filenames, images in load_images(label_path, batch_shape):
#     print('processing batch {}'.format(batch_idx))
#     feat_out = net.blobs['fc6'].data[0]


# # transform it and copy it into the net
# image = caffe.io.load_image('image.jpg')
# net.blobs['data'].data[...] = transformer.preprocess('data', image)
#
# # perform classification
# net.forward()
#
# # obtain the output probabilities
# output_prob = net.blobs['prob'].data[0]
#
# # sort top five predictions from softmax output
# top_inds = output_prob.argsort()[::-1][:5]
#
# plt.imshow(image)
#
# print 'probabilities and labels:'
# zip(output_prob[top_inds], labels[top_inds])
