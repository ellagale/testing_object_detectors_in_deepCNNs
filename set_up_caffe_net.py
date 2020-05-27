from __future__ import print_function

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import os
import getopt
import sys
import caffe


#################################################################################################################
#       USE THIS FILE TO SET UP THE NETWORK ONLY!!
#################################################################################################################

class CaffeSettings:
    no_of_guesses = 1  # sort top five predictions from softmax output
    verbose = True
    noImageName = 1
    image_directory = os.getcwd()
    # image_name='image.jpg'
    # blob_list= ['prob','fc8', 'fc7', 'fc6']# ['prob'] #['fc8','fc6']
    setting = ''
    blob = 'prob'
    labels = []
    short_labels = []
    net = None
    usingDocker = 0
    caffe_root = ''
    labels_file = ''
    model_def = ''
    model_weights = ''
    dir_list = []
    transformer = None
    short_labels = []
    model_file = ''
    this_one_file_name = ''
    file_root = ''
    setting = ''
    deploy_file = 'deploy.prototxt'
    img_size = 227
    blob_list = ['fc8']
    do_limited = False
    forceLabels = False  # whether to force the actiations to be labelled with the correct class


# ------------------------------------------------------------------------------------------------------------------------#
#   This file is the list of things you need to run to completely analyse an instantiation of AlexNet
#   at the moment it is notes, later we can automate it
# -------------------------------------------------------------------------------------------------------------------#
########################################################
#   0.  Command line parameters and settings
########################################################
no_of_guesses = 1  # sort top five predictions from softmax output
verbose = True
noImageName = 1
image_directory = os.getcwd()
# image_name='image.jpg'
# blob_list= ['prob','fc8', 'fc7', 'fc6']# ['prob'] #['fc8','fc6']
setting = ''
blob = 'fc7'
labels = []
short_labels = []
net = None
usingDocker = 1
caffe_root = ''
labels_file = ''
model_def = ''
model_weights = ''
dir_list = []
transformer = None
short_labels = []
model_file = ''
this_one_file_name = ''
file_root = ''
setting = ''


# FUCNTIOSN!
########################################################
#   2   Load net and set-up input preprocessing
########################################################

def set_up_caffe(image_directory='/storage/data/imagenet_2012',
                 model_file='models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                 label_file_address='/storage/data/ilsvrc12/synset_words.txt',
                 dir_file='/storage/data/imagenet_2012_class_list.txt',
                 root_dir='',
                 verbose=True,
                 deploy_file='deploy.prototxt'):
    """stupid little function that sets up all the addresses that you will need with sensible defaults
    run this to get caffe working withuot having to think
    image_directory: root dir for folders of images eg imagenet/
    model_file: .caffemodel file
    label_file_address: .txt of the label names
    dir_file: .txt file of the subdirectories for the class you wish to run
    """
    if root_dir == '':
        caffe_root = os.environ['CAFFE_ROOT']
    else:
        caffe_root = root_dir
    sys.path.insert(0, caffe_root + 'python')
    # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
    # loads caffee and checks there is a model
    if os.path.isfile(os.path.join(caffe_root, model_file)):
        print('CaffeNet found.')
    else:
        print('Model not found!')
    # load ImageNet labels
    labels_file = os.path.join(caffe_root, label_file_address)
    if not os.path.exists(labels_file):
        print('You need to run get_ilsvrc_aux.sh')
        print('Look here ---> / data / ilsvrc12 / get_ilsvrc_aux.sh')
        # !../ data / ilsvrc12 / get_ilsvrc_aux.sh
    if verbose:
        print('Using directories from {}'.format(dir_file))
    dir_list = np.loadtxt(dir_file, str, delimiter='\t')
    if verbose:
        print('Using labels from {}'.format(labels_file))
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    caffe.set_device(0)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()
    # caffe.set_mode_cpu()
    model_dir = '/'.join(model_file.split('/')[:-1])
    model_def = os.path.join(caffe_root, model_dir, deploy_file)
    if verbose:
        print('Using {} as model def'.format(model_def))
    model_weights = os.path.join(caffe_root, model_file)
    return caffe_root, image_directory, labels_file, model_def, model_weights, dir_list, labels


def Caffe_NN_setup(model_def,
                   model_weights,
                   imangenet_mean_image='python/caffe/imagenet/ilsvrc_2012_mean.npy',
                   batch_size=50,
                   verbose=False,
                   root_dir='',
                   img_size=227
                   ):
    """
    function to set up a standard Caffe_NN_setup and do imagenet
    imagenet_mean_image: address of npy array of means to subtract
    batch_size=50: batchsize for the nnw
    """
    if root_dir == '':
        caffe_root = os.environ['CAFFE_ROOT']
    else:
        caffe_root = root_dir
    print('Setting up NN')
    ## this sets up the NN
    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)
    if verbose:
        # for each layer, show the output shape
        for layer_name, blob in net.blobs.items():
            print
            layer_name + '\t' + str(blob.data.shape)
        for layer_name, param in net.params.items():
            print
            try:
                layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
            except IndexError:
                import pdb
                pdb.set_trace()
    # Set up input preprocessing. (We'll use Caffe's caffe.io.Transformer to do this,
    ## but this step is independent of other parts of Caffe, so any custom preprocessing code may be used).
    # Our default CaffeNet is configured to take images in BGR format. Values are expected to start in the range [0, 255]
    ## and then have the mean ImageNet pixel value subtracted from them. In addition, the channel dimension is expected
    ## as the first (outermost) dimension.
    # As matplotlib will load images with values in the range [0, 1] in RGB format with the channel as the
    # innermost dimension, we are arranging for the needed transformations here.
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(os.path.join(caffe_root, imangenet_mean_image))
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print('{}'.format(mu))
    # print('mean-subtracted values:{}', zip('BGR', mu))
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    net.blobs['data'].reshape(batch_size,  # batch size
                              3,  # 3-channel (BGR) images
                              img_size, img_size)  # image size is 227x227
    return net, transformer


def Get_Model_File(Flag=''):
    """Stupid little function to make sure you got the correct settings"""
    if Flag == 'AlexNet_standard':
        model_file = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'  # standard (L2 reg)
    elif Flag == 'L1_AlexNet':
        model_file = '/storage/models/L1/0602_caffenet_train_iter_733955.caffemodel'  # L1 reg
    elif Flag == 'no_reg_AlexNet':
        model_file = '/storage/models/no_reg/0702_no_ref_caffenet_train_iter_508100.caffemodel'  # no reg
    else:
        print('Error! You must define a model file')
        print('setting AlexNet default')
        model_file = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    print('Using model file: {}'.format(model_file))
    return model_file


def set_settings(setting, blob):
    """wrapper function"""
    model_file = ''
    this_one_file_name = ''
    if setting == 'AN':
        model_file = Get_Model_File('AlexNet_standard')
    elif setting == 'L1':
        model_file = Get_Model_File('L1_AlexNet')
    elif setting == 'NR':
        model_file = Get_Model_File('no_reg_AlexNet')
    ## now set hte merged filename
    if setting == 'AN':
        if blob == 'fc6':
            this_one_file_name = 'AN_merged_fc6_max.h5'
        elif blob == 'fc7':
            this_one_file_name = 'AN_merged_fc7_max.h5'
        elif blob == 'fc8':
            this_one_file_name = 'AN_merged_fc8_max.h5'
        elif blob == 'prob':
            this_one_file_name = 'AN_merged_prob_max.h5'
        elif blob == 'conv5':
            this_one_file_name = 'AN_merged_conv5_max.h5'
    elif setting == 'L1':
        if blob == 'fc6':
            this_one_file_name = 'L1_merged_fc6L1.h5'
        elif blob == 'fc7':
            this_one_file_name = '0602_L1_reg_merged_fc7.h5'
        elif blob == 'fc8':
            this_one_file_name = '0602_L1_reg_merged_fc8.h5'
        elif blob == 'prob':
            this_one_file_name = '0602_L1_reg_merged_prob.h5'
    elif setting == 'NR':
        if blob == 'fc6':
            this_one_file_name = '0702_no_reg_mergedfc6.h5'
        elif blob == 'fc7':
            this_one_file_name = '0702_no_reg_mergedfc7.h5'
        elif blob == 'fc8':
            this_one_file_name = '0702_no_reg_mergedfc8.h5'
        elif blob == 'prob':
            this_one_file_name = '0702_no_reg_mergedprob.h5'
    return model_file, this_one_file_name


# run Caffe_AlexNet.py
# change imagenet photo directory
# change the model file

# run merger.py [from the directory with the .h5s in it] for a layer at a time
# change the inputs into merge function with te

# to get results change Make_activation .py to use the correct file and then run h5_jitterere


def main():
    ####################################################################################################################
    # all SETTINGS! CHANGE THIS BIT!
    ####################################################################################################################
    # set-up with 2012 defaults
    # if using docker, you want the defaults if not use:`
    # model_file = '/storage/models/L1/0602_caffenet_train_iter_733955.caffemodel' # L1 reg
    # model_file = '/storage/models/no_reg/0702_no_ref_caffenet_train_iter_508100.caffemodel' # no reg
    # model_file = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel' # standard (L2 reg) AlexNet_standard
    # r.image_directory='/storage/data/top_1_imagenet_2012'
    r = CaffeSettings()
    # r.image_directory = '/storage/data/0602_L1_reg_top_1_imagenet_2012'
    # r.image_directory = '/storage/data/0702_no_reg_top_1_imagenet_2012'#'/storage/data/imagenet_2012'
    r.image_directory = '/storage/data/AN_top_1_imagenet_2012'  # '/storage/data/imagenet_2012'
    r.image_directory = '/storage/data/imagenet_2012'
    model_file = Get_Model_File('AlexNet_standard')
    # model_file = Get_Model_File('L1_AlexNet')
    do_by_hand = True
    if do_by_hand:
        # we're not using a standard model file or standard
        print('running do_by_hand in set_up_caffe')
        r.model_file = Get_Model_File('AlexNet_standard')
        # r.this_one_file_name = '0702_no_reg_mergedfc8.h5'
        # r.model_file=Get_Model_File('AlexNet_standard')
        # r.this_one_file_name= '0905_AN_merged_fc8.h5's
        # r.this_one_file_name= 'AN_merged_conv5_max.h5'
        # r.this_one_file_name = 'L1_merged_fc6L1.h5'
        r.this_one_file_name = 'AN_merged_fc8_max.h5'  # 'AN_merged_conv5_max.h5'#'AN_merged_fc6_max.h5'#'AN_merged_fc8_all.h5'
        # r.this_one_file_name = '/storage/data/0602_L1_reg_top_1_imagenet_2012'
        # r.model_file=Get_Model_File('L1_AlexNet')
        # r.this_one_file_name = '0702_no_reg_merged_oliprob.h5'
    else:
        r.setting = 'AN'  # AN for AlexNet, NR for no reg, L1 for L1
        r.blob = 'fc6'
        r.model_file, r.this_one_file_name = set_settings(setting, blob)
    # this hte address of the merged.h5 if you have it
    r.file_root = '/storage/data/AlexNet_Merged/'
    # r.file_root = '/storage/data/imagenet_2012/'

    ####################################################################################################################

    if usingDocker:
        # new set-up with safer deployment for use on all machines
        r.caffe_root, r.image_directory, r.labels_file, r.model_def, r.model_weights, r.dir_list, r.labels = set_up_caffe(
            image_directory=r.image_directory,
            model_file=r.model_file
        )
        r.net, r.transformer = Caffe_NN_setup(imangenet_mean_image='python/caffe/imagenet/ilsvrc_2012_mean.npy',
                                              batch_size=50, model_def=r.model_def, model_weights=r.model_weights,
                                              verbose=True, root_dir=r.caffe_root, img_size=r.img_size)
    else:
        # old set-up with hardcoded links and old-style unsafe deployment
        r.caffe_root, r.image_directory, r.labels_file, r.model_def, r.model_weights, r.dir_list, r.labels = \
            set_up_caffe(image_directory=r.image_directory,
                         model_file=r.model_file,
                         label_file_address='data/ilsvrc12/synset_words.txt',
                         dir_file='/storage/data/imagenet_2012_class_list.txt',
                         root_dir='/home/eg16993/src/caffe', verbose=True)
        r.net, r.transformer = Caffe_NN_setup(
            imangenet_mean_image='/home/eg16993/src/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy',
            batch_size=50, model_def=r.model_def, model_weights=r.model_weights,
            verbose=True, root_dir=r.caffe_root)
    r.short_labels = [label.split(' ')[0] for label in r.labels]

    print('Running in image_directory: {}'.format(r.image_directory))
    print('Using this model file: {}'.format(r.model_file))
    return r


if __name__ == '__main__':
    main()
