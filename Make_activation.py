########################################################################################################################
#       Make a giant activation files from smaller files (like those from an imagenet)
########################################################################################################################



import os
import numpy as np
import matplotlib.pyplot as plt
import kmeans
import kmeans.activation_table
import set_up_caffe_net as s


caffe_settings = None

#image_name='image.jpg'
#blob_list= ['prob','fc8', 'fc7', 'fc6']# ['prob'] #['fc8','fc6']


#settings
#verbose = True
do_check = False
#usingDocker = False


def combine_h5_files_in_activation_table(h5_file_location='/storage/data/Temp_ImageNet_Test/',
                                         h5_list_filename='h5_list.txt', h5_list=None, useFile=True, verbose=True,
                                         test=False):
    """
    Combines several h5 files into a single activation table for the win
    h5_file_location: folder containing the h5 files
    h5_list_filename: the filename of a list of h5 files to analyse
    useFile: whether to use a file full of h5 names -- could expand to include h5 list as an input
    h5List: feed this in if you dont want to make files
    """
    # TO-DO possibly sort the h5_list_filename bit so it reads in the h5 files instead or perahps as a switch
    if useFile:
        h5_list_file=os.path.join(h5_file_location, h5_list_filename)
        if verbose:
            print('Using directories from {}'.format(h5_list_file))
        h5_list = np.loadtxt(h5_list_file, str, delimiter='\t')
    else:
        # I think you could feed in the h5_list instead
        pass
    # this chunk loads the acitvations into an acativation table
    if test:
        acts = kmeans.test_activation_table.TestActivationTable(mean=False)
    else:
        acts = kmeans.activation_table.ActivationTable(mean=False)
    for file in h5_list:
        if verbose:
            print('adding file {}'.format(file))
        acts.add_file(os.path.join(h5_file_location, file))
    # training_files = [os.path.join(image_directory, '{}_fc8.h5'.format(x)) for x in class_list]
    return acts, h5_list

def combine_h5_files_in_activation_table(h5_file_location='/storage/data/Temp_ImageNet_Test/',
                                         h5_list_filename='h5_list.txt', h5_list=None, useFile=True, verbose=True,
                                         test=False):
    """
    Combines several h5 files into a single activation table for the win
    h5_file_location: folder containing the h5 files
    h5_list_filename: the filename of a list of h5 files to analyse
    useFile: whether to use a file full of h5 names -- could expand to include h5 list as an input
    h5List: feed this in if you dont want to make files
    """
    # TO-DO possibly sort the h5_list_filename bit so it reads in the h5 files instead or perahps as a switch
    if useFile:
        h5_list_file=os.path.join(h5_file_location, h5_list_filename)
        if verbose:
            print('Using directories from {}'.format(h5_list_file))
        h5_list = np.loadtxt(h5_list_file, str, delimiter='\t')
    else:
        # I think you could feed in the h5_list instead
        pass
    # this chunk loads the acitvations into an acativation table
    if test:
        acts = kmeans.test_activation_table.TestActivationTable(mean=False)
    else:
        acts = kmeans.activation_table.ActivationTable(mean=False)
    for file in h5_list:
        if verbose:
            print('adding file {}'.format(file))
        acts.add_file(os.path.join(h5_file_location, file))
    # training_files = [os.path.join(image_directory, '{}_fc8.h5'.format(x)) for x in class_list]
    return acts, h5_list


# def combine_h5_files_in_activation_table(h5_file_location='/storage/data/Temp_ImageNet_Test/',
#                                         h5_list_filename='h5_list.txt', h5_list=None, useFile=True, verbose=True):
#     """
#     Combines several h5 files into a single activation table for the win
#     h5_file_location: folder containing the h5 files
#     h5_list_filename: the filename of a list of h5 files to analyse
#     useFile: whether to use a file full of h5 names -- could expand to include h5 list as an input
#     h5List: feed this in if you dont want to make files
#     """
#     # TO-DO possibly sort the h5_list_filename bit so it reads in the h5 files instead or perahps as a switch
#     if useFile == True:
#         h5_list_file=h5_file_location + h5_list_filename
#         if verbose:
#             print('Using directories from {}'.format(h5_list_file))
#         h5_list = np.loadtxt(h5_list_file, str, delimiter='\t')
#     else:
#         # I think you could feed in the h5_list instead
#         pass
#     ## this chunk loads the acitvations into an acativation table
#     acts = kmeans.ActivationTable()
#     for file in h5_list:
#         if verbose:
#             print('adding file {}'.format(file))
#         acts.add_file(h5_file_location + file)
#     #training_files = [os.path.join(image_directory, '{}_fc8.h5'.format(x)) for x in class_list]
#     return acts, h5_list

def check_labels(acts, class_labels):
        # this checks that our labels are assigned properly
        # this only makes sense if the folders were labelled
        print('point: \t assigned label')
        for current_point in acts.get_all_activation_indices():
            # this code assumes that te points are put in in files named by the class!
            #assigned_label = class_labels[acts.get_activation(current_point).label].split(' ')[0]
            point = acts.get_activation(current_point)
            assigned_label = point.label
            label_from_h5 = point.index[0].split('_')[0]
            count = 0
            if not assigned_label == label_from_h5:
                print('{}, \t {}'.format(acts.get_activation(current_point),
                                         class_labels[acts.get_activation(current_point).label]))
                count = count + 1
        if count == 0:
            print('Labels are correct: check passed!')
        else:
            print('Check failed: {} labels are incorrect'.format(count))


def main():
    print('Make_activation')
    global acts, class_labels, h5_list, caffe_settings
    # class_labels is the imagenet labels for 2012, both human readable and n909402394
    image_directory= '/storage/data/imagenet_2012' #'/storage/data/0602_L1_reg_top_1_imagenet_2012'
    # set up caffe default
    #image_directory = '/storage/data/top_1_imagenet_2012/'


    # this is hte bit that sets up the caffe networks ------------------------------------------------------------------
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
    model_file = caffe_settings.model_file
    this_one_file_name = caffe_settings.this_one_file_name
    blob = caffe_settings.blob
    file_root=caffe_settings.file_root
    class_labels = short_labels
    # end of the bit tha sets up the caffe netwroks --------------------------------------------------------------------
    print('I am using the following merged h5 file: {}'.format(this_one_file_name))
    print('Which I expect to be located at: {}'.format(file_root))

    h5_list_filename=''
    # #/storage/data/imagenet_2012/test_class_list.txt
    option = 'merged' #'doAllClasses'#'merged'#doAllClasses'#'merged' # 'merged'#''doFewClasses'#'doTwoClasses'#'doFewClasses'#'doAllClasses'
    layer_option = 'this_one'
    h5_list_filename = 'h5_list.txt'
    if option == 'doTwoClasses':
        acts, h5_list = combine_h5_files_in_activation_table(h5_file_location='/storage/data/Temp_ImageNet_Test/',
                                                             h5_list_filename='h5_small_list.txt', h5_list=[],
                                                             useFile=True, verbose=True)
    elif option == 'doFewClasses':
        acts, h5_list = combine_h5_files_in_activation_table(h5_file_location='/storage/data/imagenet_2012/',
                                                             h5_list_filename='few_classes_h5_list.txt', h5_list=[], useFile=True,
                                                             verbose=True)
    elif option == 'doAllClasses':
        # actually does 993 calsses
        acts, h5_list = combine_h5_files_in_activation_table(h5_file_location=file_root,
                                                             h5_list_filename=h5_list_filename, h5_list=[], useFile=True,
                                                             verbose=True)
    elif option == 'prob':
        # actually does classes
        acts, h5_list = combine_h5_files_in_activation_table(h5_file_location='/storage/data/imagenet_2012/',
                                                             h5_list_filename='prob.txt', h5_list=[], useFile=True,
                                                             verbose=True)
    elif option == 'few_prob':
        # actually does 20 classes
        acts, h5_list = combine_h5_files_in_activation_table(h5_file_location='/storage/data/imagenet_2012/',
                                                             h5_list_filename='few_prob.txt', h5_list=[], useFile=True,
                                                             verbose=True)
    elif option == 'merged':
        acts = kmeans.activation_table.ActivationTable(mean=False)
        if layer_option == 'fc6_softmaxed':
            acts.add_merged_file('/storage/data/imagenet_2012/h5_files/merged_fc7_softmax.h5')
        elif layer_option == 'this_one':
                acts.add_merged_file(file_root + this_one_file_name)
        elif layer_option == 'fc6':
            acts.add_merged_file('/storage/data/imagenet_2012/h5_files/L1_merged_fc6.h5')
        elif layer_option == 'fc8':
            acts.add_merged_file('/storage/data/imagenet_2012/h5_files/0905_AN_merged_fc8.h5')
        elif layer_option == 'fc8_softmaxed':
            acts.add_merged_file('/storage/data/imagenet_2012/h5_files/merged_fc8_softmax.h5')
        elif layer_option == 'homemade_prob':
            acts.add_merged_file('/storage/data/imagenet_2012/h5_files/merged_fc8_prob.h5')
        elif layer_option == 'prob':
            acts.add_merged_file(file_root + 'AlexNet_merged_prob.h5')

    h5_out_location = '/storage/data/'
    h5_out_filename = 'test.h5'

    print('File loaded options: {}, {}'.format(option, layer_option))
    if do_check:
        check_labels(acts, class_labels)

    #


    # this gives the new indices of a bathc o



if __name__ == '__main__':
    main()
#
# acts.get_activation(acts.to_local_indices([2055])[0])
# acts.get_activation(acts.to_local_indices([5])[0]).index[1]
#
# acts.get_activations_for_neuron(10)[0][0][5]
# 9.2300749
# acts.get_activation(acts.to_local_indices([5])[0]).vector[10]
# 9.2300749

