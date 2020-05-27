
#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})
import h5_analysis_jitterer as h
from collections import Counter
import operator
import numpy as np

import sys
import csv
import Make_activation as m
from collections import OrderedDict
from h5_analysis_jitterer import grab_points_for_a_cluster

do_check = False

###############################################################
# script to get precision data
###############################################################
global acts, class_labels, labels, h5_list, caffe_settings

nzeros, pzeros, num_zeros = [],[],0
current_neuron_index, test_class_list, output_precs_data = 0,[],[]
no_files_in_label = 0
label_dict={}
acts=[]
found_labels=[]
class_labels=[]
local_list, x_data =[],[]
class_dict = []
verbose=True
test_class =''
no_of_images = 0
top_mode_class=''
zhou_precs_class60=''
zhou_no_of_classes100=0
zhou_precs60=0
filelistfile='conv5.csv'

###############################################################
#
# FUNCTIONS
##############################################################

def calculate_mean_average_precision(class_name='', current_neuron_index=current_neuron_index, acts=acts, verbose=verbose, minx=0.000000001):
    """Counts down using -1 to min list length+1 indexing and calculates the mean average precision
    given by: M.A.P. = sum over j of no. of A so far found at position j divided by the position in the list
     Note that we are counting backwards, the formula expects a 1-indexed list and j counts position over a 1-indexed list
     Note, we assume if no class name is given you want the class for the highest activation
     minx = the minimum activation to consider, this must be above 0.0 as these all have the same value"""
    #
    current_neuron = acts.get_activations_for_neuron(current_neuron_index) # get the neuron's data
    x_data = current_neuron.vector # get the activations without classes
    # grab your list of points
    local_list, selected_activations = grab_points_for_a_cluster(current_neuron_index,
                                                                                min_selected_x_data=minx,
                                                                                max_selected_x_data=max(x_data),
                                                                                acts=acts,
                                                                                x_data=x_data,
                                                                                verbose=verbose)
    Q = len(local_list)     # total length of list
    # get the test class (this is the correct class or 'A')
    if class_name == '':
        test_class = local_list[-1][0]
    else:
        test_class = class_name
    # set up counters
    MAP = 0 # mean average precision
    count_of_test_class = 0
    # loop backwards through the list, abs j is the position in a 1-indexed list
    for i in range(Q+1):
        j = -(i + 1)  # 1 indexed
        current_class = local_list[j][0] # current class
        if j == -Q:
            # if the whole of local_list is the same class (this accounts for zero indexing)
            if verbose:
                print(current_class)
                print('{}/{}'.format(count_of_test_class, abs(j)))
            j = j -1 # really this is here so we can check j
            break
        if (current_class == test_class):
            count_of_test_class = count_of_test_class + 1
            MAP = MAP + count_of_test_class/(abs(j)) # N.b. this is the sum, we divide by j on the output
    return MAP/Q


def calculate_average_precision_top_classes(local_list=local_list,
                                            class_names='',
                                            no_files_in_label=no_files_in_label,
                                            Q_stop='',
                                            no_of_points_for_zhou_precs=100,
                                            verbose=verbose):
    """Calcs ave precsision for all classes in top 100 activations
    Counts down using -1 to min list length+1 indexing and calculates the mean average precision
    given by: M.A.P. = sum over j of no. of A so far found at position j divided by the position in the list
     Note that we are counting backwards, the formula expects a 1-indexed list and j counts position over a 1-indexed list
     Note, we assume if no class name is given you want the class for the highest activation
     minx = the minimum activation to consider, this must be above 0.0 as these all have the same value
     Q_stop = no. of points back from the end to consider, i.e. 100 does the last 100, not this overrides xmin
     no_of_points = the number of points at the top of hte activation range to consider """
    #
    # get the test class (this is the correct class or 'A')
    if class_names == '':
        top_mode_class, precs_of_tmc, no_of_classes_in_top_x, list_of_top_classes = find_zhou_precision(
            number_of_points=no_of_points_for_zhou_precs, local_list=local_list)
        # n.b.list_of_top_classes is not sorted on magnitude of items
        # test_class_list is sorted that way
        test_class_list=[x[0] for x in list_of_top_classes.most_common()]
    else:
        if class_names == str:
            print('Input test_class is not a list of classes! You probably wanted to use the function\n'
                  'calculate_average_precision, not calculate_average_precision_top_class!\n')
        else:
            test_class_list = class_name
            top_mode_class, precs_of_tmc, no_of_classes_in_top_x, list_of_top_classes = 0,0,0,0
    Ave_prec_x_list = []
    precs_x_list = []
    recall_x_list = []
    for test_class in test_class_list:
        Ave_precs_x, precs_x, recall_x = calculate_ave_precs_general(test_class=test_class,
                                    local_list=local_list,
                                    Q_stop=Q_stop,
                                    no_files_in_label=no_files_in_label,
                                    verbose=verbose)
        # have done all Q points now
        Ave_prec_x_list.append(Ave_precs_x)
        precs_x_list.append(precs_x)
        recall_x_list.append(recall_x)
    out = (top_mode_class, precs_of_tmc, no_of_classes_in_top_x,
           list_of_top_classes, Ave_prec_x_list, precs_x_list,
           recall_x_list, test_class_list)
    return out

def find_extent_of_top_class(local_list=local_list):
    """starts at the highest index and counts down until the classes no longer match"""
    for i in range(len(local_list) + 1):
        j = -(i+1)  # 1 indexed
        if j == -1:
            test_class = local_list[j][0]
        current_class = local_list[j][0]
        if j == -len(local_list):
            # if the whole of local_list is the same class 9(this accounts for zero indexing)
            j = j -1
            break
        if not (current_class == test_class):
            #print(j)
            break
    no_of_members_of_test_class = -j - 1
    return no_of_members_of_test_class

def get_local_list_for_neuron(current_neuron_index=current_neuron_index,
                              minx='',
                              maxx='',
                              acts=acts):
    """Little function to get the local_list and selected activations from minx x to max_x for a neuron
    N.B. use minx=0 if you want the points ABOVE 0.0
    use minx=0.0 if you want the points equal to and above 0.0"""
    current_neuron = acts.get_activations_for_neuron(current_neuron_index)  # get the neuron's data
    x_data = current_neuron.vector  # get the activations without classes
    if minx == '':
        minx = min(x_data)  # this grabs all the points
    elif minx == 0:
        minx = min([x for x in x_data if x > 0.0])
        # grab everything above 0 - special case, we don't include 0
        # this changed grab_points_for_cluster from x >= 0 to x > 0
    if maxx == '':
        maxx =max(x_data)
    # grab your list of points
    local_list, selected_activations = grab_points_for_a_cluster(current_neuron_index,
                                                                 min_selected_x_data=minx,
                                                                 max_selected_x_data=maxx,
                                                                 acts=acts,
                                                                 x_data=x_data,
                                                                 verbose=verbose)
    return local_list, selected_activations, x_data

def calculate_average_precision(class_name='', current_neuron_index=current_neuron_index, acts=acts,
                                no_files_in_label=no_files_in_label, verbose=verbose, minx='',Q_stop=''):
    """Counts down using -1 to min list length+1 indexing and calculates the mean average precision
    given by: M.A.P. = sum over j of no. of A so far found at position j divided by the position in the list
     Note that we are counting backwards, the formula expects a 1-indexed list and j counts position over a 1-indexed list
     Note, we assume if no class name is given you want the class for the highest activation
     minx = the minimum activation to consider, this must be above 0.0 as these all have the same value
     Q_stop = no. of points back from the end to consider, i.e. 100 does the last 100, not this overrides xmin"""
    #
    current_neuron = acts.get_activations_for_neuron(current_neuron_index) # get the neuron's data
    x_data = current_neuron.vector # get the activations without classes
    if minx == '':
        minx = min(x_data)  # this grabs all the points
    # grab your list of points
    local_list, selected_activations = grab_points_for_a_cluster(current_neuron_index,
                                                                                min_selected_x_data=minx,
                                                                                max_selected_x_data=max(x_data),
                                                                                acts=acts,
                                                                                x_data=x_data,
                                                                                verbose=verbose)
    if not Q_stop == '':
        Q = Q_stop
    else:
        Q = len(local_list)     # total length of list
    # get the test class (this is the correct class or 'A')
    if class_name == '':
        test_class = local_list[-1][0]
    else:
        test_class = class_name
    N_test = no_files_in_label[test_class] # no of items in class A
    # set up counters
    AP = 0 # average precision
    count_of_test_class = 0
    # loop backwards through the list, abs j is the position in a 1-indexed list
    # values for i == -1
#    current_class = local_list[-1][0]
#    if (current_class == test_class):
#        count_of_test_class = count_of_test_class + 1 # we found A
#    precs_x = count_of_test_class /1
    recall_x = 0
    Ave_precs_x = 0
    for i in range(Q):
        j = -(i + 1)  # 1 indexed
        recall_x_minus_1 = recall_x
        current_class = local_list[j][0] # current class
        if j == -Q:
            # if the whole of local_list is the same class (this accounts for zero indexing)
            if verbose:
                print(current_class)
                print('{}/{}'.format(count_of_test_class, abs(j)))
            j = j -1 # really this is here so we can check j
            #break
        if count_of_test_class == N_test:
            #we've found them all
            if verbose:
                print('found all {} of {}, stopping...'.format(N_test, current_class))
                print('{}/{}'.format(count_of_test_class, abs(j)))
            break
        if (current_class == test_class):
            count_of_test_class = count_of_test_class + 1   #n A
        precs_x = count_of_test_class /(abs(j)) # N.b. this is the sum, we divide by j on the output
        recall_x = count_of_test_class / N_test
        delta_recall_x = recall_x - recall_x_minus_1    # difference in recall between this point nd the next
        weight_precs_x = precs_x * delta_recall_x        # weighted precsion at point x (we do average via weighted sum)
        Ave_precs_x = Ave_precs_x + weight_precs_x             # average_precision evaluated at point x
    return Ave_precs_x, precs_x, recall_x


def calculate_ave_precs_general(test_class=test_class,
                                local_list=local_list,
                                Q_stop='',
                                no_files_in_label=no_files_in_label,
                                verbose=verbose):
    """Internal function to calculate the average precision of a local_list
    current_class = name of the class we're taking as A
    local_list = sorted list of tuples containing the class and index of each point
    Q_stop = number at which to stop evaluating the ave precs
    no_files_in_label = cell array of labels to number of items in that label
    verbose = whether to print data to screen"""
    if not Q_stop == '':
        Q = Q_stop
    else:
        Q = len(local_list)
        # does all points
    if Q > len(local_list):
        Q = len(local_list)
        if verbose:
            print('The number of points to check (Q) {} is more than the number of points above minx {}'.format(Q, len(local_list)))
            print('Setting Q to len(local_list), you may wish to change your settings')
    N_test = no_files_in_label[test_class] # no of items in class A
        # set up counters
    AP = 0 # average precision
    count_of_test_class = 0
        # loop backwards through the list, abs j is the position in a 1-indexed list
    recall_x = 0
    Ave_precs_x = 0
    for i in range(Q):
        j = -(i + 1)  # 1 indexed
        recall_x_minus_1 = recall_x
        current_class = local_list[j][0]
            #j = j - 1  # really this is here so we can check j
            # break
        if count_of_test_class == N_test:
            # we've found them all
            if verbose:
                print('found all {} of {}, stopping...'.format(N_test, current_class))
                print('{}/{}'.format(count_of_test_class, abs(j)))
            break
        if (current_class == test_class):
            count_of_test_class = count_of_test_class + 1  # n A
        precs_x = count_of_test_class / (abs(j))  # N.b. this is the sum, we divide by j on the output
        recall_x = count_of_test_class / N_test
        delta_recall_x = recall_x - recall_x_minus_1  # difference in recall between this point nd the next
        weight_precs_x = precs_x * delta_recall_x  # weighted precsion at point x (we do average via weighted sum)
        Ave_precs_x = Ave_precs_x + weight_precs_x  # average_precision evaluated at point x
    if verbose:
        print('Class {}, evaluated at Q={}:, found {}'.format(test_class, Q, count_of_test_class))
        print('Average precision={}; precision={}; recall={}'.format(Ave_precs_x,precs_x,recall_x))
    return Ave_precs_x, precs_x, recall_x


def calculate_many_precs_recall_stats(test_class=test_class,
                                local_list=local_list,
                                Q_stop='',
                                no_files_in_label=no_files_in_label,
                                no_of_images=no_of_images,
                                verbose=verbose):
    """Function to calculate the average precision of a local_list
    current_class = name of the class we're taking as A
    local_list = sorted list of tuples containing the class and index of each point
    Q_stop = number at which to stop evaluating the ave precs
    no_files_in_label = cell array of labels to number of items in that label
    verbose = whether to print data to screen"""
    if not Q_stop == '':
        Q = Q_stop
    else:
        Q = len(local_list)
        # does all points
    if Q > len(local_list):
        Q = len(local_list)
        if verbose:
            print('The number of points to check (Q) {} is more than the number of points above minx {}'.format(Q, len(local_list)))
            print('Setting Q to len(local_list), you may wish to change your settings')
    N_test = no_files_in_label[test_class] # no of items in class A
    N_not_test = no_of_images - N_test # i.e. set not-A
        # set up counters
    AP = 0 # average precision
    found_recall_precs_p95 = False
    recall_p95 = 0.0
    count_of_test_class = 0
        # loop backwards through the list, abs j is the position in a 1-indexed list
    recall_x = 0 # n.b. this is also sensitivity
    Ave_precs_x = 0
    specificity_x = 0
    count_of_false_positives = 0 # number of not-A on misidentified
    max_informedness = 0
    x_for_max_informedness = 0
    max_f1_stat = 0
    x_for_max_f1 = 0
    recall_for_max_informedness = 0
    specificity_for_max_informedness = 0
    for i in range(Q):
        j = -(i + 1)  # 1 indexed
        recall_x_minus_1 = recall_x
        current_class = local_list[j][0]
            #j = j - 1  # really this is here so we can check j
            # break
        if count_of_test_class == N_test:
            # we've found them all
            if verbose:
                print('found all {} of {}, stopping...'.format(N_test, current_class))
                print('{}/{}'.format(count_of_test_class, abs(j)))
            break
        if (current_class == test_class):
            count_of_test_class = count_of_test_class + 1  # n A
        else:
            count_of_false_positives = count_of_false_positives + 1
        precs_x = count_of_test_class / (abs(j))  # N.b. this is the sum, we divide by j on the output
        recall_x = count_of_test_class / N_test
        false_pos_rate = count_of_false_positives / no_of_images
        specificity_x = 1-false_pos_rate
        if (precs_x <= .95) and (found_recall_precs_p95 is False):
            # thingy to grab the recall when precision drops below .95 (if ever)
            recall_p95 = recall_x
            found_recall_precs_p95 = True
        delta_recall_x = recall_x - recall_x_minus_1  # difference in recall between this point nd the next
        weight_precs_x = precs_x * delta_recall_x  # weighted precsion at point x (we do average via weighted sum)
        Ave_precs_x = Ave_precs_x + weight_precs_x  # average_precision evaluated at point x
        informedness_x = recall_x + specificity_x -1
        if informedness_x > max_informedness:
            max_informedness = informedness_x
            x_for_max_informedness = abs(j)
            recall_for_max_informedness = recall_x
            specificity_for_max_informedness = specificity_x
        if (precs_x > 0 and recall_x > 0):
            f1_x = 2*(precs_x*recall_x) / (precs_x + recall_x)
        else:
            f1_x = 0
        if f1_x > max_f1_stat:
            max_f1_stat = f1_x
            x_for_max_f1 = abs(j)
    if verbose:
        print('Class {}, evaluated at Q={}:, found {}'.format(test_class, Q, count_of_test_class))
        print('Average precision={}; precision={}; recall={}'.format(Ave_precs_x,precs_x,recall_x))
        print('Recall at precision .95 threashold is {}'.format(recall_p95))
        print('specificity = {}'.format(specificity_x))
        print('informedness = {}'.format(informedness_x))
        print('Max informedness was {}, seen at x {}'.format(max_informedness, x_for_max_informedness))
        print('F1 at x={} is {}'.format(abs(j),f1_x))
        print('Max F1 was {}, seen at x = {}'.format(max_f1_stat,x_for_max_f1))
    out = (Ave_precs_x, precs_x, recall_x, recall_p95,
           specificity_x, informedness_x, max_informedness, x_for_max_informedness,
           max_f1_stat, x_for_max_f1,
           recall_for_max_informedness, specificity_for_max_informedness,)
    return out





def calculate_average_precision_incl_zeros(test_class, local_list='', x_data='', selected_activations='',
                                           current_neuron_index=current_neuron_index,acts=acts,verbose=verbose):
    """Wrapper function to calculate the average precision when we have lots of zeros"""
    if local_list == '':
        # if you don't pass it in, get the data
        local_list, selected_activations, x_data = get_local_list_for_neuron(current_neuron_index=current_neuron_index,
                                                                          minx=0,
                                                                          acts=acts)
    total_no_of_points = len(x_data)
    no_of_points_selected = len(local_list)
    # grabs all points above 0 (special case)
    if verbose:
        print('Taking all points above 0.0, minimum is {}'.format(min(selected_activations)))
    Ave_precs_x, precs_x, recall_x = calculate_ave_precs_general(
            test_class=test_class,
            local_list=local_list,
            Q_stop='',
            no_files_in_label=no_files_in_label,
            verbose=verbose)
    # this gives the values just above zero
    precs_all = no_files_in_label[test_class] / total_no_of_points #  a largely pointless measure
    recall_all = 1.0
    delta_recall_all = recall_all - recall_x  # difference in recall between this point nd the next
    weight_precs_all = precs_all * delta_recall_all  # weighted precsion at point x (we do average via weighted sum)
    Ave_precs_all = Ave_precs_x + weight_precs_all
    if verbose:
        print('Class {}, evaluated at Q={}:, found {}'.format(test_class, total_no_of_points, no_files_in_label[test_class] ))
        print('Average precision={}; precision={}; recall={}'.format(Ave_precs_all,precs_all,recall_all))
    return Ave_precs_all, precs_all, recall_all, Ave_precs_x, precs_x, recall_x


def test_ave_precsision(class_name, local_list=local_list):
#N_test = no_files_in_label[test_class] # no of items in class A
    # set up counters
    AP = 0 # average precision
    count_of_test_class = 0
    # loop backwards through the list, abs j is the position in a 1-indexed list
    # values for i == -1
#    current_class = local_list[-1][0]
#    if (current_class == test_class):
#        count_of_test_class = count_of_test_class + 1 # we found A
#    precs_x = count_of_test_class /1
    egg={'1': 6396, '7': 5964, '0': 5823, '6': 5636, '9': 5595, '3': 5396, '2': 5242, '8': 5219, '4': 5213, '5': 5030}
    N_test = egg[class_name]
    test_class = class_name
    verbose=True
    Q=55514
    recall_x = 0
    Ave_precs_x = 0
    for i in range(Q):
        j = -(i + 1)  # 1 indexed
        recall_x_minus_1 = recall_x
        current_class = local_list[j][0]# current class
        if j == -Q:
            # if the whole of local_list is the same class (this accounts for zero indexing)
            if verbose:
                print(current_class)
                print('{}/{}'.format(count_of_test_class, abs(j)))
            j = j -1 # really this is here so we can check j
            #break
        if count_of_test_class == N_test:
            #we've found them all
            if verbose:
                print('found all {} of {}, stopping...'.format(N_test, test_class))
                print('{}/{}'.format(count_of_test_class, abs(j)))
            break
        if (current_class == test_class):
            count_of_test_class = count_of_test_class + 1   #n A
        precs_x = count_of_test_class /(abs(j)) # N.b. this is the sum, we divide by j on the output
        recall_x = count_of_test_class / N_test
        delta_recall_x = recall_x - recall_x_minus_1    # difference in recall between this point nd the next
        weight_precs_x = precs_x * delta_recall_x        # weighted precsion at point x (we do average via weighted sum)
        Ave_precs_x = Ave_precs_x + weight_precs_x             # average_precision evaluated at point x
    print('precsionn = {}'.format(precs_x))
    print('recall = {}'.format(recall_x))
    print('average precsion = {}'.format(Ave_precs_x))
    return Ave_precs_x, precs_x, recall_x

def find_zhou_precision(number_of_points=100, local_list=local_list):
    """Finds the maximally occuring (top mode) class in the top number_of_points and counts it"""
    classes_in_top_100=[local_list[x][0] for x in range(-number_of_points,0)]
    zhou = Counter(classes_in_top_100)
    zhou_precs_class = zhou.most_common(1)[0][0] # the class name
    zhou_precs = zhou.most_common(1)[0][1]/number_of_points # the precision
    zhou_no_of_classes = len(zhou)
    return zhou_precs_class, zhou_precs, zhou_no_of_classes, zhou

def get_max_index_of_list(a_list):
    """I keep forgetting how to do this"""
    if isinstance(a_list, np.ndarray):
        idx = np.argmax(a_list)
    elif isinstance(a_list, list):
        idx=a_list.index(max(a_list))
    return idx

def get_max_ave_precs(class_list=test_class_list, output_precs_data=output_precs_data):
    """grabs the max class names and 2nd max class names
    test_class_list = list of class names in the correct order
    output_precs_data = list of values to get the max and second max of"""
    apL = [x[0] for x in output_precs_data]
    idx=get_max_index_of_list(apL)
    MaxApLClass =class_list[idx]
    if not len(class_list) == 1:
        apL2 = [apL[i] for i in range(len(apL)) if not i == idx]
        class_list2 = [class_list[i] for i in range(len(class_list)) if not i == idx]
        idx=get_max_index_of_list(apL2)
        MaxApLClass2 = class_list2[idx]
    else:
        MaxApLClass2 = ''
    return MaxApLClass, MaxApLClass2

def count_zeros(local_list=local_list, x_data=x_data, class_labels=class_labels, topx=10, verbose=verbose):
    """counts the zeros in a local_list
    topx is the number of results to return, assuming the top-10
    nzeros is the number of zeros
    pzeros is the percentage of a class that is zero
    """
    # now do the zeros
    minx, maxx, nzeros_dict, pzeros_dict = min(x_data), 0.0, {}, {}
    # this gets all the zeros
    num_zeros = len(local_list)
    if verbose:
        print('Working on counting the zeros for all classes (slow!)')
    for test_class in class_labels:
        # counts the number of zeros and percentage of zeros for each class
        nzero = len([x for x in local_list if x[0] == test_class])
        nzeros_dict[test_class] = nzero
        pzeros_dict[test_class] = float(nzero/no_files_in_label[test_class])
    # sorts the lists so the classes with the least zeros are first
    nzeros = sorted(nzeros_dict.items(), key=operator.itemgetter(1))
    pzeros = sorted(pzeros_dict.items(), key=operator.itemgetter(1))
    # grab the top 10
    nzeros = nzeros[0:topx]
    pzeros = pzeros[0:topx]
    return num_zeros, nzeros, pzeros


###############################################################
#
# sCRIPT
##############################################################

def main():



    m.main()
    acts = m.acts
    class_labels = m.class_labels



    out_filename = 'precs_data.csv'
    out_filename2 = 'precs_data_nonzero.csv'
    out_filename3 = 'precs_data_all.csv'
    out_filename4 = 'precs_data_zeros.csv'
    out_filename5 = 'precs_summary.csv'

    class_dict = h.make_class_to_line_number_look_up_table(class_labels=class_labels, verbose=False)
        # this builds the look-up table between points and the class they are in
        ## This bit is sslow, it loads the label data for all acts
    sys.stdout.write('About to build the label dict (slow)')
    sys.stdout.flush()
    label_dict, found_labels, no_files_in_label = h.build_label_dict(acts)
    sys.stdout.write('Built the label dict')
    sys.stdout.flush()

    no_of_images = acts.get_image_count()
    print('Found {} images'.format(no_of_images))

    no_of_neurons = len(acts.get_activation(0).vector)
    print('{} neurons found'.format(no_of_neurons))

    current_range=[]



    fieldnames = ['Neuron no.',  # neuron index
        'top_mode_class_name',       # class name for top mode class (class with highest number in top 100)
        'max_ave_precs_100_class',
        'second_max_ave_precs_100_class',
        'zhou_precs60',  # Zhou precision for the most common class in top 60
        'zhou_precs_class60',  # class for ZP60
        'zhou_no_of_classes100',  # no fo classes in top 100
        '1_class',
        '1_Ave_precs_100',  #
        '1_precs_100',
        '1_recall_100',
        '1_recall_p95_100',
        '1_specificity_100',
        '1_informedness_100',
        '1_max_informedness_100',
        '1_x_for_max_informedness_100',
        '1_max_f1_stat_100',
        '1_x_for_max_f1_100',
        '2_class',
        '2_Ave_precs_100',  # start of x = 100 data for top 10 classes
        '2_precs_100',
        '2_recall_100',
        '2_recall_p95_100',
        '2_specificity_100',
        '2_informedness_100',
        '2_max_informedness_100',
        '2_x_for_max_informedness_100',
        '2_max_f1_stat_100',
        '2_x_for_max_f1_100',
        '3_class',
        '3_Ave_precs_100',  # start of x = 100 data for top 10 classes
        '3_precs_100',
        '3_recall_100',
        '3_recall_p95_100',
        '3_specificity_100',
        '3_informedness_100',
        '3_max_informedness_100',
        '3_x_for_max_informedness_100',
        '3_max_f1_stat_100',
        '3_x_for_max_f1_100',
        '4_class',
        '4_Ave_precs_100',  # start of x = 100 data for top 10 classes
        '4_precs_100',
        '4_recall_100',
        '4_recall_p95_100',
        '4_specificity_100',
        '4_informedness_100',
        '4_max_informedness_100',
        '4_x_for_max_informedness_100',
        '4_max_f1_stat_100',
        '4_x_for_max_f1_100',
        '5_class',
        '5_Ave_precs_100',  # start of x = 100 data for top 10 classes
        '5_precs_100',
        '5_recall_100',
        '5_recall_p95_100',
        '5_specificity_100',
        '5_informedness_100',
        '5_max_informedness_100',
        '5_x_for_max_informedness_100',
        '5_max_f1_stat_100',
        '5_x_for_max_f1_100',
        '6_class',
        '6_Ave_precs_100',  # start of x = 100 data for top 10 classes
        '6_precs_100',
        '6_recall_100',
        '6_recall_p95_100',
        '6_specificity_100',
        '6_informedness_100',
        '6_max_informedness_100',
        '6_x_for_max_informedness_100',
        '6_max_f1_stat_100',
        '6_x_for_max_f1_100',
        '7_class',
        '7_Ave_precs_100',  # start of x = 100 data for top 10 classes
        '7_precs_100',
        '7_recall_100',
        '7_recall_p95_100',
        '7_specificity_100',
        '7_informedness_100',
        '7_max_informedness_100',
        '7_x_for_max_informedness_100',
        '7_max_f1_stat_100',
        '7_x_for_max_f1_100',
        '8_class',
        '8_Ave_precs_100',  # start of x = 100 data for top 10 classes
        '8_precs_100',
        '8_recall_100',
        '8_recall_p95_100',
        '8_specificity_100',
        '8_informedness_100',
        '8_max_informedness_100',
        '8_x_for_max_informedness_100',
        '8_max_f1_stat_100',
        '8_x_for_max_f1_100',
        '9_class',
        '9_Ave_precs_100',  # start of x = 100 data for top 10 classes
        '9_precs_100',
        '9_recall_100',
        '9_recall_p95_100',
        '9_specificity_100',
        '9_informedness_100',
        '9_max_informedness_100',
        '9_x_for_max_informedness_100',
        '9_max_f1_stat_100',
        '9_x_for_max_f1_100',
        '10_class',
        '10_Ave_precs_100',  # start of x = 100 data for top 10 classes
        '10_precs_100',
        '10_recall_100',
        '10_recall_p95_100',
        '10_specificity_100',
        '10_informedness_100',
        '10_max_informedness_100',
        '10_x_for_max_informedness_100',
        '10_max_f1_stat_100',
        '10_x_for_max_f1_100'
        ]


    fieldnames2 = ['Neuron no.',  # neuron index
        'max_ave_precs_nonzero_class',
        'second_max_ave_precs_nonzero_class',
        '1_class',
        '1_Ave_precs_nonzero',  #
        '1_precs_nonzero',
        '1_recall_nonzero',
        '1_recall_p95_nonzero',
        '1_specificity_nonzero',
        '1_informedness_nonzero',
        '1_max_informedness_nonzero',
        '1_x_for_max_informedness_nonzero',
        '1_recall_for_max_informedness',
        '1_specificity_for_max_informedness',
        '1_max_f1_stat_nonzero',
        '1_x_for_max_f1_nonzero',
        '2_class',
        '2_Ave_precs_nonzero',  # start of x = 100 data for top 10 classes
        '2_precs_nonzero',
        '2_recall_nonzero',
        '2_recall_p95_nonzero',
        '2_specificity_nonzero',
        '2_informedness_nonzero',
        '2_max_informedness_nonzero',
        '2_x_for_max_informedness_nonzero',
        '2_recall_for_max_informedness',
        '2_specificity_for_max_informedness',
        '2_max_f1_stat_nonzero',
        '2_x_for_max_f1_nonzero',
        '3_class',
        '3_Ave_precs_nonzero',  # start of x = 100 data for top 10 classes
        '3_precs_nonzero',
        '3_recall_nonzero',
        '3_recall_p95_nonzero',
        '3_specificity_nonzero',
        '3_informedness_nonzero',
        '3_max_informedness_nonzero',
        '3_x_for_max_informedness_nonzero',
        '3_recall_for_max_informedness',
        '3_specificity_for_max_informedness',
        '3_max_f1_stat_nonzero',
        '3_x_for_max_f1_nonzero',
        '4_class',
        '4_Ave_precs_nonzero',  # start of x = 100 data for top 10 classes
        '4_precs_nonzero',
        '4_recall_nonzero',
        '4_recall_p95_nonzero',
        '4_specificity_nonzero',
        '4_informedness_nonzero',
        '4_max_informedness_nonzero',
        '4_x_for_max_informedness_nonzero',
        '4_recall_for_max_informedness',
        '4_specificity_for_max_informedness',
        '4_max_f1_stat_nonzero',
        '4_x_for_max_f1_nonzero',
        '5_class',
        '5_Ave_precs_nonzero',  # start of x = 100 data for top 10 classes
        '5_precs_nonzero',
        '5_recall_nonzero',
        '5_recall_p95_nonzero',
        '5_specificity_nonzero',
        '5_informedness_nonzero',
        '5_max_informedness_nonzero',
        '5_x_for_max_informedness_nonzero',
        '5_recall_for_max_informedness',
        '5_specificity_for_max_informedness',
        '5_max_f1_stat_nonzero',
        '5_x_for_max_f1_nonzero',
        '6_class',
        '6_Ave_precs_nonzero',  # start of x = 100 data for top 10 classes
        '6_precs_nonzero',
        '6_recall_nonzero',
        '6_recall_p95_nonzero',
        '6_specificity_nonzero',
        '6_informedness_nonzero',
        '6_max_informedness_nonzero',
        '6_x_for_max_informedness_nonzero',
        '6_recall_for_max_informedness',
        '6_specificity_for_max_informedness',
        '6_max_f1_stat_nonzero',
        '6_x_for_max_f1_nonzero',
        '7_class',
        '7_Ave_precs_nonzero',  # start of x = 100 data for top 10 classes
        '7_precs_nonzero',
        '7_recall_nonzero',
        '7_recall_p95_nonzero',
        '7_specificity_nonzero',
        '7_informedness_nonzero',
        '7_max_informedness_nonzero',
        '7_x_for_max_informedness_nonzero',
        '7_recall_for_max_informedness',
        '7_specificity_for_max_informedness',
        '7_max_f1_stat_nonzero',
        '7_x_for_max_f1_nonzero',
        '8_class',
        '8_Ave_precs_nonzero',  # start of x = 100 data for top 10 classes
        '8_precs_nonzero',
        '8_recall_nonzero',
        '8_recall_p95_nonzero',
        '8_specificity_nonzero',
        '8_informedness_nonzero',
        '8_max_informedness_nonzero',
        '8_x_for_max_informedness_nonzero',
        '8_recall_for_max_informedness',
        '8_specificity_for_max_informedness',
        '8_max_f1_stat_nonzero',
        '8_x_for_max_f1_nonzero',
        '9_class',
        '9_Ave_precs_nonzero',  # start of x = 100 data for top 10 classes
        '9_precs_nonzero',
        '9_recall_nonzero',
        '9_recall_p95_nonzero',
        '9_specificity_nonzero',
        '9_informedness_nonzero',
        '9_max_informedness_nonzero',
        '9_x_for_max_informedness_nonzero',
        '9_recall_for_max_informedness',
        '9_specificity_for_max_informedness',
        '9_max_f1_stat_nonzero',
        '9_x_for_max_f1_nonzero',
        '10_class',
        '10_Ave_precs_nonzero',  # start of x = 100 data for top 10 classes
        '10_precs_nonzero',
        '10_recall_nonzero',
        '10_recall_p95_nonzero',
        '10_specificity_nonzero',
        '10_informedness_nonzero',
        '10_max_informedness_nonzero',
        '10_x_for_max_informedness_nonzero',
        '10_recall_for_max_informedness',
        '10_specificity_for_max_informedness',
        '10_max_f1_stat_nonzero',
        '10_x_for_max_f1_nonzero'
        ]

    fieldnames3 = ['Neuron no.',  # neuron index
               'max_ave_precs_all_class',
               'second_max_ave_precs_all_class',
               '1_class',
               '1_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '1_precs_all',
               '1_recall_all',
               '2_class',
               '2_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '2_precs_all',
               '2_recall_all',
               '3_class',
               '3_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '3_precs_all',
               '3_recall_all',
               '4_class',
               '4_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '4_precs_all',
               '4_recall_all',
               '5_class',
               '5_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '5_precs_all',
               '5_recall_all',
               '6_class',
               '6_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '6_precs_all',
               '6_recall_all',
               '7_class',
               '7_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '7_precs_all',
               '7_recall_all',
               '8_class',
               '8_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '8_precs_all',
               '8_recall_all',
               '9_class',
               '9_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '9_precs_all',
               '9_recall_all',
               '10_class',
               '10_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '10_precs_all',
               '10_recall_all'
                ]

    fieldnames4 = ['Neuron no.',  # neuron index
           'num_zeros',
           '1_least_zero_class',  # start of x = 100 data for top 10 classes
           '1_least_zero_num',
           '2_least_zero_class',  # start of x = 100 data for top 10 classes
           '2_least_zero_num',
           '3_least_zero_class',  # start of x = 100 data for top 10 classes
           '3_least_zero_num',
           '4_least_zero_class',  # start of x = 100 data for top 10 classes
           '4_least_zero_num',
           '5_least_zero_class',  # start of x = 100 data for top 10 classes
           '5_least_zero_num',
           '6_least_zero_class',  # start of x = 100 data for top 10 classes
           '6_least_zero_num',
           '7_least_zero_class',  # start of x = 100 data for top 10 classes
           '7_least_zero_num',
           '8_least_zero_class',  # start of x = 100 data for top 10 classes
           '8_least_zero_num',
           '9_least_zero_class',  # start of x = 100 data for top 10 classes
           '9_least_zero_num',
           '10_least_zero_class',  # start of x = 100 data for top 10 classes
           '10_least_zero_num',
           '1_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '1_least_zero_prop_num',
           '2_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '2_least_zero_prop_num',
           '3_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '3_least_zero_prop_num',
           '4_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '4_least_zero_prop_num',
           '5_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '5_least_zero_prop_num',
           '6_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '6_least_zero_prop_num',
           '7_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '7_least_zero_prop_num',
           '8_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '8_least_zero_prop_num',
           '9_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '9_least_zero_prop_num',
           '10_least_zero_prop_class',  # start of x = 100 data for top 10 classes
           '10_least_zero_prop_num'
           ]

    def row1_outputter(current_neuron_index=current_neuron_index, top_mode_class=top_mode_class, zhou_precs60=zhou_precs60,
                       zhou_precs_class60=zhou_precs_class60, zhou_no_of_classes100=zhou_no_of_classes100,
                       output_precs_data = output_precs_data, test_class_list=test_class_list):
        """Little wrapper function to write out the row
        this is the stats over the top 100 activations!"""
        max_ave_precs_100_class, second_max_ave_precs_100_class = \
            get_max_ave_precs(class_list=test_class_list, output_precs_data=output_precs_data)
        while len(test_class_list) < 10:
            test_class_list.append('')
        row = {'Neuron no.': str(current_neuron_index),  # neuron index
                    'top_mode_class_name': str(top_mode_class), # class name for top mode class (class with highest number in top 100)
                    'max_ave_precs_100_class': str(max_ave_precs_100_class),
                    'second_max_ave_precs_100_class': str(second_max_ave_precs_100_class),
                    'zhou_precs60': str(zhou_precs60), # Zhou precision for the most common class in top 60
                    'zhou_precs_class60': str(zhou_precs_class60), # class for ZP60
                    'zhou_no_of_classes100': str(zhou_no_of_classes100), # no fo classes in top 100,
                    '1_class': str(test_class_list[0]),
                    '1_Ave_precs_100' : str(output_precs_data[0][0]), # start of x = 100 data for top 10 classes
                    '1_precs_100': str(output_precs_data[0][1]),
                    '1_recall_100': str(output_precs_data[0][2]),
                    '1_recall_p95_100':  str(output_precs_data[0][3]),
                    '1_specificity_100': str(output_precs_data[0][4]),
                    '1_informedness_100': str(output_precs_data[0][5]),
                    '1_max_informedness_100': str(output_precs_data[0][6]),
                    '1_x_for_max_informedness_100': str(output_precs_data[0][7]),
                    '1_max_f1_stat_100': str(output_precs_data[0][8]),
                    '1_x_for_max_f1_100': str(output_precs_data[0][9]),
                    '2_class': str(test_class_list[1]),
                    '2_Ave_precs_100': str(output_precs_data[1][0]),  # start of x = 100 data for top 10 classes
                    '2_precs_100': str(output_precs_data[1][1]),
                    '2_recall_100': str(output_precs_data[1][2]),
                    '2_recall_p95_100': str(output_precs_data[1][3]),
                    '2_specificity_100': str(output_precs_data[1][4]),
                    '2_informedness_100': str(output_precs_data[1][5]),
                    '2_max_informedness_100': str(output_precs_data[1][6]),
                    '2_x_for_max_informedness_100': str(output_precs_data[1][7]),
                    '2_max_f1_stat_100': str(output_precs_data[1][8]),
                    '2_x_for_max_f1_100': str(output_precs_data[1][9]),
                    '3_class': str(test_class_list[2]),
                    '3_Ave_precs_100': str(output_precs_data[2][0]),  # start of x = 100 data for top 10 classes
                    '3_precs_100': str(output_precs_data[2][1]),
                    '3_recall_100': str(output_precs_data[2][2]),
                    '3_recall_p95_100': str(output_precs_data[2][3]),
                    '3_specificity_100': str(output_precs_data[2][4]),
                    '3_informedness_100': str(output_precs_data[2][5]),
                    '3_max_informedness_100': str(output_precs_data[2][6]),
                    '3_x_for_max_informedness_100': str(output_precs_data[2][7]),
                    '3_max_f1_stat_100': str(output_precs_data[2][8]),
                    '3_x_for_max_f1_100': str(output_precs_data[2][9]),
                    '4_class': str(test_class_list[3]),
                    '4_Ave_precs_100': str(output_precs_data[3][0]),  # start of x = 100 data for top 10 classes
                    '4_precs_100': str(output_precs_data[3][1]),
                    '4_recall_100': str(output_precs_data[3][2]),
                    '4_recall_p95_100': str(output_precs_data[3][3]),
                    '4_specificity_100': str(output_precs_data[3][4]),
                    '4_informedness_100': str(output_precs_data[3][5]),
                    '4_max_informedness_100': str(output_precs_data[3][6]),
                    '4_x_for_max_informedness_100': str(output_precs_data[3][7]),
                    '4_max_f1_stat_100': str(output_precs_data[3][8]),
                    '4_x_for_max_f1_100': str(output_precs_data[3][9]),
                    '5_class': str(test_class_list[4]),
                    '5_Ave_precs_100': str(output_precs_data[4][0]),  # start of x = 100 data for top 10 classes
                    '5_precs_100': str(output_precs_data[4][1]),
                    '5_recall_100': str(output_precs_data[4][2]),
                    '5_recall_p95_100': str(output_precs_data[4][3]),
                    '5_specificity_100': str(output_precs_data[4][4]),
                    '5_informedness_100': str(output_precs_data[4][5]),
                    '5_max_informedness_100': str(output_precs_data[4][6]),
                    '5_x_for_max_informedness_100': str(output_precs_data[4][7]),
                    '5_max_f1_stat_100': str(output_precs_data[4][8]),
                    '5_x_for_max_f1_100': str(output_precs_data[4][9]),
                    '6_class': str(test_class_list[5]),
                    '6_Ave_precs_100': str(output_precs_data[5][0]),  # start of x = 100 data for top 10 classes
                    '6_precs_100': str(output_precs_data[5][1]),
                    '6_recall_100': str(output_precs_data[5][2]),
                    '6_recall_p95_100': str(output_precs_data[5][3]),
                    '6_specificity_100': str(output_precs_data[5][4]),
                    '6_informedness_100': str(output_precs_data[5][5]),
                    '6_max_informedness_100': str(output_precs_data[5][6]),
                    '6_x_for_max_informedness_100': str(output_precs_data[5][7]),
                    '6_max_f1_stat_100': str(output_precs_data[5][8]),
                    '6_x_for_max_f1_100': str(output_precs_data[5][9]),
                    '7_class': str(test_class_list[6]),
                    '7_Ave_precs_100': str(output_precs_data[6][0]),  # start of x = 100 data for top 10 classes
                    '7_precs_100': str(output_precs_data[6][1]),
                    '7_recall_100': str(output_precs_data[6][2]),
                    '7_recall_p95_100': str(output_precs_data[6][3]),
                    '7_specificity_100': str(output_precs_data[6][4]),
                    '7_informedness_100': str(output_precs_data[6][5]),
                    '7_max_informedness_100': str(output_precs_data[6][6]),
                    '7_x_for_max_informedness_100': str(output_precs_data[6][7]),
                    '7_max_f1_stat_100': str(output_precs_data[6][8]),
                    '7_x_for_max_f1_100': str(output_precs_data[6][9]),
                    '8_class': str(test_class_list[7]),
                    '8_Ave_precs_100': str(output_precs_data[7][0]),  # start of x = 100 data for top 10 classes
                    '8_precs_100': str(output_precs_data[7][1]),
                    '8_recall_100': str(output_precs_data[7][2]),
                    '8_recall_p95_100': str(output_precs_data[7][3]),
                    '8_specificity_100': str(output_precs_data[7][4]),
                    '8_informedness_100': str(output_precs_data[7][5]),
                    '8_max_informedness_100': str(output_precs_data[7][6]),
                    '8_x_for_max_informedness_100': str(output_precs_data[7][7]),
                    '8_max_f1_stat_100': str(output_precs_data[7][8]),
                    '8_x_for_max_f1_100': str(output_precs_data[7][9]),
                    '9_class': str(test_class_list[8]),
                    '9_Ave_precs_100': str(output_precs_data[8][0]),  # start of x = 100 data for top 10 classes
                    '9_precs_100': str(output_precs_data[8][1]),
                    '9_recall_100': str(output_precs_data[8][2]),
                    '9_recall_p95_100': str(output_precs_data[8][3]),
                    '9_specificity_100': str(output_precs_data[8][4]),
                    '9_informedness_100': str(output_precs_data[8][5]),
                    '9_max_informedness_100': str(output_precs_data[8][6]),
                    '9_x_for_max_informedness_100': str(output_precs_data[8][7]),
                    '9_max_f1_stat_100': str(output_precs_data[8][8]),
                    '9_x_for_max_f1_100': str(output_precs_data[8][9]),
                    '10_class': str(test_class_list[9]),
                    '10_Ave_precs_100': str(output_precs_data[9][0]),  # start of x = 100 data for top 10 classes
                    '10_precs_100': str(output_precs_data[9][1]),
                    '10_recall_100': str(output_precs_data[9][2]),
                    '10_recall_p95_100': str(output_precs_data[9][3]),
                    '10_specificity_100': str(output_precs_data[9][4]),
                    '10_informedness_100': str(output_precs_data[9][5]),
                    '10_max_informedness_100': str(output_precs_data[9][6]),
                    '10_x_for_max_informedness_100': str(output_precs_data[9][7]),
                    '10_max_f1_stat_100': str(output_precs_data[9][8]),
                    '10_x_for_max_f1_100': str(output_precs_data[9][9])
                    }
        return row

    def row2_outputter(current_neuron_index=current_neuron_index,
                       output_precs_data = output_precs_data, test_class_list=test_class_list):
        max_ave_precs_nonzero_class, second_max_ave_precs_nonzero_class=\
            get_max_ave_precs(class_list=test_class_list, output_precs_data=output_precs_data)
        while len(test_class_list) < 10:
            test_class_list.append('')
        row = {'Neuron no.': str(current_neuron_index),  # neuron index
                'max_ave_precs_nonzero_class': str(max_ave_precs_nonzero_class),
                'second_max_ave_precs_nonzero_class': str(second_max_ave_precs_nonzero_class),
                '1_class': str(test_class_list[0]),
                '1_Ave_precs_nonzero': str(output_precs_data[0][0]),  # start of x = 100 data for top 10 classes
                '1_precs_nonzero': str(output_precs_data[0][1]),
                '1_recall_nonzero': str(output_precs_data[0][2]),
                '1_recall_p95_nonzero': str(output_precs_data[0][3]),
                '1_specificity_nonzero': str(output_precs_data[0][4]),
                '1_informedness_nonzero': str(output_precs_data[0][5]),
                '1_max_informedness_nonzero': str(output_precs_data[0][6]),
                '1_x_for_max_informedness_nonzero': str(output_precs_data[0][7]),
                '1_max_f1_stat_nonzero': str(output_precs_data[0][8]),
                '1_x_for_max_f1_nonzero': str(output_precs_data[0][9]),
                '1_recall_for_max_informedness': str(output_precs_data[0][10]),
                '1_specificity_for_max_informedness': str(output_precs_data[0][11]),
                '2_class': str(test_class_list[1]),
                '2_Ave_precs_nonzero': str(output_precs_data[1][0]),  # start of x = 100 data for top 10 classes
                '2_precs_nonzero': str(output_precs_data[1][1]),
                '2_recall_nonzero': str(output_precs_data[1][2]),
                '2_recall_p95_nonzero': str(output_precs_data[1][3]),
                '2_specificity_nonzero': str(output_precs_data[1][4]),
                '2_informedness_nonzero': str(output_precs_data[1][5]),
                '2_max_informedness_nonzero': str(output_precs_data[1][6]),
                '2_x_for_max_informedness_nonzero': str(output_precs_data[1][7]),
                '2_max_f1_stat_nonzero': str(output_precs_data[1][8]),
                '2_x_for_max_f1_nonzero': str(output_precs_data[1][9]),
                '2_recall_for_max_informedness': str(output_precs_data[1][10]),
                '2_specificity_for_max_informedness': str(output_precs_data[1][11]),
                '3_class': str(test_class_list[2]),
                '3_Ave_precs_nonzero': str(output_precs_data[2][0]),  # start of x = 100 data for top 10 classes
                '3_precs_nonzero': str(output_precs_data[2][1]),
                '3_recall_nonzero': str(output_precs_data[2][2]),
                '3_recall_p95_nonzero': str(output_precs_data[2][3]),
                '3_specificity_nonzero': str(output_precs_data[2][4]),
                '3_informedness_nonzero': str(output_precs_data[2][5]),
                '3_max_informedness_nonzero': str(output_precs_data[2][6]),
                '3_x_for_max_informedness_nonzero': str(output_precs_data[2][7]),
                '3_max_f1_stat_nonzero': str(output_precs_data[2][8]),
                '3_x_for_max_f1_nonzero': str(output_precs_data[2][9]),
                '3_recall_for_max_informedness': str(output_precs_data[2][10]),
                '3_specificity_for_max_informedness': str(output_precs_data[2][11]),
                '4_class': str(test_class_list[3]),
                '4_Ave_precs_nonzero': str(output_precs_data[3][0]),  # start of x = 100 data for top 10 classes
                '4_precs_nonzero': str(output_precs_data[3][1]),
                '4_recall_nonzero': str(output_precs_data[3][2]),
                '4_recall_p95_nonzero': str(output_precs_data[3][3]),
                '4_specificity_nonzero': str(output_precs_data[3][4]),
                '4_informedness_nonzero': str(output_precs_data[3][5]),
                '4_max_informedness_nonzero': str(output_precs_data[3][6]),
                '4_x_for_max_informedness_nonzero': str(output_precs_data[3][7]),
                '4_max_f1_stat_nonzero': str(output_precs_data[3][8]),
                '4_x_for_max_f1_nonzero': str(output_precs_data[3][9]),
                '4_recall_for_max_informedness': str(output_precs_data[3][10]),
                '4_specificity_for_max_informedness': str(output_precs_data[3][11]),
                '5_class': str(test_class_list[4]),
                '5_Ave_precs_nonzero': str(output_precs_data[4][0]),  # start of x = 100 data for top 10 classes
                '5_precs_nonzero': str(output_precs_data[4][1]),
                '5_recall_nonzero': str(output_precs_data[4][2]),
                '5_recall_p95_nonzero': str(output_precs_data[4][3]),
                '5_specificity_nonzero': str(output_precs_data[4][4]),
                '5_informedness_nonzero': str(output_precs_data[4][5]),
                '5_max_informedness_nonzero': str(output_precs_data[4][6]),
                '5_x_for_max_informedness_nonzero': str(output_precs_data[4][7]),
                '5_max_f1_stat_nonzero': str(output_precs_data[4][8]),
                '5_x_for_max_f1_nonzero': str(output_precs_data[4][9]),
                '5_recall_for_max_informedness': str(output_precs_data[4][10]),
                '5_specificity_for_max_informedness': str(output_precs_data[4][11]),
                '6_class': str(test_class_list[5]),
                '6_Ave_precs_nonzero': str(output_precs_data[5][0]),  # start of x = 100 data for top 10 classes
                '6_precs_nonzero': str(output_precs_data[5][1]),
                '6_recall_nonzero': str(output_precs_data[5][2]),
                '6_recall_p95_nonzero': str(output_precs_data[5][3]),
                '6_specificity_nonzero': str(output_precs_data[5][4]),
                '6_informedness_nonzero': str(output_precs_data[5][5]),
                '6_max_informedness_nonzero': str(output_precs_data[5][6]),
                '6_x_for_max_informedness_nonzero': str(output_precs_data[5][7]),
                '6_max_f1_stat_nonzero': str(output_precs_data[5][8]),
                '6_x_for_max_f1_nonzero': str(output_precs_data[5][9]),
                '6_recall_for_max_informedness': str(output_precs_data[5][10]),
                '6_specificity_for_max_informedness': str(output_precs_data[5][11]),
                '7_class': str(test_class_list[6]),
                '7_Ave_precs_nonzero': str(output_precs_data[6][0]),  # start of x = 100 data for top 10 classes
                '7_precs_nonzero': str(output_precs_data[6][1]),
                '7_recall_nonzero': str(output_precs_data[6][2]),
                '7_recall_p95_nonzero': str(output_precs_data[6][3]),
                '7_specificity_nonzero': str(output_precs_data[6][4]),
                '7_informedness_nonzero': str(output_precs_data[6][5]),
                '7_max_informedness_nonzero': str(output_precs_data[6][6]),
                '7_x_for_max_informedness_nonzero': str(output_precs_data[6][7]),
                '7_max_f1_stat_nonzero': str(output_precs_data[6][8]),
                '7_x_for_max_f1_nonzero': str(output_precs_data[6][9]),
                '7_recall_for_max_informedness': str(output_precs_data[6][10]),
                '7_specificity_for_max_informedness': str(output_precs_data[6][11]),
                '8_class': str(test_class_list[7]),
                '8_Ave_precs_nonzero': str(output_precs_data[7][0]),  # start of x = 100 data for top 10 classes
                '8_precs_nonzero': str(output_precs_data[7][1]),
                '8_recall_nonzero': str(output_precs_data[7][2]),
                '8_recall_p95_nonzero': str(output_precs_data[7][3]),
                '8_specificity_nonzero': str(output_precs_data[7][4]),
                '8_informedness_nonzero': str(output_precs_data[7][5]),
                '8_max_informedness_nonzero': str(output_precs_data[7][6]),
                '8_x_for_max_informedness_nonzero': str(output_precs_data[7][7]),
                '8_max_f1_stat_nonzero': str(output_precs_data[7][8]),
                '8_x_for_max_f1_nonzero': str(output_precs_data[7][9]),
                '8_recall_for_max_informedness': str(output_precs_data[7][10]),
                '8_specificity_for_max_informedness': str(output_precs_data[7][11]),
                '9_class': str(test_class_list[8]),
                '9_Ave_precs_nonzero': str(output_precs_data[8][0]),  # start of x = 100 data for top 10 classes
                '9_precs_nonzero': str(output_precs_data[8][1]),
                '9_recall_nonzero': str(output_precs_data[8][2]),
                '9_recall_p95_nonzero': str(output_precs_data[8][3]),
                '9_specificity_nonzero': str(output_precs_data[8][4]),
                '9_informedness_nonzero': str(output_precs_data[8][5]),
                '9_max_informedness_nonzero': str(output_precs_data[8][6]),
                '9_x_for_max_informedness_nonzero': str(output_precs_data[8][7]),
                '9_max_f1_stat_nonzero': str(output_precs_data[8][8]),
                '9_x_for_max_f1_nonzero': str(output_precs_data[8][9]),
                '9_recall_for_max_informedness': str(output_precs_data[8][10]),
                '9_specificity_for_max_informedness': str(output_precs_data[8][11]),
                '10_class': str(test_class_list[9]),
                '10_Ave_precs_nonzero': str(output_precs_data[9][0]),  # start of x = 100 data for top 10 classes
                '10_precs_nonzero': str(output_precs_data[9][1]),
                '10_recall_nonzero': str(output_precs_data[9][2]),
                '10_recall_p95_nonzero': str(output_precs_data[9][3]),
                '10_specificity_nonzero': str(output_precs_data[9][4]),
                '10_informedness_nonzero': str(output_precs_data[9][5]),
                '10_max_informedness_nonzero': str(output_precs_data[9][6]),
                '10_x_for_max_informedness_nonzero': str(output_precs_data[9][7]),
                '10_max_f1_stat_nonzero': str(output_precs_data[9][8]),
                '10_x_for_max_f1_nonzero': str(output_precs_data[9][9]),
                '10_recall_for_max_informedness': str(output_precs_data[9][10]),
                '10_specificity_for_max_informedness': str(output_precs_data[9][11])
                }
        return row


    def row3_outputter(current_neuron_index=current_neuron_index,
                       output_precs_data = output_precs_data, test_class_list=test_class_list):
        max_ave_precs_all_class, second_max_ave_precs_all_class=\
            get_max_ave_precs(class_list=test_class_list, output_precs_data=output_precs_data)
        while len(test_class_list) < 10:
            test_class_list.append('')
        row = {'Neuron no.': str(current_neuron_index),  # neuron index
               'max_ave_precs_all_class': str(max_ave_precs_all_class),
               'second_max_ave_precs_all_class': str(second_max_ave_precs_all_class),
               '1_class': str(test_class_list[0]),
               '1_Ave_precs_all': str(output_precs_data[0][0]),  # start of x = 100 data for top 10 classes
               '1_precs_all': str(output_precs_data[0][1]),
               '1_recall_all': str(output_precs_data[0][2]),
               '2_class': str(test_class_list[1]),
               '2_Ave_precs_all': str(output_precs_data[1][0]),  # start of x = 100 data for top 10 classes
               '2_precs_all': str(output_precs_data[1][1]),
               '2_recall_all': str(output_precs_data[1][2]),
               '3_class': str(test_class_list[2]),
               '3_Ave_precs_all': str(output_precs_data[2][0]),  # start of x = 100 data for top 10 classes
               '3_precs_all': str(output_precs_data[2][1]),
               '3_recall_all': str(output_precs_data[2][2]),
               '4_class': str(test_class_list[3]),
               '4_Ave_precs_all': str(output_precs_data[3][0]),  # start of x = 100 data for top 10 classes
               '4_precs_all': str(output_precs_data[3][1]),
               '4_recall_all': str(output_precs_data[3][2]),
               '5_class': str(test_class_list[4]),
               '5_Ave_precs_all': str(output_precs_data[4][0]),  # start of x = 100 data for top 10 classes
               '5_precs_all': str(output_precs_data[4][1]),
               '5_recall_all': str(output_precs_data[4][2]),
               '6_class': str(test_class_list[5]),
               '6_Ave_precs_all': str(output_precs_data[5][0]),  # start of x = 100 data for top 10 classes
               '6_precs_all': str(output_precs_data[5][1]),
               '6_recall_all': str(output_precs_data[5][2]),
               '7_class': str(test_class_list[6]),
               '7_Ave_precs_all': str(output_precs_data[6][0]),  # start of x = 100 data for top 10 classes
               '7_precs_all': str(output_precs_data[6][1]),
               '7_recall_all': str(output_precs_data[6][2]),
               '8_class': str(test_class_list[7]),
               '8_Ave_precs_all': str(output_precs_data[7][0]),  # start of x = 100 data for top 10 classes
               '8_precs_all': str(output_precs_data[7][1]),
               '8_recall_all': str(output_precs_data[7][2]),
               '9_class': str(test_class_list[8]),
               '9_Ave_precs_all': str(output_precs_data[8][0]),  # start of x = 100 data for top 10 classes
               '9_precs_all': str(output_precs_data[8][1]),
               '9_recall_all': str(output_precs_data[8][2]),
               '10_class': str(test_class_list[9]),
               '10_Ave_precs_all': str(output_precs_data[9][0]),  # start of x = 100 data for top 10 classes
               '10_precs_all': str(output_precs_data[9][1]),
               '10_recall_all': str(output_precs_data[9][2]),
               }
        return row

    def row4_outputter(current_neuron_index=current_neuron_index,
                       nzeros = nzeros, pzeros=pzeros, num_zeros=num_zeros):
        row = {'Neuron no.': str(current_neuron_index),  # neuron index
                'num_zeros': str(num_zeros),
                '1_least_zero_class': str(nzeros[0][0]),  # start of x = 100 data for top 10 classes
                '1_least_zero_num': str(nzeros[0][1]),
                '2_least_zero_class': str(nzeros[1][0]),  # start of x = 100 data for top 10 classes
                '2_least_zero_num': str(nzeros[1][1]),
                '3_least_zero_class': str(nzeros[2][0]),  # start of x = 100 data for top 10 classes
                '3_least_zero_num': str(nzeros[2][1]),
                '4_least_zero_class': str(nzeros[3][0]),  # start of x = 100 data for top 10 classes
                '4_least_zero_num': str(nzeros[3][1]),
                '5_least_zero_class': str(nzeros[4][0]),  # start of x = 100 data for top 10 classes
                '5_least_zero_num': str(nzeros[4][1]),
                '6_least_zero_class': str(nzeros[5][0]),  # start of x = 100 data for top 10 classes
                '6_least_zero_num': str(nzeros[5][1]),
                '7_least_zero_class': str(nzeros[6][0]),  # start of x = 100 data for top 10 classes
                '7_least_zero_num': str(nzeros[6][1]),
                '8_least_zero_class': str(nzeros[7][0]),  # start of x = 100 data for top 10 classes
                '8_least_zero_num': str(nzeros[7][1]),
                '9_least_zero_class': str(nzeros[8][0]),  # start of x = 100 data for top 10 classes
                '9_least_zero_num': str(nzeros[8][1]),
                '10_least_zero_class': str(nzeros[9][0]),  # start of x = 100 data for top 10 classes
                '10_least_zero_num': str(nzeros[9][1]),
                '1_least_zero_prop_class': str(pzeros[0][0]),  # start of x = 100 data for top 10 classes
                '1_least_zero_prop_num': str(pzeros[0][1]),
                '2_least_zero_prop_class': str(pzeros[1][0]),  # start of x = 100 data for top 10 classes
                '2_least_zero_prop_num': str(pzeros[1][1]),
                '3_least_zero_prop_class': str(pzeros[2][0]),  # start of x = 100 data for top 10 classes
                '3_least_zero_prop_num': str(pzeros[2][1]),
                '4_least_zero_prop_class': str(pzeros[3][0]),  # start of x = 100 data for top 10 classes
                '4_least_zero_prop_num': str(pzeros[3][1]),
                '5_least_zero_prop_class': str(pzeros[4][0]),  # start of x = 100 data for top 10 classes
                '5_least_zero_prop_num': str(pzeros[4][1]),
                '6_least_zero_prop_class': str(pzeros[5][0]),  # start of x = 100 data for top 10 classes
                '6_least_zero_prop_num': str(pzeros[5][1]),
                '7_least_zero_prop_class': str(pzeros[6][0]),  # start of x = 100 data for top 10 classes
                '7_least_zero_prop_num': str(pzeros[6][1]),
                '8_least_zero_prop_class': str(pzeros[7][0]),  # start of x = 100 data for top 10 classes
                '8_least_zero_prop_num': str(pzeros[7][1]),
                '9_least_zero_prop_class': str(pzeros[8][0]),  # start of x = 100 data for top 10 classes
                '9_least_zero_prop_num': str(pzeros[8][1]),
                '10_least_zero_prop_class': str(pzeros[9][0]),  # start of x = 100 data for top 10 classes
                '10_least_zero_prop_num': str(pzeros[9][1])
                }
        return row


    current_range = [0]

    with open(filelistfile, 'r') as infile:
        csv_reader = csv.reader(infile, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are {", ".join(row)}')
                line_count += 1
            else:
                current_range.append(int(row[0]))
                line_count += 1
                print('Processed {} lines.'.format(line_count))


    with open(out_filename, 'w') as csvfile:
        # fieldnames=out_list
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        with open(out_filename2, 'w') as csvfile2:
            writer2 = csv.DictWriter(csvfile2, delimiter=',', fieldnames=fieldnames2)
            writer2.writeheader()
            with open(out_filename3, 'w') as csvfile3:
                writer3 = csv.DictWriter(csvfile3, delimiter=',', fieldnames=fieldnames3)
                writer3.writeheader()
                with open(out_filename4, 'w') as csvfile4:
                    writer4 = csv.DictWriter(csvfile4, delimiter=',', fieldnames=fieldnames4)
                    writer4.writeheader()
            # calculations go in here!
                    for current_neuron_index in current_range:
                        if verbose:
                            print('Grabbing the points for neuron {}'.format(current_neuron_index))
                        # # this grabs all the points - note that this gets the data out of acts and is slow so we don;twant to do this twice
                        local_list, selected_activations, x_data = get_local_list_for_neuron(current_neuron_index=current_neuron_index,
                                                                                             minx='',
                                                                                             maxx='',
                                                                                             acts=acts)
                        # this grabs all the zeros from x_data
                        local_list0, selected_activations0 = h.grab_points_for_a_cluster(current_neuron_index,
                                                                                       min_selected_x_data=min(x_data),
                                                                                       max_selected_x_data=0.0,
                                                                                       acts=acts,
                                                                                       x_data=x_data,
                                                                                       verbose=verbose)
                        # this gets the zhou precision over the top 60 activations
                        zhou_precs_class60, zhou_precs60, zhou_no_of_classes60, zhou60 = find_zhou_precision(
                            number_of_points=60, local_list=local_list)
                        # grab the classes in the top 100 and find the top mode class
                        top_mode_class, zhou_precs100, zhou_no_of_classes100, zhou100 = find_zhou_precision(
                            number_of_points=100, local_list=local_list)
                        # now we set up some counters
                        output_precs_data = []
                        output_precs_data2 = []
                        output_precs_data3 = []
                        test_class_list = [x for x in zhou100]
                        if len(test_class_list) > 10:
                            test_class_list = test_class_list[0:10]  # we always take the top 10 classes, but these may not all be in the top 100
                        placeholder = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0)
                        # max_f1_stat, x_for_max_f1)
                #       # now we loop over all 10 classes found by Zhou
                        for idx in range(10):
                            if idx + 1 <= len(test_class_list):
                                # there is a class do stuff
                                # egg = (Ave_precs_x, precs_x, recall_x, recall_p95,
                                # specificity_x, informedness_x, max_informedness, x_for_max_informedness,
                                # max_f1_stat, x_for_max_f1)
                                test_class = test_class_list[idx]
                                # this calculates recall stats over the top 100
                                egg = calculate_many_precs_recall_stats(test_class=test_class,
                                                                        local_list=local_list,
                                                                        Q_stop=100,
                                                                        no_files_in_label=no_files_in_label,
                                                                        no_of_images=no_of_images,
                                                                        verbose=verbose)
                                # this does recall stats over all / or all nonzero points
                                egg2 = calculate_many_precs_recall_stats(test_class=test_class,
                                                                        local_list=local_list,
                                                                        Q_stop='',
                                                                        no_files_in_label=no_files_in_label,
                                                                        no_of_images=no_of_images,
                                                                        verbose=verbose)
                                # this calcs the correct precisions for all points
                                Ave_precs_all, precs_all, recall_all, _, _, _ = calculate_average_precision_incl_zeros(test_class,
                                                                                                                       local_list=local_list,
                                                                                                                       x_data=x_data,
                                                                                                                       selected_activations=selected_activations,
                                                                                                                       current_neuron_index=current_neuron_index,
                                                                                                                       acts=acts,
                                                                                                                       verbose=verbose)
                                egg3 = (Ave_precs_all, precs_all, recall_all)
                                #### output buts
                                output_precs_data.append(egg)  # output this
                                output_precs_data2.append(egg2)  # output this
                                output_precs_data3.append(egg3)
                            else:
                                output_precs_data.append(placeholder)
                                output_precs_data2.append(placeholder)
                                output_precs_data3.append((0.0,0.0,0.0,0.0))
                        # find the max and second max ave precs of our 10 selected testclasses over this range
                        if verbose:
                            print('Nearly finished analysing unit {}'.format(current_neuron_index))
                        num_zeros, nzeros, pzeros = \
                             count_zeros(local_list=local_list0, x_data=x_data, class_labels=class_labels,
                                         topx=10, verbose=verbose)
                        # NOW OUTPUT  the results for this neuron
                        # for the top 100
                        row = row1_outputter(current_neuron_index=current_neuron_index, top_mode_class=top_mode_class,
                                       zhou_precs60=zhou_precs60,
                                       zhou_precs_class60=zhou_precs_class60, zhou_no_of_classes100=zhou_no_of_classes100,
                                       output_precs_data=output_precs_data,
                                       test_class_list=test_class_list
                                             )
                        sorted_row = OrderedDict(sorted(row.items(), key=lambda item: fieldnames.index(item[0])))
                        writer.writerow(sorted_row)
                        # for all nonzero data
                        row2 = row2_outputter(current_neuron_index=current_neuron_index,
                                       output_precs_data=output_precs_data2,
                                              test_class_list=test_class_list)
                        sorted_row2 = OrderedDict(sorted(row2.items(), key=lambda item: fieldnames2.index(item[0])))
                        writer2.writerow(sorted_row2)
                        # for the data adjusted to include the 0.0 points
                        row3 = row3_outputter(current_neuron_index=current_neuron_index,
                                              output_precs_data=output_precs_data3,
                                              test_class_list=test_class_list)
                        sorted_row3 = OrderedDict(sorted(row3.items(), key=lambda item: fieldnames3.index(item[0])))
                        writer3.writerow(sorted_row3)
                        # data to do with teh number of zeros for classes
                        row4 = row4_outputter(current_neuron_index=current_neuron_index,
                                       nzeros=nzeros, pzeros=pzeros, num_zeros=num_zeros)
                        sorted_row4 = OrderedDict(sorted(row4.items(), key=lambda item: fieldnames4.index(item[0])))
                        writer4.writerow(sorted_row4)


            # all (precs) and zero count data
        #output_precs_data = output_precs_data2

        #sorted_row = OrderedDict(sorted(row2.items(), key=lambda item: fieldnames.index(item[0])))
        #writer2.writerow(sorted_row)

if __name__ == '__main__':
    main()


