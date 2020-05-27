#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})
import h5_analysis_jitterer as h


import sys
import csv
import Make_activation as m
from collections import OrderedDict



import precision_calulator as p

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
test_class = ''
local_list = []
Zhou = 0


def calculate_missing_precs_recall_stats(test_class=test_class,
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
    found_recall_precs_p9 = False
    recall_p9 = 0.0
    count_of_test_class = 0
    recall_p1 = 0.0
    found_recall_precs_p1 = False
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
        if (precs_x <= .9) and (found_recall_precs_p9 is False):
            # thingy to grab the recall when precision drops below .95 (if ever)
            recall_p9 = recall_x
            found_recall_precs_p9 = True
        if (precs_x <= 1) and (found_recall_precs_p1 is False):
            # thingy to grab the recall when precision drops below .1 (if ever)
            recall_p1 = recall_x
            found_recall_precs_p1 = True
        delta_recall_x = recall_x - recall_x_minus_1  # difference in recall between this point nd the next
        weight_precs_x = precs_x * delta_recall_x  # weighted precsion at point x (we do average via weighted sum)
        Ave_precs_x = Ave_precs_x + weight_precs_x  # average_precision evaluated at point x
        informedness_x = recall_x + specificity_x -1
        # if informedness_x > max_informedness:
        #     max_informedness = informedness_x
        #     x_for_max_informedness = abs(j)
        #     recall_for_max_informedness = recall_x
        #     specificity_for_max_informedness = specificity_x
        # if (precs_x > 0 and recall_x > 0):
        #     f1_x = 2*(precs_x*recall_x) / (precs_x + recall_x)
        # else:
        #     f1_x = 0
        # if f1_x > max_f1_stat:
        #     max_f1_stat = f1_x
        #     x_for_max_f1 = abs(j)
    out = (recall_p1, recall_p9)
    return out


###############################################################
#
# sCRIPT
##############################################################



m.main()
acts = m.acts
class_labels = m.class_labels



out_filename = 'extra_precss_data_2.csv'

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
    'tmc_recall_p1',
    'tmc_recall_p09',
    'Zhou_no_60']


def row6_outputter(current_neuron_index=current_neuron_index, top_mode_class=top_mode_class,
                   output_precs_data = output_precs_data, Zhou = Zhou):
    """Little wrapper function to write out the row
    this is the stats over the top 100 activations!"""
    row = {'Neuron no.': str(current_neuron_index),  # neuron index
                'top_mode_class_name': str(top_mode_class), # class name for top mode class (class with highest number in top 100)
                'tmc_recall_p1' : str(output_precs_data[0][0]), # start of x = 100 data for top 10 classes
                'tmc_recall_p09': str(output_precs_data[0][1]),
                'Zhou_no_60': str(Zhou)}
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
    for current_neuron_index in current_range:
        if verbose:
            print('Grabbing the points for neuron {}'.format(current_neuron_index))
        # # this grabs all the points - note that this gets the data out of acts and is slow so we don;twant to do this twice
        local_list, selected_activations, x_data = p.get_local_list_for_neuron(
            current_neuron_index=current_neuron_index,
            minx=0,
            maxx='',
            acts=acts)
        top_mode_class, zhou_precs60, zhou_no_of_classes60, zhou60 = p.find_zhou_precision(
            number_of_points=60, local_list=local_list)
        output_precs_data = []
        placeholder = (0.0, 0.0, 0.0, 0.0)
        # now we loop over all 10 classes found by Zhou
        egg = calculate_missing_precs_recall_stats(test_class=top_mode_class,
                                local_list=local_list,
                                Q_stop='',
                                no_files_in_label=no_files_in_label,
                                no_of_images=no_of_images,
                                verbose=verbose)
        output_precs_data.append(egg)
        if verbose:
            print('Nearly finished analysing unit {}'.format(current_neuron_index))
        row = row6_outputter(current_neuron_index=current_neuron_index, top_mode_class=top_mode_class,
                             output_precs_data=output_precs_data,
                             Zhou=zhou_no_of_classes60
                             )
        print(row)
        sorted_row = OrderedDict(sorted(row.items(), key=lambda item: fieldnames.index(item[0])))
        writer.writerow(sorted_row)
        sys.stdout.flush()
