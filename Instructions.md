# Instructions

You will need to change the directories in the files
Timings are given for how long each step took on a well-specced machine bought in 2016

(Download and install caffe)
(Download ImageNet 2012 (this will take a day))

## To create a set of square-cropped correct images
run `Make_squarified_test_set.py`

## To create a set of top-1 images
run `Make_top_1_set.py`

## To create merged .h5 files of activations

ImageNet2012 should contain 1000 subdirectories labelled with the ImageNet code (something like n01234556)

1. Change `set_up_caffe_net.py` if needed, this script contains links to the ImageNet and caffe directories
Edit the settings in `set_up_caffe_net.py`

2. Run `Caffe_AlexNet2.py` in your ImageNet directory
 - this sets up caffe and builds .h5 files of activations for the individual directories (for each class) in ImageNet2012
 - this is where the NN is given the ImageNet files and tested
 - you can choose whether to take all activations or only the best ones
 - this makes 1000 `n0123456.h5' files (or less if you've changed the settings).
 - this takes about a day
 - change `blob` (blob is caffes word for which layer of a NN you want, ie. `fc6`)
 
3. run `merger.py`
  - this merges the individual .h5 files into one big file
  - to save space I would normally delete the invidual .h5 files (as Caffe_AlexNet2.py does not overwrite the individual h5 files) and move the merged h5 to a different directory.
 
 ## To load .h5 files into an activation table
  1. run `Make_activation.py`
    - this can load merged or individual .h5 files. 
    - you will need to change the directories
    
## To get jitterplots and many of the selectivity measures
 This is best done in an interactive python 3 terminal
 1. create merged h5 files and load them into an activation table as above
 2. run h5_analysis_jitterer.py (check settings)
    - creates a csv output file containing: 

        ```fieldnames = ['Neuron no.',  # neuron index
                  'top_class_name',
                  'all_K',  # no of K for 'All': whole of (midX to maxX) range
                  'all_No_images',  # No of images over All
                  'biggest_gap',  # Size of biggest gap: this defines the start of 'Struct' range
                  'big_gap_code',  # Coded position of gap: 0 is top cluster, counting down
                  'second_biggest_gap',  # Second biggest gap size --> could be used as struct start
                  '2_Big_gap_code',  # Gap position code
                  'top_class',  # Class with highest activation- could be a list
                  'c_0_no',  # No. images in cluster 0 (top cluster)
                  'c_0_no_class',  # No. of classes in top cluster
                  'struct_no',  # No. of images in structured region
                  'struct_K',  # No. of clusters in struct range --> this may be after a 2nd kmeans
                  'struct_no_class',  # No of classes in structured region
                  'No_top_in_cluster_0',  # No. of top class in top cluster
                  'No_top_class_in_struct',  # No. of top class in structure
                  'No_top_class_in_half_range',  # No. of top class in half range
                  'No_top_class',  # No in the top class overall
                  'pc_top_class_in_top_100',  # pc of top class in top 100
                  'is_class_selective',
                  'ccma_selectivity_top',  # ccma_selectivity to top activating class
                  'mu_max_top',  # average activation of top activating class
                  'ccma_selectivity',  # ccma_selectivity of highest mean activation class
                  'mu_max',  # mean of highest mean activatinging class
                  'mean_act_class_name',  # name of highest mean class
                  'ccma_selectivity_2',  # ccma_selectivity of 2nd highest mean activation class
                  'mu_max_2',  # mean of second highest mean activatinging class
                  'mean_act_class_name_2',  # name of highest mean class
                  'range_top',  # range of top activating class
                  'range_mean',  # range of class with highest mean activation
                  'range_2_mean',  # range of class with second highest mean activation
                  'gap_selectivity',  # sub-group selectivity on anything above the largest gap
                  'extent_of_top_class'  # number of top activations before the class changes
                  ]```

## To get precision and other selectivity measures

  1. as above, create a merged .h5 file and load it into an activation table
  2. run `precision_calculator.py`
      - This creates several selectivity measures for the top 10 classes and outputs four separate csv files
    1. 'precs_data.csv'
      - contains values for max average precision, precision, Zhou's measure, and values calculated from the 100 highest activations
    2. = 'precs_data_nonzero.csv'
      - values calculate from all non-zero activations (the majority of activations are zero)
    3. = 'precs_data_all.csv'
      - values calculated from all activations
    4. = 'precs_data_zeros.csv'
      - count of which classes are the most activated in that they have the least number of zeros
      
    Full set of values are given below: 
    
        ```fieldnames1 = ['top_mode_class_name',       # class name for top mode class (class with highest number in top 100)
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
        '1_x_for_max_f1_100'
        ]```


      ```fieldnames2 = ['Neuron no.',  # neuron index
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
        '1_x_for_max_f1_nonzero'
        ]```

    ```fieldnames3 = ['Neuron no.',  # neuron index
               'max_ave_precs_all_class',
               'second_max_ave_precs_all_class',
               '1_class',
               '1_Ave_precs_all',  # start of x = 100 data for top 10 classes
               '1_precs_all',
               '1_recall_all'
                ]```
  
    ```fieldnames4 = ['Neuron no.',  # neuron index
           'num_zeros',
           '1_least_zero_class',  # start of x = 100 data for top 10 classes
           '1_least_zero_num'
           ]```
 
If needed, extra values are calculated in precision_calculator2:

     'top_mode_class_name',       # class name for top mode class (class with highest number in top 100)
    'tmc_recall_p1',(top mode class, recall calculated at precision of 1)
    'tmc_recall_p095' (top mode class, recall calculated at precision of 0.95)
    
    If needed, extra values are calculated in precision_calculator3:
    'top_mode_class_name',       # class name for top mode class (class with highest number in top 100)
    'tmc_recall_p1',
    'tmc_recall_p09',
    'Zhou_no_60' # zhou's precision measure calculated from the top 60 activations
  
 ## To do selectivity experiments with architectures other than AlexNet
 ### To check for selectivity using buses and cars as in figure 6.
 
 1. run test_zhou_[NN_name] (e.g. `test_zhou_densenet.py`) in a python3 terminal
 2. you can then change the unit and classname to test different units and classes (we chose cars and buses as the NNs seemed to best selective to these classes and they are truely objects). 
 
 ### To calculate the selectivty metrics for architectures other than AlexNet:
 
 1. run IM_test_zhou_[architecture], e.g. `IM_test_zhou_googlenet_places365.py`
