# testing_object_detectors_in_deepCNNs
Code for Gale et al "Are there any `object detectors' in the hidden layers of CNNs trained to identify objects or scenes?", Vision Research, 2020

This code is the code that was used in "Are there any `object detectors' in the hidden layers of CNNs trained to identify objects or scenes?" and it is offered here with no warranty. 

This code will not be updated or maintained.

If you wish to repeat the experiment, you can use the code exactly as is, with the following (slightly involved) instructions (I suggest doing it in a docker container). 

If you wish to build on this research your best bet is to take the relevant analysis functions and put them into your pipeline. This code expects you to use keras that calls tensorflow1 as a backend, and includes the code to get ImageNet into the NN, create the activations, save them out as .h5 files and read them back in to perform the analysis. 

TensorFlow2 has recently come out and it is MUCH more user friendly and it is much easier to build a datapipe line (using tensorflow datasets) and keras now comes as standard inside tensorflow2. I am currently in the process of building a new codebase that uses this pipeline (for a different project), so you might want to look at the pipeline in my clusterflow project (due online in the next six months) and combine that with the analysis functions in this codebase. I'm sorry but I am unlikely to combine that new pipeline with this project.

If you find this code useful, please cite our paper your follow on work and within your codebase. 
