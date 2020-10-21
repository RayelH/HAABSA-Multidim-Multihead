seed_value = 1

import os
os.environ['PYTHONHASHSEED']=str(seed_value)
seed_value += 1

import random
random.seed(seed_value)
seed_value += 1

import numpy as np
np.random.seed(seed_value)
seed_value += 1

import tensorflow as tf
tf.set_random_seed(seed_value)

# from OntologyReasoner import OntReasoner
from loadData import *

# import parameter configuration and data paths
from config import *

# import modules
import sys

import lcrModelAlt_hierarchical_v1
import lcrModelAlt_hierarchical_v2
import lcrModelAlt_hierarchical_v3
import lcrModelAlt_hierarchical_v4
import lcrModelAlt_hierarchical_v4_multidim
import lcrModelAlt_hierarchical_v4_multihead
import lcrModelAlt_hierarchical_v4_multihead_multidim


def main(_):
    accuracyOnt = 0.868159204
    trainAccuracyOnt = 0.845567207
    remaining_size = 301
    test = FLAGS.test_path
    number_iters = 10
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, False)
    all_accuracies = np.zeros(number_iters)
    all_training_accuracies = np.zeros(number_iters)
    all_combined_accuracy_train = np.zeros(number_iters)
    all_combined_accuracy_test = np.zeros(number_iters)
    all_test_accuracy_no_ont = np.zeros(number_iters)
    for i in range(number_iters):
        training_accuracy, accuracy, combined_accuracy_train, combined_accuracy_test, test_accuracy_no_ont, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4_multihead.main(FLAGS.train_path, test, accuracyOnt,
                                                                                  trainAccuracyOnt,
                                                                                  test_size,
                                                                                  remaining_size)
        all_accuracies[i] = accuracy
        all_training_accuracies[i] = training_accuracy
        all_combined_accuracy_train[i] = combined_accuracy_train
        all_combined_accuracy_test[i] = combined_accuracy_test
        all_test_accuracy_no_ont[i] = test_accuracy_no_ont
        tf.reset_default_graph()
        print('iteration: ' + str(i))

    average_accuracy = np.mean(all_accuracies)
    average_training_accuracy = np.mean(all_training_accuracies)
    average_all_combined_accuracy_train = np.mean(all_combined_accuracy_train)
    average_all_combined_accuracy_test = np.mean(all_combined_accuracy_test)
    average_all_test_accuracy_no_ont = np.mean(all_test_accuracy_no_ont)

    max_accuracy = np.max(all_accuracies)
    max_training_accuracy = np.max(all_training_accuracies)
    max_all_combined_accuracy_train = np.max(all_combined_accuracy_train)
    max_all_combined_accuracy_test = np.max(all_combined_accuracy_test)
    max_all_test_accuracy_no_ont = np.max(all_test_accuracy_no_ont)

    min_accuracy = np.min(all_accuracies)
    min_training_accuracy = np.min(all_training_accuracies)
    min_all_combined_accuracy_train = np.min(all_combined_accuracy_train)
    min_all_combined_accuracy_test = np.min(all_combined_accuracy_test)
    min_all_test_accuracy_no_ont = np.min(all_test_accuracy_no_ont)

    stdev = np.std(all_accuracies, ddof=1)
    training_stdev = np.std(all_training_accuracies, ddof=1)
    stdev_all_combined_accuracy_train = np.std(all_combined_accuracy_train, ddof=1)
    stdev_all_combined_accuracy_test = np.std(all_combined_accuracy_test, ddof=1)
    stdev_all_test_accuracy_no_ont = np.std(all_test_accuracy_no_ont, ddof=1)

    fp = open('consistency_runs_results/' + str(FLAGS.year) + '/multihead', 'w')
    fp.write('average test accuracy = {}, max = {}, min  = {}, stdev = {} \n'.format(average_accuracy, max_accuracy, min_accuracy, stdev))
    fp.write('average training accuracy = {}, max = {}, min = {}, stdev = {} \n'.format(average_training_accuracy, max_training_accuracy, min_training_accuracy, training_stdev))
    fp.write('average combined test accuracy = {}, max = {}, min  = {}, stdev = {} \n'.format(average_all_combined_accuracy_test, max_all_combined_accuracy_test, min_all_combined_accuracy_test, stdev_all_combined_accuracy_test))
    fp.write('average combined train accuracy = {}, max = {}, min  = {}, stdev = {} \n'.format(
        average_all_combined_accuracy_train, max_all_combined_accuracy_train, min_all_combined_accuracy_train,
        stdev_all_combined_accuracy_train))
    fp.write('average accuracy obs that ont cant do = {}, max = {}, min  = {}, stdev = {} \n'.format(
        average_all_test_accuracy_no_ont, max_all_test_accuracy_no_ont, min_all_test_accuracy_no_ont,
        stdev_all_test_accuracy_no_ont))


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()