# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC


# seed_value = 1
#
# import os
# os.environ['PYTHONHASHSEED']=str(seed_value)
# seed_value += 1
#
# import random
# random.seed(seed_value)
# seed_value += 1
#
# import numpy as np
# np.random.seed(seed_value)
# seed_value += 1
#
# import tensorflow as tf
# tf.set_random_seed(seed_value)

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
import lcrModelAlt_test



# main function
def main(_):
    loadData = False  # only for non-contextualised word embeddings.
    #   Use prepareBERT for BERT (and BERT_Large) and prepareELMo for ELMo
    useOntology = False  # When run together with runLCRROTALT, the two-step method is used
    runLCRROTALT = False

    runSVM = False
    runCABASC = False
    runLCRROT = False
    runLCRROTINVERSE = False

    runLCRROTALT_v4 = False
    runLCRROTALT_v4_multidim = True
    runLCRROTALT_v4_multihead = False
    runLCRROTALT_v4_multidim_multihead = False
    runLCRROTALT_test = False
    runLCRROTALT_test2 = False

    # determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM or runLCRROTALT_v4 or runLCRROTALT_v4_multidim or runLCRROTALT_v4_multihead or runLCRROTALT_v4_multidim_multihead or runLCRROTALT_test or runLCRROTALT_test2:
        backup = True
    else:
        backup = False

    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
    print(test_size)
    # remaining_size = 301
    # accuracyOnt = 0.8277
    remaining_size = 301
    accuracyOnt = 0.868159204
    trainAccuracyOnt = 0.845567207

    if useOntology == True:
        print('Starting Ontology Reasoner')
        # in sample accuracy
        Ontology = OntReasoner()
        accuracyOnt, remaining_size = Ontology.run(backup, FLAGS.test_path_ont, runSVM)
        print(accuracyOnt)
        print(remaining_size)
        # out of sample accuracy
        # Ontology = OntReasoner()
        # accuracyInSampleOnt, remainingInSample_size = Ontology.run(backup,FLAGS.train_path_ont, runSVM)
        if runSVM == True:
            test = FLAGS.remaining_svm_test_path
        else:
            test = FLAGS.remaining_test_path
        print('train acc = {:.4f}, test acc={:.4f}, remaining size={}'.format(accuracyOnt, accuracyOnt, remaining_size))
    else:
        if runSVM == True:
            test = FLAGS.test_svm_path
        else:
            test = FLAGS.test_path

    if runLCRROTALT_v4 == True:
        _, _, _, _, _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, test, accuracyOnt, trainAccuracyOnt, test_size,
                                                                        remaining_size)
        tf.reset_default_graph()

    if runLCRROTALT_v4_multidim == True:
        _, _, _, _, _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4_multidim.main(FLAGS.train_path, test, accuracyOnt, trainAccuracyOnt,
                                                                                 test_size,
                                                                                 remaining_size)

    if runLCRROTALT_v4_multihead == True:
        _, _, _, _, _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4_multihead.main(FLAGS.train_path, test, accuracyOnt, trainAccuracyOnt,
                                                                                  test_size,
                                                                                  remaining_size)
    if runLCRROTALT_test == True:

        lcrModelAlt_test.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                                  remaining_size)

    if runLCRROTALT_test2 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_text2.main(FLAGS.train_path, test, accuracyOnt,
                                                             test_size,
                                                             remaining_size)


    if runLCRROTALT_v4_multidim_multihead == True:
        _, _, _, _, _, pred2 = lcrModelAlt_hierarchical_v4_multihead_multidim.main(FLAGS.train_path, test, accuracyOnt, trainAccuracyOnt, test_size,
                                                                       remaining_size)



print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
