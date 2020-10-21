from loadData import *
from config import *
import lcrModelAlt_hierarchical_v4
import lcrModelAlt_hierarchical_v4_multidim_cross
import lcrModelAlt_hierarchical_v4_multihead
import lcrModelAlt_hierarchical_v4_multihead_multidim
import lcrModelAlt_hierarchical_v4_cross
import lcrModelAlt_hierarchical_v4_multihead_cross


def cross_val(number_of_folds=10, compute_folds=False, multidim=False, multihead=False, multidimmultihead=False,
              learning_rate=FLAGS.learning_rate, keep_prob=FLAGS.keep_prob1, beta=0.9,
              l2=0.01, number_of_heads=6, number_epochs=100):
    BASE_train = "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_train_'
    BASE_val = "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_val_'

    if compute_folds:
        train_size, test_size, train_polarity_vector, test_polarity_vector = loadCrossValidation(FLAGS, number_of_folds)


    if multidim:
        average_training_losses = np.zeros(number_epochs)
        average_test_losses = np.zeros(number_epochs)
        average_training_accuracies = np.zeros(number_epochs)
        average_test_accuracies = np.zeros(number_epochs)

        all_test_losses = np.zeros([number_epochs, number_of_folds])
        all_test_accuracies = np.zeros([number_epochs, number_of_folds])

        for i in range(number_of_folds):
            min_training_loss, max_training_accuracy, min_test_loss, max_test_accuracy, training_losses, train_accuracies, test_losses, test_accuracies = lcrModelAlt_hierarchical_v4_multihead_cross.main(train_path=BASE_train + str(i) + '.txt', test_path=BASE_val + str(i) + '.txt',
                                                         learning_rate=learning_rate, keep_prob=keep_prob, beta=beta, l2=l2, number_of_heads=number_of_heads, number_epochs=number_epochs)

            average_training_losses = np.add(average_training_losses, training_losses)
            average_training_accuracies = np.add(average_training_accuracies, train_accuracies)
            average_test_losses = np.add(average_test_losses, test_losses)
            average_test_accuracies = np.add(average_test_accuracies, test_accuracies)

            all_test_losses[:,i] = test_losses
            all_test_accuracies[:,i] = test_accuracies

            tf.reset_default_graph()
            print('iteration: ' + str(i))

        std_dev_validation = np.std(all_test_losses,1)
        std_dev_accuracy = np.std(all_test_accuracies,1)

        average_training_losses = np.true_divide(average_training_losses, number_of_folds)
        average_training_accuracies = np.true_divide(average_training_accuracies, number_of_folds)
        average_test_losses = np.true_divide(average_test_losses, number_of_folds)
        average_test_accuracies = np.true_divide(average_test_accuracies, number_of_folds)

        avg_min_training_loss = min(average_training_losses)
        avg_min_test_loss = min(average_test_losses)
        avg_max_train_accuracy = max(average_training_accuracies)
        avg_max_test_accuracy = max(average_test_accuracies)

        index_avg_min_training_loss = np.argmin(average_training_losses)
        index_avg_min_test_loss = np.argmin(average_test_losses)
        index_avg_max_train_accuracy = np.argmax(average_training_accuracies)
        index_avg_max_test_accuracy = np.argmax(average_test_accuracies)

        last_training_loss = average_training_losses[-1]
        last_test_loss = average_test_losses[-1]
        last_train_accuracy = average_training_accuracies[-1]
        last_test_accuracy = average_test_accuracies[-1]

        with open("cross_results_" + str(FLAGS.year) + "/multihead_new" + str(FLAGS.year) + '.txt', 'a') as result:
            result.write('Config: l_2 = {}, learning rate = {}, number_epochs = {}, dropout keep probability = {}, number of heads = {} \n'.format(l2, learning_rate, number_epochs, keep_prob, number_of_heads))
            result.write('Max Validation Accuracy = {}, at Epoch {} \n'.format(avg_max_test_accuracy, index_avg_max_test_accuracy))
            result.write('Max Train Accuracy = {}, at Epoch {} \n'.format(avg_max_train_accuracy,
                                                                              index_avg_max_train_accuracy))
            result.write('Min Validation Cost = {}, at Epoch {} \n'.format(avg_min_test_loss,
                                                                              index_avg_min_test_loss))
            result.write('Min Train Cost = {}, at Epoch {} \n'.format(avg_min_training_loss,
                                                                              index_avg_min_training_loss))
            result.write('training losses = {}'.format(average_training_losses))
            result.write('\n')
            result.write('validation losses = {}'.format(average_test_losses))
            result.write('\n')
            result.write('training accuracies = {}'.format(average_training_accuracies))
            result.write('\n')
            result.write('validation accuracies = {}'.format(average_test_accuracies))
            result.write('\n')
            result.write('std dev validation error = {}'.format(std_dev_validation))
            result.write('\n')
            result.write('std dev validation accuracy = {}'.format(std_dev_accuracy))
            result.write('\n')



def main(_):
    learning_rates = [0.07, 0.12]
    l2s = [0.00001, 0.00005, 0.0002, 0.001, 0.01]
    keep_prob = 0.5
    number_epochs = [400]
    head_sizes = [2, 3, 4]


    count = 0
    compute_folds = True

    for i in range(len(learning_rates)):
        for j in range(len(l2s)):
            for l in range(len(head_sizes)):
                for k in range(len(number_epochs)):
                    count += 1
                    print('learning_rate = {}, l2 = {}, number_epochs = {}, keep_prob = {}, count = {}'.format(
                        learning_rates[i], l2s[j], number_epochs[k], keep_prob, count))
                    cross_val(number_of_folds=5, compute_folds=compute_folds, multidim=True,
                              learning_rate=learning_rates[i], l2=l2s[j], keep_prob=keep_prob, number_of_heads=head_sizes[l],
                              number_epochs=number_epochs[k])
                    compute_folds = False




if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()