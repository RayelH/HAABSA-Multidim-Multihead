#!/usr/bin/env python
# encoding: utf-8

import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter
import numpy as np

def lcr_rot(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2, _id='all'):
    print('I am lcr_rot_alt.')
    cell = tf.contrib.rnn.LSTMCell
    # left hidden
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    pool_l = reduce_mean_with_len(hiddens_l, sen_len_fw)

    # right hidden
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
    pool_r = reduce_mean_with_len(hiddens_r, sen_len_bw)

    # target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl')
    outputs_t_l_init = tf.matmul(att_l, hiddens_l)
    outputs_t_l = tf.squeeze(outputs_t_l_init)

    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr')
    outputs_t_r_init = tf.matmul(att_r, hiddens_r)
    outputs_t_r = tf.squeeze(outputs_t_r_init)

    # attention target left
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l')
    outputs_l_init = tf.matmul(att_t_l, hiddens_t)
    outputs_l = tf.squeeze(outputs_l_init)
    # attention target right
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r')
    outputs_r_init = tf.matmul(att_t_r, hiddens_t)
    outputs_r = tf.squeeze(outputs_r_init)

    outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    for i in range(2):
        # attention target
        att_l = bilinear_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl'+str(i))
        outputs_t_l_init = tf.matmul(att_l, hiddens_l)
        outputs_t_l = tf.squeeze(outputs_t_l_init)

        att_r = bilinear_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr'+str(i))
        outputs_t_r_init = tf.matmul(att_r, hiddens_r)
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # attention left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l'+str(i))
        outputs_l_init = tf.matmul(att_t_l, hiddens_t)
        outputs_l = tf.squeeze(outputs_l_init)

        # attention right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r'+str(i))
        outputs_r_init = tf.matmul(att_t_r, hiddens_t)
        outputs_r = tf.squeeze(outputs_r_init)

        outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin1'+str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'fin2'+str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    prob = softmax_layer(outputs_fin, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, l2, FLAGS.n_class)
    return prob, att_l, att_r, att_t_l, att_t_r

def main(train_path, test_path, accuracyOnt, trainAccuracyOnt, test_size, remaining_size, learning_rate=FLAGS.learning_rate, keep_prob=FLAGS.keep_prob1, momentum=0.9, l2=FLAGS.l2_reg):
    print_config()

    word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
    word_embedding = tf.constant(w2v, name='word_embedding')

    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
        sen_len = tf.placeholder(tf.int32, None)

        x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        sen_len_bw = tf.placeholder(tf.int32, [None])

        target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
        tar_len = tf.placeholder(tf.int32, [None])

    inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
    inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
    target = tf.nn.embedding_lookup(word_embedding, target_words)

    alpha_fw, alpha_bw = None, None
    prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = lcr_rot(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len,
                                                             keep_prob1, keep_prob2, l2, 'all')

    loss = loss_func(y, prob)
    acc_num, acc_prob = acc_func(y, prob)
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss,
                                                                                                    global_step=global_step)
    # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
    true_y = tf.argmax(y, 1)
    pred_y = tf.argmax(prob, 1)

    title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
        FLAGS.keep_prob1,
        FLAGS.keep_prob2,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.l2_reg,
        FLAGS.max_sentence_len,
        FLAGS.embedding_dim,
        FLAGS.n_hidden,
        FLAGS.n_class
    )


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        sess.run(tf.global_variables_initializer())

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _ = load_inputs_twitter(
            train_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _ = load_inputs_twitter(
            test_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        te_x_ont, te_sen_len_ont, te_x_bw_ont, te_sen_len_bw_ont, te_y_ont, te_target_word_ont, te_tar_len_ont, _, _, _ = load_inputs_twitter(
            FLAGS.remaining_test_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        tr_x_ont, tr_sen_len_ont, tr_x_bw_ont, tr_sen_len_bw_ont, tr_y_ont, tr_target_word_ont, tr_tar_len_ont, _, _, _ = load_inputs_twitter(
            FLAGS.remaining_train_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_fw, max_bw = None, None
        max_tl, max_tr = None, None
        max_ty, max_py = None, None
        max_prob = None
        step = None

        all_training_losses, all_training_accuracies = [], []
        all_test_losses, all_test_accuracies = [], []

        for i in range(FLAGS.n_iter):
            learning_rate = (0.99) * learning_rate
            number_of_training_examples_correct, number_of_training_examples, training_loss = 0., 0, 0.
            for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len,
                                           FLAGS.batch_size, keep_prob, keep_prob):

                _, step, _trainacc, _training_loss = sess.run([optimizer, global_step, acc_num, loss], feed_dict=train)

                number_of_training_examples_correct += _trainacc
                number_of_training_examples += numtrain
                training_loss += _training_loss * numtrain

            number_of_test_examples_correct, test_loss, number_of_test_examples = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                            te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
                if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                    _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                        [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                    tl += list(_tl)
                    tr += list(_tr)
                else:
                    _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run([loss, acc_num, true_y, pred_y, prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r], feed_dict=test)
                ty = np.asarray(_ty)
                py = np.asarray(_py)
                p = np.asarray(_p)
                fw = np.asarray(_fw)
                bw = np.asarray(_bw)
                tl = np.asarray(_tl)
                tr = np.asarray(_tr)
                number_of_test_examples_correct += _acc
                test_loss += _loss * num
                number_of_test_examples += num

            number_of_test_examples_correct_ont, number_of_test_examples_ont = 0., 0
            for test_ont, num_ont in get_batch_data(te_x_ont, te_sen_len_ont, te_x_bw_ont, te_sen_len_bw_ont, te_y_ont,
                                            te_target_word_ont, te_tar_len_ont, 2000, 1.0, 1.0, False):
                _acc_ont = sess.run(acc_num, feed_dict=test_ont)
                number_of_test_examples_correct_ont += _acc_ont
                number_of_test_examples_ont += num_ont

            number_of_train_examples_correct_ont, number_of_train_examples_ont = 0., 0
            for train_ont, num_train_ont in get_batch_data(tr_x_ont, tr_sen_len_ont, tr_x_bw_ont, tr_sen_len_bw_ont,
                                                           tr_y_ont,
                                                           tr_target_word_ont, tr_tar_len_ont, 2000, 1.0, 1.0, False):
                _acc_ont_train = sess.run(acc_num, feed_dict=train_ont)
                number_of_train_examples_correct_ont += _acc_ont_train
                number_of_train_examples_ont += num_train_ont

            print(
                'number of training examples={}, correct training examples={}, number of test examples={}, correct test examples={}, number of examples without onto = {}'
                    .format(number_of_training_examples, number_of_training_examples_correct, number_of_test_examples,
                            number_of_test_examples_correct, number_of_test_examples_ont))
            training_accuracy = number_of_training_examples_correct / number_of_training_examples
            test_accuracy = number_of_test_examples_correct / number_of_test_examples
            test_accuracy_ont = number_of_test_examples_correct_ont / number_of_test_examples_ont
            train_accuracy_ont = number_of_train_examples_correct_ont / number_of_train_examples_ont
            totalacc_train = ((train_accuracy_ont * number_of_train_examples_ont) + (trainAccuracyOnt * (
                        number_of_training_examples - number_of_train_examples_ont))) / number_of_training_examples
            totalacc = ((test_accuracy_ont * number_of_test_examples_ont) + (accuracyOnt * (
                        number_of_test_examples - number_of_test_examples_ont))) / number_of_test_examples
            average_test_loss = test_loss / number_of_test_examples
            average_training_loss = training_loss / number_of_training_examples
            print(
                'Epoch {}: average training loss={:.6f}, train acc={:.6f}, average test loss={:.6f}, test acc={:.6f}, combined acc={:.6f}, accuracy without onto={:.6f}, in-sample with onto = {}'.format(
                    i, average_training_loss, training_accuracy, average_test_loss, test_accuracy, totalacc, test_accuracy_ont, totalacc_train))


        max_acc = test_accuracy
        max_fw = fw
        max_bw = bw
        max_tl = tl
        max_tr = tr
        max_ty = ty
        max_py = py
        max_prob = p

        # fp = open(FLAGS.prob_file + '_hierarchical' + str(FLAGS.year), 'w')
        # for item in max_prob:
        #     fp.write(' '.join([str(it) for it in item]) + '\n')
        # fp = open(FLAGS.prob_file + '_fw_hierarchical' + str(FLAGS.year), 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_fw):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_bw_hierarchical' + str(FLAGS.year), 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_bw):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_tl_hierarchical' + str(FLAGS.year), 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_tl):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_tr_hierarchical' + str(FLAGS.year), 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_tr):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        print('Optimization Finished! Max acc={}'.format(max_acc))

        print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        ))

        return training_accuracy, max_acc, totalacc_train, totalacc, test_accuracy_ont, np.where(np.subtract(max_py, max_ty) == 0, 0, 1), max_fw.tolist(), max_bw.tolist(), max_tl.tolist(), max_tr.tolist()

if __name__ == '__main__':
    tf.app.run()
