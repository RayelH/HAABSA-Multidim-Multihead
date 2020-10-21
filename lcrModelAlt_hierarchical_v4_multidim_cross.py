#!/usr/bin/env python
# encoding: utf-8

import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import bilinear_attention_layer, dot_produce_attention_layer, multidimensional_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter
import numpy as np

tf.set_random_seed(1)

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
    att_l = multidimensional_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl')
    outputs_t_l_init = tf.transpose(tf.reduce_sum((tf.multiply(att_l, tf.transpose(hiddens_l, perm=[0, 2, 1]))), reduction_indices=-1, keep_dims=True) + 1e-9,[0,2,1])
    outputs_t_l = tf.squeeze(outputs_t_l_init)

    # attention right
    att_r = multidimensional_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr')
    outputs_t_r_init = tf.transpose(tf.reduce_sum((tf.multiply(att_r, tf.transpose(hiddens_r, perm=[0, 2, 1]))), reduction_indices=-1, keep_dims=True) + 1e-9,[0,2,1])
    outputs_t_r = tf.squeeze(outputs_t_r_init)

    # attention target left
    att_t_l = multidimensional_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l')
    outputs_l_init = tf.transpose(tf.reduce_sum((tf.multiply(att_t_l, tf.transpose(hiddens_t, perm=[0, 2, 1]))), reduction_indices=-1, keep_dims=True) + 1e-9,[0,2,1])
    outputs_l = tf.squeeze(outputs_l_init)

    # attention target right
    att_t_r = multidimensional_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r')
    outputs_r_init = tf.transpose(tf.reduce_sum((tf.multiply(att_t_r, tf.transpose(hiddens_t, perm=[0, 2, 1]))), reduction_indices=-1, keep_dims=True) + 1e-9,[0,2,1])
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
        att_l = multidimensional_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl'+str(i))
        outputs_t_l_init = tf.transpose(tf.reduce_sum((tf.multiply(att_l, tf.transpose(hiddens_l, perm=[0, 2, 1]))), reduction_indices=-1, keep_dims=True) + 1e-9,[0,2,1])
        outputs_t_l = tf.squeeze(outputs_t_l_init)

        att_r = multidimensional_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr'+str(i))
        outputs_t_r_init = tf.transpose(tf.reduce_sum((tf.multiply(att_r, tf.transpose(hiddens_r, perm=[0, 2, 1]))), reduction_indices=-1, keep_dims=True) + 1e-9,[0,2,1])
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # attention left
        att_t_l = multidimensional_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l'+str(i))
        outputs_l_init = tf.transpose(tf.reduce_sum((tf.multiply(att_t_l, tf.transpose(hiddens_t, perm=[0, 2, 1]))), reduction_indices=-1, keep_dims=True) + 1e-9,[0,2,1])
        outputs_l = tf.squeeze(outputs_l_init)

        # attention right
        att_t_r = multidimensional_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r'+str(i))
        outputs_r_init = tf.transpose(tf.reduce_sum((tf.multiply(att_t_r, tf.transpose(hiddens_t, perm=[0, 2, 1]))), reduction_indices=-1, keep_dims=True) + 1e-9,[0,2,1])
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

def main(train_path, test_path, learning_rate=FLAGS.learning_rate, keep_prob=FLAGS.keep_prob1, l2=FLAGS.l2_reg, beta=0.9, number_epochs=100):
    print_config()
    batch_size = FLAGS.batch_size
    with tf.device('/gpu:1'):
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
        prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = lcr_rot(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len, keep_prob1, keep_prob2, l2, 'all')

        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,  momentum=beta).minimize(loss, global_step=global_step)
        # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

        title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            FLAGS.keep_prob1,
            FLAGS.keep_prob2,
            FLAGS.batch_size,
            learning_rate,
            l2,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class
        )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))

        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, '/-')

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

        for i in range(number_epochs):
            learning_rate = (0.99) * learning_rate
            number_of_training_examples_correct, number_of_training_examples, training_loss = 0., 0, 0.
            for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len,
                                           FLAGS.batch_size, keep_prob, keep_prob):
                # _, step = sess.run([optimizer, global_step], feed_dict=train)
                _, step, _trainacc, _training_loss = sess.run([optimizer, global_step, acc_num, loss], feed_dict=train)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)
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
            print(
                'number of training examples={}, correct training examples={}, number of test examples={}, correct test examples={}'
                .format(number_of_training_examples, number_of_training_examples_correct, number_of_test_examples,
                        number_of_test_examples_correct))

            training_accuracy = number_of_training_examples_correct / number_of_training_examples
            test_accuracy = number_of_test_examples_correct / number_of_test_examples
            average_test_loss = test_loss / number_of_test_examples
            average_training_loss = training_loss / number_of_training_examples

            all_training_losses.append(average_training_loss)
            all_training_accuracies.append(training_accuracy)
            all_test_losses.append(average_test_loss)
            all_test_accuracies.append(test_accuracy)

            print(
                'Epoch {}: average training loss={:.6f}, train acc={:.6f}, average test loss={:.6f}, test acc={:.6f}'.format(
                    i, average_training_loss, training_accuracy, average_test_loss, test_accuracy))

            # if acc > max_acc:
            #     max_acc = acc
            #     max_fw = fw
            #     max_bw = bw
            #     max_tl = tl
            #     max_tr = tr
            #     max_ty = ty
            #     max_py = py
            #     max_prob = p

        min_training_loss = min(all_training_losses)
        max_training_accuracy = max(all_training_accuracies)
        min_test_loss = min(all_test_losses)
        max_test_accuracy = max(all_test_accuracies)

        # P = precision_score(max_ty, max_py, average=None)
        # R = recall_score(max_ty, max_py, average=None)
        # F1 = f1_score(max_ty, max_py, average=None)
        # print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        # print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        # print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)
        #
        # fp = open(FLAGS.prob_file, 'w')
        # for item in max_prob:
        #     fp.write(' '.join([str(it) for it in item]) + '\n')
        # fp = open(FLAGS.prob_file + '_fw', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_fw):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_bw', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_bw):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_tl', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_tl):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        # fp = open(FLAGS.prob_file + '_tr', 'w')
        # for y1, y2, ws in zip(max_ty, max_py, max_tr):
        #     fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        print('Optimization Finished! Max acc={}'.format(max_acc))

        print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            learning_rate,
            number_epochs,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            l2
        ))

        return min_training_loss, max_training_accuracy, min_test_loss, max_test_accuracy, all_training_losses, all_training_accuracies, all_test_losses, all_test_accuracies

if __name__ == '__main__':
    tf.app.run()
