import os
import numpy as np
import tensorflow as tf

import shutil, sys
from datetime import datetime
import h5py
import hdf5storage

from lseqsleepnet import LSeqSleepNet
from config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_wrapper import DataGeneratorWrapper
import time

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "./train_data_eeg.txt", "file containing the list of training EEG data")
tf.app.flags.DEFINE_string("eeg_test_data", "./eval_data_eeg.txt", "file containing the list of test EEG data")
tf.app.flags.DEFINE_string("eog_train_data", "", "file containing the list of training EOG data")
tf.app.flags.DEFINE_string("eog_test_data", "", "file containing the list of test EOG data")
tf.app.flags.DEFINE_string("emg_train_data", "", "file containing the list of training EMG data")
tf.app.flags.DEFINE_string("emg_test_data", "", "file containing the list of test EMG data")

tf.app.flags.DEFINE_string("out_dir", "./output/", "Output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Checkpoint directory")

tf.app.flags.DEFINE_float("dropout_rnn", 0.75, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden2", 64, "Sequence length (default: 20)")

# subsuqence length
tf.app.flags.DEFINE_integer("sub_seq_len", 10, "Sequence length (default: 32)")
# number of subsequence
tf.app.flags.DEFINE_integer("nsubseq", 10, "number of overall segments (default: 9)")

tf.app.flags.DEFINE_integer("dualrnn_blocks", 1, "Number of dual rnn blocks (default: 1)")

tf.app.flags.DEFINE_float("gpu_usage", 0.5, "Dropout keep probability (default: 0.5)")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

config = Config()
config.dropout_rnn = FLAGS.dropout_rnn
config.sub_seq_len = FLAGS.sub_seq_len
config.nfilter = FLAGS.nfilter
config.nhidden1 = FLAGS.nhidden1
config.nhidden2 = FLAGS.nhidden2
config.attention_size = FLAGS.attention_size

config.nsubseq = FLAGS.nsubseq
config.dualrnn_blocks = FLAGS.dualrnn_blocks

eeg_active = (FLAGS.eeg_train_data != "")
eog_active = (FLAGS.eog_train_data != "")
emg_active = (FLAGS.emg_train_data != "")

if (not eog_active and not emg_active):
    print("eeg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             num_fold=1,
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.sub_seq_len * config.nsubseq,
                                             shuffle=True)
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                             num_fold=1,
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.sub_seq_len * config.nsubseq,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    nchannel = 1

elif(eog_active and not emg_active):
    print("eeg and eog active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             num_fold=1,
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.sub_seq_len * config.nsubseq,
                                             shuffle=True)
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                            eog_filelist=os.path.abspath(FLAGS.eog_test_data),
                                            num_fold=1,
                                            data_shape_2=[config.frame_seq_len, config.ndim],
                                            seq_len = config.sub_seq_len * config.nsubseq,
                                            shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    nchannel = 2
elif(eog_active and emg_active):
    print("eeg, eog, and emg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             emg_filelist=os.path.abspath(FLAGS.emg_train_data),
                                             num_fold=1,
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.sub_seq_len * config.nsubseq,
                                             shuffle=True)
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                            eog_filelist=os.path.abspath(FLAGS.eog_test_data),
                                            emg_filelist=os.path.abspath(FLAGS.emg_test_data),
                                            num_fold=1,
                                            data_shape_2=[config.frame_seq_len, config.ndim],
                                            seq_len = config.sub_seq_len * config.nsubseq,
                                            shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    train_gen_wrapper.compute_emg_normalization_params()
    test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    test_gen_wrapper.set_emg_normalization_params(train_gen_wrapper.emg_meanX, train_gen_wrapper.emg_stdX)
    nchannel = 3

config.nchannel = nchannel

# do not need training data anymore
del train_gen_wrapper

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_usage, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = LSeqSleepNet(config=config)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables())
        # Load the saved model
        best_dir = os.path.join(checkpoint_path, "best_model_acc")
        saver.restore(sess, best_dir)
        print("Model all loaded")

        def dev_step(x_batch, y_batch):
            x_shape = x_batch.shape
            y_shape = y_batch.shape
            x = np.zeros(x_shape[:1] + (config.nsubseq, config.sub_seq_len,) + x_shape[2:])
            y = np.zeros(y_shape[:1] + (config.nsubseq, config.sub_seq_len,) + y_shape[2:])
            for s in range(config.nsubseq):
                x[:, s] = x_batch[:, s * config.sub_seq_len: (s + 1) * config.sub_seq_len]
                y[:, s] = y_batch[:, s * config.sub_seq_len: (s + 1) * config.sub_seq_len]

            frame_seq_len = np.ones(len(x_batch) * config.sub_seq_len * config.nsubseq,
                                        dtype=int) * config.frame_seq_len
            sub_seq_len = np.ones(len(x_batch) * config.nsubseq, dtype=int) * config.sub_seq_len
            inter_subseq_len = np.ones(len(x_batch) * config.sub_seq_len, dtype=int) * config.nsubseq
            feed_dict = {
                net.input_x: x,
                net.input_y: y,
                net.dropout_rnn: 1.0,
                net.inter_subseq_len: inter_subseq_len,
                net.sub_seq_len: sub_seq_len,
                net.frame_seq_len: frame_seq_len,
                net.istraining: 0
            }
            output_loss, total_loss, yhat, score = sess.run([net.output_loss, net.loss, net.prediction, net.score], feed_dict)
            return output_loss, total_loss, yhat, score

        def _evaluate(gen):
            # Validate the model on the entire data in gen
            output_loss =0
            total_loss = 0
            yhat = np.zeros([len(gen.data_index), config.sub_seq_len*config.nsubseq])
            score = np.zeros([len(gen.data_index), config.sub_seq_len*config.nsubseq, config.nclass])

            factor = 20*4

            # use 10x of minibatch size to speed up
            num_batch_per_epoch = np.floor(len(gen.data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for s in range(config.nsubseq):
                    yhat[(test_step - 1) * factor * config.batch_size: test_step * factor * config.batch_size,
                    s*config.sub_seq_len:(s+1)*config.sub_seq_len] = yhat_[:, s]
                    score[(test_step - 1) * factor * config.batch_size: test_step * factor * config.batch_size,
                    s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = score_[:, s]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for s in range(config.nsubseq):
                    yhat[(test_step - 1) * factor * config.batch_size: len(gen.data_index),
                    s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = yhat_[:, s]
                    score[(test_step - 1) * factor * config.batch_size: len(gen.data_index),
                    s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = score_[:, s]
            yhat = yhat + 1 # make label starting from 1 rather than 0

            return yhat, score, output_loss, total_loss


        def evaluate(gen_wrapper):

            N = int(np.sum(gen_wrapper.file_sizes) - (config.sub_seq_len*config.nsubseq - 1)*len(gen_wrapper.file_sizes))
            yhat = np.zeros([N, config.sub_seq_len*config.nsubseq])
            y = np.zeros([N, config.sub_seq_len*config.nsubseq])

            score = np.zeros([N, config.sub_seq_len*config.nsubseq, config.nclass])

            count = 0
            output_loss = 0
            total_loss = 0
            gen_wrapper.new_subject_partition()

            gen_wrapper.next_fold()
            yhat_, score_, output_loss_, total_loss_ = _evaluate(gen_wrapper.gen)

            output_loss += output_loss_
            total_loss += total_loss_

            yhat[count : count + len(gen_wrapper.gen.data_index)] = yhat_
            score[count : count + len(gen_wrapper.gen.data_index)] = score_

            # groundtruth
            for n in range(config.sub_seq_len*config.nsubseq):
                y[count : count + len(gen_wrapper.gen.data_index), n] =\
                    gen_wrapper.gen.label[gen_wrapper.gen.data_index - (config.sub_seq_len*config.nsubseq - 1) + n]
            count += len(gen_wrapper.gen.data_index)

            test_acc = np.zeros([config.sub_seq_len*config.nsubseq])
            for n in range(config.sub_seq_len*config.nsubseq):
                test_acc[n] = accuracy_score(yhat[:, n], y[:, n]) # excluding the indexes of the recordings

            return test_acc, yhat, score, output_loss, total_loss

        # evaluation on test data
        start_time = time.time()
        test_acc, test_yhat, test_score, test_output_loss, test_total_loss = evaluate(gen_wrapper=test_gen_wrapper)
        end_time = time.time()
        with open(os.path.join(out_dir, "test_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))

        hdf5storage.savemat(os.path.join(out_path, "test_ret.mat"),
                            {'yhat': test_yhat,
                             'acc': test_acc,
                             'score': test_score,
                             'output_loss': test_output_loss,
                             'total_loss': test_total_loss},
                            format='7.3')
        test_gen_wrapper.gen.reset_pointer()
