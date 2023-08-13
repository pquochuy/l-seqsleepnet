import os, gc, json, argparse, pickle, time, shutil, select, sys
import numpy as np
from os.path import join as pj

from scipy.io import loadmat, savemat
from sklearn import metrics
from imblearn.metrics import specificity_score
from imblearn.metrics import sensitivity_score

import h5py
import hdf5storage

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/', help='the main directory where the output file is stored')
parser.add_argument('--num_fold', type=int, default=20)
parser.add_argument('--num_repeat', type=int, default=5)
parser.add_argument('--out_filename', type=str, default='test_ret.mat', help='the mat file containing the network predictions')
parser.add_argument('--datalist_dir', type=str, default='../../file_list_20sub/eeg/', help='the data directory to look for the files containing data list')
parser.add_argument('--aggregation', type=str, default='multiplication', help='aggregtation type: average/multiplication')
parser.add_argument('--subseqlen', type=int, default=10, help='subsequence length')
parser.add_argument('--nsubseq', type=int, default=20, help='numer of subsequences')
parser.add_argument('--nstage', type=int, default=5, help='number of sleep stages')
config = parser.parse_args()

# affective sequence length
config.seq_len = config.subseqlen*config.nsubseq

def read_groundtruth(filelist):
    labels = dict()
    file_sizes = []
    with open(filelist) as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        print(l)
        items = l.split()
        file_sizes.append(int(items[1]))
        #data = h5py.File(items[0], 'r')
        data = hdf5storage.loadmat(file_name=items[0])
        label = np.array(data['label'])  # labels
        label = np.transpose(label, (1, 0))  # rearrange dimension
        label = np.squeeze(label)
        labels[i] = label
    return labels, file_sizes

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

#score = np.zeros([config.seq_len, len(gen.data_index), config.nclass])
def aggregate_avg(score):
    fused_score = None
    for i in range(config.seq_len):
        prob_i = np.concatenate((np.zeros((config.seq_len - 1, config.nstage)), softmax(np.squeeze(score[i, :, :]))), axis=0)
        prob_i = np.roll(prob_i, -(config.seq_len - i - 1), axis=0)
        if fused_score is None:
            fused_score = prob_i
        else:
            fused_score += prob_i
    label = np.argmax(fused_score, axis=-1) + 1 # +1 as label stored in mat files start from 1
    return label

#score = np.zeros([config.seq_len, len(gen.data_index), config.nclass])
def aggregate_mul(score):
    fused_score = None
    for i in range(config.seq_len):
        prob_i = np.log10(softmax(np.squeeze(score[i, :, :])))
        prob_i = np.concatenate((np.ones((config.seq_len - 1, config.nstage)), prob_i), axis=0)
        prob_i = np.roll(prob_i, -(config.seq_len - i - 1), axis=0)
        if fused_score is None:
            fused_score = prob_i
        else:
            fused_score += prob_i
    label = np.argmax(fused_score, axis=-1) + 1 # +1 as label stored in mat files start from 1
    return label

def aggregate_lseqsleepnet(output_file, file_sizes):
    # outputs = loadmat(output_file)
    outputs = hdf5storage.loadmat(file_name=output_file)
    outputs = outputs['score']
    # score = [len(gen.data_index), config.epoch_seq_len, config.nclass] -> need transpose
    outputs = np.transpose(outputs, (1, 0, 2))
    preds = dict()
    sum_size = 0
    for i, N in enumerate(file_sizes):
        score = outputs[:, sum_size:(sum_size + N - (config.seq_len - 1))]
        sum_size += N - (config.seq_len - 1)
        preds[i] = aggregate_avg(score) if config.aggregation == "average" else aggregate_mul(score)
    return preds


def calculate_metrics(labels, preds):
    ret = dict()
    #labels = np.hstack(list(labels.values()))
    #preds = np.hstack(list(preds.values()))
    ret['acc'] = metrics.accuracy_score(y_true=labels, y_pred=preds)
    ret['F1'] = metrics.f1_score(y_true=labels, y_pred=preds, labels=np.arange(1, config.nstage+1), average=None)
    ret['mean-F1'] = np.mean(ret['F1'])
    ret['kappa'] = metrics.cohen_kappa_score(y1=labels, y2=preds, labels=np.arange(1, config.nstage+1))
    ret['sensitivity'] = sensitivity_score(labels, preds, average='macro')
    ret['specificity'] = specificity_score(labels, preds, average='macro')
    return ret

results = dict()
for repeat in range(config.num_repeat):
    label_list = []
    pred_list = []
    for n in range(config.num_fold):
        data_list_file = pj(config.datalist_dir, 'test_list_n' + str(n+1) + '.txt')
        labels, file_sizes = read_groundtruth(data_list_file)
        label_list.extend(list(labels.values()))
        preds = aggregate_lseqsleepnet(pj(config.out_dir, 'repeat'+str(repeat+1), 'n'+str(n+1), config.out_filename), file_sizes)
        pred_list.extend(list(preds.values()))
    results['Repeat'+str(repeat+1)] = calculate_metrics(np.hstack(label_list), np.hstack(pred_list))

# print results
for repeat in range(config.num_repeat):
    print(results['Repeat' + str(repeat+1)])