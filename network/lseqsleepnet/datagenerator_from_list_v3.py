# Sequence data generation for sequence-to-sequence sleep staging. A prepared data file for 1 PSG recording has following variables
# X1: raw data of shape (num_epoch, 3000)
# X2: time-frequency data of shape (num_epoch, 29, 129)
# label: discrete labels of sleep stage of shape (num_epoch, )
# y: one-hot encoding of labels oof shape (num_epoch, 5)

import numpy as np
import h5py

class DataGenerator3:
    def __init__(self, list_of_files, file_sizes, data_shape_2=np.array([29, 129]), seq_len = 200):
        '''
        Mini-batch data generator (L-seqsleepnet only uses time-frequency data). This is very much similar to SeqSleepNet.
        Args:
            list_of_files: list of paths to the data files
            file_sizes: number of sleep epochs of the data files in list_of_files
            data_shape_2: The data shape of one sleep epoch
            seq_len: sequence length
        '''
        # Init params
        self.list_of_files = list_of_files
        self.file_sizes = file_sizes

        self.data_shape_2 = data_shape_2
        self.X2 = None
        self.y = None
        self.label = None

        self.boundary_index = np.array([])

        self.seq_len = seq_len
        self.Ncat = 5 # five-class sleep staging

        self.pointer = 0
        self.data_index = None
        self.data_size = np.sum(self.file_sizes) # total number of epoch

        # read data from mat files in list_of_files
        self.read_mat_filelist()

    # read data from mat files in list_of_files
    def read_mat_filelist(self):
        self.X2 = np.ndarray([self.data_size, self.data_shape_2[0], self.data_shape_2[1]],dtype=np.float32)
        self.y = np.ndarray([self.data_size, self.Ncat],dtype=np.float32)
        self.label = np.ndarray([self.data_size],dtype=np.float32)
        count = 0
        for i in range(len(self.list_of_files)):
            # read in one file
            X2, y, label = self.read_mat_file(self.list_of_files[i].strip())
            self.X2[count : count + len(X2)] = X2.astype(np.float32)
            self.y[count : count + len(X2)] = y.astype(np.float32)
            self.label[count : count + len(X2)] = label.astype(np.float32)
            # boundary_index keeps list of end-of-recording indexes that cannot constitute a full sequence
            self.boundary_index = np.append(self.boundary_index, np.arange(count, count + self.seq_len - 1))
            count += len(X2)
        # indices of all sleep epochs
        self.data_index = np.arange(len(self.X2))
        # exclude those starting indices in the boundary list
        mask = np.in1d(self.data_index,self.boundary_index, invert=True)
        self.data_index = self.data_index[mask]

    def read_mat_file(self,filename):
        """
        Read mat HD5F file and parsing
        """
        data = h5py.File(filename,'r')
        data.keys()
        X2 = np.array(data['X2']) # time-frequency input
        X2 = np.transpose(X2, (2, 1, 0))  # rearrange dimension
        y = np.array(data['y']) # one-hot encoding labels
        y = np.transpose(y, (1, 0))  # rearrange dimension
        label = np.array(data['label']) # labels
        label = np.transpose(label, (1, 0))  # rearrange dimension
        label = np.squeeze(label)

        return X2, y, label

    def normalize(self, meanX2, stdX2):
        # data normalization for time-frequency input here
        X2 = self.X2
        X2 = np.reshape(X2,(self.data_size*self.data_shape_2[0], self.data_shape_2[1]))
        X2 = (X2 - meanX2) / stdX2
        self.X2 = np.reshape(X2, (self.data_size, self.data_shape_2[0], self.data_shape_2[1]))

    # shuffle data indices
    def shuffle_data(self):
        idx = np.random.permutation(len(self.data_index))
        self.data_index = self.data_index[idx]

                
    def reset_pointer(self):
        """
        reset pointer to begin of the list (needed to start a new training epoch)
        """
        self.pointer = 0
        #self.shuffle_data()
        
    # generate a new batch of data
    def next_batch(self, batch_size):
        data_index = self.data_index[self.pointer:self.pointer + batch_size]

        #update pointer
        self.pointer += batch_size

        # after stack eeg, eog, emg, data_shape now has one more dimension
        batch_x2 = np.ndarray([batch_size, self.seq_len, self.data_shape_2[0], self.data_shape_2[1], self.data_shape_2[2]],dtype=np.float32)
        batch_y = np.ndarray([batch_size, self.seq_len, self.y.shape[1]],dtype=np.float32)
        batch_label = np.ndarray([batch_size, self.seq_len],dtype=np.float32)

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                batch_x2[i, n]  = self.X2[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]

        batch_x2.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        return batch_x2, batch_y, batch_label

    # get rest of data that is collectively smaller than 1 full batch
    # this necessary for evaluation/testing
    def rest_batch(self, batch_size):

        data_index = self.data_index[self.pointer : len(self.data_index)]
        # actual length of this batch
        actual_len = len(self.data_index) - self.pointer

        # update pointer
        self.pointer = len(self.data_index)

        # after stack eeg, eog, emg, data_shape now has one more dimension
        batch_x2 = np.ndarray([actual_len, self.seq_len, self.data_shape_2[0], self.data_shape_2[1], self.data_shape_2[2]],dtype=np.float32)
        batch_y = np.ndarray([actual_len, self.seq_len, self.y.shape[1]],dtype=np.float32)
        batch_label = np.ndarray([actual_len, self.seq_len],dtype=np.float32)

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                batch_x2[i, n]  = self.X2[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]

        batch_x2.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        return actual_len, batch_x2, batch_y, batch_label
