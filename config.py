class Config(object):
    def __init__(self):
        self.sub_seq_len = 10  # subsequence length
        self.nchannel = 1  # number of channels
        self.nclass = 5

        self.nsubseq = 10

        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.0001
        self.training_epoch = 10*self.sub_seq_len*self.nsubseq
        self.batch_size = 8
        self.evaluate_every = 100
        self.checkpoint_every = 100

        # spectrogram size
        self.ndim = 129  # freq bins
        self.frame_seq_len = 29  # time frames in one sleep epoch spectrogram

        self.nhidden1 = 64
        self.lstm_nlayer1 = 1
        self.attention_size = 64
        self.nhidden2 = 64
        self.lstm_nlayer2 = 1

        self.nfilter = 32
        self.nfft = 256
        self.samplerate = 100
        self.lowfreq = 0
        self.highfreq = 50

        self.dropout_rnn = 0.75
        self.fc_size = 512

        self.dualrnn_blocks = 1
        self.early_stop_count = 50

        #maximum number of evaluation steps to stop training. This can be used to control the number of training steps to be equivalent to SeqSleepNet
        self.max_eval_steps = 110 # 110 for SleepEDF-20
