CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n1.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n1/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110   
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n1.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n1/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n2.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n2.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n2/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110    
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n2.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n2/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n3.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n3.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n3/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110    
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n3.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n3/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n4.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n4.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n4/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110    
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n4.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n4/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n5.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n5.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n5/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110    
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n5.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n5/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n6.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n6.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n6/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n6.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n6/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n7.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n7.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n7/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n7.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n7/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n8.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n8.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n8/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n8.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n8/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n9.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n9.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n9/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n9.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n9/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n10.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n10.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n10/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n10.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n10/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n11.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n11.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n11/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n11.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n11/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n12.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n12.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n12/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n12.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n12/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n13.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n13.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n13/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n13.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n13/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n14.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n14.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n14/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n14.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n14/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n15.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n15.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n15/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n15.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n15/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n16.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n16.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n16/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n16.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n16/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n17.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n17.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n17/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n17.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n17/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n18.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n18.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n18/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n18.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n18/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n19.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n19.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n19/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n19.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n19/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
CUDA_VISIBLE_DEVICES="0,-1" python3 train_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n20.txt" --eeg_eval_data "../../file_list_20sub/eeg/eval_list_n20.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n20/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45 --max_eval_steps 110
CUDA_VISIBLE_DEVICES="0,-1" python3 test_lseqsleepnet.py --eeg_train_data "../../file_list_20sub/eeg/train_list_n20.txt" --eeg_test_data "../../file_list_20sub/eeg/test_list_n1.txt" --out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/repeat5/n20/' --dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --nsubseq 20  --dualrnn_blocks 1  --gpu_usage 0.45
