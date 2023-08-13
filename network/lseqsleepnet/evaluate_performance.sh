# scratch training case
python3 evaluate.py \
--out_dir './scratch_training_1chan_subseqlen10_nsubseq20_1blocks_20sub/' \
--out_filename 'test_ret.mat' \
--datalist_dir '../../file_list_20sub/eeg/' \
--num_fold 20 \
--num_repeat 5 \
--subseqlen 10 \
--nsubseq 20

# finetuning case
python3 evaluate.py \
--out_dir './finetunue_1chan_subseqlen10_nsubseq20_1blocks_20sub/' \
--out_filename 'test_ret.mat' \
--datalist_dir '../../file_list_20sub/eeg/' \
--num_fold 20 \
--num_repeat 5 \
--subseqlen 10 \
--nsubseq 20
