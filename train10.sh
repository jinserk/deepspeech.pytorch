#!/bin/bash

store_path="models/20171127_train10_swb"
#continue_from="models/20171127_train10/deepspeech_checkpoint_epoch_003_iter_110000.pth.tar"

cmd="python train.py \
	--train_manifest data/swb/swb-train.csv \
	--val data/swb/swb-dev.csv \
	--sample_rate 8000 \
	--augment \
	--num_workers 4 \
	--batch_size 16 \
	--rnn_type lstm \
	--hidden_size 512 \
	--hidden_layers 5 \
	--epochs 100 \
	--optim adam \
	--lr 1e-4 \
	--cuda \
	--tensorboard \
	--log_dir $store_path/tensorboard \
	--checkpoint \
	--checkpoint_per_batch 10000 \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar"

if [ -n "$continue_from" ]; then
	cmd="$cmd --continue_from $continue_from"
fi 

echo $cmd
eval "$cmd"

