#!/bin/bash

store_path="models/20171030_train01"
#continue_from="models/20171025_train01/deepspeech_001.pth.tar"

cmd="python train.py \
	--train_manifest data/ted/ted_train_manifest.csv \
	--val data/ted/ted_val_manifest.csv \
	--sample_rate 8000 \
	--augment \
	--num_workers 64 \
	--batch_size 64 \
	--rnn_type lstm \
	--hidden_size 1024 \
	--hidden_layers 5 \
	--epochs 100 \
	--optim adam \
	--lr 1e-3 \
	--sortagrad \
	--tensorboard \
	--log_dir $store_path/tensorboard \
	--checkpoint \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar"

if [ -n "$continue_from" ]; then
	cmd="$cmd --continue_from $continue_from"
fi 

echo $cmd
eval "$cmd"

