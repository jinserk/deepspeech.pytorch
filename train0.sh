#!/bin/bash

store_path="models/20170929_train0"
log_file="train.log"

cmd="python train.py \
	--train_manifest data/ted/ted_train_manifest.csv \
	--val data/ted/ted_val_manifest.csv \
	--sample_rate 8000 \
	--augment \
	--num_workers 8 \
	--batch_size 2 \
	--rnn_type lstm \
	--hidden_size 800 \
	--hidden_layers 5 \
	--epochs 10 \
	--cuda \
	--checkpoint \
	--checkpoint_per_batch 10000 \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
> $log_file  2>&1 &"

echo $cmd
eval "$cmd"

