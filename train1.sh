#!/bin/bash

store_path="models/20171003_train1"
log_file="train1.log"

cmd="python train.py \
	--train_manifest data/manifests/train1.csv \
	--val data/manifests/val1.csv \
	--sample_rate 8000 \
	--augment \
	--num_workers 8 \
	--batch_size 2 \
	--rnn_type lstm \
	--hidden_size 800 \
	--hidden_layers 5 \
	--epochs 20 \
	--cuda \
	--checkpoint \
	--checkpoint_per_batch 10000 \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
	--continue_from models/20170729_train0/deepspeech_final.pth.tar \
> $log_file  2>&1 &"

echo $cmd
eval "$cmd"

