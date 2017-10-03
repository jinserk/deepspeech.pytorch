#!/bin/bash

store_path="models/20170929"
log_file="train.log"

cmd="python train.py \
	--train_manifest data/manifests/train_all.csv \
	--val data/manifests/val_all.csv \
	--sample_rate 8000 \
	--augment \
	--noise_dir data/noise \
	--noise_min 0.0 \
	--noise_max 0.5 \
	--num_workers 8 \
	--batch_size 2 \
	--rnn_type lstm \
	--hidden_size 800 \
	--hidden_layers 5 \
	--epochs 100 \
	--cuda \
	--checkpoint \
	--checkpoint_per_batch 10000 \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
	--continue_from models/20170927/deepspeech_checkpoint_epoch_001_iter_190000.pth.tar \
>> $log_file  2>&1 &"

echo $cmd
eval "$cmd"

