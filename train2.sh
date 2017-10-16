#!/bin/bash

store_path="models/20171003_train2"
log_file="$store_path/train2.log"

mkdir -p $store_path

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
	--epochs 150 \
	--cuda \
	--checkpoint \
	--checkpoint_per_batch 10000 \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
	--continue_from models/20171003_train1/deepspeech.final.pth.tar \
>> $log_file  2>&1 &"

echo $cmd
eval "$cmd"

