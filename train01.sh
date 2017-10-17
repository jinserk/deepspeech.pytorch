#!/bin/bash

store_path="models/20171016_train01"
log_file="$store_path/train01.log"

mkdir -p $store_path

cmd="python train.py \
	--train_manifest data/manifests/train01.csv \
	--val data/manifests/val01.csv \
	--sample_rate 8000 \
	--augment \
	--num_workers 48 \
	--batch_size 16 \
	--rnn_type lstm \
	--hidden_size 1000 \
	--hidden_layers 5 \
	--epochs 100 \
	--optim adam \
	--optim_restart \
	--lr 0.001 \
	--cuda \
	--tensorboard \
	--log_dir $store_path/tensorboard \
	--checkpoint \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
> $log_file  2>&1 &"

echo $cmd
eval "$cmd"

