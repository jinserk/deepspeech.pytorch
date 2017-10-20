#!/bin/bash

store_path="models/20171019_train02"
log_file="$store_path/train02.log"

mkdir -p $store_path

cmd="python train.py \
	--train_manifest data/manifests/train02.csv \
	--val data/manifests/val02.csv \
	--sample_rate 8000 \
	--augment \
	--no_bucketing \
	--num_workers 48 \
	--batch_size 16 \
	--rnn_type lstm \
	--hidden_size 1024 \
	--hidden_layers 5 \
	--epochs 100 \
	--optim adam \
	--optim_restart \
	--lr 3e-4 \
	--cuda \
	--tensorboard \
	--log_dir $store_path/tensorboard \
	--checkpoint \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
	--continue_from models/20171017_train01/deepspeech_020.pth.tar \
> $log_file  2>&1 &"

echo $cmd
eval "$cmd"

