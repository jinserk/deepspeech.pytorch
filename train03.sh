#!/bin/bash

store_path="models/20171017_train02"
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
	--continue_from models/20171016_train01/deepspeech_010.pth.tar \
> $log_file  2>&1 &"

echo $cmd
eval "$cmd"

