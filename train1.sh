#!/bin/bash

store_path="models/20171013_train1"
log_file="$store_path/train1.log"

mkdir -p $store_path

cmd="python train.py \
	--train_manifest data/manifests/train1.csv \
	--val data/manifests/val1.csv \
	--sample_rate 8000 \
	--augment \
	--num_workers 12 \
	--batch_size 4 \
	--rnn_type lstm \
	--hidden_size 800 \
	--hidden_layers 5 \
	--epochs 50 \
	--optim adam \
	--optim_restart true \
	--lr 0.001 \
	--cuda \
	--checkpoint \
	--checkpoint_per_batch 100000 \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
	--continue_from models/20171013_train1/deepspeech_011.pth.tar \
> $log_file  2>&1 &"

echo $cmd
eval "$cmd"

