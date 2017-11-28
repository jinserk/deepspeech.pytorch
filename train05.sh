#!/bin/bash

store_path="models/20171026_train05"
#continue_from="models/20171025_train01/deepspeech_050.pth.tar"

cmd="python train.py \
	--train_manifest data/manifests/train05.csv \
	--val data/manifests/val05.csv \
	--sample_rate 8000 \
	--no_bucketing \
	--augment \
	--num_workers 32 \
	--batch_size 32 \
	--rnn_type lstm \
	--hidden_size 1024 \
	--hidden_layers 5 \
	--epochs 100 \
	--optim adam \
	--optim_restart \
	--lr 5e-4 \
	--cuda \
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

