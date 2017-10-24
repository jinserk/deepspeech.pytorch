#!/bin/bash

store_path="models/20171023_train02"
continue_from="models/20171019_train02/deepspeech_007.pth.tar"

cmd="python train.py \
	--train_manifest data/manifests/train02.csv \
	--val data/manifests/val02.csv \
	--sample_rate 8000 \
	--augment \
	--no_bucketing \
	--num_workers 16 \
	--batch_size 16 \
	--rnn_type lstm \
	--hidden_size 1024 \
	--hidden_layers 5 \
	--epochs 100 \
	--optim adam \
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

