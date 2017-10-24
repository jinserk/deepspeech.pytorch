#!/bin/bash

store_path="models/20171024_train01"

cmd="python train.py \
	--train_manifest data/manifests/train01.csv \
	--val data/manifests/val01.csv \
	--sample_rate 8000 \
	--augment \
	--no_bucketing \
	--num_workers 16 \
	--batch_size 16 \
	--rnn_type lstm \
	--hidden_size 1024 \
	--hidden_layers 5 \
	--epochs 100 \
	--optim yellowfin \
	--cuda \
	--tensorboard \
	--log_dir $store_path/tensorboard \
	--checkpoint \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar"

echo $cmd
eval "$cmd"

