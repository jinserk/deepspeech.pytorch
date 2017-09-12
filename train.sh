#!/bin/bash

store_path="models/20170912"

python train.py \
	--train_manifest data/manifests/train.csv \
	--val data/manifests/val.csv \
	--sample_rate 8000 \
	--augment \
	--noise_dir data/noise \
	--num_workers 16 \
	--batch_size 12 \
	--rnn_type gru \
	--hidden_size 1200 \
	--hidden_layers 5 \
	--epochs 100 \
	--cuda \
	--visdom \
	--checkpoint \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
	#--continue_from $store_path/deepspeech_12.pth.tar \

