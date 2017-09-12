#!/bin/bash

store_path="models/20170911"

python train.py \
	--train_manifest data/swb/swb-train.csv \
	--val data/swb/swb-dev.csv \
	--sample_rate 8000 \
	--augment \
	--num_workers 4 \
	--batch_size 64 \
	--rnn_type gru \
	--hidden_size 800 \
	--hidden_layers 5 \
	--epochs 100 \
	--visdom \
	--checkpoint \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
	#--continue_from $store_path/deepspeech_12.pth.tar \

