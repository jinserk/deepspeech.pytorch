#!/bin/bash

store_path="models/20170912"
log_file="train.log"

nohup \
python train.py \
	--train_manifest data/manifests/train2.csv \
	--val data/manifests/val2.csv \
	--sample_rate 8000 \
	--augment \
	--noise_dir data/noise \
	--noise_min 0.0 \
	--noise_max 0.5 \
	--num_workers 24 \
	--batch_size 8 \
	--rnn_type gru \
	--hidden_size 1000 \
	--hidden_layers 5 \
	--epochs 100 \
	--cuda \
	--visdom \
	--checkpoint \
	--checkpoint_per_batch 10000 \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar \
	--continue_from $store_path/deepspeech_checkpoint_epoch_1_iter_130000.pth.tar \
	--no_bucketing \
>$log_file 2>&1 &

