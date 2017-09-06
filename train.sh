#!/bin/bash

store_path="models/20170829"

python train.py \
	--train_manifest data/manifests/train.csv \
	--val data/manifests/val.csv \
	--sample_rate 8000 \
	--augment \
	--num_workers 4 \
	--batch_size 20 \
	--epochs 100 \
	--cuda \
	--checkpoint \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar
	#--continue_from models/20170823/deepspeech_67.pth.tar
