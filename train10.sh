#!/bin/bash

store_path="models/20171226_train10_phn"
#continue_from="models/20171220_train10_all/deepspeech_checkpoint_epoch_001_iter_050000.pth.tar"
#continue_from="models/20171220_train10_all/deepspeech_001.pth.tar"

cmd="python train.py \
	--train_manifest data/manifests/train10.csv \
	--val data/manifests/val10.csv \
	--sample_rate 8000 \
	--augment \
	--num_workers 8 \
	--batch_size 32 \
	--rnn_type lstm \
	--hidden_size 512 \
	--hidden_layers 4 \
	--epochs 100 \
	--optim adam \
	--lr 1e-4 \
	--cuda \
	--phone \
	--label_file graph/phones.txt \
	--tensorboard \
	--log_dir $store_path/tensorboard \
	--checkpoint \
	--checkpoint_per_batch 10000 \
	--save_folder $store_path \
	--model_path $store_path/deepspeech.final.pth.tar"

if [ -n "$continue_from" ]; then
	cmd="$cmd --continue_from $continue_from"
fi 

echo $cmd
eval "$cmd"

