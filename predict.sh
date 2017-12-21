#!/bin/bash

model="models/20171220_train10_all/deepspeech_checkpoint_epoch_001_iter_050000.pth.tar"
#model="models/20171212_train10_all/deepspeech_010.pth.tar"
decoder="greedy"
cuda="no"
lm_path="lm/cantab/lm.binary"
trie_path="lm/cantab/lm.trie"

. parse_options.sh

wav=$1
[ "$cuda" == "yes" ] && cuda_opt="--cuda" || cuda_opt=""

if [ "$decoder" == "beam" ]; then
	python transcribe.py \
		--model_path $model \
		--decoder $decoder \
		--lm_path $lm_path \
		--dict_path $trie_path \
	  --lm_workers 30 \
		$cuda_opt \
		--audio_path $wav
else
	python transcribe.py \
		--model_path $model \
		--decoder $decoder \
		$cuda_opt \
		--audio_path $wav
fi
