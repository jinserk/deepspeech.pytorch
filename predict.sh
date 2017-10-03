#!/bin/bash

model="models/20170929_train0/deepspeech_checkpoint_epoch_009_iter_010000.pth.tar"
#model="models/20170929_train0/deepspeech_003.pth.tar"
decoder="greedy"
cuda="yes"
lm_path="lm/cantab/lm.binary"
trie_path="lm/cantab/lm.trie"

. parse_options.sh

wav=$1
[ "$cuda" == "yes" ] && cuda_opt="--cuda" || cuda_opt=""

if [ "$decoder" == "beam" ]; then
	python transcribe.py \
		--model_path $model \
		--decoder $decoder \
		--beam_width 20 \
		--lm_path $lm_path \
		--trie_path $trie_path \
		--lm_alpha 1.2 \
		--lm_beta1 1.0 \
		--lm_beta2 1.0 \
		$cuda_opt \
		--audio_path $wav
else
	python transcribe.py \
		--model_path $model \
		--decoder $decoder \
		$cuda_opt \
		--audio_path $wav
fi
