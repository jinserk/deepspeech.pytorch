#!/bin/bash

model="models/20170829/deepspeech_12.pth.tar"
decoder="beam"
cuda="no"
lm_path="lm/mozilla/lm.binary"
trie_path="lm/mozilla/lm.trie"

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