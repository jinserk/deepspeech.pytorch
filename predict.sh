#!/bin/bash

model="models/20170829/deepspeech_10.pth.tar"
decoder="beam"
cuda="yes"

. parse_options.sh

wav=$1
[ "$cuda" == "yes" ] && cuda_opt="--cuda" || cuda_opt=""

if [ "$decoder" == "beam" ]; then
	python transcribe.py \
		--model_path $model \
		--decoder $decoder \
		--beam_width 20 \
		--lm_path lm/lm.binary \
		--trie_path lm/lm.trie \
		--lm_alpha 0.8 \
		$cuda_opt \
		--audio_path $wav
else
	python transcribe.py \
		--model_path $model \
		--decoder $decoder \
		$cuda_opt \
		--audio_path $wav
fi
