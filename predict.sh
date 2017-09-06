#!/bin/bash

model="models/20170829/deepspeech_10.pth.tar"
decoder="beam"

. parse_options.sh

wav=$1

if [ "$decoder" == "beam" ]; then
	python predict.py \
		--model_path $model \
		--decoder $decoder \
		--beam_width 20 \
		--lm_path lm/lm.binary \
		--trie_path lm/lm.trie \
		--lm_alpha 0.8 \
		--audio_path $wav
else
	python predict.py \
		--model_path $model \
		--decoder $decoder \
		--audio_path $wav
fi
