#!/bin/bash

model="models/20171226_train10_phn/deepspeech_checkpoint_epoch_002_iter_010000.pth.tar"
#model="models/20171228_train10_phn/deepspeech_001.pth.tar"
decoder="lattice"
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
