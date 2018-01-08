#!/bin/bash

model_path="models/20171229_train10_phn/deepspeech_checkpoint_epoch_017_iter_010000.pth.tar"
#model_path="models/20171220_train10_all/deepspeech_001.pth.tar"

cmd="python test.py \
	--test_manifest data/manifests/test10.csv \
	--num_workers 10 \
	--batch_size 32 \
	--decoder lattice \
	--model_path $model_path"

echo $cmd
eval "$cmd"

