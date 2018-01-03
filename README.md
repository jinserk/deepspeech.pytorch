# deepspeech.pytorch

Original project's repository is [here](https://github.com/SeanNaren/deepspeech.pytorch). Please refer to [README.md](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/README.md)
for how to install, to prepare dataset, and to train or test.

## Modified Features

* Using additional phase info of spectrogram, as well as its amplitude info
* Changing manifest csv file format to keep the length of wav file info to reduce the processing time
* Add preparing code for LDC's fisher and swbd, and Mozilla's common voice datasets
* Supports phone labeling and the lattice decoder using [Kaldi](https://github.com/kaldi-asr/kaldi.git) framework

# Installation

We assume you already have a working code of this project. Also assume that:
&lt;KALDI\_PATH&gt; : the path you install Kaldi
&lt;DSPYT\_PATH&gt; : the path you install this project

Install Kaldi:
```
cd <KALDI_PATH/..>
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools
make
cd ../src
./configure
make
```

Make a decoding graph by downloading [Kaldi's pretrained ASpIRE chain model](https://kaldi-asr.org/models.html):
```
cd <DSPYT_PATH>
./mkgraph.sh
```

Train a manifest with `--phone` option. You can refer to [`train10.sh`](https://github.com/jinserk/deepspeech.pytorch/blob/master/train10.sh)

Test with `--decoder lattice` option. You can refer to [`predict.sh`](https://github.com/jinserk/deepspeech.pytorch/blob/master/predict.sh)

Currently prediction is relying on the Kaldi's lattice decoder binary. We'll implment a python interface by using CFFI.
You can check the result by the following command:
```
. ./kaldi/path.sh
decode-faster --max-active=10000 --beam=16 --acoustic-scale=1.0 --allow-partial=true --word-symbol-table=kaldi/graph/words.txt kaldi/graph/TLG.fst ark:/tmp/<tmp ark file generated from predict.sh> "ark:|gzip -c > ./lat.gz"
```
