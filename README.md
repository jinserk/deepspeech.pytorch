# deepspeech.pytorch

Original project's repository is [here](https://github.com/SeanNaren/deepspeech.pytorch). Please refer to the [README.md](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/README.md)
to see how to install, prepare dataset, train or test.

## Modified Features

* Using additional phase info of spectrogram, as well as its amplitude info as the CNN input
* Replace RELU to Swish in CNN
* Changing manifest csv file format to keep the length of wav file info to reduce its processing time
* Add preparing code for LDC's fisher and swbd, and Mozilla's common voice datasets
* Supports phone labeling and the lattice decoder using [Kaldi](https://github.com/kaldi-asr/kaldi.git) framework

# Installation

We assume you already have a working installation of this project and Kaldi. Also we assume that:
* &lt;KALDI\_PATH&gt; : the path you install Kaldi
* &lt;DSPYT\_PATH&gt; : the path you install this project

Make a decoding graph by downloading [Kaldi's pretrained ASpIRE chain model](http://kaldi-asr.org/models.html):
```
cd <DSPYT_PATH>/kaldi
./mkgraph.sh
```

Train a manifest with `--phone` option. You can refer to [`train10.sh`](https://github.com/jinserk/deepspeech.pytorch/blob/master/train10.sh)

Test with `--decoder lattice` option. You can refer to [`predict.sh`](https://github.com/jinserk/deepspeech.pytorch/blob/master/predict.sh)

Currently the prediction by using lattice decoder is relying on the Kaldi's binary. We'll implement a python interface by using CFFI.
You can test the result of lattice decoder by the following command:
```
. ./kaldi/path.sh
decode-faster --max-active=10000 --beam=16 --acoustic-scale=1.0 --allow-partial=true --word-symbol-table=kaldi/graph/words.txt kaldi/graph/TLG.fst ark:/tmp/<tmp ark file generated from predict.sh> "ark:|gzip -c > ./lat.gz"
```
