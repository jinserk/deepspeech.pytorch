# deepspeech.pytorch

Original project's repository is [here](https://github.com/SeanNaren/deepspeech.pytorch). Please refer to the [README.md](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/README.md)
to see how to install, prepare dataset, train or test.

## Modified Features

* Using additional phase info of spectrogram, as well as its amplitude info as the CNN input
* Replace RELU to Swish in CNN
* Changing manifest csv file format to keep the length of wav file info to reduce its processing time
* Add codes for LDC's fisher and swbd, and Mozilla's common voice datasets in `data`
* Supports phone labeling to interface with lattice decoder based on [Kaldi](https://github.com/kaldi-asr/kaldi.git) framework

# Installation

Assumed you already have each of proper clones in the following paths:
* &lt;DSPYT\_PATH&gt; : the path to install this project
* &lt;KALDI\_PATH&gt; : the path to install Kaldi

Install and compile Kaldi with shared library option:
```
cd <KALDI_PATH>/tools
make
cd ../src
configure --shared
make
```

Modify the Kaldi installation path in `<DSPYT_PATH>/kaldi/path.sh`
```
export KALDI_ROOT=<KALDI_PATH>
```

Make `liblatgen.so`
```
cd <DSPYT_PATH>/kaldi
make
```

Make a decoding graph by downloading [Kaldi's pretrained ASpIRE chain model](http://kaldi-asr.org/models.html):
```
cd <DSPYT_PATH>/kaldi
./mkgraph.sh
```

Train a manifest with `--phone` option. You can refer to [`train10.sh`](https://github.com/jinserk/deepspeech.pytorch/blob/master/train10.sh)

Test or predict with `--decoder lattice` option. You can refer to [`test10.sh`](https://github.com/jinserk/deepspeech.pytorch/blob/master/test10.sh)


# What's different from EESEN or DeepSpeech2

Basically the config using phone labeling is almost the same to [EESEN](https://github.com/srvk/eesen.git) except:
* Utilizing DeepSpeech2 as an acoustic model, not using the G2P embedding from its end-to-end char labeling, and keeping its feature embedding with CNN layers
* Used position-dependent-phones instead of position-independent-phones, so don't need to insert any word boundary symbol
* TLG.fst can easily updated by `L_disambig.fst` and `G.fst` from any Kaldi recipes
* Implemented on PyTorch, while EESEN is built on Tensorflow

