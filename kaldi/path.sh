#!/bin/bash

#export EESEN_ROOT=$HOME/eesen
#export PATH=$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/src/lmbin:$PATH
#export PATH=$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/irstlm/bin/:$PATH
export KALDI_ROOT=$HOME/kaldi
export PATH=$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

