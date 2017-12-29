#!/bin/bash

# This script compiles the ARPA-formatted language models into FSTs. Finally it composes the LM, lexicon
# and token FSTs together into the decoding graph.
export EESEN_ROOT=$HOME/eesen

export PATH=$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/src/lmbin:$PATH
export PATH=$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/irstlm/bin/:$PATH
export LC_ALL=C

model_dir="./aspire"
lang_dir="$model_dir/data/lang_pp_test"
out_dir="./graph"

# move files required for training
cp $lang_dir/words.txt $out_dir
cp $lang_dir/phones.txt $out_dir
cp $lang_dir/phones/align_lexicon.int $out_dir

# aspire model directory check
if [ ! -e $lang_dir ]; then
  mkdir -p $model_dir; cd $model_dir
  if [ ! -e ./0001_aspire_chain_model.tar.gz ]; then
    echo "downloading pretrained ASpIRE model"
    wget --no-check-certificate --quiet http://dl.kaldi-asr.org/models/0001_aspire_chain_model.tar.gz
    tar zxf 0001_aspire_chain_model.tar.gz
  else
    echo "model file already downloaded, but not unzipped"
    tar -zxf 0001_aspire_chain_model.tar.gz
  fi
  cd ..
else
  echo "model directory already exists, skipping downloading"
fi
 
mkdir -p $out_dir

# Get the full list of CTC tokens used in FST. These tokens include <eps>, the blank <blk>, the actual labels (e.g.,
# phonemes), and the disambiguation symbols.  
phn_dir=$lang_dir/phones
(echo '<blk>';) | cat - $phn_dir/silence.txt $phn_dir/nonsilence.txt | \
  awk '{print $1 " " (NR-1)}' > $out_dir/labels.txt
(echo '<eps>'; echo '<blk>';) | cat - $phn_dir/silence.txt $phn_dir/nonsilence.txt $phn_dir/disambig.txt | \
  awk '{print $1 " " (NR-1)}' > $out_dir/tokens.txt

# Compile the tokens into FST
echo "generating token fst to T.fst"
t_fst=$out_dir/T.fst
t_tmp=$t_fst.$$
trap "rm -f $t_tmp" EXIT HUP INT PIPE TERM
if [[ ! -s $t_fst ]]; then
  ctc_token_fst.py $out_dir/tokens.txt | \
    fstcompile --isymbols=$out_dir/tokens.txt --osymbols=$out_dir/tokens.txt \
    --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $t_tmp || exit 1;
  mv $t_tmp $t_fst
fi
    
# Compose the final decoding graph. The composition of L.fst and G.fst is determinized and minimized.
lg_fst=$out_dir/LG.fst
lg_tmp=$lg_fst.$$
trap "rm -f $lg_tmp" EXIT HUP INT PIPE TERM
if [[ ! -s $lg_fst || $lg_fst -ot $lang_dir/G.fst || $lg_fst -ot $lang_dir/L_disambig.fst ]]; then
  fsttablecompose $lang_dir/L_disambig.fst $lang_dir/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstpushspecial | fstarcsort --sort_type=ilabel > $lg_tmp || exit 1;
  mv $lg_tmp $lg_fst
  fstisstochastic $lg_fst || echo "[info]: $lg_fst is not stochastic"
fi

if false; then
  
# Do we need to make CLG?  
N=3 #$(tree-info $tree | grep "context-width" | cut -d' ' -f2) || { echo "Error when getting context-width"; exit 1; }
P=1 #$(tree-info $tree | grep "central-position" | cut -d' ' -f2) || { echo "Error when getting central-position"; exit 1; }
clg_fst=$out_dir/CLG.fst
clg_tmp=$clg_fst.$$
ilabels=$out_dir/ilabels_$N_$P
ilabels_tmp=$ilabels.$$
trap "rm -f $clg_tmp $ilabels_tmp" EXIT HUP INT PIPE TERM
if [[ ! -s $clg_fst || $clg_fst -ot $lg_fst || ! -s $ilabels || $ilabels -ot $lg_fst ]]; then
  fstcomposecontext --context-size=$N --central-position=$P \
   --read-disambig-syms=$lang_dir/phones/disambig.int \
   --write-disambig-syms=$out_dir/disambig_ilabels_${N}_${P}.int \
    $ilabels_tmp < $lg_fst | \
    fstarcsort --sort_type=ilabel > $clg_tmp
  mv $clg_tmp $clg_fst
  mv $ilabels_tmp $ilabels
  fstisstochastic $clg_fst || echo "[info]: $clg_fst is not stochastic."
fi
fsttablecompose $t_fst $clg_fst > $out_dir/TCLG.fst || exit 1;
echo "Composing decoding graph TCLG.fst succeeded"

else
  
fsttablecompose $t_fst $lg_fst > $out_dir/TLG.fst || exit 1;
echo "Composing decoding graph TLG.fst succeeded"

fi

