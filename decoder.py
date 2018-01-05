#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein as Lev
import torch
import numpy as np
from six.moves import xrange
from kaldi.file_io import tmpWriteArk, writeScp
from kaldi import latgen

class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labeler):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labeler = labeler
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labeler.labels)])
        self.blank_index = labeler.blank_index
        space_index = len(labeler.labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labeler.labels:
            space_index = labeler.labels.index(' ')
        self.space_index = space_index

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        if not self.labeler.is_char():
            return 0.

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        if self.labeler.is_char():
            s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
            return Lev.distance(s1, s2)
        else:
            c1 = [chr(c) for c in s1]
            c2 = [chr(c) for c in s2]
            return Lev.distance(''.join(c1), ''.join(c2))

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self, labeler, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0,
                 beam_width=100, num_processes=4):
        super(BeamCTCDecoder, self).__init__(labeler)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        self._decoder = CTCBeamDecoder(labeler.labels, lm_path, alpha, beta,
                                       cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, labeler.blank_index)

    def convert_to_strings(self, out, seq_len, best=None):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                if best is not None and p != best[b]:
                    continue
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int_to_char[x], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            if utterances:
                results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes, best=None):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                if best is not None and p != best[b]:
                    continue
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.IntTensor())
            if utterances:
                results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu().transpose(0, 1).contiguous()
        out, scores, offsets, seq_lens = self._decoder.decode(probs)

        best = np.argmax(scores, axis=1)
        #best = np.zeros(len(out))
        strings = self.convert_to_strings(out, seq_lens, best)
        offsets = self.convert_tensor(offsets, seq_lens, best)
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, labeler):
        super(GreedyDecoder, self).__init__(labeler)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i]]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1]]:
                    pass
                elif char == self.labeler.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.IntTensor(offsets)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of seq_length x batch x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)
        return strings, offsets


class LatticeDecoder(Decoder):
    def __init__(self, labeler, fst_file="kaldi/graph/TLG.fst", wd_file="kaldi/graph/words.txt"):
        super(LatticeDecoder, self).__init__(labeler)
        latgen.initialize(beam=16.0, max_active=8000, min_active=200, acoustic_scale=1.0, allow_partial=True,
                          fst_file=fst_file, wd_file=wd_file)

    def greedy_check(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False, index_output=False):
        phones_list = []
        offsets_list = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            phones, offsets = self.process_phone(sequences[x], seq_len, remove_repetitions, index_output)
            phones_list.append([phones])  # We only return one path
            if return_offsets:
                offsets_list.append([offsets])
        if return_offsets:
            return phones_list, offsets_list
        else:
            return phones_list

    def process_phone(self, sequence, size, remove_repetitions=False, index_output=False):
        phones = []
        offsets = []
        for i in range(size):
            phone = sequence[i]
            if phone != self.labeler.blank_index:
                # if this phone is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and phone == sequence[i - 1]:
                    pass
                else:
                    if index_output:
                        phones.append(phone)
                    else:
                        phones.append(self.labeler.idx2label[phone])
                    offsets.append(i)
        if index_output:
            return torch.IntTensor(phones), torch.IntTensor(offsets)
        else:
            return phones, torch.IntTensor(offsets)

    def decode_token(self, probs, sizes=None, index_output=False):
        probs = probs.transpose(0, 1).contiguous()
        _, max_probs = torch.max(probs, 2)
        phones, offsets = self.greedy_check(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                            remove_repetitions=True, return_offsets=True, index_output=index_output)
        if not index_output:
            print(' '.join(phones[0][0]))
        return phones, offsets

    def write_to_file(self, loglikes):
        uttids = [f"utt{i:04d}" for i in range(len(loglikes))]
        ark_name, ptrs = tmpWriteArk(loglikes.numpy(), uttids)
        scp_name = ark_name.replace(".ark", ".scp")
        writeScp(scp_name, uttids, ptrs)
        print(f"AM results has been written to {ark_name} and {scp_name}")
        return ark_name, scp_name

    def decode(self, probs, sizes=None, check=False):
        probs = probs.transpose(0, 1).contiguous()
        # log of probabilities
        loglikes = probs.log_()
        # scaled likelihood
        priors = torch.FloatTensor(self.labeler.get_label_priors())
        loglikes -= priors
        #eps = torch.zeros(feats.shape[0], feats.shape[1], 1)
        #mod_feats = torch.cat((eps, feats), 2)

        if check:
            _, max_probs = torch.max(loglikes, 2)
            phones, offsets = self.greedy_check(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                remove_repetitions=False, return_offsets=True, index_output=False)
            print(' '.join(phones[0][0]))

        #ark, scp = self.write_to_file(loglikes)
        strings, words, alignments = latgen.decode(loglikes.numpy())
        return [strings], [alignments]

