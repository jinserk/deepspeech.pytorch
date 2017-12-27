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
        if self.labeler.type == 'phn':
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
        if self.labeler.type == 'chr':
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
            if self.labeler.type == 'chr':
                string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            else:
                string, string_offsets = self.process_phone(sequences[x], seq_len, remove_repetitions)
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

    def process_phone(self, sequence, size, remove_repetitions=False):
        phones = []
        offsets = []
        for i in range(size):
            phone = sequence[i]
            if phone != self.labeler.blank_index:
                # if this phone is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and phone == self.int_to_char[sequence[i - 1]]:
                    pass
                else:
                    phones.append(phone)
                    offsets.append(i)
        return torch.IntTensor(phones), torch.IntTensor(offsets)

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
