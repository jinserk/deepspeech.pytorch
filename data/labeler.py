import sys
import json
import operator
import numpy as np
from tqdm import tqdm

class Labeler(object):
    """
    Basic labeler class
    """

    def __init__(self, package=None):
        if package is None:
            self.type = None
            self.blank_index = 0
            self.labels = list()
            self.label2idx = dict()
            self.idx2label = dict()
        else:
            self.type = package['type']
            self.blank_index = package['blank_index']
            self.labels = package['labels']
            if 'label_map' in package:
                self.label2idx = package['label_map']
            elif 'label2idx' in package:
                self.label2idx = package['label2idx']
            else:
                raise IOError
            self.idx2label = dict([(v, k) for (k, v) in self.label2idx.items()])

    def is_char(self):
        return True if self.type == 'chr' else False

    def load_labels(self, label_file):
        raise NotImplementedError

    def count_label_prior(self, trans_list):
        raise NotImplementedError

    def convert_trans_to_labels(self, text):
        raise NotImplementedError

    def serialize(self):
        return {
            'type': self.type,
            'blank_index': self.blank_index,
            'labels': self.labels,
            'label2idx': self.label2idx,
        }


class CharLabeler(Labeler):

    def __init__(self, package=None, label_file=None):
        super().__init__(package)
        if package is None:
            self.type = 'chr'
            self.load_labels(label_file)

    def load_labels(self, label_file):
        with open(label_file) as f: # assume json format
            self.labels = str(''.join(json.load(f)))
        self.label2idx = dict([(self.labels[i], i) for i in range(len(self.labels))])
        self.idx2label = dict([(i, c) for (i, c) in enumerate(self.labels)])

    def count_label_prior(self, trans_list):
        pass

    def convert_trans_to_labels(self, text):
        return list(filter(None, [self.label2idx.get(x) for x in list(text)]))


class PhoneLabeler(Labeler):

    def __init__(self, package=None, label_file=None, dict_file=None, lexicon_file=None):
        super().__init__(package)
        if package is None:
            self.type = 'phn'
            self.load_labels(label_file)
            self.load_dict(dict_file)
            self.load_lexicon(lexicon_file)
            self.label_counts = None
        else:
            if 'word_map' in package:
                self.word2idx = package['word_map']
            elif 'word2idx' in package:
                self.word2idx = package['word2idx']
            else:
                raise IOError
            self.idx2word = dict([(v, k) for (k, v) in self.word2idx.items()])
            self.lexicon_map = package['lexicon_map']
            self.label_counts = package['label_counts']

    def load_labels(self, label_file):
        self.label2idx['_'] = 0 # blank
        self.labels = [0]
        with open(label_file, "r") as f:
            for line in f:
                token = line.strip().split()
                label_index = int(token[1])
                if ('<' in token[0] and '>' in token[0]) or '#' in token[0]:
                    continue
                if label_index == 0:
                    print("label index 0 is reserved for blank label, <blk> or \"_\"")
                    sys.exit(1)
                self.labels.append(label_index)
                self.label2idx[token[0]] = label_index
                self.idx2label[label_index] = token[0]

    def load_dict(self, dict_file):
        self.word2idx = {}
        self.idx2word = {}
        with open(dict_file, "r") as f:
            for line in f:
                token = line.strip().split()
                self.word2idx[token[0]] = int(token[1])
                self.idx2word[int(token[1])] = token[0]

    def load_lexicon(self, lexicon_file):
        self.lexicon_map = {}
        with open(lexicon_file, "r") as f:
            for line in f:
                token = line.strip().split()
                lex = [int(i) for i in token[2:]]
                if self.blank_index in lex:
                    raise IOError
                for p in lex:
                    if not p in self.labels:
                        raise IOError
                self.lexicon_map[int(token[0])] = lex

    def count_label_priors(self, trans_list):
        if self.label_counts is not None:
            print("warning: label_counts exists already")
            return
        self.label_counts = [0] * len(self.labels)
        print("counting label priors from the manifest of transcripts")
        for tf in tqdm(trans_list):
            with open(tf, 'r') as tr:
                transcript = tr.read().replace('\n', '')
                labels = self.convert_trans_to_labels(transcript, blank=True)
                for l in labels:
                    self.label_counts[l] += 1
        print(self.label_counts)

    def get_label_priors(self, prior_scale=1., prior_cutoff=1e-10, blank_scale=1.):
        priors = np.array(self.label_counts)
        zidx = np.nonzero(priors == 0)
        priors = np.array([prior_cutoff if c < prior_cutoff else c for c in priors])
        priors[0] *= blank_scale
        priors = np.log(priors / priors.sum()) * prior_scale
        priors[zidx] = 1e30 #np.finfo('d').max
        return priors

    def convert_trans_to_labels(self, text, blank=False):
        labels = [self.blank_index] if blank else []
        for word in text.strip().split():
            wid = self.word2idx[word] if word in self.word2idx else self.word2idx['<unk>']
            try:
                if blank:
                    for c in self.lexicon_map[wid]:
                        labels.extend([c, self.blank_index])
                else:
                    labels.extend(self.lexicon_map[wid])
            except:
                print("words and lexicon files are not mismatched")
                raise
        return labels

    def convert_labels_to_trans(self, indices):
        try:
            trans = [self.idx2word[idx] for idx in indices]
        except:
            print(f"no such index = {idx} exists on dictionary")
            raise
        return ' '.join(trans)

    def load_package(self, package):
        self.type = package['type']
        self.labels = package['labels']
        if 'label_map' in package:
            self.label2idx = package['label_map']
        elif 'label2idx' in package:
            self.label2idx = package['label2idx']
        else:
            raise IOError
        self.idx2label = dict([(v, k) for (k, v) in self.label2idx.items()])
        if 'word_map' in package:
            self.word2idx = package['word_map']
        elif 'word2idx' in package:
            self.word2idx = package['word2idx']
        else:
            raise IOError
        self.idx2word = dict([(v, k) for (k, v) in self.word2idx.items()])
        self.lexicon_map = package['lexicon_map']
        self.label_counts = package['label_counts']

    def serialize(self):
        ret = super().serialize()
        ret.update({
            'word2idx': self.word2idx,
            'lexicon_map': self.lexicon_map,
            'label_counts': self.label_counts,
        })
        return ret


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='create labelings for AM')
    parser.add_argument('--input', help = "input text file that will be translated to labels")
    parser.add_argument('--output', help = "path to output labels")
    parser.add_argument('--phone', dest='phone', default=True, action='store_true', help = "true if phone labels are used")
    parser.add_argument('--label_file', default="../kaldi/graph/labels.txt", help = "path of label units file")
    parser.add_argument('--dict_file', default="../kaldi/graph/words.txt", help = "path of word dict file")
    parser.add_argument('--lexicon_file', default="../kaldi/graph/phones/align_lexicon.int", help = "path of lexicon file")
    parser.add_argument('--ignore_noises', default = False, action='store_true', help='ignore all noises e.g. [noise], [laughter], [vocalized-noise]')

    args = parser.parse_args()

    if args.phone:
        labeler = PhoneLabeler(label_file=args.label_file, dict_file=args.dict_file, lexicon_file=args.lexicon_file)
    else:
        labeler = CharLabeler(label_file=args.label_file)

    with open(args.input, "r") as in_txt, open(args.output, "w") as out_txt:
        for line in in_txt:
            print(line)
            labels = labeler.convert_trans_to_labels(line)
            print(labels)
            out_txt.write(" ".join([str(x) for x in labels]))
