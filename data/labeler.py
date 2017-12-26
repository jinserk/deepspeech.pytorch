import sys
import json
import operator

class Labeler(object):
    """
    Basic labeler class
    """

    def __init__(self):
        self.blank_index = 0
        self.type = None
        self.labels = list()
        self.label_map = dict()

    def load_labels(self, label_file):
        raise NotImplementedError

    def convert_trans_to_labels(self, text):
        raise NotImplementedError

    def load_package(self, package):
        self.type = package['type']
        self.labels = package['labels']
        self.label_map = package['label_map']

    def store_package(self):
        return {
            'type': self.type,
            'labels': self.labels,
            'label_map': self.label_map,
        }


class CharLabeler(Labeler):

    def __init__(self, label_file):
        super(CharLabeler, self).__init__()
        self.type = 'chr'
        self.load_labels(label_file)

    def load_labels(self, label_file):
        with open(label_file) as f: # assume json format
            self.labels = str(''.join(json.load(f)))
        self.label_map = dict([(self.labels[i], i) for i in range(len(self.labels))])

    def convert_trans_to_labels(self, text):
        return list(filter(None, [self.label_map.get(x) for x in list(text)]))


class PhoneLabeler(Labeler):

    def __init__(self, label_file, dict_file, lexicon_file):
        super(PhoneLabeler, self).__init__()
        self.type = 'phn'
        self.load_labels(label_file)
        self.load_dict(dict_file)
        self.load_lexicon(lexicon_file)

    def load_labels(self, label_file):
        with open(label_file, "r") as f:
            for line in f:
                token = line.strip().split()
                label_index = int(token[1])
                self.label_map[token[0]] = label_index
                self.labels.append(label_index)

    def load_dict(self, dict_file):
        self.word_map = {}
        with open(dict_file, "r") as f:
            for line in f:
                token = line.strip().split()
                self.word_map[token[0]] = int(token[1])

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

    def convert_trans_to_labels(self, text):
        labels = []
        for word in text.strip().split():
            if word in self.word_map:
                wid = self.word_map[word]
            else: # oov
                wid = self.word_map['<unk>']
            try:
                labels.extend(self.lexicon_map[wid])
            except:
                print("words and lexicon files are not mismatched")
                raise
        return labels

    def load_package(self, package):
        self.type = package['type']
        self.labels = package['labels']
        self.label_map = package['label_map']
        self.word_map = package['word_map']
        self.lexicon_map = package['lexicon_map']

    def store_package(self):
        return {
            'type': self.type,
            'labels': self.labels,
            'label_map': self.label_map,
            'word_map': self.word_map,
            'lexicon_map': self.lexicon_map,
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='create labelings for AM')
    parser.add_argument('--input', help = "input text file that will be translated to labels")
    parser.add_argument('--output', help = "path to output labels")
    parser.add_argument('--phone', dest='phone', action='store_true', help = "true if phone labels are used")
    parser.add_argument('--label_file', default="../labels.json", help = "path of label units file")
    parser.add_argument('--dict_file', default="../graph/words.txt", help = "path of word dict file")
    parser.add_argument('--lexicon_file', default="../graph/phones/align_lexicon.int", help = "path of lexicon file")
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
