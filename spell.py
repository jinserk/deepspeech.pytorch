from __future__ import absolute_import
import re
import kenlm
import os.path
from heapq import heapify
from six.moves import range

# Define beam with for alt sentence search
BEAM_WIDTH = 1024
MODEL = None

CWD = os.path.dirname(__file__)

# Lazy-load language model (TED corpus, Kneser-Ney, 4-gram, 30k word LM)
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = kenlm.Model(os.path.join(CWD, 'lm/lm.binary'))
    return MODEL

def words(text):
    "List of words in text."
    return re.findall(r'\w+', text.lower())

# Load known word set
with open(os.path.join(CWD, 'lm/words.txt')) as f:
    WORDS = set(words(f.read()))

def log_probability(sentence):
    "Log base 10 probability of `sentence`, a list of words"
    return get_model().score(' '.join(sentence), bos = False, eos = False)

def correction(sentence, alphabet):
    "Most probable spelling correction for sentence."
    sentence = sentence.lower()
    alphabet = alphabet.lower()
    layer = [(0,[])]
    for word in words(sentence):
        layer = [(-log_probability(node + [cword]), node + [cword]) for cword in candidate_words(word, alphabet) for priority, node in layer]
        heapify(layer)
        layer = layer[:BEAM_WIDTH]
    return ' '.join(layer[0][1])

def candidate_words(word, alphabet):
    "Generate possible spelling corrections for word."
    return (known_words([word]) or known_words(edits1(word, alphabet)) or known_words(edits2(word, alphabet)) or [word])

def known_words(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word, alphabet):
    "All edits that are one edit away from `word`."
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in alphabet]
    inserts    = [L + c + R               for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def edits2(word, alphabet):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word, alphabet) for e2 in edits1(e1, alphabet))
