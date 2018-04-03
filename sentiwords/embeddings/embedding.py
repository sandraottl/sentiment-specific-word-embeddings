"""Module defining an Embedding class and helper functions for working
with embeddings.
author: Maurice Gercuk
"""
import numpy as np
import csv
from nltk import ngrams
from tqdm import tqdm
from os.path import splitext, dirname
from os import makedirs


class Embedding():
    """Class wrapping Embedding initialization, loading, lookup and
    saving operations."""

    def __init__(self, size=None, ngram=1):
        """Intitialize an empty embedding object.
        arguments:
           size: dimensionality of the embedding vectors. Needed when
                 building embedding from input sentences.
           ngram: Degree of ngram used for building vocabulary from
                  sentences.
        """
        self.size = size
        self.ngram = ngram
        self.vocabulary = dict()
        self.vocab_size = 0
        self.embedding_matrix = None

    def load(self, load_path):
        """Load embedding from a csv file in GloVe format.
        arguments:
           load_path: Path to embedding csv.
        """
        self.vocab_size = file_len(load_path)

        with open(load_path, mode='r', newline='') as embedding_file:
            reader = csv.reader(
                embedding_file, delimiter=' ', quoting=csv.QUOTE_NONE)
            for index, embedding in tqdm(enumerate(reader)):
                if self.embedding_matrix is None:
                    self.size = len(embedding) - 1
                    self.embedding_matrix = np.zeros(
                        (self.vocab_size, self.size), dtype=float)
                self.vocabulary[embedding[0]] = index
                self.embedding_matrix[index] = embedding[1:]

    def initialize_embeddings_from_sentences(self, sentences):
        """Inititalize an embedding from sentences (tokenized).
        arguments:
           sentences: List of sentences. Each sentence should be a
                      list of tokens.
        """
        assert sentences is not None and self.size is not None
        print('Building Vocabulary')
        self._build_vocabulary(sentences)
        self.embedding_matrix = np.random.uniform(-1, 1,
                                                  (self.vocab_size, self.size))

    def _build_vocabulary(self, sentences):
        """Build a vocabulary from the unique words in given sentences.
        arguments:
           sentences: List of sentences. Each sentence should be a
                      list of tokens.
        """
        unique_words = set()
        unique_words.add('<unknown>')
        for sentence in tqdm(sentences):
            grams = ngrams(sentence, self.ngram)
            for gram in grams:
                unique_words.add('_'.join(gram))
        self.vocabulary = {
            word: index
            for index, word in enumerate(sorted(unique_words))
        }
        self.vocab_size = len(self.vocabulary)

    def lookup(self, words):
        """Look up vectors for input words in the embedding matrix.
        arguments:
           words: List of words to lookup in the embedding matrix.
        returns:
           A list of embedding vectors, one for each input word.
        """
        ids = lookup_ids(self.vocabulary, words)
        return self.embedding_matrix[ids, ]

    def save(self, save_path):
        """Save the embedding to file. Saves both a vocabulary file
        and a embedding csv in GloVe format. The vocabulary file is
        named like the embedding csv but with .vocab file extension.

        arguments:
           save_path: Path for output csv.
        """
        makedirs(dirname(save_path), exist_ok=True)
        with open(save_path, 'w', newline='') as save_file:
            writer = csv.writer(
                save_file,
                delimiter=' ',
                quoting=csv.QUOTE_NONE,
                escapechar='',
                quotechar='')
            print('Writing embeddings to file...')
            for word in sorted(
                    self.vocabulary, key=lambda k: self.vocabulary[k]):
                writer.writerow([word] + list(self.lookup([word])[0]))
            print('Writing vocabulary to file...')
            vocab_path = splitext(save_path)[0] + '.vocab'
            save_vocab(self.vocabulary, vocab_path)


def save_vocab(vocab, save_path):
    """Saves a vocabulary to disk. Each vocabulary word is on a seperate line.
    Ordered by word-id.

    arguments:
       vocab: vocabulary to save to file. Dictionary of 'word: id'.
       save_path: Path for output file.
    """
    sorted_vocab = sorted(vocab, key=lambda k: vocab[k])
    with open(save_path, 'w') as output:
        for word in sorted_vocab:
            output.write(word + '\n')


def load_vocab(load_path):
    """Load a vocabulary from disk.
    arguments:
       load_path: Path to vocabulary file.
    returns: Dictionary of 'word: id'.
    """
    vocabulary = dict()
    with open(load_path) as vocabulary_file:
        for index, line in enumerate(vocabulary_file):
            vocabulary[line.strip()] = index
    return vocabulary


def lookup_ids(vocabulary, words):
    """Lookup ids of given word in a vocabulary (dict of 'word: id').
    arguments:
       vocabulary: Dictionary of 'word: id'.
       words: List of words.
    returns: List of word-ids.
    """
    ids = [
        vocabulary[w] if w in vocabulary else vocabulary['<unknown>']
        for w in words
    ]
    return ids


def file_len(fname):
    """Helper function that computes the number of lines in a file.
    arguments:
       fname: File to check for length.
    returns: Number of lines in input file.
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
