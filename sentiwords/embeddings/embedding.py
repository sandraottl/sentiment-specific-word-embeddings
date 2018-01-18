import numpy as np
import pandas as pd
import csv
from nltk import ngrams
from ..processing.preprocessing import Preprocessor
from tqdm import tqdm


class Embedding():

    def __init__(self, size=200, sentences=None, load_path=None, ngram=1):
        self.size = size
        self.sentences = sentences
        self.load_path = load_path
        self.ngram = ngram
        self.vocabulary = dict()
        self.vocab_size = 0
        self.embedding_matrix = None

        if load_path:
            self._load_embeddings()
        else:
            self._initialize_embeddings()
            self.embedding_matrix = np.random.uniform(-1, 1, (self.vocab_size, self.size))

    def _load_embeddings(self):
        self.vocab_size = file_len(self.load_path)
        self.embedding_matrix = np.zeros((self.vocab_size, self.size), dtype=float)

        with open(self.load_path, mode='r', newline='') as embedding_file:
            reader = csv.reader(embedding_file, delimiter=' ', quoting=csv.QUOTE_NONE)
            for index, embedding in tqdm(enumerate(reader)):
                self.vocabulary[embedding[0]] = index
                self.embedding_matrix[index] = embedding[1:]

    def _initialize_embeddings(self):
        assert self.sentences is not None
        print('Building Vocabulary')
        self._build_vocabulary()

    def _build_vocabulary(self):
        unique_words = set()
        for sentence in tqdm(self.sentences):
            grams = ngrams(sentence, self.ngram)
            for gram in grams:
                unique_words.add('_'.join(gram))
        self.vocabulary = {word: index for index, word in enumerate(sorted(unique_words))}
        self.vocab_size = len(self.vocabulary)

    def lookup(self, words):
        ids = [self.vocabulary[w] for w in words if w in self.vocabulary]
        return self.embedding_matrix[ids,]

    def save_embeddings(self, save_path):
        with open(save_path, 'w', newline='') as save_file:
            writer = csv.writer(save_file, delimiter=' ', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
            print('Writing embeddings to file...')
            for word in sorted(self.vocabulary):
                writer.writerow([word] + list(self.lookup([word])[0]))


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if __name__=='__main__':
    processor = Preprocessor()
    sentences = processor.preprocess_csv('/home/maurice/Downloads/training.1600000.processed.noemoticon.csv')
    embedding = Embedding(size=25, sentences=sentences)
    embedding.save_embeddings('/home/maurice/Downloads/twitter_embeddings.csv')
    # embedding = Embedding(size=200, load_path='/home/maurice/Downloads/twitter_embeddings.csv')
    print(embedding.vocab_size)

