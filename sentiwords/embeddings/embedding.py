import numpy as np
import csv
from nltk import ngrams
from ..processing.preprocessing import Preprocessor
from tqdm import tqdm
from os.path import splitext


class Embedding():
    def __init__(self, size=200, ngram=1):
        self.size = size
        self.ngram = ngram
        self.vocabulary = dict()
        self.vocab_size = 0
        self.embedding_matrix = None

    def load(self, load_path):
        self.vocab_size = file_len(load_path)
        self.embedding_matrix = np.zeros(
            (self.vocab_size, self.size), dtype=float)

        with open(load_path, mode='r', newline='') as embedding_file:
            reader = csv.reader(
                embedding_file, delimiter=' ', quoting=csv.QUOTE_NONE)
            for index, embedding in tqdm(enumerate(reader)):
                self.vocabulary[embedding[0]] = index
                self.embedding_matrix[index] = embedding[1:]

    def initialize_embeddings_from_sentences(self, sentences):
        assert sentences is not None
        print('Building Vocabulary')
        self._build_vocabulary(sentences)
        self.embedding_matrix = np.random.uniform(-1, 1,
                                                  (self.vocab_size, self.size))

    def _build_vocabulary(self, sentences):
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
        ids = lookup_ids(self.vocabulary, words)
        return self.embedding_matrix[ids, ]

    def save(self, save_path):
        with open(save_path, 'w', newline='') as save_file:
            writer = csv.writer(
                save_file,
                delimiter=' ',
                quoting=csv.QUOTE_NONE,
                escapechar='',
                quotechar='')
            print('Writing embeddings to file...')
            for word in sorted(self.vocabulary, key=lambda k: self.vocabulary[k]):
                writer.writerow([word] + list(self.lookup([word])[0]))
            print('Writing vocabulary to file...')
            vocab_path = splitext(save_path)[0]+'.vocab'
            save_vocab(self.vocabulary, vocab_path)


def save_vocab(vocab, save_path):
    sorted_vocab = sorted(vocab, key=lambda k: vocab[k])
    with open(save_path, 'w') as output:
        for word in sorted_vocab:
            output.write(word + '\n')


def load_vocab(load_path):
    vocabulary = dict()
    with open(load_path) as vocabulary_file:
        for index, line in enumerate(vocabulary_file):
            vocabulary[line.strip()] = index
    return vocabulary


def lookup_ids(vocabulary, words):
    ids = [
        vocabulary[w] if w in vocabulary else vocabulary['<unknown>']
        for w in words
    ]
    return ids


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def convert_tweet_csv(input_csv,
                      output_csv,
                      embedding,
                      delimiter='\t',
                      preprocessor=Preprocessor()):
    with open(input_csv) as input, open(output_csv, 'w', newline='') as output:
        reader = csv.reader(input, delimiter=delimiter)
        writer = csv.writer(output, delimiter=',')

        # write header
        header = ['id'] + ['emb_' + str(i)
                           for i in range(embedding.size)] + ['sentiment']
        writer.writerow(header)

        for line in tqdm(reader):
            id = line[0]
            sentiment = line[1]
            tweet = line[2]
            if tweet != 'Not Available':
                tokenized_tweet = preprocessor.tokenize_tweet(tweet)
                embeddings = embedding.lookup(tokenized_tweet)
                for row in embeddings:
                    writer.writerow([id]+list(row)+[sentiment])

if __name__ == '__main__':
    # processor = Preprocessor()
    # sentences = processor.preprocess_csv(
    #     '/home/maurice/Downloads/training.1600000.processed.noemoticon.csv')
    # embedding = Embedding(size=25)
    # embedding.initialize_embeddings_from_sentences(sentences)
    # embedding.save_embeddings(
    #     '/home/maurice/Downloads/twitter_embeddings_better_preprocessing.csv')
    # save_vocab(
    #     embedding.vocabulary,
    #     '/home/maurice/Downloads/twitter_embeddings_better_preprocessing.vocab'
    # )
    # # embedding = Embedding(size=200, load_path='/home/maurice/Downloads/twitter_embeddings.csv')
    # print(embedding.vocab_size)
    # vocab = load_vocab('/home/maurice/Downloads/twitter_embeddings.vocab')
    embedding = Embedding(size=25)
    embedding.load('/home/maurice/Downloads/GloVe/glove.twitter.27B.25d.txt')
    convert_tweet_csv('/home/maurice/Desktop/train2013.txt', '/home/maurice/Desktop/train2013_embs.csv', embedding)
