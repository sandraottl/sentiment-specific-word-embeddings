# Classification with Convolutional and LSTM Neural Networks
# **********************************************************

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Embedding, LSTM, Flatten, GlobalMaxPooling1D, Conv1D, MaxPooling1D
from keras.models import Model

# from preprocessing import Preprocessor
from ..processing.preprocessing import Preprocessor


class NeuralNetworkClassifier:

    def __init__(self):

        self.tweet_length = 0  # padding length to which all tweets will be converted
        self.embedding_dim = 0  # number of dimensions of embeddings

        self.prelearned_embeddings = {}
        self.vocabulary = {}
        self.embedding_matrix = None
        self.x_train = None
        self.x_develop = None
        self.x_test = None
        self.y_train = None
        self.y_develop = None
        self.y_test = None

    def load_embedding(self,
                       embedding_dim,
                       path_embeddingfile):

        self.embedding_dim = embedding_dim

        with open(path_embeddingfile, encoding='utf8') as embedfile:
            for line in embedfile:
                values = line.split()
                word = values[0]
                embed = np.asarray(values[1:], dtype='float32')
                self.prelearned_embeddings[word] = embed

        print('Loaded %s pre-learned word vectors.' % len(self.prelearned_embeddings))

    def load_and_preprocess_data(self,
                                 path_trainingfile,
                                 path_developfile,
                                 path_testfile,
                                 tweet_length=40):

        self.tweet_length = tweet_length

        pp = Preprocessor()

        df_train = pd.read_csv(path_trainingfile, sep='\t', header=None, names=['id', 'sentiment', 'text'], quoting=3)
        df_train['tokens'] = df_train['text'].apply(pp.tokenize_tweet)
        x_train_raw = df_train['tokens'].values
        y_train_raw = df_train['sentiment'].values

        print('Loaded %s Tweets as Training Data' % len(x_train_raw))

        df_develop = pd.read_csv(path_developfile, sep='\t', header=None, names=['id', 'sentiment', 'text'], quoting=3)
        df_develop['tokens'] = df_develop['text'].apply(pp.tokenize_tweet)
        x_develop_raw = df_develop['tokens'].values
        y_develop_raw = df_develop['sentiment'].values

        print('Loaded %s Tweets as Development Data' % len(x_develop_raw))

        df_test = pd.read_csv(path_testfile, sep='\t', header=None, names=['id', 'sentiment', 'text'], quoting=3)
        df_test['tokens'] = df_test['text'].apply(pp.tokenize_tweet)
        x_test_raw = df_test['tokens'].values
        y_test_raw = df_test['sentiment'].values

        print('Loaded %s Tweets as Test Data' % len(x_test_raw))

        # Build overall vocabulary

        unique_tokens = set()

        for tweet in x_train_raw:
            for token in tweet:
                unique_tokens.add(token)

        for tweet in x_develop_raw:
            for token in tweet:
                unique_tokens.add(token)

        for tweet in x_test_raw:
            for token in tweet:
                unique_tokens.add(token)

        self.vocabulary = {
            word: index
            for index, word in enumerate(sorted(unique_tokens))
        }

        print('Overall vocabulary size: %s' % len(self.vocabulary))

        # Translate Tokens to Indices in the Tweets and pad them to the same length

        x_train_indices = [[self.vocabulary[token] for token in tweet] for tweet in x_train_raw]
        x_develop_indices = [[self.vocabulary[token] for token in tweet] for tweet in x_develop_raw]
        x_test_indices = [[self.vocabulary[token] for token in tweet] for tweet in x_test_raw]

        self.x_train = pad_sequences(x_train_indices, tweet_length)
        self.x_develop = pad_sequences(x_develop_indices, tweet_length)
        self.x_test = pad_sequences(x_test_indices, tweet_length)

        # Transform Sentiment Labels in binary format

        encoder = LabelEncoder()
        encoder.fit(y_train_raw)

        self.y_train = to_categorical(encoder.transform(y_train_raw))
        self.y_develop = to_categorical(encoder.transform(y_develop_raw))
        self.y_test = to_categorical(encoder.transform(y_test_raw))

        # Short Summary for Debugging
        print("Shapes of Input Tensors X :", self.x_train.shape, self.x_develop.shape, self.x_test.shape)
        print("Shapes of Output Tensors Y:", self.y_train.shape, self.y_develop.shape, self.y_test.shape)

    def __prepare_embedding_matrix(self):

        self.embedding_matrix = np.zeros((len(self.vocabulary), self.embedding_dim))

        # for debugging/statistics; will count how many words of vocabulary are not found in prelearned embeddings
        embed_not_found_counter = 0

        for w, i in self.vocabulary.items():
            # words for which we do not have an embedding will stay zero in matrix
            if self.prelearned_embeddings.get(w) is not None:
                self.embedding_matrix[i] = self.prelearned_embeddings.get(w)
            else:
                embed_not_found_counter += 1

        print("For %s words of the Tweet Vocabulary, no prelearned embedding could be found" % embed_not_found_counter)

    def convnn(self):

        # Check if loading and preprocessing is finished, then prepare embedding matrix
        assert self.tweet_length > 0
        assert self.embedding_dim > 0

        if self.embedding_matrix is None:
            self.__prepare_embedding_matrix()

        embedding_layer = Embedding(len(self.vocabulary),
                                    self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.tweet_length,
                                    trainable=False)

        inputtensor = Input(shape=(self.tweet_length,), dtype='int32')

        embedded_inputs = embedding_layer(inputtensor)

        x = Conv1D(20, 2, activation='relu')(embedded_inputs)
        x = MaxPooling1D(2)(x)
        x = Conv1D(20, 2, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(20, 2, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)

        x = Dense(20, activation='relu')(x)

        outputlayer = Dense(self.y_train.shape[1], activation='softmax')(x)

        model = Model(inputtensor, outputlayer)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        model.fit(self.x_train, self.y_train,
                  batch_size=128,
                  epochs=10,
                  verbose=2,
                  validation_data=(self.x_develop, self.y_develop))

        scores = model.evaluate(self.x_test, self.y_test)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def lstm(self):

        # Check if loading and preprocessing is finished, then prepare embedding matrix
        assert self.tweet_length > 0
        assert self.embedding_dim > 0

        if self.embedding_matrix is None:
            self.__prepare_embedding_matrix()

        embedding_layer = Embedding(len(self.vocabulary),
                                    self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.tweet_length,
                                    trainable=False)

        inputtensor = Input(shape=(self.tweet_length,), dtype='int32')

        embedded_inputs = embedding_layer(inputtensor)

        x = LSTM(self.embedding_dim, return_sequences=True)(embedded_inputs)
        x = LSTM(self.embedding_dim, return_sequences=True)(x)

        x = Flatten()(x)

        x = Dense(self.y_train.shape[1], activation='softmax')(x)

        model = Model(inputtensor, x)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

        model.fit(self.x_train, self.y_train,
                  batch_size=128,
                  epochs=10,
                  verbose=2,
                  validation_data=(self.x_develop, self.y_develop))

        scores = model.evaluate(self.x_test, self.y_test)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])


if __name__ == '__main__':

    NNC = NeuralNetworkClassifier()

    NNC.load_embedding(embedding_dim=100,
                       path_embeddingfile='C:/Users/rickrack/Downloads/glove.6B/glove.6B.100d.txt')
    NNC.load_and_preprocess_data(path_trainingfile='C:/Users/rickrack/Downloads/semeval2013/twitter-2013train-A.txt',
                                 path_developfile='C:/Users/rickrack/Downloads/semeval2013/twitter-2013dev-A.txt',
                                 path_testfile='C:/Users/rickrack/Downloads/semeval2013/twitter-2013test-A.txt')

    NNC.convnn()
    NNC.lstm()
