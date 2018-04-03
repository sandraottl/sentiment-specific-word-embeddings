"""
Implementation of Convolutional and LSTM Neural Network based classification

Responsibility: Rick Beer

"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Embedding, LSTM, Flatten, GlobalMaxPooling1D, Conv1D, MaxPooling1D
from keras.models import Model

# from ..processing.preprocessing import Preprocessor
sys.path.append('C:/Users/WohnzimmerPC/Documents/GitHub/sentiment-specific-word-embeddings/sentiwords/')
from processing.preprocessing import Preprocessor


class NeuralNetworkClassifier:

    """
    Handles loading of embeddings and datasets and performs classification with CNN and LSTM networks

    Methods:
        load_embedding:  loads pre-learned embeddings from a file for use in classification
        load_and_preprocess_data:  loads three datasets (training,development,test) with tweets for classification
        convnn:  performs classification with Convolutional Neural Network
        lstm:  performs classification with Long Short-Term Memory Neural Network

    Note: you can reuse an instance for multiple classifications on the same combination of datasets and embeddings,
          but you cannot change the data/embeddings afterwards (create a new instance instead)

    """

    def __init__(self):
        """Initialization of new instance (no arguments)"""

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
        """
        Load pre-learned embeddings to be used in classification

        :param embedding_dim: number of dimensions of the loaded embeddings
        :param path_embeddingfile: path to the file containing the embeddings
        :return void

        """
        self.embedding_dim = embedding_dim

        with open(path_embeddingfile, encoding='utf8') as embedfile:
            # iterate over lines and save the word and its embedding into dictionary prelearned_embeddings
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
        """
        Load and preprocess the three datasets (training, development, test) for classification

        :param path_trainingfile: path to the file containing the training dataset
        :param path_developfile: path to the file containing the development dataset
        :param path_testfile: path to the file containing the test dataset
        :param tweet_length: padding length of tweets to which all tweets will be truncated/expanded
        :return: void

        """
        self.tweet_length = tweet_length

        # instantiate new Preprocessor for cleaning and tokenizing the tweets
        pp = Preprocessor()

        # Load, preprocess and split Training dataset
        df_train = pd.read_csv(path_trainingfile, sep='\t', header=None, names=['id', 'sentiment', 'text'], quoting=3)
        df_train['tokens'] = df_train['text'].apply(pp.tokenize_tweet)
        x_train_raw = df_train['tokens'].values
        y_train_raw = df_train['sentiment'].values

        print('Loaded %s Tweets as Training Data' % len(x_train_raw))

        # Load, preprocess and split Development/Evaluation dataset
        df_develop = pd.read_csv(path_developfile, sep='\t', header=None, names=['id', 'sentiment', 'text'], quoting=3)
        df_develop['tokens'] = df_develop['text'].apply(pp.tokenize_tweet)
        x_develop_raw = df_develop['tokens'].values
        y_develop_raw = df_develop['sentiment'].values

        print('Loaded %s Tweets as Development Data' % len(x_develop_raw))

        # Load, preprocess and split Test dataset
        df_test = pd.read_csv(path_testfile, sep='\t', header=None, names=['id', 'sentiment', 'text'], quoting=3)
        df_test['tokens'] = df_test['text'].apply(pp.tokenize_tweet)
        x_test_raw = df_test['tokens'].values
        y_test_raw = df_test['sentiment'].values

        print('Loaded %s Tweets as Test Data' % len(x_test_raw))

        # Build overall vocabulary by adding every token to a set (automatically avoids duplicates)
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
        """
        Private Method called to initialize the embedding matrix before classification
        :return: void
        """
        # set dimensions and fill embedding matrix with zeros
        self.embedding_matrix = np.zeros((len(self.vocabulary), self.embedding_dim))

        # for debugging/statistics: will count how many words of vocabulary are not found in prelearned embeddings
        embed_not_found_counter = 0

        for w, i in self.vocabulary.items():
            # words for which we do not have an embedding will stay zero in matrix
            if self.prelearned_embeddings.get(w) is not None:
                self.embedding_matrix[i] = self.prelearned_embeddings.get(w)
            else:
                embed_not_found_counter += 1

        print("For %s words of the Tweet Vocabulary, no prelearned embedding could be found" % embed_not_found_counter)

    def convnn(self, dim_convlayers=20, rank_kernels=2, verbose_mode=False, save_predictions_file=""):
        """
        Perform classification with Convolutional Neural Network

        :param dim_convlayers: number of dimensions of the Convolutional Layers
        :param rank_kernels: rank of convolution kernels used in the Convolutional Layers
        :param verbose_mode: print detailed results of each learning epoch during operation
        :param save_predictions_file: path to file in which the predictions are saved (optional)
        :return: classification accuracy
        """
        # convert back float arguments by Bayesian Optimization to integer (this would break tensorflow)
        dimconvlayers = int(round(dim_convlayers))
        rankkernels = int(round(rank_kernels))

        # Check if loading and preprocessing is finished, then prepare embedding matrix
        assert self.tweet_length > 0
        assert self.embedding_dim > 0

        if self.embedding_matrix is None:
            self.__prepare_embedding_matrix()

        # Prepare Embedding Layer as first layer in the network (initializing with embedding matrix)
        embedding_layer = Embedding(len(self.vocabulary),
                                    self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.tweet_length,
                                    trainable=False)

        inputtensor = Input(shape=(self.tweet_length,), dtype='int32')

        embedded_inputs = embedding_layer(inputtensor)

        # Sequence of Convolutional and Pooling Layers as described in the report
        x = Conv1D(dimconvlayers, rankkernels, activation='relu')(embedded_inputs)
        x = MaxPooling1D(rankkernels)(x)
        x = Conv1D(dimconvlayers, rankkernels, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)

        # Fully connected dense layer with 20 neurons
        x = Dense(20, activation='relu')(x)

        # Output layer with as many neurons as tweet classes (assuming equal number of classes in train/develop/test)
        outputlayer = Dense(self.y_train.shape[1], activation='softmax')(x)

        # Build and compile the model
        model = Model(inputtensor, outputlayer)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        # Fitting the model (i.e. learning with training set)
        model.fit(self.x_train, self.y_train,
                  batch_size=128,
                  epochs=10,
                  verbose=2 if verbose_mode else 0,
                  validation_data=(self.x_develop, self.y_develop))

        # Calculating classification accuracy on test set
        scores = model.evaluate(self.x_test, self.y_test, verbose=1 if verbose_mode else 0)

        # if user provided a path, save the actual predictions on the test set to a file
        if save_predictions_file != "":
            predictions = model.predict(self.x_test)
            np.savetxt(save_predictions_file, predictions, delimiter=",")

        # print loss and accuracy on test set
        if verbose_mode:
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])

        return scores[1]

    def lstm(self, verbose_mode=False, save_predictions_file=""):
        """
        Perform classification with Long Short-Term Memory Neural Network

        :param verbose_mode: print detailed results of each learning epoch during operation
        :param save_predictions_file: path to file in which the predictions are saved (optional)
        :return: classification accuracy
        """
        # Check if loading and preprocessing is finished, then prepare embedding matrix
        assert self.tweet_length > 0
        assert self.embedding_dim > 0

        if self.embedding_matrix is None:
            self.__prepare_embedding_matrix()

        # Prepare Embedding Layer as first layer in the network (initializing with embedding matrix)
        embedding_layer = Embedding(len(self.vocabulary),
                                    self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.tweet_length,
                                    trainable=False)

        inputtensor = Input(shape=(self.tweet_length,), dtype='int32')

        embedded_inputs = embedding_layer(inputtensor)

        # Two LSTM Layers (first argument refers to dimensionality of cell state, not number of cells!)
        x = LSTM(self.embedding_dim, return_sequences=True)(embedded_inputs)
        x = LSTM(self.embedding_dim, return_sequences=True)(x)

        # Flatten layer to serialize output into one-dimensional
        x = Flatten()(x)

        # Output layer with as many neurons as tweet classes (assuming equal number of classes in train/develop/test)
        x = Dense(self.y_train.shape[1], activation='softmax')(x)

        # Build and compile the model
        model = Model(inputtensor, x)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

        # Fitting the model (i.e. learning with training set)
        model.fit(self.x_train, self.y_train,
                  batch_size=128,
                  epochs=10,
                  verbose=2 if verbose_mode else 0,
                  validation_data=(self.x_develop, self.y_develop))

        # Calculating classification accuracy on test set
        scores = model.evaluate(self.x_test, self.y_test, verbose=1 if verbose_mode else 0)

        # if user provided a path, save the actual predictions on the test set to a file
        if save_predictions_file != "":
            predictions = model.predict(self.x_test)
            np.savetxt(save_predictions_file, predictions, delimiter=",")

        # print loss and accuracy on test set
        if verbose_mode:
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])

        return scores[1]


if __name__ == '__main__':

    # Sample Usage

    basedir = "C:/Users/WohnzimmerPC/Desktop/Datenpool_TMP2018_SSWE/"

    embeddingfile = os.path.join(basedir, 'glove-alpha05_50d/glove-alpha05_50d.csv')
    trainingfile = os.path.join(basedir, 'semeval_fullcleaned_2sent/twitter-2013train-A.txt')
    developfile = os.path.join(basedir, 'semeval_fullcleaned_2sent/twitter-2013dev-A.txt')
    testfile = os.path.join(basedir, 'semeval_fullcleaned_2sent/twitter-2013test-A.txt')

    NNC = NeuralNetworkClassifier()

    NNC.load_embedding(embedding_dim=50,
                       path_embeddingfile=embeddingfile)
    NNC.load_and_preprocess_data(path_trainingfile=trainingfile,
                                 path_developfile=developfile,
                                 path_testfile=testfile)

    NNC.convnn(verbose_mode=True)
    NNC.lstm(verbose_mode=True)
