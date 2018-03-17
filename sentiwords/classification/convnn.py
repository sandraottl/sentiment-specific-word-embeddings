# Classification with Convolutional Neural Network
# ************************************************

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding
from keras.models import Model

#from preprocessing import Preprocessor
from ..processing.preprocessing import Preprocessor


# Some parameters / settings

path_embeddingfile = 'C:/Users/rickrack/Downloads/glove.6B/glove.6B.100d.txt'
path_trainingfile = 'C:/Users/rickrack/Downloads/semeval2013/twitter-2013train-A.txt'
path_developfile = 'C:/Users/rickrack/Downloads/semeval2013/twitter-2013dev-A.txt'
path_testfile = 'C:/Users/rickrack/Downloads/semeval2013/twitter-2013test-A.txt'

embedding_dim = 100
tweet_length = 40   # padding length



# BEGIN Loading and preprocessing of Training, Development and Test Data

pp = Preprocessor()

df_train = pd.read_csv(path_trainingfile, sep='\t', header=None, names=['id','sentiment','text'], quoting=3)
df_train['tokens'] = df_train['text'].apply(pp.tokenize_tweet)
x_train_raw = df_train['tokens'].values
y_train_raw = df_train['sentiment'].values

print('Loaded %s Tweets as Training Data' % len(x_train_raw))

df_develop = pd.read_csv(path_developfile, sep='\t', header=None, names=['id','sentiment','text'], quoting=3)
df_develop['tokens'] = df_develop['text'].apply(pp.tokenize_tweet)
x_develop_raw = df_develop['tokens'].values
y_develop_raw = df_develop['sentiment'].values

print('Loaded %s Tweets as Development Data' % len(x_develop_raw))

df_test = pd.read_csv(path_testfile, sep='\t', header=None, names=['id','sentiment','text'], quoting=3)
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

vocabulary = {
            word: index
            for index, word in enumerate(sorted(unique_tokens))
        }

print('Overall vocabulary size: %s' % len(vocabulary))


# Translate Tokens to Indices in the Tweets and pad them to the same length

x_train_indices = [[vocabulary[token] for token in tweet] for tweet in x_train_raw]
x_develop_indices = [[vocabulary[token] for token in tweet] for tweet in x_develop_raw]
x_test_indices = [[vocabulary[token] for token in tweet] for tweet in x_test_raw]

x_train = pad_sequences(x_train_indices, tweet_length)
x_develop = pad_sequences(x_develop_indices, tweet_length)
x_test = pad_sequences(x_test_indices, tweet_length)

# Transform Sentiment Labels in binary format

encoder = LabelEncoder()
encoder.fit(y_train_raw)

y_train = to_categorical(encoder.transform(y_train_raw))
y_develop = to_categorical(encoder.transform(y_develop_raw))
y_test = to_categorical(encoder.transform(y_test_raw))


# Short Summary for Debugging
print("Shapes of Input Tensors X :", x_train.shape, x_develop.shape, x_test.shape)
print("Shapes of Output Tensors Y:", y_train.shape, y_develop.shape, y_test.shape)


# END Loading and preprocessing Data


# BEGIN Preparing Embedding Matrix

# Load pre-learned Embeddings from .txt file into a dictionary (mapping word -> embedding)

prelearned_embeddings = {}

with open(path_embeddingfile, encoding='utf8') as embedfile:
    for line in embedfile:
        values = line.split()
        word = values[0]
        embed = np.asarray(values[1:], dtype='float32')
        prelearned_embeddings[word] = embed

print('Loaded %s pre-learned word vectors.' % len(prelearned_embeddings))

# Create and fill Embedding matrix

embedding_matrix = np.zeros((len(vocabulary), embedding_dim))

embed_not_found_counter = 0  # for debugging/statistics; will count how many words of vocabulary are not found in prelearned embeddings

for w, i in vocabulary.items():
    if prelearned_embeddings.get(w) is not None:     # words for which we do not have an embedding will stay zero in matrix
        embedding_matrix[i] = prelearned_embeddings.get(w)
    else:
        embed_not_found_counter += 1

print("For %s words of the Tweet Vocabulary, no prelearned embedding could be found" % embed_not_found_counter)

# END Preparing Embedding Matrix


# BEGIN Convoluation Neural Network Setup and Operation

embedding_layer = Embedding(len(vocabulary),
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=tweet_length,
                            trainable=False)

inputtensor = Input(shape=(tweet_length,), dtype='int32')

embedded_inputs = embedding_layer(inputtensor)

x = Conv1D(20, 2, activation='relu')(embedded_inputs)
x = MaxPooling1D(2)(x)
x = Conv1D(20, 2, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(20, 2, activation='relu')(x)
x = GlobalMaxPooling1D()(x)

x = Dense(20, activation='relu')(x)

outputlayer = Dense(y_train.shape[1], activation='softmax')(x)

model = Model(inputtensor, outputlayer)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=2,
          validation_data=(x_develop, y_develop))

scores = model.evaluate(x_test, y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# END Convoluation Neural Network