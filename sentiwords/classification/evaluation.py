"""
Evaluation of different embeddings on the Semeval 2013 dataset with CNN and LSTM networks

Responsibility: Rick Beer
"""

import sys
import os

# from ..classification.neuralnetworks import NeuralNetworkClassifier
sys.path.append('C:/Users/WohnzimmerPC/Documents/GitHub/sentiment-specific-word-embeddings/sentiwords/')
from classification.neuralnetworks import NeuralNetworkClassifier


basedir = "C:/Users/WohnzimmerPC/Desktop/Datenpool_TMP2018_SSWE/"
trainingfile = os.path.join(basedir, 'semeval_fullcleaned_2sent/twitter-2013train-A.txt')
developfile = os.path.join(basedir, 'semeval_fullcleaned_2sent/twitter-2013dev-A.txt')
testfile = os.path.join(basedir, 'semeval_fullcleaned_2sent/twitter-2013test-A.txt')


# 1. Stock Glove Embeddings (25d)

NNC_Stock25 = NeuralNetworkClassifier()
NNC_Stock25.load_embedding(25, os.path.join(basedir, 'glove-standard/twitter.27B.25d.txt'))
NNC_Stock25.load_and_preprocess_data(trainingfile, developfile, testfile)

print("Classification accuracy of CNN with Stock Glove (25d):", NNC_Stock25.convnn())
print("Classification accuracy of LSTM with Stock Glove (25d):", NNC_Stock25.lstm())


# 2. Stock Glove Embeddings (50d)

NNC_Stock50 = NeuralNetworkClassifier()
NNC_Stock50.load_embedding(50, os.path.join(basedir, 'glove-standard/twitter.27B.50d.txt'))
NNC_Stock50.load_and_preprocess_data(trainingfile, developfile, testfile)

print("Classification accuracy of CNN with Stock Glove (50d):", NNC_Stock50.convnn())
print("Classification accuracy of LSTM with Stock Glove (50d):", NNC_Stock50.lstm())


# 3. SSWE (25d)

NNC_SSWE25 = NeuralNetworkClassifier()
NNC_SSWE25.load_embedding(25, os.path.join(basedir, 'sswe-alpha05_25d/sswe-alpha05_25d.csv'))
NNC_SSWE25.load_and_preprocess_data(trainingfile, developfile, testfile)

print("Classification accuracy of CNN with SSWE (25d):", NNC_SSWE25.convnn())
print("Classification accuracy of LSTM with SSWE (25d):", NNC_SSWE25.lstm())


# 4. SSWE (50d)

NNC_SSWE50 = NeuralNetworkClassifier()
NNC_SSWE50.load_embedding(50, os.path.join(basedir, 'sswe-alpha05_50d/sswe-alpha05_50d.csv'))
NNC_SSWE50.load_and_preprocess_data(trainingfile, developfile, testfile)

print("Classification accuracy of CNN with SSWE (50d):", NNC_SSWE50.convnn())
print("Classification accuracy of LSTM with SSWE (50d):", NNC_SSWE50.lstm())


# 5. Sentiment-trained Glove Embeddings (25d)

NNC_SentGlove25 = NeuralNetworkClassifier()
NNC_SentGlove25.load_embedding(25, os.path.join(basedir, 'glove-alpha05_25d/glove-alpha05_25d.csv'))
NNC_SentGlove25.load_and_preprocess_data(trainingfile, developfile, testfile)

print("Classification accuracy of CNN with Sentiment-Trained Glove (25d):", NNC_SentGlove25.convnn())
print("Classification accuracy of LSTM with Sentiment-Trained Glove (25d):", NNC_SentGlove25.lstm())


# 6. Sentiment-trained Glove Embeddings (50d)

NNC_SentGlove50 = NeuralNetworkClassifier()
NNC_SentGlove50.load_embedding(50, os.path.join(basedir, 'glove-alpha05_50d/glove-alpha05_50d.csv'))
NNC_SentGlove50.load_and_preprocess_data(trainingfile, developfile, testfile)

print("Classification accuracy of CNN with Sentiment-Trained Glove (50d):", NNC_SentGlove50.convnn())
print("Classification accuracy of LSTM with Sentiment-Trained Glove (50d):", NNC_SentGlove50.lstm())