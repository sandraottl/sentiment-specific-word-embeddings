# BAYESIAN OPTIMIZATION
# for finding better hyperparameter settings

# import sys
from bayes_opt import BayesianOptimization

from ..classification.neuralnetworks import NeuralNetworkClassifier
# sys.path.append('C:/Users/WohnzimmerPC/Documents/GitHub/sentiment-specific-word-embeddings/sentiwords/')
# from classification.neuralnetworks import NeuralNetworkClassifier


NNC = NeuralNetworkClassifier()

NNC.load_embedding(embedding_dim=100,
                   path_embeddingfile='C:/Users/WohnzimmerPC/Desktop/Datenpool_TMP2018_SSWE/glove-standard/glove.6B.100d.txt')
NNC.load_and_preprocess_data(
    path_trainingfile='C:/Users/WohnzimmerPC/Desktop/Datenpool_TMP2018_SSWE/semeval_full/twitter-2013train-A.txt',
    path_developfile='C:/Users/WohnzimmerPC/Desktop/Datenpool_TMP2018_SSWE/semeval_full/twitter-2013dev-A.txt',
    path_testfile='C:/Users/WohnzimmerPC/Desktop/Datenpool_TMP2018_SSWE/semeval_full/twitter-2013test-A.txt')


Optimizer = BayesianOptimization(NNC.convnn,{'dim_convlayers':(10,50),'dim_denselayer':(10,50)})

Optimizer.maximize()

print('Ergebnis: %f' % Optimizer.res['max'])
print('Ergebnis: %f' % Optimizer.res['max']['max_val'])
