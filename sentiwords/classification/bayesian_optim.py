"""
Bayesian Optimization for finding best hyperparameter settings for CNN (dim_convlayers, rank_kernels)

(Using library from https://github.com/fmfn/BayesianOptimization)

Responsibility: Rick Beer

"""

import sys
import os
from bayes_opt import BayesianOptimization

# from ..classification.neuralnetworks import NeuralNetworkClassifier
sys.path.append('C:/Users/WohnzimmerPC/Documents/GitHub/sentiment-specific-word-embeddings/sentiwords/')
from classification.neuralnetworks import NeuralNetworkClassifier


# Prepare and load data sets / embeddings
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


# Create BayesOpt. instance and specify the parameters to be optimized with the range of allowed values (minval,maxval)
NNCOptimizer = BayesianOptimization(NNC.convnn, {'dim_convlayers': (10, 50), 'rank_kernels': (1, 5)})

# Set number of iterations and initialize searching
gp_params = {"alpha": 1e-4}
NNCOptimizer.maximize(n_iter=10, **gp_params)

# Print results
print("Optimal combination of input parameters: dim_convlayers =",
      NNCOptimizer.res['max']['max_params']['dim_convlayers'],
      ", rank_kernels =",
      NNCOptimizer.res['max']['max_params']['rank_kernels'])

print("-> reaches max. performance of:", NNCOptimizer.res['max']['max_val'])
