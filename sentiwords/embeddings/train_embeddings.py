import csv
import tensorflow as tf
import argparse
import numpy as np
from decimal import Decimal
from ..processing.preprocessing import Preprocessor
from .embedding import load_vocab, lookup_ids
from nltk import ngrams
from tqdm import tqdm
from random import randint

tf.logging.set_verbosity(tf.logging.INFO)

class InputGenerator():

    def __init__(self, input_path, vocab, preprocessor=Preprocessor(), csv_delimiter=',', ngram=1, csv_columns=['sentiment', 'id', 'date', 'status', 'user', 'text']):
        input_file = open(input_path, 'r', encoding='utf-8')
        self.reader = csv.reader(input_file, delimiter=csv_delimiter)
        if csv_columns is None:
            self.csv_columns = next(self.reader)
        self.preprocessor = preprocessor
        self.ngram = ngram
        self.csv_columns = csv_columns
        self.vocab = vocab
        self.samples = iter(self.generate_samples())

    def generate_samples(self):
        for line in self.reader:
            current_tweet = list(map('_'.join, ngrams(self.preprocessor.tokenize_tweet(line[self.csv_columns.index('text')]), self.ngram)))[::self.ngram]
            current_label = 1 if int(line[self.csv_columns.index('sentiment')]) > 0 else - 1
            chunks = (current_tweet[i*3:(i+1)*3] for i in range(0, int(len(current_tweet)/3)))
            for current_chunk in chunks:
                sample = lookup_ids(self.vocab, current_chunk)
                yield (sample, self.negative_sample(sample)), current_label

    def negative_sample(self, sample):
        random_word = randint(0, len(self.vocab)-1)
        while random_word == sample[1]:
            random_word = randint(0, len(self.vocab)-1)
        return [sample[0], random_word, sample[2]]

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.samples)

def input_fn(csv, vocabulary, csv_delimiter=',', preprocessor=Preprocessor(), ngram=1, shuffle=True, num_epochs=None, batch_size=32, csv_columns=['sentiment', 'id', 'date', 'status', 'user', 'text']):
    input_generator = InputGenerator(csv, vocabulary, preprocessor, csv_delimiter, ngram, csv_columns)
    def gen():
        for sample in input_generator:
            yield sample
    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.int64, tf.int64), output_shapes=([2,3],[]))
    dataset = dataset.prefetch(100000)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    with tf.name_scope('input'):
        features, labels = iterator.get_next()
        return {'original': features[:,0], 'corrupted': features[:,1]}, labels

def model_fn_wrapper(mode, features, labels, params={'vocabulary_size': 100000, 'alpha': 0.5, 'hidden_units': 20, 'learning_rate':0.1, 'embedding_size': 50}):
    def shared_network(input):
        # input_layer = tf.feature_column.input_layer(features, word_ids)
        # define embedding variable
        word_embeddings = tf.get_variable('word_embeddings', [params['vocabulary_size'], params['embedding_size']])

        # lookup embeddings for true and negative sample
        embeds = tf.nn.embedding_lookup(word_embeddings, input, name='embeddings')
        flattened_embeds = tf.layers.flatten(embeds, name='flattened_embeds')

        hidden = tf.layers.dense(flattened_embeds, params['hidden_units'], activation=lambda x: tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1), name='hidden', reuse=tf.AUTO_REUSE)
        output = tf.layers.dense(hidden, 2, name='output', reuse=tf.AUTO_REUSE)
        return output

    with tf.variable_scope('shared_network', reuse=tf.AUTO_REUSE) as scope:
        # original ngram output

        original_output = shared_network(features['original'])
        original_semantic_score = original_output[:,0]
        original_sentiment_score = original_output[:,1]

        # corrupted ngram output
        corrupted_output = shared_network(features['corrupted'])
        corrupted_semantic_score = corrupted_output[:,0]
        corrupted_sentiment_score = corrupted_output[:,1]


    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope('loss'):
            sentiment_loss = tf.maximum(tf.cast(0, tf.float32), 1 - tf.cast(labels, tf.float32) * original_sentiment_score + tf.cast(labels, tf.float32) * corrupted_sentiment_score, name='sentiment')
            semantic_loss = tf.maximum(tf.cast(0, tf.float32), 1 - original_semantic_score + corrupted_semantic_score, name='semantic')
            loss = tf.reduce_mean(params['alpha']*semantic_loss + (1-params['alpha'])*sentiment_loss, name='combined')
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = original_output
    else:
        predictions = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.AdagradOptimizer(params['learning_rate'])
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step(), name='train_op')
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description=
        'Train a Sentiment-specific word embeddings on a csv twitter sentiment dataset.')
    parser.add_argument(
        '-data',
        required=True,
        default=None,
        help='Twitter sentiment dataset in csv format.')
    parser.add_argument('-vocabulary', help='Vocabulary file (each word on separate line).', required=True)
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batchsize for training.')
    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
        help='Number of epochs to train the model.')
    parser.add_argument(
        '--model_dir',
        default=None,
        help=
        'Directory for saving and restoring model checkpoints, summaries and exports.'
    )
    parser.add_argument(
        '--alpha',
        default=0.5,
        type=float,
        help='Alpha parameter used to weigh syntactic versus sentiment loss.')
    parser.add_argument(
        '--lr',
        default=0.1,
        type=float,
        help='Learning rate.')
    parser.add_argument(
        '--hidden', default=20, type=int, help='Number of units of the hidden layer.')
    parser.add_argument(
        '--embedding_size',
        default=50,
        type=int,
        help='Size of word embedding vectors.')
    parser.add_argument(
        '--keep_checkpoints',
        default=10,
        type=int,
        help='How many checkpoints to keep stored on disk.')
    args = parser.parse_args()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        keep_checkpoint_max=args.keep_checkpoints,
        session_config=session_config
        )
    vocab = load_vocab(args.vocabulary)
    i = InputGenerator(args.data, vocab, Preprocessor(), ngram=1)
    model = tf.estimator.Estimator(model_fn=model_fn_wrapper, model_dir=args.model_dir, params={'vocabulary_size': len(vocab), 'alpha': args.alpha, 'hidden_units': args.hidden, 'learning_rate': args.lr, 'embedding_size': args.embedding_size}, config=config)
    model.train(lambda: input_fn(args.data, vocab, num_epochs=args.epochs, batch_size=args.batch_size))
