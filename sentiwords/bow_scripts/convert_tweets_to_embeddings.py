"""
Module for converting a csv with tweets and their sentiment labels into a corresponding embeddings csv.
author: Sandra Ottl
"""
import csv
import argparse
from tqdm import tqdm
from ..processing.preprocessing import Preprocessor
from ..embeddings.embedding import Embedding


def convert_tweet_csv(input_csv,
                      output_csv,
                      embedding,
                      delimiter='\t',
                      preprocessor=Preprocessor()):
    """
    Convert csv with tweets and their sentiment labels into a corresponding embeddings csv.
    arguments:
        input_csv: csv containing tweets
        output_csv: csv containing embeddings
        embedding_csv: csv containing words and their embedding vectors
        delimiter: used delimiter (e.g. tab)
        preprocessor: object that has a tokenize_tweet method
    """
    with open(input_csv) as input, open(output_csv, 'w', newline='') as output:
        reader = csv.reader(input, delimiter=delimiter)
        writer = csv.writer(output, delimiter=';')

        # write header
        header = ['id'] + ['emb_' + str(i)
                           for i in range(embedding.size)] + ['class']
        writer.writerow(header)
        for line in reader:
            id = line[0]
            sentiment = line[1]
            tweet = line[2]
            if tweet != 'Not Available':  # if tweet is available (should be always)
                tokenized_tweet = preprocessor.tokenize_tweet(tweet)
                embeddings = embedding.lookup(tokenized_tweet)
                for row in embeddings:
                    writer.writerow([id] + list(row) + [sentiment])
