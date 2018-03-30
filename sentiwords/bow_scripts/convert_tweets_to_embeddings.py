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
    parameters:
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
                    writer.writerow([id]+list(row)+[sentiment])

# def main():
#     parser = argparse.ArgumentParser(
#         description=
#         'Look up word embeddings for tweets in a csv of format "tweet_id<tab>sentiment<tab>tweet" and write them to an output csv. '
#     )
#     parser.add_argument('-i', required=True, help='Input csv filled with tweets.')
#     parser.add_argument('-o', required=True, help='Output path.')
#     parser.add_argument('-embedding', required=True, help='Path to embedding csv.')
#     args = parser.parse_args()
#     embedding = Embedding()
#     embedding.load(args.embedding)
#     convert_tweet_csv(args.i, args.o, embedding)


# if __name__=='__main__':
#     main()
