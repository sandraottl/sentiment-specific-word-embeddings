"""
Main module for running the complete Bag-of-Embeddings experiment: 
Creating embedding files, constructing Bag-of-Embeddings, 
training and evaluating linear SVM classifier on the created BoEs and aggregating 
the best results.
author: Sandra Ottl
"""
import subprocess
import pandas as pd
import numpy as np
from decimal import Decimal
from os import listdir, makedirs
from os.path import join, isdir, abspath, basename, splitext
from .convert_tweets_to_embeddings import convert_tweet_csv
from .linear_svm import run_SVM
from ..embeddings.embedding import Embedding
from collections import OrderedDict
from argparse import ArgumentParser

__OPEN_XBOW_PATH = '/media/storage/TMP/openXBOW/openXBOW.jar'


def create_embedding_files(train, devel, test, embeddings, basepath='experiment'):
    """
    Creating embedding files for training, development and testing files.
    Converts the tweet csvs by looking up word embeddings for every word of each tweet.
    arguments:
        train: path to training csv
        devel: path to development csv
        test: path to testing csv
        embeddings: path to embeddings file
        basepath: folder for the converted csvs
    """
    print('Loading embeddings ...')
    makedirs(basepath, exist_ok=True)
    embedding = Embedding()
    embedding.load(embeddings)
    print('Writing embeddings of training partition ...')
    convert_tweet_csv(train, join(basepath, basename(train)), embedding)
    print('Writing embeddings of development partition ...')
    convert_tweet_csv(devel, join(basepath, basename(devel)), embedding)
    print('Writing embeddings of testing partition ...')
    convert_tweet_csv(test, join(basepath, basename(test)), embedding)


def create_BoW(train, devel, test, cbs=[10, 20, 50, 100, 200, 500, 1000, 2000], no_vectors=[1, 10, 25, 50, 100, 200, 500, 1000], BoW_folder='BoW'):
    """
    Creates BoEs from csvs with word vectors for different codebook sizes and 
    number of vectors by using openXBOW.
    arguments:
        train: path to training csv
        devel: path to development csv
        test: path to testing csv
        cbs: codebook sizes
        no_vectors: number of vectors
        BoW_folder: output folder for BoEs
    """
    makedirs(BoW_folder, exist_ok=True)  # create BoW directory
    for size in cbs:
        makedirs(join(BoW_folder, str(size)), exist_ok=True)
        for a in no_vectors:
            if a > size/2:
                break
            output_directory = join(BoW_folder, str(size), str(a))
            makedirs(output_directory, exist_ok=True)
            print('Creating BoWs for codebook size {} and a={} ...'.format(str(size), str(a)))
            subprocess.run(["java", "-Xmx15g", "-jar", __OPEN_XBOW_PATH, "-i", train, "-o", join(output_directory, basename(train)), "-B", join(output_directory, 'codebook'), "-size", str(size), "-a", str(a), "-log", "-norm", '1', "-writeName"])  # log: logarithmic term weighting, norm 1: normalize bag by its length, writeName: write tweet ID into output
            for eval_file in [devel, test]:
                subprocess.run(["java", "-Xmx15g", "-jar", __OPEN_XBOW_PATH, "-i", eval_file, "-o", join(output_directory, basename(eval_file)), "-b", join(output_directory, 'codebook'), "-norm", '1', "-writeName"])  # norm 1: normalize bag by its length, writeName: write tweet ID into output


def evaluate_BoW(train, devel, test, complexity, BoW_folder='BoW'):
    """
    Loops through all BoE configurations and evaluates them with a linear SVM classifier.
    arguments:
        train: common filename of all training csvs
        devel: common filename of all development csvs
        test: common filename of all testing csvs
        complexity: complexity values
        BoW_folder: folder containing BoEs
    """
    size_folders = [join(BoW_folder, folder) for folder in listdir(BoW_folder) if isdir(join(BoW_folder, folder))]
    for size in size_folders:
        a_folders = [join(size, folder) for folder in listdir(size) if isdir(join(size, folder))]
        for a in a_folders:
            print('Evaluating BoWs for codebook size {} and a={} ...'.format(basename(size), basename(a)))
            run_SVM(join(a, train), join(a, devel), join(a, test), complexity=complexity, cm_path=join(a, 'confusion_matrix.pdf'), output=join(a, 'results.csv'))


def get_best_results(file_path, n_best=1):
    """
    Get n_best (complexity, Accuracy Development, Accuracy Test) triples from results csv.
    arguments:
        file_path: results csv
        n_best: number of best triples that are returned (default: 1)
    returns: n_best (complexity, Accuracy Development, Accuracy Test) triples
    """
    df = pd.read_csv(file_path, dtype=object)
    f = lambda x: x.strip('%')  # get percent number without percentage sign
    df[['Accuracy Development', 'Accuracy Test']] = df[['Accuracy Development', 'Accuracy Test']].applymap(f)
    df[['Accuracy Development', 'Accuracy Test']] = df[['Accuracy Development', 'Accuracy Test']].astype(float)  # applymap: apply f on all elements of selected df

    best_results = []
    for idx in range(n_best):
        best_result = df.loc[df['Accuracy Development'].idxmax()]  # line (of df) where Accuracy Development has maximum
        best_results.append(
            (best_result['Complexity'], best_result['Accuracy Development'].astype(str),
             best_result['Accuracy Test'].astype(str)))
        df = df[df['Complexity'] != best_result['Complexity']]  # remove this line from df
    return best_results


def aggregate_performance(basepath):
    """
    Aggregate best BoW results.
    Write csv containing best two results for each codebook size and assignment value and 
    the best overall result.
    arguments:
        basepath: basepath of experiment results
    """
    PATH = abspath(basepath)

    # codebook sizes
    sizes = sorted([
        int(folder_name) for folder_name in listdir(PATH)
        if isdir(join(PATH, folder_name))
    ])

    # folder dictionary for folder structure: BoW/codebook size/a
    folder_dict = {
        size: sorted([
            assignment_value
            for assignment_value in listdir(join(PATH, str(size)))
        ])
        for size in sizes
    }
    data = OrderedDict()
    # best_overall_result: (codebook size, a, complexity, accuracy devel, accuracy test)
    best_overall_result = (0, 0, 0.0, 0.0, 0.0)
    for size in sizes:
        for assignment_value in folder_dict[size]:
            # best_results: list of triples (complexity, accuracy development, accuracy test)
            best_results = get_best_results(
                join(PATH, str(size), str(assignment_value), 'results.csv'),
                n_best=2)
            best_overall_result = (
                size, assignment_value, best_results[0][0], best_results[0][1], best_results[0][2]
            ) if float(best_results[0][1]) > float(best_overall_result[3]) else best_overall_result
            best_results_string = '   '.join(map(' '.join, best_results))
            if size in data:
                data[size][assignment_value] = best_results_string
            else:
                data[size] = OrderedDict()
                data[size][assignment_value] = best_results_string

    df = pd.DataFrame.from_dict(data, orient='index').T  # T: transpose (table with size in rows and a in columns)
    df.to_csv(join(PATH, 'BoW-performance.csv'))

    # write best_overall_result
    with open(join(PATH, 'BoW-performance.csv'), mode='a') as csv:
        csv.write('\n' + '  '.join(map(str, best_overall_result)))


def main():
    """
    Main method for running the complete Bag-of-Embeddings experiment.
    """
    parser = ArgumentParser(
        description=
        'Run BoW experiments.')
    parser.add_argument(
        'input',
        nargs=3,
        help='csv files of training, development and test sets')
    parser.add_argument(
        '-C',
        nargs='+',
        type=Decimal,
        help='Complexities for SVM.',
        required=False,
        default=np.logspace(0, -9, num=10))
    parser.add_argument(
        '-sizes',
        nargs='+',
        type=int,
        help='Codebook sizes',
        default=[10, 20, 50, 100, 200, 500, 1000, 2000])
    parser.add_argument(
        '-no_vectors',
        nargs='+',
        type=int,
        help='Assignment values for codebookvectors',
        default=[1, 10, 25, 50, 100, 200, 500, 1000])
    parser.add_argument(
        '-output', help='Output base path.', required=False, default=None)
    parser.add_argument(
        '-embeddings',
        required=True,
        help='csv file with word embeddings')
    args = vars(parser.parse_args())
    train_tweets, devel_tweets, test_tweets, output = args['input'][0], args['input'][1], args['input'][2], args['output']
    create_embedding_files(train_tweets, devel_tweets, test_tweets, args['embeddings'], output)
    train, devel, test = join(output, basename(train_tweets)), join(output, basename(devel_tweets)), join(output, basename(test_tweets))
    create_BoW(train, devel, test, cbs=args['sizes'], no_vectors=args['no_vectors'], BoW_folder=join(args['output'], 'BoW'))
    evaluate_BoW(basename(train), basename(devel), basename(test), complexity=args['C'], BoW_folder=join(output, 'BoW'))
    aggregate_performance(join(output, 'BoW'))


if __name__=='__main__':
    main()
