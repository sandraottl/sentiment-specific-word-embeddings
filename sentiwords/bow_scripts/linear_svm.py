"""
Module for loading training, development and testing csv, training a linear SVM classifier 
on those, and constructing a confusion matrix.
author: Sandra Ottl
"""
import argparse
import csv
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
from decimal import Decimal
from os.path import abspath, dirname
from os import makedirs
from .confusion_matrix import plot_confusion_matrix, save_fig

RANDOM_SEED = 42


def _load(file):
    """
    Get values of names, features and labels out of input csv.
    arguments:
        file: file that has to be loaded
    returns: numpy arrays of names, features and labels contained in input csv
    """
    df = pd.read_csv(file, sep=';', header=None)
    names = df.iloc[:, 0].astype(str)
    features = df.iloc[:, 1:-1].astype(float)
    labels = df.iloc[:, -1].astype(str)
    return names, features, labels


def parameter_search_train_devel_test(train_X,
                                      train_y,
                                      devel_X,
                                      devel_y,
                                      test_X,
                                      test_y,
                                      Cs=np.logspace(0, -9, num=10),
                                      output=None):
    """
    Run optimization loop for linear SVM classifier on train, devel and test.
    For each complexity, SVM is trained on both train and combined traindevel 
    and evaluated on devel and test respectively. The achieved accuracies are 
    optionally written to a csv file for each C.
    arguments:
        train_X: training features
        train_y: training labels
        devel_X: development features
        devel_y: development labels
        test_X: testing features
        test_y: testing labels
        Cs: c values
        output: output filepath (.csv)
    returns: test predictions and accuracy for optimized model
    """
    best_war_devel = 0
    traindevel_X = np.append(train_X, devel_X, axis=0)
    traindevel_y = np.append(train_y, devel_y)
    try:
        if output:
            dir = dirname(output)
            makedirs(dir, exist_ok=True)
            csv_file = open(output, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ['Complexity', 'Accuracy Development', 'Accuracy Test'])
        for C in Cs:
            clf = LinearSVC(
                C=C, class_weight='balanced',
                random_state=RANDOM_SEED)  # classifier
            clf.fit(train_X, train_y)  # train SVM on training partition
            predicted_devel = clf.predict(
                devel_X
            )  # predict labels for classifier on development features
            WAR_devel = accuracy_score(devel_y,
                                       predicted_devel)  # compute accuracy

            clf = LinearSVC(
                C=C, class_weight='balanced', random_state=RANDOM_SEED)
            clf.fit(traindevel_X,
                    traindevel_y)  # train SVM on development partition
            predicted_test = clf.predict(test_X)
            WAR_test = accuracy_score(test_y, predicted_test)
            print('C: {:.1E} WAR development: {:.2%} WAR test: {:.2%}'.format(
                Decimal(C), WAR_devel, WAR_test))
            if output:
                csv_writer.writerow([
                    '{:.1E}'.format(Decimal(C)), '{:.2%}'.format(WAR_devel),
                    '{:.2%}'.format(WAR_test)
                ])
            if WAR_devel > best_war_devel:  # save test accuracy of best devel accuracy
                best_war_devel = WAR_devel
                final_war_test = WAR_test
                best_prediction = predicted_test
        return final_war_test, best_prediction

    finally:
        if output:
            csv_file.close()


def run_SVM(train,
            devel,
            test,
            complexity=np.logspace(0, -9, num=10),
            cm_path=None,
            output=None):
    """
    Loading input training, development and testing csvs and training of a linear SVM classifier 
    on those. Construction of a confusion matrix.
    arguments:
        train: path to training csv
        devel: path to development csv
        test: path to testing csv
        complexity: complexity values
        cm_path: path to confusion_matrix
        output: output filepath (.csv)
    """
    print('Loading input ...')
    train_names, train_X, train_y = _load(train)
    devel_names, devel_X, devel_y = _load(devel)
    test_names, test_X, test_y = _load(test)
    labels = sorted(set(train_y))

    print('Starting training ...')
    WAR, best_prediction = parameter_search_train_devel_test(
        train_X,
        train_y,
        devel_X,
        devel_y,
        test_X,
        test_y,
        Cs=complexity,
        output=output)

    cm = confusion_matrix(test_y, best_prediction, labels=labels)
    if cm_path:
        print('Writing confusion matrix ...')
        cm_path = abspath(cm_path)
        makedirs(
            dirname(cm_path), exist_ok=True
        )  # if directory of confusion matrix does not exist yet
        fig = plot_confusion_matrix(
            cm,
            classes=labels,
            normalize=True,
            title='Accuracy {:.1%}'.format(WAR))
        save_fig(fig, cm_path)
