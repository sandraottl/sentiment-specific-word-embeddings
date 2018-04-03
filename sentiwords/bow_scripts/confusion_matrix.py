"""
Module for plotting a confusion matrix.
author: Sandra Ottl
"""
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from os.path import splitext, dirname, abspath
from os import makedirs
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.metrics import accuracy_score, confusion_matrix
from argparse import ArgumentParser


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues',
                          predicted_label='Prediction',
                          true_label='Actual'):
    """
    Adapted from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    Plots a confusion matrix.
    arguments:
        cm: confusion matrix
        classes: sorted labels for all classes (e.g. negative, positive)
        normalize: applying normalization for color purposes (color depends on class recall rather than counts)
        title: title over confusion matrix
        cmap: color map
        predicited_label: x axis
        true_label: y axis
    returns: matplotlib figure
    """
    fig = matplotlib.figure.Figure(dpi=200)  # dpi: pixel per inch
    original_cm = cm
    if normalize:
        cm = cm.astype('float') / cm.sum(
            axis=1)[:,
                    np.newaxis]  # calculate recall for every element in matrix
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, vmin=0, vmax=1, cmap=cmap)  # plot confusion matrix
    fig.colorbar(im)  # add color bar
    ax.set_title(title)  # add title
    tick_marks = np.arange(len(classes))  # add tick marks
    ax.set_xlabel(predicted_label)  # set x axis label (Prediction)
    ax.set_xticks(tick_marks)  # set x axis tick marks
    ax.set_xticklabels(
        classes,
        rotation=45)  # set x axis tick labels (negative, positive) rotated
    ax.set_ylabel(true_label)  # set y axis label (Actual)
    ax.set_yticks(tick_marks)  # set y axis tick marks
    ax.set_yticklabels(classes)  # set y axis tick labels (negative, positive)

    # add counts to plot
    fmt = 'd'
    thresh = 0.5  # threshold for displaying counts in black/white

    # label every confusion matrix box
    for i, j in itertools.product(
            range(original_cm.shape[0]), range(original_cm.shape[1])):
        ax.text(
            j,
            i,
            format(original_cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    fig.set_tight_layout(True)
    return fig


def save_fig(fig, save_path):
    """
    Saves a figure to a certain location.
    arguments:
        fig: matplotlib figure
        save_path: path of the saved figure
    """
    canvas = FigureCanvasAgg(fig)
    fig.savefig(save_path, format=splitext(save_path)[1][1:])


def main():
    """
    Main method for plotting a confusion matrix of an input file.
    """
    parser = ArgumentParser(description='Run BoW experiments.')
    parser.add_argument('input', help='csv file')
    parser.add_argument('output', help='output path.', default=None)
    args = vars(parser.parse_args())

    data = np.genfromtxt(args['input'], delimiter=';', dtype=str)
    predictions = data[1:, 0]
    actual = data[1:, 1]
    print(predictions, actual)
    labels = sorted(set(actual))
    cm = confusion_matrix(actual, predictions, labels=labels)
    print(accuracy_score(actual, predictions))

    fig = plot_confusion_matrix(
        cm,
        classes=labels,
        normalize=True,
        title='Confusion Matrix',
        cmap='Blues')
    save_fig(fig, args['output'])


if __name__ == '__main__':
    main()
