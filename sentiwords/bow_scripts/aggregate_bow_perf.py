import pandas as pd
from decimal import Decimal
from os import listdir
from os.path import isdir, join, abspath
from collections import OrderedDict
from argparse import ArgumentParser


def get_best_results(file_path, n_best=1):
    df = pd.read_csv(file_path, dtype=object)
    f = lambda x: x.strip('%')
    df[['UAR Train', 'UAR Development']] = df[['UAR Train',
                                               'UAR Development']].applymap(f)
    best_results = []
    for idx in range(n_best):
        best_result = df.loc[df['UAR Development'].idxmax()]
        best_results.append(
            (best_result['Complexity'], best_result['UAR Train'],
             best_result['UAR Development']))
        df = df[df['Complexity'] != best_result['Complexity']]
    return best_results


if __name__ == '__main__':

    parser = ArgumentParser(description='Aggregate BoW performance.')
    parser.add_argument('input', help='Basepath for BoWs.')
    args = parser.parse_args()

    PATH = abspath(args.input)

    sizes = sorted([
        int(folder_name) for folder_name in listdir(PATH)
        if isdir(join(PATH, folder_name))
    ])

    folder_dict = {
        size: sorted([
            assignment_value
            for assignment_value in listdir(join(PATH, str(size)))
        ])
        for size in sizes
    }
    data = OrderedDict()
    best_result = (0, 0, 0.0, 0.0)
    for size in sizes:
        for assignment_value in folder_dict[size]:
            best_results = get_best_results(
                join(PATH, str(size), str(assignment_value), 'results.csv'),
                n_best=2)
            best_result = (
                size, assignment_value, best_results[0][0], best_results[0][2]
            ) if float(best_results[0][2]) > float(best_result[3]) else best_result
            best_results_string = '   '.join(map(' '.join, best_results))
            if size in data:
                data[size][assignment_value] = best_results_string
            else:
                data[size] = OrderedDict()
                data[size][assignment_value] = best_results_string

    df = pd.DataFrame.from_dict(data, orient='index').T
    df.to_csv(join(PATH, 'BoW-performance.csv'))
    with open(join(PATH, 'BoW-performance.csv'), mode='a') as csv:
        csv.write('\n' + '  '.join(map(str, best_result)))
