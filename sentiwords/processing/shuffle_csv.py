import pandas as pd
import argparse
from csv import QUOTE_ALL

def shuffle_csv(csv_input_path, csv_output_path, delimiter=','):
    df = pd.read_csv(csv_input_path, sep=delimiter, header=None, encoding='utf-8')
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(csv_output_path, sep=',', encoding='utf-8', quoting=QUOTE_ALL, header=False, index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Shuffle a large csv file.')
    parser.add_argument('input', help='Input path to csv file.')
    parser.add_argument('output', help='Output path.')
    args = parser.parse_args()
    shuffle_csv(args.input, args.output)
