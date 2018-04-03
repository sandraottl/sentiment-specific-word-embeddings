"""Clean downloaded SemEval files by removing not available tweets and neutral tweets.
author: Sandra Ottl
"""
import csv
import argparse


def clean_csv(input_csv, output_csv):
    """Clean a single SemEval csv and write the cleaned version.
    arguments:
       input_csv: Filepath to input csv (tab-delimited).
       output_csv: Filepath to output csv (tab-delimited).
    """
    with open(input_csv) as unclean, open(output_csv, 'w', newline='') as clean:
        reader = csv.reader(unclean, delimiter='\t')
        writer = csv.writer(clean, delimiter='\t')
        for line in reader:
            if line[2] == 'Not Available' or line[1] == 'neutral':
                pass
            else:
                writer.writerow(line)


def main():
    """Create commandline parser, parse arguments and call clean_csv.
    arguments: None
    returns: Nothing
    """
    parser = argparse.ArgumentParser(
        description='Remove deleted and neutral tweets from semeval tweet files.')
    parser.add_argument('input', help='Path to input file.')
    parser.add_argument('output', help='Path to output file.')
    args = parser.parse_args()
    clean_csv(args.input, args.output)


if __name__ == '__main__':
    main()
