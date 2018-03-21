import csv
import argparse


def clean_csv(input_csv, output_csv):
    with open(input_csv) as unclean, open(output_csv, 'w', newline='') as clean:
        reader = csv.reader(unclean, delimiter='\t')
        writer = csv.writer(clean, delimiter='\t')
        for line in reader:
            if line[2] == 'Not Available' or line[1] == 'neutral':
                pass
            else:
                writer.writerow(line)


def main():
    parser = argparse.ArgumentParser(
        description='Remove deleted and neutral tweets from semeval tweet files.')
    parser.add_argument('input', help='Path to input file.')
    parser.add_argument('output', help='Path to output file.')
    args = parser.parse_args()
    clean_csv(args.input, args.output)


if __name__ == '__main__':
    main()
