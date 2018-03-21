#!/bin/bash

# enable crazier wildcards
shopt -s extglob

bow_path="/home/maurice/Downloads/semeval/GloVe_vectors/2013/BoW_standardize"

bow_files=("train.csv" "devel.csv" "test.csv")

cd ${bow_path}
for size in */; do
	for a in ${size}*/; do
		command="linear_svm "
		for file in "${bow_files[@]}"; do
			command="$command${bow_path}/${a}$file "
		done;
		command="$command-o ${bow_path}/${a}results.csv -cm ${bow_path}/${a}confusion_matrix.pdf -pred ${bow_path}/${a}predictions.csv"
		echo $command
		$command
	done;
done;
