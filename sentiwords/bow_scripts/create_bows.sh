#!/bin/bash

# enable crazier wildcards
shopt -s extglob

openXBOWpath=/home/maurice/software/openXBOW/openXBOW.jar

base_path="/home/maurice/Downloads/semeval/GloVe_vectors/2013"

train_file=train.csv

eval_files=(devel.csv test.csv)

codebook_sizes=(10 20 50 100 200 500 1000 2000)

As=(1 10 25 50 100 200 500 1000)

cd $base_path
mkdir BoW_standardize
cd BoW_standardize

for size in "${codebook_sizes[@]}"; do
	mkdir ${size}
	for a in "${As[@]}"; do
		if (( $a > $((size/2)) )); then
			break
		fi
		mkdir ${size}/${a}
		java -Xmx15g -jar $openXBOWpath -i ${base_path}/$train_file -o ${size}/${a}/$train_file -B ${size}/${a}/codebook -size ${size} -a ${a} -standardizeInput -log -norm 1 -writeName 
		for eval_file in "${eval_files[@]}"; do
			java -Xmx15g -jar $openXBOWpath -i ${base_path}/$eval_file -o ${size}/${a}/$eval_file -b ${size}/${a}/codebook -norm 1 -writeName
		done;
	done;
done;
	
	
