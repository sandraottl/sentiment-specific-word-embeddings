#!/bin/bash
source /home/gerczuk/deep_spectrum_extractor/venv/bin/activate

folders=(/home/gerczuk/Deadline/features/*/*/*/*/BoW/)

for folder in "${folders[@]}"; do 
	python3 aggregate_bow_perf.py $folder 
done;
