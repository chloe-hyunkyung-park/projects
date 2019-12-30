#!/bin/bash

#$ -l h_rt=2:00:00  #time needed
#$ -pe smp 10 #number of cores
#$ -l rmem=16G #number of memery
#$ -o Q1_180228768.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M hpark19@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

#$ -P rse-com6012
#$ -q rse-com6012.q

module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda
source activate myspark
spark-submit --driver-memory 16g --executor-memory 16g --master local[10] Q1_180228768.py
