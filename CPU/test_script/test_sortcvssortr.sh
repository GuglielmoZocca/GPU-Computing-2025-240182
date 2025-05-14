#!/bin/bash

#Multiple tests SortC vs SortR on the matrices in ../matrix/

rm bin/SpMV

make clean_batch_outputs
rm errors/*
make "MACROS=-D dtype=float -D SortR"

./sbatch_test.sh Cities.mtx SortR
./sbatch_test.sh BenElechi1.mtx SortR
./sbatch_test.sh degme.mtx SortR
./sbatch_test.sh Hardesty2.mtx SortR
./sbatch_test.sh rail4284.mtx SortR
./sbatch_test.sh specular.mtx SortR
./sbatch_test.sh t2em.mtx SortR
./sbatch_test.sh torso1.mtx SortR
./sbatch_test.sh mawi_201512012345.mtx SortR

OUTPUTF="outputs/test-mawi_201512012345.mtx-SortC.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D dtype=float -D SortC"

./sbatch_test.sh Cities.mtx SortR
./sbatch_test.sh BenElechi1.mtx SortR
./sbatch_test.sh degme.mtx SortR
./sbatch_test.sh Hardesty2.mtx SortR
./sbatch_test.sh rail4284.mtx SortR
./sbatch_test.sh specular.mtx SortR
./sbatch_test.sh t2em.mtx SortR
./sbatch_test.sh torso1.mtx SortR
./sbatch_test.sh mawi_201512012345.mtx SortR

OUTPUTF="outputs/test-mawi_201512012345.mtx-SortR.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

python Parse_script_out_CPU.py Complete

rm temp_job.sh