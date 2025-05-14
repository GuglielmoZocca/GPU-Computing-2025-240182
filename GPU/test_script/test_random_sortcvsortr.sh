#!/bin/bash

module load CUDA/12.3.2
make clean_batch_outputs
rm bin/SpMV
make "MACROS=-D dtype=float -D SortR -D COO1 -D Managed -D RAND"
for (( i=1; i <= $4; ++i ))
do
for (( j=32; j <= 1024; j = j*2 ))
do
    sbatch sbatch_script_rand.sh $1 $2 $3 $j $i
done
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortC -D COO1 -D Managed -D RAND"

for (( i=1; i <= $4; ++i ))
do
for (( j=32; j <= 1024; j = j*2 ))
do
    sbatch sbatch_script_rand.sh $1 $2 $3 $j $i
done
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortR -D COO2 -D Pinned -D RAND"

for (( i=1; i <= $4; ++i ))
do
for (( j=32; j <= 1024; j = j*2 ))
do
    sbatch sbatch_script_rand.sh $1 $2 $3 $j $i
done
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortR -D COO3 -D Pinned -D RAND"
for (( i=1; i <= $4; ++i ))
do
for (( j=32; j <= 1024; j = j*2 ))
do
    sbatch sbatch_script_rand.sh $1 $2 $3 $j $i
done
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortR -D COO4 -D Pinned -D RAND"
for (( i=1; i <= $4; ++i ))
do
for (( j=32; j <= 1024; j = j*2 ))
do
    sbatch sbatch_script_rand.sh $1 $2 $3 $j $i
done
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortC -D COO4 -D Pinned -D RAND"
for (( i=1; i <= $4; ++i ))
do
for (( j=32; j <= 1024; j = j*2 ))
do
    sbatch sbatch_script_rand.sh $1 $2 $3 $j $i
done
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortR -D COO5 -D Pinned -D RAND"
for (( i=1; i <= $4; ++i ))
do
for (( j=32; j <= 1024; j = j*2 ))
do
    sbatch sbatch_script_rand.sh $1 $2 $3 $j $i
done
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortC -D COO5 -D Pinned -D RAND"
for (( i=1; i <= $4; ++i ))
do
for (( j=32; j <= 1024; j = j*2 ))
do
    sbatch sbatch_script_rand.sh $1 $2 $3 $j $i
done
done

python Parse_script_out_GPU.py Complete