#!/bin/bash

#Multiple tests SortC vs SortR on the matrices in ../matrix/

module load CUDA/12.3.2
rm bin/SpMV
make clean_batch_outputs
rm errors/*
make "MACROS=-D dtype=float -D SortR -D COO1 -D Managed"

for (( i=32; i <= 1024; i=i*2 ))
do
./sbatch_test.sh Cities.mtx $i SortR COO1
./sbatch_test.sh BenElechi1.mtx $i SortR COO1
./sbatch_test.sh degme.mtx $i SortR COO1
./sbatch_test.sh Hardesty2.mtx $i SortR COO1
./sbatch_test.sh rail2586.mtx $i SortR COO1
./sbatch_test.sh specular.mtx $i SortR COO1
./sbatch_test.sh torso1.mtx $i SortR COO1
./sbatch_test.sh mawi_201512012345.mtx $i SortR COO1
done

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortR-COO1.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortC -D COO1 -D Managed"

for (( i=32; i <= 1024; i=i*2 ))
do
./sbatch_test.sh Cities.mtx $i SortC COO1
./sbatch_test.sh BenElechi1.mtx $i SortC COO1
./sbatch_test.sh degme.mtx $i SortC COO1
./sbatch_test.sh Hardesty2.mtx $i SortC COO1
./sbatch_test.sh rail2586.mtx $i SortC COO1
./sbatch_test.sh specular.mtx $i SortC COO1
./sbatch_test.sh torso1.mtx $i SortC COO1
./sbatch_test.sh mawi_201512012345.mtx $i SortC COO1
done

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortC-COO1.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortR -D COO2 -D Pinned"

for (( i=32; i <= 1024; i=i*2 ))
do
./sbatch_test.sh Cities.mtx $i SortR COO2
./sbatch_test.sh BenElechi1.mtx $i SortR COO2
./sbatch_test.sh degme.mtx $i SortR COO2
./sbatch_test.sh Hardesty2.mtx $i SortR COO2
./sbatch_test.sh rail2586.mtx $i SortR COO2
./sbatch_test.sh specular.mtx $i SortR COO2
./sbatch_test.sh torso1.mtx $i SortR COO2
./sbatch_test.sh mawi_201512012345.mtx $i SortR COO2
done

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortR-COO2.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortR -D COO4 -D Pinned"

for (( i=32; i <= 1024; i=i*2 ))
do
./sbatch_test.sh Cities.mtx $i SortR COO4
./sbatch_test.sh BenElechi1.mtx $i SortR COO4
./sbatch_test.sh degme.mtx $i SortR COO4
./sbatch_test.sh Hardesty2.mtx $i SortR COO4
./sbatch_test.sh rail2586.mtx $i SortR COO4
./sbatch_test.sh specular.mtx $i SortR COO4
./sbatch_test.sh torso1.mtx $i SortR COO4
./sbatch_test.sh mawi_201512012345.mtx $i SortR COO4
done

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortR-COO4.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortC -D COO4 -D Pinned"

for (( i=32; i <= 1024; i=i*2 ))
do
./sbatch_test.sh Cities.mtx $i SortC COO4
./sbatch_test.sh BenElechi1.mtx $i SortC COO4
./sbatch_test.sh degme.mtx $i SortC COO4
./sbatch_test.sh Hardesty2.mtx $i SortC COO4
./sbatch_test.sh rail2586.mtx $i SortC COO4
./sbatch_test.sh specular.mtx $i SortC COO4
./sbatch_test.sh torso1.mtx $i SortC COO4
./sbatch_test.sh mawi_201512012345.mtx $i SortC COO4
done

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortC-COO4.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortR -D COO5 -D Pinned"

for (( i=32; i <= 1024; i=i*2 ))
do
./sbatch_test.sh Cities.mtx $i SortR COO5
./sbatch_test.sh BenElechi1.mtx $i SortR COO5
./sbatch_test.sh degme.mtx $i SortR COO5
./sbatch_test.sh Hardesty2.mtx $i SortR COO5
./sbatch_test.sh rail2586.mtx $i SortR COO5
./sbatch_test.sh specular.mtx $i SortR COO5
./sbatch_test.sh torso1.mtx $i SortR COO5
./sbatch_test.sh mawi_201512012345.mtx $i SortR COO5
done

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortR-COO5.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done


rm bin/SpMV
make "MACROS=-D dtype=float -D SortC -D COO5 -D Pinned"

for (( i=32; i <= 1024; i=i*2 ))
do
./sbatch_test.sh Cities.mtx $i SortC COO5
./sbatch_test.sh BenElechi1.mtx $i SortC COO5
./sbatch_test.sh degme.mtx $i SortC COO5
./sbatch_test.sh Hardesty2.mtx $i SortC COO5
./sbatch_test.sh rail2586.mtx $i SortC COO5
./sbatch_test.sh specular.mtx $i SortC COO5
./sbatch_test.sh torso1.mtx $i SortC COO5
./sbatch_test.sh mawi_201512012345.mtx $i SortC COO5
done

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortC-COO5.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

python Parse_script_out_GPU.py Complete

rm temp_job.sh