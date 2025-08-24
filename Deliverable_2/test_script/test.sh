#!/bin/bash

#Multiple tests on the matrices in ../matrix/, where $1 correspond on the number of streams in the COO_NEW_1 solution

module load CUDA/12.3.2
rm bin/SpMV
make clean_batch_outputs
rm errors/*
make "MACROS= -D SortR -D COO_OLD -D Eval"

for (( i=32; i <= 1024; i=i*2 ))
do
./sbatch_script/sbatch_test.sh Cities.mtx $i SortR COO_OLD
./sbatch_script/sbatch_test.sh BenElechi1.mtx $i SortR COO_OLD
./sbatch_script/sbatch_test.sh degme.mtx $i SortR COO_OLD
./sbatch_script/sbatch_test.sh rail2586.mtx $i SortR COO_OLD
./sbatch_script/sbatch_test.sh specular.mtx $i SortR COO_OLD
./sbatch_script/sbatch_test.sh mawi_201512012345.mtx $i SortR COO_OLD
done

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortR-COO_OLD.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done


rm bin/SpMV
make "MACROS= -D SortC -D COO_OLD -D Eval"

for (( i=32; i <= 1024; i=i*2 ))
do
./sbatch_script/sbatch_test.sh Cities.mtx $i SortC COO_OLD
./sbatch_script/sbatch_test.sh BenElechi1.mtx $i SortC COO_OLD
./sbatch_script/sbatch_test.sh degme.mtx $i SortC COO_OLD
./sbatch_script/sbatch_test.sh rail2586.mtx $i SortC COO_OLD
./sbatch_script/sbatch_test.sh specular.mtx $i SortC COO_OLD
./sbatch_script/sbatch_test.sh mawi_201512012345.mtx $i SortC COO_OLD
done

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortC-COO_OLD.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done


rm bin/SpMV
make "MACROS= -D SortR -D COO_CUSPARSE -D Eval"

./sbatch_script/sbatch_test.sh Cities.mtx NULL SortR COO_CUSPARSE
./sbatch_script/sbatch_test.sh BenElechi1.mtx NULL SortR COO_CUSPARSE
./sbatch_script/sbatch_test.sh degme.mtx SortR NULL COO_CUSPARSE
./sbatch_script/sbatch_test.sh rail2586.mtx SortR NULL COO_CUSPARSE
./sbatch_script/sbatch_test.sh specular.mtx SortR NULL COO_CUSPARSE
./sbatch_script/sbatch_test.sh mawi_201512012345.mtx NULL SortR COO_CUSPARSE

OUTPUTF="outputs/test-mawi_201512012345.mtx-NULL-SortR-COO_CUSPARSE.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1 -D BLOCK_SIZE=32 -D N_STREAM=$1 -D Eval"

./sbatch_script/sbatch_test.sh Cities.mtx 32 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh BenElechi1.mtx 32 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh degme.mtx SortR 32 COO_NEW_1
./sbatch_script/sbatch_test.sh rail2586.mtx SortR 32 COO_NEW_1
./sbatch_script/sbatch_test.sh specular.mtx SortR 32 COO_NEW_1
./sbatch_script/sbatch_test.sh mawi_201512012345.mtx 32 SortR COO_NEW_1

OUTPUTF="outputs/test-mawi_201512012345.mtx-32-SortR-COO_NEW_1.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=64 -D N_STREAM=$1 -D Eval"

./sbatch_script/sbatch_test.sh Cities.mtx 64 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh BenElechi1.mtx 64 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh degme.mtx SortR 64 COO_NEW_1
./sbatch_script/sbatch_test.sh rail2586.mtx SortR 64 COO_NEW_1
./sbatch_script/sbatch_test.sh specular.mtx SortR 64 COO_NEW_1
./sbatch_script/sbatch_test.sh mawi_201512012345.mtx 64 SortR COO_NEW_1

OUTPUTF="outputs/test-mawi_201512012345.mtx-64-SortR-COO_NEW_1.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=128 -D N_STREAM=$1 -D Eval"

./sbatch_script/sbatch_test.sh Cities.mtx 128 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh BenElechi1.mtx 128 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh degme.mtx SortR 128 COO_NEW_1
./sbatch_script/sbatch_test.sh rail2586.mtx SortR 128 COO_NEW_1
./sbatch_script/sbatch_test.sh specular.mtx SortR 128 COO_NEW_1
./sbatch_script/sbatch_test.sh mawi_201512012345.mtx 128 SortR COO_NEW_1

OUTPUTF="outputs/test-mawi_201512012345.mtx-128-SortR-COO_NEW_1.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=256 -D N_STREAM=$1 -D Eval"

./sbatch_script/sbatch_test.sh Cities.mtx 256 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh BenElechi1.mtx 256 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh degme.mtx SortR 256 COO_NEW_1
./sbatch_script/sbatch_test.sh rail2586.mtx SortR 256 COO_NEW_1
./sbatch_script/sbatch_test.sh specular.mtx SortR 256 COO_NEW_1
./sbatch_script/sbatch_test.sh mawi_201512012345.mtx 256 SortR COO_NEW_1

OUTPUTF="outputs/test-mawi_201512012345.mtx-256-SortR-COO_NEW_1.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=512 -D N_STREAM=$1 -D Eval"

./sbatch_script/sbatch_test.sh Cities.mtx 512 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh BenElechi1.mtx 512 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh degme.mtx SortR 512 COO_NEW_1
./sbatch_script/sbatch_test.sh rail2586.mtx SortR 512 COO_NEW_1
./sbatch_script/sbatch_test.sh specular.mtx SortR 512 COO_NEW_1
./sbatch_script/sbatch_test.sh mawi_201512012345.mtx 512 SortR COO_NEW_1

OUTPUTF="outputs/test-mawi_201512012345.mtx-512-SortR-COO_NEW_1.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1 -D BLOCK_SIZE=1024 -D N_STREAM=$1 -D Eval"

./sbatch_script/sbatch_test.sh Cities.mtx 1024 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh BenElechi1.mtx 1024 SortR COO_NEW_1
./sbatch_script/sbatch_test.sh degme.mtx SortR 1024 COO_NEW_1
./sbatch_script/sbatch_test.sh rail2586.mtx SortR 1024 COO_NEW_1
./sbatch_script/sbatch_test.sh specular.mtx SortR 1024 COO_NEW_1
./sbatch_script/sbatch_test.sh mawi_201512012345.mtx 1024 SortR COO_NEW_1

OUTPUTF="outputs/test-mawi_201512012345.mtx-1024-SortR-COO_NEW_1.out"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

python python_script/Parse_script_out.py Complete_$1

rm temp_job.sh