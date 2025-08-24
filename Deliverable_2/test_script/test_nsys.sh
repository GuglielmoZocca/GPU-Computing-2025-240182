#!/bin/bash

#Multiple nsys tests on the matrices in ../matrix/, where $1 correspond on the number of streams in the COO_NEW_1 solution

rm -f test/report_nsys/*
module load CUDA/12.3.2
make clean_batch_outputs
rm bin/SpMV
make "MACROS=-D COO_OLD -D SortR -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_nsys.sh Cities SortR COO_OLD
./sbatch_script/sbatch_test_nsys.sh BenElechi1 SortR COO_OLD
./sbatch_script/sbatch_test_nsys.sh degme SortR COO_OLD
./sbatch_script/sbatch_test_nsys.sh rail2586 SortR COO_OLD
./sbatch_script/sbatch_test_nsys.sh specular SortR COO_OLD
./sbatch_script/sbatch_test_nsys.sh mawi_201512012345 SortR COO_OLD

OUTPUTF="test/report_nsys/report-mawi_201512012345-SortR-COO_OLD.nsys-rep"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

sleep 10

rm bin/SpMV
make "MACROS=-D COO_OLD -D SortC -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_nsys.sh Cities SortC COO_OLD
./sbatch_script/sbatch_test_nsys.sh BenElechi1 SortC COO_OLD
./sbatch_script/sbatch_test_nsys.sh degme SortC COO_OLD
./sbatch_script/sbatch_test_nsys.sh rail2586 SortC COO_OLD
./sbatch_script/sbatch_test_nsys.sh specular SortC COO_OLD
./sbatch_script/sbatch_test_nsys.sh mawi_201512012345 SortC COO_OLD

OUTPUTF="test/report_nsys/report-mawi_201512012345-SortC-COO_OLD.nsys-rep"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

sleep 10

rm bin/SpMV
make "MACROS=-D COO_CUSPARSE -D SortR -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_nsys.sh Cities SortR COO_CUSPARSE
./sbatch_script/sbatch_test_nsys.sh BenElechi1 SortR COO_CUSPARSE
./sbatch_script/sbatch_test_nsys.sh degme SortR COO_CUSPARSE
./sbatch_script/sbatch_test_nsys.sh rail2586 SortR COO_CUSPARSE
./sbatch_script/sbatch_test_nsys.sh specular SortR COO_CUSPARSE
./sbatch_script/sbatch_test_nsys.sh mawi_201512012345 SortR COO_CUSPARSE

OUTPUTF="test/report_nsys/report-mawi_201512012345-SortR-COO_CUSPARSE.nsys-rep"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=32 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_nsys_new_solution.sh Cities SortR COO_NEW_1 32
./sbatch_script/sbatch_test_nsys_new_solution.sh BenElechi1 SortR COO_NEW_1 32
./sbatch_script/sbatch_test_nsys_new_solution.sh degme SortR COO_NEW_1 32
./sbatch_script/sbatch_test_nsys_new_solution.sh rail2586 SortR COO_NEW_1 32
./sbatch_script/sbatch_test_nsys_new_solution.sh specular SortR COO_NEW_1 32
./sbatch_script/sbatch_test_nsys_new_solution.sh mawi_201512012345 SortR COO_NEW_1 32

OUTPUTF="test/report_nsys/report-mawi_201512012345-SortR-COO_NEW_1-32.nsys-rep"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=64 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_nsys_new_solution.sh Cities SortR COO_NEW_1 64
./sbatch_script/sbatch_test_nsys_new_solution.sh BenElechi1 SortR COO_NEW_1 64
./sbatch_script/sbatch_test_nsys_new_solution.sh degme SortR COO_NEW_1 64
./sbatch_script/sbatch_test_nsys_new_solution.sh rail2586 SortR COO_NEW_1 64
./sbatch_script/sbatch_test_nsys_new_solution.sh specular SortR COO_NEW_1 64
./sbatch_script/sbatch_test_nsys_new_solution.sh mawi_201512012345 SortR COO_NEW_1 64

OUTPUTF="test/report_nsys/report-mawi_201512012345-SortR-COO_NEW_1-64.nsys-rep"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=128 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_nsys_new_solution.sh Cities SortR COO_NEW_1 128
./sbatch_script/sbatch_test_nsys_new_solution.sh BenElechi1 SortR COO_NEW_1 128
./sbatch_script/sbatch_test_nsys_new_solution.sh degme SortR COO_NEW_1 128
./sbatch_script/sbatch_test_nsys_new_solution.sh rail2586 SortR COO_NEW_1 128
./sbatch_script/sbatch_test_nsys_new_solution.sh specular SortR COO_NEW_1 128
./sbatch_script/sbatch_test_nsys_new_solution.sh mawi_201512012345 SortR COO_NEW_1 128

OUTPUTF="test/report_nsys/report-mawi_201512012345-SortR-COO_NEW_1-128.nsys-rep"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=256 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_nsys_new_solution.sh Cities SortR COO_NEW_1 256
./sbatch_script/sbatch_test_nsys_new_solution.sh BenElechi1 SortR COO_NEW_1 256
./sbatch_script/sbatch_test_nsys_new_solution.sh degme SortR COO_NEW_1 256
./sbatch_script/sbatch_test_nsys_new_solution.sh rail2586 SortR COO_NEW_1 256
./sbatch_script/sbatch_test_nsys_new_solution.sh specular SortR COO_NEW_1 256
./sbatch_script/sbatch_test_nsys_new_solution.sh mawi_201512012345 SortR COO_NEW_1 256

OUTPUTF="test/report_nsys/report-mawi_201512012345-SortR-COO_NEW_1-256.nsys-rep"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=512 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_nsys_new_solution.sh Cities SortR COO_NEW_1 512
./sbatch_script/sbatch_test_nsys_new_solution.sh BenElechi1 SortR COO_NEW_1 512
./sbatch_script/sbatch_test_nsys_new_solution.sh degme SortR COO_NEW_1 512
./sbatch_script/sbatch_test_nsys_new_solution.sh rail2586 SortR COO_NEW_1 512
./sbatch_script/sbatch_test_nsys_new_solution.sh specular SortR COO_NEW_1 512
./sbatch_script/sbatch_test_nsys_new_solution.sh mawi_201512012345 SortR COO_NEW_1 512

OUTPUTF="test/report_nsys/report-mawi_201512012345-SortR-COO_NEW_1-512.nsys-rep"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1 -D BLOCK_SIZE=1024 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_nsys_new_solution.sh Cities SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_nsys_new_solution.sh BenElechi1 SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_nsys_new_solution.sh degme SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_nsys_new_solution.sh rail2586 SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_nsys_new_solution.sh specular SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_nsys_new_solution.sh mawi_201512012345 SortR COO_NEW_1 1024

OUTPUTF="test/report_nsys/report-mawi_201512012345-SortR-COO_NEW_1-1024.nsys-rep"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

sleep 10

rm temp_job.sh