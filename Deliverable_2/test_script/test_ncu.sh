#!/bin/bash

#Multiple ncu tests on the matrices in ../matrix/, where $1 correspond on the number of streams in the COO_NEW_1 solution

rm -f test/report_ncu/*
module load CUDA/12.3.2
make clean_batch_outputs
rm bin/SpMV
make "MACROS=-D COO_OLD -D SortR -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_ncu.sh Cities SortR COO_OLD
./sbatch_script/sbatch_test_ncu.sh BenElechi1 SortR COO_OLD
./sbatch_script/sbatch_test_ncu.sh degme SortR COO_OLD
./sbatch_script/sbatch_test_ncu.sh rail2586 SortR COO_OLD
./sbatch_script/sbatch_test_ncu.sh specular SortR COO_OLD
./sbatch_script/sbatch_test_ncu.sh mawi_201512012345 SortR COO_OLD

OUTPUTF="test/report_ncu/report-mawi_201512012345-SortR-COO_OLD.csv"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D COO_OLD -D SortC -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_ncu.sh Cities SortC COO_OLD
./sbatch_script/sbatch_test_ncu.sh BenElechi1 SortC COO_OLD
./sbatch_script/sbatch_test_ncu.sh degme SortC COO_OLD
./sbatch_script/sbatch_test_ncu.sh rail2586 SortC COO_OLD
./sbatch_script/sbatch_test_ncu.sh specular SortC COO_OLD
./sbatch_script/sbatch_test_ncu.sh mawi_201512012345 SortC COO_OLD

OUTPUTF="test/report_ncu/report-mawi_201512012345-SortC-COO_OLD.csv"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D COO_CUSPARSE -D SortR -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_ncu.sh Cities SortR COO_CUSPARSE
./sbatch_script/sbatch_test_ncu.sh BenElechi1 SortR COO_CUSPARSE
./sbatch_script/sbatch_test_ncu.sh degme SortR COO_CUSPARSE
./sbatch_script/sbatch_test_ncu.sh rail2586 SortR COO_CUSPARSE
./sbatch_script/sbatch_test_ncu.sh specular SortR COO_CUSPARSE
./sbatch_script/sbatch_test_ncu.sh mawi_201512012345 SortR COO_CUSPARSE

OUTPUTF="test/report_ncu/report-mawi_201512012345-SortR-COO_CUSPARSE.csv"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=32 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_ncu_new_solution.sh Cities SortR COO_NEW_1 32
./sbatch_script/sbatch_test_ncu_new_solution.sh BenElechi1 SortR COO_NEW_1 32
./sbatch_script/sbatch_test_ncu_new_solution.sh degme SortR COO_NEW_1 32
./sbatch_script/sbatch_test_ncu_new_solution.sh rail2586 SortR COO_NEW_1 32
./sbatch_script/sbatch_test_ncu_new_solution.sh specular SortR COO_NEW_1 32
./sbatch_script/sbatch_test_ncu_new_solution.sh mawi_201512012345 SortR COO_NEW_1 32

OUTPUTF="test/report_ncu/report-mawi_201512012345-SortR-COO_NEW_1-32.csv"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=64 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_ncu_new_solution.sh Cities SortR COO_NEW_1 64
./sbatch_script/sbatch_test_ncu_new_solution.sh BenElechi1 SortR COO_NEW_1 64
./sbatch_script/sbatch_test_ncu_new_solution.sh degme SortR COO_NEW_1 64
./sbatch_script/sbatch_test_ncu_new_solution.sh rail2586 SortR COO_NEW_1 64
./sbatch_script/sbatch_test_ncu_new_solution.sh specular SortR COO_NEW_1 64
./sbatch_script/sbatch_test_ncu_new_solution.sh mawi_201512012345 SortR COO_NEW_1 64

OUTPUTF="test/report_ncu/report-mawi_201512012345-SortR-COO_NEW_1-64.csv"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=128 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_ncu_new_solution.sh Cities SortR COO_NEW_1 128
./sbatch_script/sbatch_test_ncu_new_solution.sh BenElechi1 SortR COO_NEW_1 128
./sbatch_script/sbatch_test_ncu_new_solution.sh degme SortR COO_NEW_1 128
./sbatch_script/sbatch_test_ncu_new_solution.sh rail2586 SortR COO_NEW_1 128
./sbatch_script/sbatch_test_ncu_new_solution.sh specular SortR COO_NEW_1 128
./sbatch_script/sbatch_test_ncu_new_solution.sh mawi_201512012345 SortR COO_NEW_1 128

OUTPUTF="test/report_ncu/report-mawi_201512012345-SortR-COO_NEW_1-128.csv"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=256 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_ncu_new_solution.sh Cities SortR COO_NEW_1 256
./sbatch_script/sbatch_test_ncu_new_solution.sh BenElechi1 SortR COO_NEW_1 256
./sbatch_script/sbatch_test_ncu_new_solution.sh degme SortR COO_NEW_1 256
./sbatch_script/sbatch_test_ncu_new_solution.sh rail2586 SortR COO_NEW_1 256
./sbatch_script/sbatch_test_ncu_new_solution.sh specular SortR COO_NEW_1 256
./sbatch_script/sbatch_test_ncu_new_solution.sh mawi_201512012345 SortR COO_NEW_1 256

OUTPUTF="test/report_ncu/report-mawi_201512012345-SortR-COO_NEW_1-256.csv"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=512 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_ncu_new_solution.sh Cities SortR COO_NEW_1 512
./sbatch_script/sbatch_test_ncu_new_solution.sh BenElechi1 SortR COO_NEW_1 512
./sbatch_script/sbatch_test_ncu_new_solution.sh degme SortR COO_NEW_1 512
./sbatch_script/sbatch_test_ncu_new_solution.sh rail2586 SortR COO_NEW_1 512
./sbatch_script/sbatch_test_ncu_new_solution.sh specular SortR COO_NEW_1 512
./sbatch_script/sbatch_test_ncu_new_solution.sh mawi_201512012345 SortR COO_NEW_1 512

OUTPUTF="test/report_ncu/report-mawi_201512012345-SortR-COO_NEW_1-512.csv"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

rm bin/SpMV
make "MACROS=-D SortR -D COO_NEW_1  -D BLOCK_SIZE=1024 -D N_STREAM=$1 -D EVAL_NCU -lineinfo"

./sbatch_script/sbatch_test_ncu_new_solution.sh Cities SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_ncu_new_solution.sh BenElechi1 SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_ncu_new_solution.sh degme SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_ncu_new_solution.sh rail2586 SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_ncu_new_solution.sh specular SortR COO_NEW_1 1024
./sbatch_script/sbatch_test_ncu_new_solution.sh mawi_201512012345 SortR COO_NEW_1 1024

OUTPUTF="test/report_ncu/report-mawi_201512012345-SortR-COO_NEW_1-1024.csv"

# Wait until the file exists
while [ ! -s $OUTPUTF ]; do
  sleep 1  # Wait 1 second before checking again
done

sleep 10

python python_script/Parse_ncu_report.py $1

python python_script/Parse_statistics_report.py $1

rm temp_job.sh