# SPMV PROJECT
## Deliverable_1
### Directory structure

In this section, the project's directory structure and the contents of each directory and file are explained:
* graph_report.ipynb: It is a Jupyter notebook for the compilation of some data graph used by the report based on the test
* matrix: it will contain the matrices used in the test
* CPU: Directory for the CPU solution
  - makefile
  - sbatch_script.sh: sbatch script for a run of the CPU Solution with a matrix
  - sbatch_script_rand.sh: sbatch script for a run of the CPU Solution with a random matrix
  - sbatch_test.sh: sbatch script for the test
  - Parse_script_out_CPU.py: python script for parse test results
  - SpMV.c: c script for the CPU solution
  - include: contains the header for the time, print, mean and standard deviation library
  - src: contains the source code for the time, print, mean and standard deviation library
  - test_script: contain the test script for the test
  - test: contain the parsed results of the test
* GPU: Directory for the GPU solutions
  - makefile
  - sbatch_script.sh: sbatch script for a run of a GPU Solution with a matrix
  - sbatch_script_rand.sh: sbatch script for a run of a GPU Solution with a random matrix
  - sbatch_test.sh: sbatch script for the test
  - Parse_script_out_GPU.py: python script for parse test results
  - SpMV.cu: cu script for the GPU solutions
  - include: contains the header for the time, print, mean and standard deviation library
  - src: contains the source code for the time, print, mean and standard deviation library
  - test_script: contain the test script for the test
    + test_sortcvssortr.sh: test all solutions
    + test_sortcvssortr_no_COO3.sh: test all solutions except COO3
  - test: contain the parsed results of the test


### Test procedure

To execute the test used for the paper follow this instructions:
1. Clone the repository
2. Download the following matrices from [here](https://sparse.tamu.edu) in `matrix` directory: BenElechi1.mtx, degme.mtx, Cities.mtx, Hardesty2.mtx, mawi_201512012345.mtx, rail2586.mtx, specular.mtx, torso1.mtx
3. In case you want to test GPU solutions:
   1. Go to the GPU directory: `cd Path/to/GPU`
   2. Give the permission to the test scripts:
      1. `chmod +x ./test_script/test_sortcvssortr.sh`
      2. `chmod +x ./test_script/test_sortcvssortr_no_COO3.sh`
   4. Execute the test script:
      1. test all solutions (take some time because of COO3): `./test_script/test_sortcvssortr.sh`
      2. test all solutions except COO3: `./test_script/test_sortcvssortr_no_COO3.sh`
   5. When the test ends, the result can be found in `test/GPU_test_Complete.csv`
   6. In case you can run Jupyter notebook, you can visualize some graphs related to the test through the notebook `../graph_report.ipynb`
1. In case you want to test CPU solution:
   1. Go to the CPU directory: `cd Path/to/CPU`
   2. Give the permission to the test script: `chmod +x ./test_script/test_sortcvssortr.sh`
   3. Execute the script: `./test_script/test_sortcvssortr.sh`
   4. When the test ends, the result can be found in `test/CPU_test_Complete.csv`
   5. In case you can run Jupyter notebook, you can visualize some graphs related to the test through the notebook `../graph_report.ipynb`

### Correctness check

To check the correctness of the GPU solutions follow these instructions:
   1. Clone the repository
   2. Download the following matrices from [here](https://sparse.tamu.edu) in `matrix` directory: BenElechi1.mtx, degme.mtx, Cities.mtx, Hardesty2.mtx, mawi_201512012345.mtx, rail2586.mtx, specular.mtx, torso1.mtx
   3. Go to the GPU directory: `cd Path/to/GPU`
   4. Load the required module: `module load CUDA/12.3.2`
   5. Remove the possible executable: `rm bin/SpMV`
   6. Execute `make` with the desired macros:
      * COO1 solution (sorted by row or by column): `make "MACROS=-D Managed -D COO1  (-D SortR or -D SortC) -D dtype=int -D Check"`
      * COO2 solution: `make "MACROS=-D Pinned -D COO2 -D SortR -D dtype=int -D Check"`
      * COO3 solution: `make "MACROS=-D Pinned -D COO3 -D SortR -D dtype=int -D Check"`
      * COO4 solution (sorted by row or by column): `make "MACROS=-D Pinned -D COO4  (-D SortR or -D SortC) -D dtype=int -D Check"`
      * COO5 solution (sorted by row or by column): `make "MACROS=-D Pinned -D COO5  (-D SortR or -D SortC) -D dtype=int -D Check"`
   7. Execute the sbatch script with an integer matrix *.mtx (eg. Cities.mtx) and the block size B (eg. 32): `sbatch sbatch_script.sh *.mtx B`
   8. Check in the directory `outputs` the output file with the id correspondent to the job executed
   9. In that file, besides the other test information there will be a field `SpMV verification:`
      * In the case appears `SUCCESS`, the solution is correct
      * In the case appears `FAILURE`, the solution is uncorrect

### Random run

To execute a solution with a random matrix follow these instruction:
1. Clone the repository
2. Download the following matrices from [here](https://sparse.tamu.edu) in `matrix` directory: BenElechi1.mtx, degme.mtx, Cities.mtx, Hardesty2.mtx, mawi_201512012345.mtx, rail2586.mtx, specular.mtx, torso1.mtx
3. In case you want to test GPU solutions:
   1. Go to the CPU directory: `cd Path/to/GPU`
   2. Load the required module: `module load CUDA/12.3.2`
   3. Remove the possible executable: `rm bin/SpMV`
   4. Execute `make` with the desired macros :
      * COO1 solution (sorted by row or by column) (integer matrix or floating point matrix): `make "MACROS=-D Managed -D COO1  (-D SortR or -D SortC) -D (dtype=int or dtype=float) -D RAND"`
      * COO2 solution (integer matrix or floating point matrix): `make "MACROS=-D Pinned -D COO2 -D SortR -D (dtype=int or dtype=float) -D RAND"`
      * COO3 solution (integer matrix or floating point matrix): `make "MACROS=-D Pinned -D COO3 -D SortR -D (dtype=int or dtype=float) -D RAND"`
      * COO4 solution (sorted by row or by column) (integer matrix or floating point matrix): `make "MACROS=-D Pinned -D COO4  (-D SortR or -D SortC) -D (dtype=int or dtype=float) -D RAND"`
      * COO5 solution (sorted by row or by column) (integer matrix or floating point matrix): `make "MACROS=-D Pinned -D COO5  (-D SortR or -D SortC) -D (dtype=int or dtype=float) -D RAND"`
   5. If you want check the correctness just insert `-D Check` and substitute value type with  `dtype=int`
   6. Execute the sbatch script `sbatch sbatch_script.sh` with the following arguments:
      * Number of matrix row
      * Number of matrix column
      * Number of non-zeros elements
      * block size (eg. 32)
      * Random seed (eg. 1)
   7. Check in the directory `outputs` the output file with the id correspondent to the job executed
   8. In that file, besides the other test information, if you have specified `-D Check` there will be a field `SpMV verification:`
      * In the case appears `SUCCESS`, the solution is correct
      * In the case appears `FAILURE`, the solution is uncorrect
4. In case you want to test CPU solution:
   1. Go to the CPU directory: `cd Path/to/CPU`
   2. Remove the possible executable: `rm bin/SpMV`
   3. Execute `make` with the desired macros (solution sorted by row or by column) (integer matrix or floating point matrix): `(-D SortR or -D SortC) -D (dtype=int or dtype=float) -D RAND"`
   4. Execute the sbatch script `sbatch sbatch_script.sh` with the following arguments:
      * Number of matrix row
      * Number of matrix column
      * Number of non-zeros elements
      * Random seed (eg. 1)
   7. Check in the directory `outputs` the output file with the id correspondent to the job executed
      
## Deliverable_2
### Directory structure

In this section, the project's directory structure and the contents of each directory and file are explained:
* graph_script: directory for the graphs creation script
  - graph_report.ipynb: it is a Jupyter notebook for the compilation of some data graph used by the report based on the test
* matrix: it will contain the matrices used in the test
* makefile
* sbatch_script: directory for the sbatch scripts 
  - sbatch_script.sh: sbatch script for a run of a GPU Solution with a matrix
  - sbatch_script_rand.sh: sbatch script for a run of a GPU Solution with a random matrix
  - sbatch_test.sh: sbatch script for the test
  - sbatch_test_ncu.sh: sbatch script for the ncu test
  - sbatch_test_ncu_new_solution.sh: sbatch script for the ncu test of the new solution
  - sbatch_test_nsys.sh: sbatch script for the nsys test
  - sbatch_test_nsys_new_solution.sh: sbatch script for the nsys test of the new solution
* python_script: directory for the python scripts utilized to parse the test data
  - Parse_script_out.py: python script for parse test results
  - Parse_ncu_report.py: python script for parse ncu test results
  - Parse_statistics_report.py: python script for clean the output of Parse_ncu_report.py
* test_script: directory for the test scripts
  - test.sh: test of all solutions
  - test_ncu.sh: test ncu of all solutions
  - test_nsys.sh: test nsys of all solutions
* test: directory for the parsed results of the normal, ncu and sys tests, and the ncu and nsys reports
* SpMV.cu: cu script for the GPU solutions
* include: directory for the headers for the time, print, mean, standard deviation library, sparse matrix analysis and utility functions
* src: directory for the source code for some header files in include directory

### Test procedure

To execute the tests used for the paper follow this instructions:
1. Clone the repository
2. Download the following matrices from [here](https://sparse.tamu.edu) in `Path/to/Deliverable_2/matrix` directory: BenElechi1.mtx, degme.mtx, Cities.mtx, mawi_201512012345.mtx, rail2586.mtx, specular.mtx
3. Go to the Deliverable_2 directory: `cd Path/to/Deliverable_2`
4. Give the permission to the test scripts:
   1. `chmod +x ./test_script/test.sh`
   2. `chmod +x ./test_script/test_ncu.sh`
   3. `chmod +x ./test_script/test_ncu.sh`
5. Execute the test scripts, where n is the number of streams for the COO_NEW_1 solution:
   1. normal test of all solutions: `./test_script/test.sh n`
   2. ncu test of all solutions: `./test_script/test_ncu.sh n`
   3. nsys test of all solutions: `./test_script/test_nsys.sh n`
6. When the tests end, the results can be found as:
   1. `test/GPU_test_Complete_n.csv` for the normal tests
   2. `test/combined_ncu_report_new_n.csv` for the ncu tests
   3. ncu and sys reports respectively in `test/report_ncu` and `test/report_nsys`
7. In case you can run Jupyter notebook, you can visualize some graphs related to the tests through the notebook `graph_script/graph_report.ipynb`

### Correctness check

To check the correctness of the GPU solutions follow these instructions:
1. Clone the repository
2. Download the following matrices from [here](https://sparse.tamu.edu) in `matrix` directory: BenElechi1.mtx, degme.mtx, Cities.mtx, mawi_201512012345.mtx, rail2586.mtx, specular.mtx
3. Go to the Deliverable_2 directory: `cd Path/to/Deliverable_2`
4. Load the required module: `module load CUDA/12.3.2`
5. Remove the possible executable: `rm bin/SpMV`
6. Execute `make` with the desired macros:
  * COO_OLD solution (sorted by row or by column): `make "MACROS=-D COO_OLD  (-D SortR or -D SortC) -D Check"`
  * Cusparse solution: `make "MACROS=-D COO_CUSPARSE -D SortR -D Check"`
  * COO_NEW_1 solution (where n is the block size, m is the number of the stream, k is the predicted max number of blocks): `make "MACROS=-D COO_NEW_1 -D SortR -D BLOCK_SIZE=n -D N_STREAM=m -D MAX_BLOCKS=k -D Check"`
7. Execute the sbatch script with an integer matrix *.mtx (eg. Cities.mtx) and, only in COO_OLD case, the block size B (eg. 32): `sbatch sbatch_script.sh *.mtx B`
8. Check in the directory `outputs` the output file with the id correspondent to the job executed
9. In that file, besides the other test information there will be a field `SpMV verification:`
  * In the case appears `SUCCESS`, the solution is correct
  * In the case appears `FAILURE`, the solution is uncorrect

### Random run

To execute a solution with a random matrix follow these instruction:
1. Clone the repository
2. Download the following matrices from [here](https://sparse.tamu.edu) in `matrix` directory: BenElechi1.mtx, degme.mtx, Cities.mtx, Hardesty2.mtx, mawi_201512012345.mtx, rail2586.mtx, specular.mtx, torso1.mtx
3. In case you want to test GPU solutions:
   1. Go to the CPU directory: `cd Path/to/GPU`
   2. Load the required module: `module load CUDA/12.3.2`
   3. Remove the possible executable: `rm bin/SpMV`
   4. Execute `make` with the desired macros :
      * COO1 solution (sorted by row or by column) (integer matrix or floating point matrix): `make "MACROS=-D Managed -D COO1  (-D SortR or -D SortC) -D (dtype=int or dtype=float) -D RAND"`
      * COO2 solution (integer matrix or floating point matrix): `make "MACROS=-D Pinned -D COO2 -D SortR -D (dtype=int or dtype=float) -D RAND"`
      * COO3 solution (integer matrix or floating point matrix): `make "MACROS=-D Pinned -D COO3 -D SortR -D (dtype=int or dtype=float) -D RAND"`
      * COO4 solution (sorted by row or by column) (integer matrix or floating point matrix): `make "MACROS=-D Pinned -D COO4  (-D SortR or -D SortC) -D (dtype=int or dtype=float) -D RAND"`
      * COO5 solution (sorted by row or by column) (integer matrix or floating point matrix): `make "MACROS=-D Pinned -D COO5  (-D SortR or -D SortC) -D (dtype=int or dtype=float) -D RAND"`
   5. If you want check the correctness just insert `-D Check` and substitute value type with  `dtype=int`
   6. Execute the sbatch script `sbatch sbatch_script.sh` with the following arguments:
      * Number of matrix row
      * Number of matrix column
      * Number of non-zeros elements
      * block size (eg. 32)
      * Random seed (eg. 1)
   7. Check in the directory `outputs` the output file with the id correspondent to the job executed
   8. In that file, besides the other test information, if you have specified `-D Check` there will be a field `SpMV verification:`
      * In the case appears `SUCCESS`, the solution is correct
      * In the case appears `FAILURE`, the solution is uncorrect
4. In case you want to test CPU solution:
   1. Go to the CPU directory: `cd Path/to/CPU`
   2. Remove the possible executable: `rm bin/SpMV`
   3. Execute `make` with the desired macros (solution sorted by row or by column) (integer matrix or floating point matrix): `(-D SortR or -D SortC) -D (dtype=int or dtype=float) -D RAND"`
   4. Execute the sbatch script `sbatch sbatch_script.sh` with the following arguments:
      * Number of matrix row
      * Number of matrix column
      * Number of non-zeros elements
      * Random seed (eg. 1)
   7. Check in the directory `outputs` the output file with the id correspondent to the job executed

