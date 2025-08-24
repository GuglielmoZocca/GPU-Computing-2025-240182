CC = c++
OPTCC = -std=c++14 -O2

CU = nvcc
BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs
BATCH_ERR_FOLDER := errors
OPTCU = --gpu-architecture=sm_80 -m64 -O3 -std=c++14
CUSPARSE = -lcusparse

MAIN    =  SpMV.cu

OBJECTS = $(OBJ_FOLDER)/time_lib.o $(OBJ_FOLDER)/mmio.o $(OBJ_FOLDER)/utils.o


all: $(BIN_FOLDER)/SpMV

$(OBJ_FOLDER)/time_lib.o: $(SRC_FOLDER)/time_lib.cpp
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/time_lib.cpp -o $@ $(OPTCC)

$(OBJ_FOLDER)/mmio.o: $(SRC_FOLDER)/mmio.c
	mkdir -p $(@D)
	$(CC) -c $(SRC_FOLDER)/mmio.c -o $@ $(OPTCC)

$(OBJ_FOLDER)/import_sparse_matrix.o: $(SRC_FOLDER)/import_sparse_matrix.cpp
	mkdir -p $(@D)
	$(CC) -c $(SRC_FOLDER)/import_sparse_matrix.cpp -o $@ $(OPTCC)

$(OBJ_FOLDER)/utils.o: $(SRC_FOLDER)/utils.cpp
	mkdir -p $(@D)
	$(CC) -c $(SRC_FOLDER)/utils.cpp -o $@ $(OPTCC)

$(BIN_FOLDER)/SpMV: SpMV.cu $(OBJECTS)
	mkdir -p $(@D)
	$(CU) $^ -o $@ $(MACROS) $(OPTCU) $(CUSPARSE)

clean_batch_outputs:
	rm $(BATCH_OUT_FOLDER)/*
	rm $(BATCH_ERR_FOLDER)/*

clean:
	rm -rf ${BIN_FOLDER}
	rm -rf ${OBJ_FOLDER}
