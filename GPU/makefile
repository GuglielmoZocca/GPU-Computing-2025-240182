CU = nvcc --gpu-architecture=sm_80 -m64 -O3 
BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs

MAIN    =  SpMV.cu
OBJECTS = $(OBJ_FOLDER)/my_time_lib.o

all: $(BIN_FOLDER)/SpMV

$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/my_time_lib.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CU) -c $(SRC_FOLDER)/my_time_lib.cu -o $@ $(OPT)

$(BIN_FOLDER)/SpMV: SpMV.cu $(OBJECTS)
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CU) $^ -o $@ $(MACROS)

clean_batch_outputs:
	rm $(BATCH_OUT_FOLDER)/*

clean:
	rm -rf ${BIN_FOLDER}
	rm -rf ${OBJ_FOLDER}
