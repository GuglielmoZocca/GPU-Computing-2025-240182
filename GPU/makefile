
BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs

MAIN    =  SpMV.c
OBJECTS = $(OBJ_FOLDER)/my_time_lib.o

all: $(BIN_FOLDER)/SpMV

$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/my_time_lib.c -o $@ $(OPT)

$(BIN_FOLDER)/SpMV: SpMV.c $(OBJECTS)
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) $^ -o $@ $(OPT)

clean_batch_outputs:
	rm $(BATCH_OUT_FOLDER)/*

clean:
	rm -rf ${BIN_FOLDER}
	rm -rf ${OBJ_FOLDER}
