//
// Created by Guglielmo Zocca on 11/06/25.
//

#include "../include/utils.h"

#include "../include/mmio.h"


void get_mtx_dims (FILE *f, int *m, int *n, int *nnz) {
    int ret_code;
    if ((ret_code = mm_read_mtx_crd_size(f, m, n, nnz)) !=0) exit(1);
    return;
}


//Initialization function for value coordinate vector, input and output vectors
void init_matrixI(int rows, int cols, int *matrix, int val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}


//Support function for matrix randomization
int compare_rand(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}




//Concatenation string function
char* concat(const char *s1, const char *s2)
{
    char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

