//
// Created by Guglielmo Zocca on 11/06/25.
//

#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <string>
#include "mmio.h"

//Struct for the matrix sorting
template<typename ValType>
struct COOTuple{
    int row;
    int col;
    ValType val;
};

void get_mtx_dims (FILE *f, int *m, int *n, int *nnz);


//Initialization function for row ans column coordinate vectors
template<typename ValType>
void init_matrixV(int rows, int cols, ValType *matrix, ValType val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}

//Initialization function for value coordinate vector, input and output vectors
void init_matrixI(int rows, int cols, int *matrix, int val);

//Support function for sorting by row
template<typename ValType>
int compare_cooR(const void *a, const void *b) {
    const struct COOTuple<ValType> *ia = (const struct COOTuple<ValType> *)a;
    const struct COOTuple<ValType> *ib = (const struct COOTuple<ValType> *)b;

    if (ia->row != ib->row)
        return ia->row - ib->row;
    return ia->col - ib->col;
}

//Support function for sorting by column
template<typename ValType>
int compare_cooC(const void *a, const void *b) {
    const struct COOTuple<ValType> *ia = (const struct COOTuple<ValType> *)a;
    const struct COOTuple<ValType> *ib = (const struct COOTuple<ValType> *)b;

    if (ia->col != ib->col)
        return ia->col - ib->col;
    return ia->row - ib->row;
}

//Matrix sort by row function
template<typename ValType>
void sort_cooR(int *row, int *col, ValType *val, size_t nnz) {
    struct COOTuple<ValType> *entries = (struct COOTuple<ValType> *)malloc(nnz * sizeof(struct COOTuple<ValType>));
    if (!entries) {
        perror("Failed to allocate memory for COO sorting");
        return;
    }

    for (size_t i = 0; i < nnz; i++) {
        entries[i].row = row[i];
        entries[i].col = col[i];
        entries[i].val = val[i];
    }

    qsort(entries, nnz, sizeof(struct COOTuple<ValType>), compare_cooR<ValType>);

    for (size_t i = 0; i < nnz; i++) {
        row[i] = entries[i].row;
        col[i] = entries[i].col;
        val[i] = entries[i].val;
    }

    free(entries);
}

//Matrix sort by column function
template<typename ValType>
void sort_cooC(int *row, int *col, ValType *val, size_t nnz) {
    struct COOTuple<ValType> *entries = (struct COOTuple<ValType> *)malloc(nnz * sizeof(struct COOTuple<ValType>));
    if (!entries) {
        perror("Failed to allocate memory for COO sorting");
        return;
    }

    for (size_t i = 0; i < nnz; i++) {
        entries[i].row = row[i];
        entries[i].col = col[i];
        entries[i].val = val[i];
    }

    qsort(entries, nnz, sizeof(struct COOTuple<ValType>), compare_cooC<ValType>);

    for (size_t i = 0; i < nnz; i++) {
        row[i] = entries[i].row;
        col[i] = entries[i].col;
        val[i] = entries[i].val;
    }

    free(entries);
}

//Support function for matrix randomization
int compare_rand(const void *a, const void *b);

//Shuffle function
template<typename ValType>
void shuffle(int *row,int *col,ValType *val,int nnz)
{
    if (nnz>1)
    {
        int i;
        int t1;
        int t2;
        int t3;
        int j;
        for (i = 0; i < nnz - 1; i++)
        {
            j = i + rand() / (RAND_MAX / (nnz - i) + 1);
            t1 = row[j];
            t2 = col[j];
            t3 = val[j];
            row[j] = row[i];
            col[j] = col[i];
            val[j] = val[i];
            row[i] = t1;
            col[i] = t2;
            val[i] = t3;
        }
    }
}

//Randomization function of the coo matrix
template<typename ValType>
void initialize_random_coo(int *row, int *col, ValType *val, int nnz, int num_rows, int num_cols,int code) {

    printf("Randomization\n");
    int *usedR = (int *)malloc(num_rows*sizeof(int));
    char *usedC = (char *)malloc(num_cols*sizeof(char));
    int *usedC_i = (int *)malloc(num_cols*sizeof(int));
    if (!usedR) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }
    if (!usedC) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }
    if (!usedC_i) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }

    int i;

    for (i = 0; i < num_rows; i++) {
       usedR[i] = 0;
    }
    for (i = 0; i < num_cols; i++) {
        usedC[i] = 0;
    }
    for (i = 0; i < num_cols; i++) {
        usedC_i[i] = 0;
    }

    int r;
    int c;
    if(code == -1){
        srand(time(NULL));
    }else{
        srand(code);
    }
    size_t count = 0;
    while (count < nnz) {
        r = rand() % num_rows;
        if(usedR[r] < num_cols){
            usedR[r] = usedR[r] + 1;
            row[count] = r;
            count++;
        }
    }

    qsort(row, nnz, sizeof(int), compare_rand);

    int colum = 0;
    count = 0;
    while (count < nnz) {
        r = row[count];
        c = rand() % num_cols;

        if (!usedC[c]) {
            usedC[c] = 1;
            usedC_i[colum]=c;
            colum++;
            col[count] = c;
            val[count] = 1;

            if(count < nnz-1){
                if(r != row[count+1]){
                    for (i = 0; i < colum; i++) {
                        usedC[usedC_i[i]] = 0;
                        usedC_i[i] = 0;
                    }
                    colum = 0;
                }
            }
            count++;
        }

    }

    shuffle<ValType>(row,col,val,nnz);

    free(usedR);
    free(usedC);
    free(usedC_i);
}

//Randomization function of the csr matrix
template<typename ValType>
void initialize_random_csr(int *row_index, int *col, ValType *val, int nnz, int num_rows, int num_cols,int code) {

    printf("Randomization\n");

    int * row = (int *)malloc(num_rows*sizeof(int));

    init_matrixI(1, nnz, row, num_rows);

    int *usedR = (int *)malloc(num_rows*sizeof(int));
    char *usedC = (char *)malloc(num_cols*sizeof(char));
    int *usedC_i = (int *)malloc(num_cols*sizeof(int));
    if (!usedR) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }
    if (!usedC) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }
    if (!usedC_i) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }

    int i;

    for (i = 0; i < num_rows; i++) {
       usedR[i] = 0;
    }
    for (i = 0; i < num_cols; i++) {
        usedC[i] = 0;
    }
    for (i = 0; i < num_cols; i++) {
        usedC_i[i] = 0;
    }

    int r;
    int c;
    if(code == -1){
        srand(time(NULL));
    }else{
        srand(code);
    }
    size_t count = 0;
    while (count < nnz) {
        r = rand() % num_rows;
        if(usedR[r] < num_cols){
            usedR[r] = usedR[r] + 1;
            row[count] = r;
            count++;
        }
    }

    qsort(row, nnz, sizeof(int), compare_rand);

    int colum = 0;
    count = 0;
    while (count < nnz) {
        r = row[count];
        c = rand() % num_cols;

        if (!usedC[c]) {
            usedC[c] = 1;
            usedC_i[colum]=c;
            colum++;
            col[count] = c;
            val[count] = 1;

            if(count < nnz-1){
                if(r != row[count+1]){
                    for (i = 0; i < colum; i++) {
                        usedC[usedC_i[i]] = 0;
                        usedC_i[i] = 0;
                    }
                    colum = 0;
                }
            }
            count++;
        }

    }

    sort_cooR<ValType>(row, col,val,nnz);

    row_index[0] = 0;
    int j = 0;
    for(i=0;i<nnz;i++){
      for(j=(row[i]+1);j<=num_rows;j++){
          row_index[j]++;
      }
    }

	free(row);
    free(usedR);
    free(usedC);
    free(usedC_i);
}


//Concatenation string function
char* concat(const char *s1, const char *s2);

//convert mtx to COO
template<typename ValType>
int mtx_to_COO(FILE *fp,int *COOr,int *COOc,ValType *COOv,int nnz,size_t size,MM_typecode code){

    int i;
    int j;
    char* buffer = (char *)malloc(size + 1);
    if (!buffer) {
        perror("Failed to allocate buffer");
        fclose(fp);
        return 1;
    }

    fread(buffer, 1, size, fp);
    char *save_outer;
    char *save_inner;
    buffer[size] = '\0';
    char *tokenOUT = strtok_r(buffer, "\n",&save_outer);
    char *tokenIN;

    std::string floa;

    for(i=0;i<nnz;i++) {
        tokenIN = strtok_r(tokenOUT, " ",&save_inner);
        for(j=0;j<3;j++){
            if(j==0){
                COOr[i]=(atoi(tokenIN)-1);
            }
            if(j==1){
                COOc[i]=atoi(tokenIN)-1;
            }
            if(j==2){
                if (mm_is_integer(code)) {
                    COOv[i]=atoi(tokenIN);
                }else{
                  	floa = std::string(tokenIN);
                    COOv[i]=std::stof(floa);
                }
            }
            tokenIN = strtok_r(NULL, " ",&save_inner);
        }
        tokenOUT = strtok_r(NULL, "\n",&save_outer);

    }

    free(buffer);

    return 0;

}

//convert mtx to CSR
template<typename ValType>
int mtx_to_CSR(FILE *fp,int *CSRr,int *CSRc,ValType *CSRv,int nnz,int n,size_t size,MM_typecode code){

    int i;
    int j;

    init_matrixI(1, n, CSRr, 0);

    int* COOR = (int *)malloc(nnz * sizeof(int));


    char* buffer = (char *)malloc(size * sizeof(int));
    if (!buffer) {
        perror("Failed to allocate buffer");
        fclose(fp);
        return 1;
    }

    fread(buffer, 1, size, fp);
    char *save_outer;
    char *save_inner;
    buffer[size] = '\0';
    char *tokenOUT = strtok_r(buffer, "\n",&save_outer);
    char *tokenIN;

    int k;
	std::string floa;

    for(i=0;i<nnz;i++) {
        tokenIN = strtok_r(tokenOUT, " ",&save_inner);
        for(j=0;j<3;j++){
            if(j==0){
              	for(k=((atoi(tokenIN)-1)+1);k<=n;k++){
                  CSRr[k]++;
              	}
                COOR[i]=atoi(tokenIN)-1;

            }
            if(j==1){
                CSRc[i]=atoi(tokenIN)-1;
            }
            if(j==2){
                if (mm_is_integer(code)) {
                    CSRv[i]=atoi(tokenIN);
                }else{
                  	floa = std::string(tokenIN);
                    CSRv[i]=std::stof(floa);
                }
            }
            tokenIN = strtok_r(NULL, " ",&save_inner);
        }
        tokenOUT = strtok_r(NULL, "\n",&save_outer);

    }

    sort_cooR<ValType>(COOR, CSRc,CSRv,nnz);

    free(COOR);

    free(buffer);

    return 0;

}



#endif //UTILS_H
