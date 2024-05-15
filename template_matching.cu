#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "PCC.h"
#include "SSD.h"

void init(int **t, int **s, int *rows, int *columns, int *kernel, char **argv);
void get_row_and_column_from_file_name(char *file_name, int *rows, int *columns);
void read_data_from_file(int *array, char *file_name);
bool compare_ans(float *cpu_res, float *cuda_res, int rows, int columns, int kernel_size);
void show_output_float(float *res, int rows, int columns, int kernel_size, const char *device);
void show_output_int(int *res, int rows, int columns, int kernel_size, const char *device);
void PCC_test(int *t, int*s, int rows, int columns, int kernel_size);
void SSD_test(int *t, int*s, int rows, int columns, int kernel_size);


int main(int argc, char **argv){
    
    int rows, columns, kernel_size, *t, *s;
    
    init(&t, &s, &rows, &columns, &kernel_size, argv);
    // PCC_test(t, s, rows, columns, kernel_size);
    SSD_test(t, s, rows, columns, kernel_size);
    
    return 0;
}

void init(int **t, int **s, int *rows, int *columns, int *kernel_size, char **argv){
    /* 
     * S file name => SX_K_K.txt
     * T file name => TX_R_C.txt
     */

    /* get rows & columns & kernel_size for file_name */ 
    char t_file_name[100], s_file_name[100] ;
    printf("============== init ==============\n");
    printf("target file : %s\nsearch file : %s\n", argv[1], argv[2]);
    strcpy(t_file_name, argv[1]);
    strcpy(s_file_name, argv[2]);

    get_row_and_column_from_file_name(argv[1], rows, columns);
    get_row_and_column_from_file_name(argv[2], kernel_size, kernel_size);

    
    printf("target size (%d, %d) search size (%d, %d)\n", *rows, *columns, *kernel_size, *kernel_size);
    *t = (int*) malloc((*rows) * (*columns) * sizeof(int));
    *s = (int*) malloc((*kernel_size) * (*kernel_size) * sizeof(int));

    read_data_from_file(*t, t_file_name);
    read_data_from_file(*s, s_file_name);
    
    // for(int i=0; i<*rows; i++){
    //     for(int j=0; j<*columns; j++) printf("%d, ", (*t)[i*(*columns) + j]);
    //     printf("\n");
    // } 
    
    // for(int i=0; i<*kernel_size; i++){
    //     for(int j=0; j<*kernel_size; j++) printf("%d, ", (*s)[i*(*kernel_size) + j]);
    //     printf("\n");
    // } 
    printf("=================================\n");
    
    return;
}

void get_row_and_column_from_file_name(char *file_name, int *rows, int *columns){
    
    char *split_value = strtok(file_name, "."), *tmp;
    strtok(split_value, "/");
    while((tmp = strtok(NULL, "/") )!= NULL) split_value = tmp;
    strtok(split_value, "_");
    *rows = atoi(strtok(NULL, "_"));
    *columns = atoi(strtok(NULL, "_"));

    return;
}

void read_data_from_file(int *array, char *file_name){
    FILE *fptr = fopen(file_name, "r");
    char buf[1024], *tmp;
    int idx = 0;
    while(fgets(buf, 1024, fptr)){

        array[idx] = atoi(strtok(buf, ",")), idx++;   
        while((tmp = strtok(NULL, ",")) != NULL) array[idx] = atoi(tmp), idx++;

    }

    fclose(fptr);

    return;
}

bool compare_ans(float *cpu_res, float *cuda_res, int rows, int columns, int kernel_size){
    float diff = 0;
    bool res = true;
    for(int i=0; i<rows - kernel_size + 1; i++){
        for(int j=0; j<rows - kernel_size + 1; j++){
            // printf("%d:%d, %f - %f = %f\n", i, j, cpu_res[i*columns + j], cuda_res[i*columns + j], cpu_res[i*columns + j] - cuda_res[i*columns + j]);
            diff +=  fabs(cpu_res[i*columns + j] - cuda_res[i*columns + j]);
            if(fabs(cpu_res[i*columns + j] - cuda_res[i*columns + j]) > PRECISION){
                res = false;
                // printf("%d-%d, %f - %f, %f\n", i, j, cpu_res[i*columns + j], cuda_res[i*columns + j], cpu_res[i*columns + j] - cuda_res[i*columns + j]);
            }
        }
        // return false;
    }

    printf("total diff %f\n", diff);
    return res;
}

void show_output_float(float *res, int rows, int columns, int kernel_size, const char *device){
    int *max_x, *max_y, idx=0;
    float *v; 
    float max_correlation = 0.0;

    printf("============== %s output ==============\n", device);
    max_x = (int*) malloc(150 * sizeof(int));
    max_y = (int*) malloc(150 * sizeof(int));
    v = (float*) malloc(150 * sizeof(float));

    for(int i=0; i<rows - kernel_size + 1; i++){
        for(int j=0; j<columns - kernel_size + 1; j++){
            if(res[i*columns + j] - max_correlation > PRECISION)
                max_x[0] = i, max_y[0] = j, v[0] = res[i*columns + j], idx = 1, max_correlation = res[i*columns + j];
            else if(abs(res[i*columns + j] - max_correlation) < PRECISION)
                max_x[idx] = i, max_y[idx] = j, v[idx] = res[i*columns + j], idx++;
        }
    }

    printf("max correlation %.3f\n", max_correlation);
    for(int i=0; i<idx; i++){
        printf("(%d, %d) : %.5f\n", max_x[i], max_y[i], v[i]);
    }

    free(max_x);
    free(max_y);
    free(v);

    printf("====================================\n");
    return;
}

void show_output_int(int *res, int rows, int columns, int kernel_size, const char *device){
    int *max_x, *max_y, idx=0;
    int *v; 
    int max_correlation = 0.0;

    printf("============== %s output ==============\n", device);
    max_x = (int*) malloc(150 * sizeof(int));
    max_y = (int*) malloc(150 * sizeof(int));
    v = (int*) malloc(150 * sizeof(int));

    for(int i=0; i<rows - kernel_size + 1; i++){
        for(int j=0; j<columns - kernel_size + 1; j++){
            if(res[i*columns + j] > max_correlation)
                max_x[0] = i, max_y[0] = j, v[0] = res[i*columns + j], idx = 1, max_correlation = res[i*columns + j];
            else if(res[i*columns + j] == max_correlation)
                max_x[idx] = i, max_y[idx] = j, v[idx] = res[i*columns + j], idx++;
        }
    }

    printf("max correlation %.3f\n", max_correlation);
    for(int i=0; i<idx; i++){
        printf("(%d, %d) : %d\n", max_x[i], max_y[i], v[i]);
    }

    free(max_x);
    free(max_y);
    free(v);

    printf("====================================\n");
    return;
}

void PCC_test(int *t, int*s, int rows, int columns, int kernel_size){
    float *cpu_res_pcc = (float*) malloc(rows * columns * sizeof(float));
    float *cpu_faster_res_pcc = (float*) malloc(rows * columns * sizeof(float));
    float *cuda_res_pcc = (float*) malloc(rows * columns * sizeof(float));
    float *cuda_res_pcc_faster = (float*) malloc(rows * columns * sizeof(float));

    float cpu_time, cpu_faster_time, cuda_time, cuda_faster_time;
    const char *device_type[10] = {"CPU", "CPU Faster", "CUDA", "CUDA Faster"};
    cpu_time = PCC(t, s, rows, columns, kernel_size, cpu_res_pcc);
    cpu_faster_time = PCC_faster(t, s, rows, columns, kernel_size, cpu_faster_res_pcc);
    cuda_time = PCC_CUDA(t, s, rows, columns, kernel_size, cuda_res_pcc);
    cuda_faster_time = PCC_CUDA_faster(t, s, rows, columns, kernel_size, cuda_res_pcc_faster);
    show_output_float(cpu_res_pcc, rows, columns, kernel_size, device_type[0]);
    show_output_float(cpu_faster_res_pcc, rows, columns, kernel_size, device_type[1]);
    show_output_float(cuda_res_pcc, rows, columns, kernel_size, device_type[2]);
    show_output_float(cuda_res_pcc_faster, rows, columns, kernel_size, device_type[3]);
    // bool r = compare_ans(cpu_res_pcc, cuda_res_pcc, rows, columns, kernel_size);
    // bool r = compare_ans(cpu_res_pcc, cuda_res_pcc, rows, columns, kernel_size);
    printf("============== result ==============\n");
    // printf("result %d\n", r);
    printf("cpu vs. cpu faster speedup %3.20f\n", cpu_time / cpu_faster_time);
    printf("cpu vs. gpu speedup  %3.20f\n", cpu_time / cuda_time);
    printf("gpu vs. gpu faster speedup  %3.20f\n", cuda_time / cuda_faster_time);
    printf("cpu vs. gpu faster speedup  %3.20f\n", cpu_time / cuda_faster_time);
    printf("====================================\n");

    free(cpu_res_pcc);
    free(cpu_faster_res_pcc);
    free(cuda_res_pcc);
    free(cuda_res_pcc_faster);
    return;
}

void SSD_test(int *t, int*s, int rows, int columns, int kernel_size){
    int *cpu_res_pcc = (int*) malloc(rows * columns * sizeof(int));
    int *cpu_faster_res_pcc = (int*) malloc(rows * columns * sizeof(int));
    int *cuda_res_pcc = (int*) malloc(rows * columns * sizeof(int));
    int *cuda_res_pcc_faster = (int*) malloc(rows * columns * sizeof(int));

    float cpu_time, cpu_faster_time, cuda_time, cuda_faster_time;
    const char *device_type[10] = {"CPU", "CPU Faster", "CUDA", "CUDA Faster"};
    cpu_time = SSD(t, s, rows, columns, kernel_size, cpu_res_pcc);
    // cpu_faster_time = PCC_faster(t, s, rows, columns, kernel_size, cpu_faster_res_pcc);
    cuda_time = SSD_CUDA(t, s, rows, columns, kernel_size, cuda_res_pcc);
    // cuda_faster_time = PCC_CUDA_faster(t, s, rows, columns, kernel_size, cuda_res_pcc_faster);
    show_output_int(cpu_res_pcc, rows, columns, kernel_size, device_type[0]);
    show_output_int(cuda_res_pcc, rows, columns, kernel_size, device_type[2])
    ;
    printf("============== result ==============\n");
    // printf("result %d\n", r);
    printf("cpu vs. cpu faster speedup %3.20f\n", cpu_time / cpu_faster_time);
    printf("cpu vs. gpu speedup  %3.20f\n", cpu_time / cuda_time);
    printf("====================================\n");

    free(cpu_res_pcc);
    free(cpu_faster_res_pcc);
    free(cuda_res_pcc);
    free(cuda_res_pcc_faster);
    return;
}