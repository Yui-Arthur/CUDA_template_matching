#ifndef _PCC_H_
#define _PCC_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "config.h"

float PCC(int *, int *, int, int, int, float*);
float PCC_faster(int *, int *, int, int, int, float*);
float PCC_CUDA(int *t, int *s, int rows, int columns, int kernel_size, float*);
float PCC_CUDA_faster(int *t, int *s, int rows, int columns, int kernel_size, float *res_pcc);
__global__ void PCC_kernel_calculate(float *t, float *s, float *z, float res_s, int rows, int columns, int kernel_size, int kernel_columns);
__global__ void PCC_faster_kernel_calculate(int *t, int *s, float *z, int sum_s, float avg_s, float res_s, int rows, int columns, int res_columns, int kernel_size, int kernel_columns);

float PCC(int *t, int *s, int rows, int columns, int kernel_size, float *res_pcc){
    printf("============== PCC CPU ==============\n");
    clock_t func_start = clock();
    int sum_s = 0;
    float avg_s = 0;
    float *s_fp = (float*) malloc( kernel_size * kernel_size * sizeof(float));
    float res_x = 0, res_y = 0, res_z = 0;
    /* cal avg s, s - avg s, res_s */
    for(int i = 0; i < kernel_size; i++) for(int j = 0; j < kernel_size; j++) sum_s += s[i*kernel_size + j];
    avg_s =  sum_s / (float)(kernel_size*kernel_size);
    for(int i = 0; i < kernel_size; i++) {
        for(int j = 0; j < kernel_size; j++){
            s_fp[i*kernel_size + j] = s[i*kernel_size + j] - avg_s;
            res_y += s_fp[i*kernel_size + j] * s_fp[i*kernel_size + j];
        } 
    }
    printf("avg s %.5f\n", avg_s);
    res_y = sqrt(res_y);

    float avg_t = 0;
    int sum_t = 0;

    clock_t cal_start = clock();
    for(int r=0; r<rows-kernel_size+1; r++){
        for(int c=0; c<columns-kernel_size+1; c++){
            sum_t = 0;
            res_x = 0, res_z = 0, avg_t = 0;
            for(int i=0; i<kernel_size; i++) for(int j=0; j<kernel_size; j++) sum_t += t[(r+i)* columns + c+j];
            avg_t = sum_t / (float)(kernel_size*kernel_size);

            for(int i=0; i<kernel_size; i++){
                for(int j=0; j<kernel_size; j++){
                    float t_fp = t[(r+i)* columns + c+j] - avg_t;
                    res_x += t_fp * t_fp;
                    res_z += t_fp * s_fp[i * kernel_size + j];
                }
            }
            res_pcc[r*columns + c] = res_z / (sqrt(res_x) * res_y);
        }
    }
    clock_t cal_end = clock();
    clock_t func_end = clock();
    float function_elapsedTime = (func_end-func_start)/(double)(CLOCKS_PER_SEC);
    float cal_elapsedTime = (cal_end-cal_start)/(double)(CLOCKS_PER_SEC);
    printf("PCC Total Function Time : %10.10f ms\n", function_elapsedTime * 1000);
    printf("PCC Calculation Time on CPU: %10.10f ms\n", cal_elapsedTime * 1000) ;
    printf("=====================================\n");

    return cal_elapsedTime * 1000;
}

float PCC_faster(int *t, int *s, int rows, int columns, int kernel_size, float *res_pcc){
    printf("============== PCC Faster CPU ==============\n");
    clock_t func_start = clock();
    int sum_s = 0;
    float avg_s = 0;
    float res_y = 0;
    /* cal avg s, s - avg s, res_s */
    for(int i = 0; i < kernel_size; i++) for(int j = 0; j < kernel_size; j++) sum_s += s[i*kernel_size + j];
    avg_s =  sum_s / (float)(kernel_size*kernel_size);
    for(int i = 0; i < kernel_size; i++) {
        for(int j = 0; j < kernel_size; j++){
            res_y += (s[i*kernel_size + j] - avg_s) * (s[i*kernel_size + j] - avg_s);
        } 
    }
    res_y = sqrt(res_y);
    printf("avg s %.5f\n", avg_s);
    printf("res y : %.5f\n", res_y);

    int total_kernel_size = kernel_size * kernel_size ;

    clock_t cal_start = clock();
    for(int r=0; r<rows-kernel_size+1; r++){
        for(int c=0; c<columns-kernel_size+1; c++){
            
            int sum_t = 0, sum_pow_t = 0, sum_ts = 0;
            for(int i=0; i<kernel_size; i++){
                for(int j=0; j<kernel_size; j++){
                    int curr_t_value = t[(r+i)*columns + c+j], curr_s_value = s[i*kernel_size + j];
                    sum_t += curr_t_value;
                    sum_pow_t += curr_t_value * curr_t_value;
                    sum_ts += curr_t_value * curr_s_value;
                }
            }
            // printf("(%d , %d , %d)", sum_t, sum_pow_t, sum_ts);
            float avg_t = sum_t / float(total_kernel_size);
            float res_z = sum_ts - avg_t*sum_s - avg_s*sum_t + total_kernel_size*avg_s*avg_t;
            float res_x = sum_pow_t + avg_t * (total_kernel_size*avg_t - 2*sum_t);
            res_pcc[r*columns + c] = res_z / sqrt(res_x) / res_y;   
            // printf("%.3f ", res_pcc[r*columns + c]);
        }
        // printf("\n");
    }
    clock_t cal_end = clock();
    clock_t func_end = clock();
    float function_elapsedTime = (func_end-func_start)/(double)(CLOCKS_PER_SEC);
    float cal_elapsedTime = (cal_end-cal_start)/(double)(CLOCKS_PER_SEC);
    printf("PCC Faster Total Function Time : %10.10f ms\n", function_elapsedTime * 1000);
    printf("PCC Faster Calculation Time on CPU: %10.10f ms\n", cal_elapsedTime * 1000);
    printf("=====================================\n");

    return cal_elapsedTime * 1000;
}

float PCC_CUDA(int *t, int *s, int rows, int columns, int kernel_size, float *res_pcc){
    printf("============== PCC CUDA ==============\n");
    clock_t func_srart = clock();
    int sum_s = 0;
    float avg_s = 0;
    float *s_fp = (float*) malloc( kernel_size * kernel_size * sizeof(float));
    float res_s = 0;
    cudaError_t R;
    /* cal avg s, s - avg s, res_s */
    for(int i = 0; i < kernel_size; i++) for(int j = 0; j < kernel_size; j++) sum_s += s[i*kernel_size + j];
    avg_s =  sum_s / (float)(kernel_size*kernel_size);
    for(int i = 0; i < kernel_size; i++) {
        for(int j = 0; j < kernel_size; j++){
            s_fp[i*kernel_size + j] = s[i*kernel_size + j] - avg_s;
            res_s += s_fp[i*kernel_size + j] * s_fp[i*kernel_size + j];
        } 
    }
    printf("avg s %.5f\n", avg_s);
    res_s = sqrt(res_s);
    
    /* copy target int to float */
    float *t_fp = (float*) malloc(columns * rows * sizeof(float));
    for(int i = 0; i < rows; i++) for(int j = 0; j < columns; j++) t_fp[i*columns + j] = t[i*columns + j];
    
    /* alloc cuda memory */
    size_t pitch_t, pitch_s;
    float *device_t, *device_s, *device_res;
    R = cudaMallocPitch((void **)(&device_t), &pitch_t, columns * sizeof(float), rows);
    R = cudaMallocPitch((void **)(&device_s), &pitch_s, kernel_size * sizeof(float), kernel_size);
    R = cudaMallocPitch((void **)(&device_res), &pitch_t, columns * sizeof(float), rows);
    printf("cudaMallocPitch / Cuda Error : %s\n",cudaGetErrorString(R));

    /* copy t and s to cuda */
    R = cudaMemcpy2D(device_t, pitch_t, t_fp, columns * sizeof(float), columns * sizeof(float), rows, cudaMemcpyHostToDevice);
    R = cudaMemcpy2D(device_s, pitch_s, s_fp, kernel_size * sizeof(float), kernel_size * sizeof(float), kernel_size, cudaMemcpyHostToDevice);
    printf("cudaMemcpy2D / Cuda Error : %s\n",cudaGetErrorString(R));

    /* 
     * kernel function setting 
     * grid  2D = (rows / block_size, column / block_size)
     * block 2D = (block_size, block_size)
     * 
     * shard memory size 
     *  t = new_column * block_size 
     *  s = new_kernel * kernel_size
     */
    
    dim3 grid((rows + BLOCK_SIZE - 1 )  / (BLOCK_SIZE - kernel_size + 1), (columns + BLOCK_SIZE - 1)  / (BLOCK_SIZE - kernel_size + 1));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int shared_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(float) + kernel_size * kernel_size * sizeof(float);
    printf("grid (%d,%d)\n", (rows + BLOCK_SIZE - 1)  / BLOCK_SIZE, (columns + BLOCK_SIZE - 1)  / BLOCK_SIZE);
    printf("block (%d,%d)\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("shared memory size %d\n", shared_size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    /* run kernel function */
    PCC_kernel_calculate <<<grid, block, shared_size>>>(device_t, device_s, device_res, res_s, rows, pitch_t / sizeof(float), kernel_size, pitch_s / sizeof(float));
    R = cudaGetLastError();
    printf("kernel func start / Cuda Error : %s\n",cudaGetErrorString(R));
    R = cudaDeviceSynchronize();
    printf("kernel func run / Cuda Error : %s\n",cudaGetErrorString(R));
    cudaEventRecord(stop, 0);

    free(t_fp);
    free(s_fp);

    /* copy result from cuda */
    cudaMemcpy2D(res_pcc, columns * sizeof(float), device_res, pitch_t, columns * sizeof(float), rows, cudaMemcpyDeviceToHost);
    printf("cudaMemcpy2D / Cuda Error : %s\n",cudaGetErrorString(R));
    
    cudaFree(device_t);
    cudaFree(device_s);
    cudaFree(device_res);
    clock_t func_end = clock();
    float func_elapsedTime = (func_end-func_srart) / (double)(CLOCKS_PER_SEC);
    float kernel_elapsedTime;
    cudaEventElapsedTime(&kernel_elapsedTime, start, stop);

    printf("PCC Total Function Time : %10.10f ms\n", func_elapsedTime * 1000);
    printf("PCC Calculation Time on GPU: %10.10f ms\n", kernel_elapsedTime);

    printf("======================================\n");
    return kernel_elapsedTime;
}

float PCC_CUDA_faster(int *t, int *s, int rows, int columns, int kernel_size, float *res_pcc){
    printf("============== PCC CUDA Faster ==============\n");
    clock_t func_srart = clock();
    int sum_s = 0, sum_pow_s = 0;
    float avg_s = 0;
    float res_s = 0;
    cudaError_t R;
    /* cal avg s, s - avg s, res_s */
    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            sum_s += s[i*kernel_size + j];
            sum_pow_s += s[i*kernel_size + j] * s[i*kernel_size + j];
        }
    }  
    avg_s =  sum_s / (float)(kernel_size*kernel_size);
    res_s = sqrt(sum_pow_s + avg_s * (kernel_size*kernel_size*avg_s - 2*sum_s));
    
    
    printf("avg s : %.5f\n", avg_s);
    printf("res y : %.5f\n", res_s);
    
    /* alloc cuda memory */
    size_t pitch_t, pitch_s, pitch_res;
    int *device_t, *device_s;
    float *device_res;
    R = cudaMallocPitch((void **)(&device_t), &pitch_t, columns * sizeof(int), rows);
    R = cudaMallocPitch((void **)(&device_s), &pitch_s, kernel_size * sizeof(int), kernel_size);
    R = cudaMallocPitch((void **)(&device_res), &pitch_res, columns * sizeof(float), rows);
    printf("cudaMallocPitch / Cuda Error : %s\n",cudaGetErrorString(R));

    /* copy t and s to cuda */
    R = cudaMemcpy2D(device_t, pitch_t, t, columns * sizeof(int), columns * sizeof(int), rows, cudaMemcpyHostToDevice);
    R = cudaMemcpy2D(device_s, pitch_s, s, kernel_size * sizeof(int), kernel_size * sizeof(int), kernel_size, cudaMemcpyHostToDevice);
    printf("cudaMemcpy2D / Cuda Error : %s\n",cudaGetErrorString(R));

    /* 
     * kernel function setting 
     * grid  2D = (rows / block_size, column / block_size)
     * block 2D = (block_size, block_size)
     * 
     * shard memory size 
     *  t = new_column * block_size 
     *  s = new_kernel * kernel_size
     */
    
    int block_size = (FASTER_BLOCK_SIZE*THREAD_WORK_PER_AXIS - kernel_size + 1) ;
    dim3 grid((rows + block_size - 1 )  / block_size, (columns + block_size - 1)  / block_size);
    // dim3 grid(1,1);
    dim3 block(FASTER_BLOCK_SIZE, FASTER_BLOCK_SIZE);
    int shared_size = FASTER_BLOCK_SIZE * FASTER_BLOCK_SIZE * sizeof(int) * THREAD_WORK_PER_AXIS*THREAD_WORK_PER_AXIS 
                    + kernel_size * kernel_size * sizeof(int);
    printf("grid (%d,%d)\n", grid.x, grid.y);
    printf("block (%d,%d)\n", block.x, block.y);
    printf("shared memory size %d\n", shared_size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    /* run kernel function */
    PCC_faster_kernel_calculate <<<grid, block, shared_size>>>(device_t, device_s, device_res,
                                                            sum_s, avg_s, res_s, rows, 
                                                            pitch_t / sizeof(int), pitch_res / sizeof(float), kernel_size, pitch_s / sizeof(int));
    R = cudaGetLastError();
    printf("kernel func start / Cuda Error : %s\n",cudaGetErrorString(R));
    R = cudaDeviceSynchronize();
    printf("kernel func run / Cuda Error : %s\n",cudaGetErrorString(R));
    cudaEventRecord(stop, 0);

    /* copy result from cuda */
    cudaMemcpy2D(res_pcc, columns * sizeof(float), device_res, pitch_res, columns * sizeof(float), rows, cudaMemcpyDeviceToHost);
    printf("cudaMemcpy2D / Cuda Error : %s\n",cudaGetErrorString(R));
    
    cudaFree(device_t);
    cudaFree(device_s);
    cudaFree(device_res);
    clock_t func_end = clock();
    float func_elapsedTime = (func_end-func_srart) / (double)(CLOCKS_PER_SEC);
    float kernel_elapsedTime;
    cudaEventElapsedTime(&kernel_elapsedTime, start, stop);

    printf("PCC Total Function Time : %10.10f ms\n", func_elapsedTime * 1000);
    printf("PCC Calculation Time on GPU: %10.10f ms\n", kernel_elapsedTime);

    printf("======================================\n");
    return kernel_elapsedTime;
}

__global__ void PCC_kernel_calculate(float *t, float *s, float *z, float res_s, int rows, int columns, int kernel_size, int kernel_columns){
    /* 
     *   
     *  s : 2D array rows*cloumns => target array value
     *  t : 2D array kernel * kernel => search array - avg search value
     *  z = 2D array (rows-kernel+1)*(column-kernel+1) => PCC result
     * 
     *  first : 
     *      load t and s into shared memory
     *  second :
     *      calculate avg t with kernel*kernel
     *  third : 
     *      calculate sum of s * t => res_z
     *      calculate sum of t * t => res_x
     *  fourth :
     *      write res_z / (sqrt(res_t) * res_s) to z
     *  
     */

    int row_id = blockIdx.x * (blockDim.x-kernel_size+1) + threadIdx.x;
    int col_id = blockIdx.y * (blockDim.y-kernel_size+1) + threadIdx.y;
    int idx_t = row_id * columns + col_id;
    int idx_b = threadIdx.x * blockDim.x + threadIdx.y;
    extern __shared__ float shared_fp[];
    float *shared_t_fp = shared_fp;
    float *shared_s_fp = shared_fp + blockDim.x * blockDim.y ;

    /* if block size bigger then target size */
    if(idx_t >= rows*columns)
        return;

    shared_t_fp[idx_b] = t[idx_t];
    
    if(threadIdx.x < kernel_size && threadIdx.y < kernel_size)
        shared_s_fp[threadIdx.x*kernel_size + threadIdx.y] = s[threadIdx.x * kernel_columns + threadIdx.y]; 

    __syncthreads();

    
    if(threadIdx.y + kernel_size - 1 >= blockDim.y  || threadIdx.x + kernel_size - 1 >= blockDim.x)
        return;
    
    int sum_t = 0;
    for(int i=0; i<kernel_size; i++) for(int j=0; j<kernel_size; j++) sum_t += (int)shared_t_fp[idx_b + (i)*blockDim.y + j];
    float avg_t = (float)sum_t / (float)(kernel_size*kernel_size), res_x = 0, res_z = 0;

    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            float tmp_t = shared_t_fp[idx_b + i*blockDim.y + j] - avg_t;
            res_x += tmp_t * tmp_t;
            res_z += tmp_t * shared_s_fp[i*kernel_size + j];
        }

    }
    z[idx_t] = res_z / (sqrt(res_x) * res_s);
    // printf("sum_t %d, avg_t %.3f, res_x : %.3f, res_y %.3f, res_z : %.3f, res : %.3f\n", sum_t, avg_t, sqrt(res_x), res_s, res_z, res_z / (sqrt(res_x) * res_s));

    return;
}

__global__ void PCC_faster_kernel_calculate(int *t, int *s, float *z, int sum_s, float avg_s, float res_s, int rows, int columns, int res_columns, int kernel_size, int kernel_columns){
    /* 
     *   
     *  s : 2D array rows*cloumns => target array value
     *  t : 2D array kernel * kernel => search array - avg search value
     *  z = 2D array (rows-kernel+1)*(column-kernel+1) => PCC result
     * 
     *  first : 
     *      load t and s into shared memory
     *  second :
     *      calculate sum of t / sum of t*t / sum of t*s with kernel*kernel
     *  third :
     *      write res_z / (sqrt(res_t) * res_s) to z
     *  
     */

    int row_id = blockIdx.x * (THREAD_WORK_PER_AXIS*blockDim.x - kernel_size+1) + THREAD_WORK_PER_AXIS * threadIdx.x;
    int col_id = blockIdx.y * (THREAD_WORK_PER_AXIS*blockDim.y - kernel_size+1) + THREAD_WORK_PER_AXIS * threadIdx.y;
    int idx_t = row_id * columns + col_id;
    int idx_b = THREAD_WORK_PER_AXIS*THREAD_WORK_PER_AXIS * threadIdx.x * blockDim.x + THREAD_WORK_PER_AXIS * threadIdx.y;
    
    int shared_columns_size = blockDim.x * THREAD_WORK_PER_AXIS;
    extern __shared__ int shared[];
    int *shared_t = shared;
    int *shared_s = shared + blockDim.x * blockDim.y * THREAD_WORK_PER_AXIS * THREAD_WORK_PER_AXIS ;

    if(threadIdx.x < kernel_size && threadIdx.y < kernel_size)
        shared_s[threadIdx.x*kernel_size + threadIdx.y] = s[threadIdx.x * kernel_columns + threadIdx.y]; 
    
    /* if block size bigger then target size */
    if(idx_t >= rows*columns )
        return;

    /* move data into shared memory */
    #pragma unroll
    for(int i=0; i<THREAD_WORK_PER_AXIS; i++){
        #pragma unroll
        for(int j=0; j<THREAD_WORK_PER_AXIS; j++){
            if(idx_t+ i*columns + j >= rows*columns)
                continue;
            shared_t[idx_b + i*shared_columns_size + j] = t[idx_t+ i*columns + j];
        }
    }
    
    __syncthreads();
    
    #pragma unroll
    for(int r=0; r<THREAD_WORK_PER_AXIS; r++){
        #pragma unroll
        for(int c=0; c<THREAD_WORK_PER_AXIS; c++){
            int sum_t = 0, sum_pow_t = 0, sum_ts = 0;
            // if(threadIdx.y*THREAD_WORK_PER_AXIS + r + kernel_size - 1 >= shared_columns_size  || threadIdx.x*THREAD_WORK_PER_AXIS + c + kernel_size - 1 >= shared_columns_size)
            if(idx_b + (r+kernel_size-1)*shared_columns_size + c + kernel_size -1 >= blockDim.x * blockDim.y * THREAD_WORK_PER_AXIS * THREAD_WORK_PER_AXIS)
                continue;
            if((row_id + r)*res_columns + col_id + c >= rows*res_columns)
                continue;

            for(int i=0; i<kernel_size; i++){
                for(int j=0; j<kernel_size; j++){
                    int current_t = shared_t[idx_b + (r+i)*shared_columns_size + j+c], current_s = shared_s[i*kernel_size + j];  

                    sum_t += current_t;
                    sum_pow_t += current_t * current_t;
                    sum_ts += current_t * current_s;
                }
            }  
            float avg_t = (float)sum_t / (float)(kernel_size*kernel_size);

            float res_z = sum_ts - avg_t*sum_s - avg_s*sum_t + kernel_size*kernel_size*avg_s*avg_t;
            float res_x = sum_pow_t + avg_t * (kernel_size*kernel_size*avg_t - 2*sum_t);

            z[(row_id + r)*res_columns + col_id + c] = res_z / (sqrt(res_x) * res_s);
        }
    }
    
    return;
}


#endif