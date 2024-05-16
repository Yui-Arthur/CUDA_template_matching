#ifndef _SSD_H_
#define _SSD_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "config.h"

float SSD(int *t, int *s, int rows, int columns, int kernel_size, int *res_pcc);
float SSD_CUDA(int *t, int *s, int rows, int columns, int kernel_size, int *res_pcc);
__global__ void SSD_kernel_calculate(int *t, int *s, int *z, int rows, int columns, int kernel_size, int kernel_columns);

float SSD(int *t, int *s, int rows, int columns, int kernel_size, int *res_pcc){
    printf("============== SSD CPU ==============\n");
    clock_t func_start = clock();

    clock_t cal_start = clock();
    for(int r=0; r<rows-kernel_size+1; r++){
        for(int c=0; c<columns-kernel_size+1; c++){
            int res = 0;
            for(int i=0; i<kernel_size; i++){
                for(int j=0; j<kernel_size; j++){
                    int val = t[(r+i)* columns + c+j] - s[i*kernel_size + j];
                    res += val*val;
                }
            }  
            
            res_pcc[r*columns + c] = res;
        }
    }
    clock_t cal_end = clock();
    clock_t func_end = clock();
    float function_elapsedTime = (func_end-func_start)/(double)(CLOCKS_PER_SEC);
    float cal_elapsedTime = (cal_end-cal_start)/(double)(CLOCKS_PER_SEC);
    printf("SSD Total Function Time : %10.10f ms\n", function_elapsedTime * 1000);
    printf("SSD Calculation Time on CPU: %10.10f ms\n", cal_elapsedTime * 1000) ;
    printf("=====================================\n");

    return cal_elapsedTime * 1000;
}

float SSD_CUDA(int *t, int *s, int rows, int columns, int kernel_size, int *res_pcc){
    printf("============== SSD CUDA ==============\n");
    clock_t func_srart = clock();
    
    cudaError_t R;
    /* alloc cuda memory */
    size_t pitch_t, pitch_s;
    int *device_t, *device_s, *device_res;
    R = cudaMallocPitch((void **)(&device_t), &pitch_t, columns * sizeof(int), rows);
    R = cudaMallocPitch((void **)(&device_s), &pitch_s, kernel_size * sizeof(int), kernel_size);
    R = cudaMallocPitch((void **)(&device_res), &pitch_t, columns * sizeof(int), rows);
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
    
    dim3 grid((rows + BLOCK_SIZE - 1 )  / (BLOCK_SIZE - kernel_size + 1), (columns + BLOCK_SIZE - 1)  / (BLOCK_SIZE - kernel_size + 1));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int shared_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(int) + kernel_size * kernel_size * sizeof(int);
    printf("grid (%d,%d)\n", (rows + BLOCK_SIZE - 1)  / BLOCK_SIZE, (columns + BLOCK_SIZE - 1)  / BLOCK_SIZE);
    printf("block (%d,%d)\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("shared memory size %d\n", shared_size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    /* run kernel function */
    SSD_kernel_calculate <<<grid, block, shared_size>>>(device_t, device_s, device_res, rows, pitch_t / sizeof(int), kernel_size, pitch_s / sizeof(int));
    R = cudaGetLastError();
    printf("kernel func start / Cuda Error : %s\n",cudaGetErrorString(R));
    R = cudaDeviceSynchronize();
    printf("kernel func run / Cuda Error : %s\n",cudaGetErrorString(R));
    cudaEventRecord(stop, 0);


    /* copy result from cuda */
    cudaMemcpy2D(res_pcc, columns * sizeof(int), device_res, pitch_t, columns * sizeof(int), rows, cudaMemcpyDeviceToHost);
    printf("cudaMemcpy2D / Cuda Error : %s\n",cudaGetErrorString(R));
    
    cudaFree(device_t);
    cudaFree(device_s);
    cudaFree(device_res);
    clock_t func_end = clock();
    float func_elapsedTime = (func_end-func_srart) / (double)(CLOCKS_PER_SEC);
    float kernel_elapsedTime;
    cudaEventElapsedTime(&kernel_elapsedTime, start, stop);

    printf("SSD Total Function Time : %10.10f ms\n", func_elapsedTime * 1000);
    printf("SSD Calculation Time on GPU: %10.10f ms\n", kernel_elapsedTime);

    printf("======================================\n");
    return kernel_elapsedTime;
}

__global__ void SSD_kernel_calculate(int *t, int *s, int *z, int rows, int columns, int kernel_size, int kernel_columns){
    /* 
     *   
     *  s : 2D array rows*cloumns => target array value
     *  t : 2D array kernel * kernel => search array 
     *  z = 2D array (rows-kernel+1)*(column-kernel+1) => SSD result
     * 
     *  first : 
     *      load t and s into shared memory
     *  second :
     *      calculate (t-s)^2 with kernel*kernel
     *  third :
     *      write sum of (t-s)^2 to z
     *  
     */

    int row_id = blockIdx.x * (blockDim.x-kernel_size+1) + threadIdx.x;
    int col_id = blockIdx.y * (blockDim.y-kernel_size+1) + threadIdx.y;
    int idx_t = row_id * columns + col_id;
    int idx_b = threadIdx.x * blockDim.x + threadIdx.y;
    extern __shared__ int shared[];
    int *shared_t = shared;
    int *shared_s = shared + blockDim.x * blockDim.y ;

    /* if block size bigger then target size */
    if(idx_t >= rows*columns)
        return;

    shared_t[idx_b] = t[idx_t];
    
    if(threadIdx.x < kernel_size && threadIdx.y < kernel_size)
        shared_s[threadIdx.x*kernel_size + threadIdx.y] = s[threadIdx.x * kernel_columns + threadIdx.y]; 

    __syncthreads();

    
    if(threadIdx.y + kernel_size - 1 >= blockDim.y  || threadIdx.x + kernel_size - 1 >= blockDim.x)
        return;
    
    int res = 0;

    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            float value = shared_t[idx_b + i*blockDim.y + j] - shared_s[i*kernel_size + j];
            res += value*value;
        }

    }
    z[idx_t] = res;
    // printf("sum_t %d, avg_t %.3f, res_x : %.3f, res_y %.3f, res_z : %.3f, res : %.3f\n", sum_t, avg_t, sqrt(res_x), res_s, res_z, res_z / (sqrt(res_x) * res_s));

    return;
}

#endif