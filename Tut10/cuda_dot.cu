#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000000

#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

__global__ void dot_product(double *a, double *b, double *sum){
    __shared__ double temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < N){
        temp[threadIdx.x] = a[index] * b[index];
    }
    else{
        temp[threadIdx.x] = 0.0;
    }

    __syncthreads();
    
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(threadIdx.x < stride){
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(sum, temp[0]);
    }
}

int main(){
    FILE *f1 = fopen("file1.txt", "r");
    FILE *f2 = fopen("file2.txt", "r");
    if (!f1 || !f2) {
        printf("Failed to open file.\n");
        return 1;
    }

    double *a = (double *)malloc(N * sizeof(double));
    double *b = (double *)malloc(N * sizeof(double));
    double *final_sum = (double *)malloc(sizeof(double));
    *final_sum = 0.0;

    for(int i = 0; i < N; i++){
        fscanf(f1, "%lf", &a[i]);
        fscanf(f2, "%lf", &b[i]);
    }

    fclose(f1);
    fclose(f2);

    double *d_a, *d_b, *d_final_sum;
    cudaMalloc(&d_a, N * sizeof(double));
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_final_sum, sizeof(double));

    cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_final_sum, 0, sizeof(double));

    cudaDeviceSynchronize();
    clock_t start, end;
    start = clock();
    dot_product<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_a, d_b, d_final_sum);
    cudaDeviceSynchronize();
    end = clock();

    cudaMemcpy(final_sum, d_final_sum, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Dot product: %lf\n", *final_sum);
    printf("Time taken: %lf seconds\n", ((double)(end - start))/CLOCKS_PER_SEC);
    return 0;
}