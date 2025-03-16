#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000000
#define THREADS_PER_BLOCK 512

__global__ void sum_n(double *numbers, double *result, int n)
{
    __shared__ double sharedSum[THREADS_PER_BLOCK]; 
    int tid = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    sharedSum[tid] = (i < n) ? numbers[i] : 0.0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(result, sharedSum[0]);
    }
}

int main()
{
    FILE *f = fopen("input.txt", "r");
    if (!f)
    {
        printf("Failed to open file.\n");
        return 1;
    }

    double *a = (double *)malloc(N * sizeof(double));
    double *final_sum = (double *)malloc(sizeof(double));
    *final_sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        fscanf(f, "%lf", &a[i]);
    }
    fclose(f);

    double *d_a, *d_final_sum;
    cudaMalloc(&d_a, N * sizeof(double));
    cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_final_sum, sizeof(double));
    cudaMemset(d_final_sum, 0, sizeof(double));

    cudaDeviceSynchronize();
    clock_t start = clock();

    int block = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sum_n<<<block, THREADS_PER_BLOCK>>>(d_a, d_final_sum, N);

    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(final_sum, d_final_sum, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Sum: %lf\n", *final_sum);
    printf("Time taken: %lf seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    cudaFree(d_a);
    cudaFree(d_final_sum);
    free(a);
    free(final_sum);

    return 0;
}