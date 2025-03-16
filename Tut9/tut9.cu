#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 1024

#define N 10000000

__global__ void vector_add(double *a, double *b, double *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        c[index] = a[index] + b[index];
    }
}

__global__ void vector_mul(double *a, double *b, double *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        c[index] = a[index] * b[index];
    }
}

void vector_mul_serial(double *a, double *b, double *c){
    for (int i = 0; i < N; i++){
        c[i] = a[i] * b[i];
    }
}

void vector_add_serial(double *a, double *b, double *c){
    for (int i = 0; i < N; i++){
        c[i] = a[i] + b[i];
    }
}

double calculate_speedup(double t1, double tn){
    return t1 / tn;
}

int main(){
    FILE *input1 = fopen("file1.txt", "r");
    FILE *input2 = fopen("file2.txt", "r");

    double *a = (double*)malloc(N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *c_add = (double*)malloc(N * sizeof(double));
    double *s_add = (double*)malloc(N * sizeof(double));

    // for multiplication
    double *c_mul = (double*)malloc(N * sizeof(double));
    double *s_mul = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++){
        fscanf(input1, "%lf", &a[i]);
        fscanf(input2, "%lf", &b[i]);
    }
    
    fclose(input1);
    fclose(input2);
    printf("=============Addition Part================\n");
    //run the serial code
    clock_t start = clock();
    vector_add_serial(a, b, s_add);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken by serial add code: %lf\n", time_taken);

    // making copies on device
    double *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(double));
    cudaMalloc((void**)&d_b, N * sizeof(double));
    cudaMalloc((void**)&d_c, N * sizeof(double));

    //copying data to the device copies
    cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, N * sizeof(double));
    
    //run the addition kernel
    cudaDeviceSynchronize();
    clock_t start_gpu = clock();
    vector_add<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    clock_t end_gpu = clock();
    double time_taken_gpu = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
    printf("Time taken by GPU add code: %lf\n", time_taken_gpu);
    //copy result from device to host
    cudaMemcpy(c_add, d_c, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Comparing Serial and parallel addition\n");
    for (int i = 0; i < 10; i++){
        printf("%lf %lf\n", s_add[i], c_add[i]);
    }

    double add_speedup = calculate_speedup(time_taken, time_taken_gpu);

    printf("Speedup: %lf\n\n\n", add_speedup);

    // Multiplication Part
    printf("=============Multiplication Part================\n");

    //serial code mul
    clock_t start_mul = clock();
    vector_mul_serial(a, b, s_mul);
    clock_t end_mul = clock();
    double time_taken_mul = (double)(end_mul - start_mul) / CLOCKS_PER_SEC;
    printf("Time taken by serial code: %lf\n", time_taken_mul);

    //CUDA mul
    cudaMemset(d_c, 0, N * sizeof(double));
    cudaDeviceSynchronize();
    clock_t start_gpu_mul = clock();
    vector_mul<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    clock_t end_gpu_mul = clock();
    double time_taken_gpu_mul = (double)(end_gpu_mul - start_gpu_mul) / CLOCKS_PER_SEC;
    printf("Time taken by GPU code: %lf\n", time_taken_gpu_mul);

    cudaMemcpy(c_mul, d_c, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Comparing serial and parallel multiplication\n");
    for (int i = 0; i < 10; i++){
        printf("%lf %lf\n", s_mul[i], c_mul[i]);
    }

    double mul_speedup = calculate_speedup(time_taken_mul, time_taken_gpu_mul);
    printf("Multiplication Speedup: %lf\n", mul_speedup);

    // free memory

    free(a);
    free(b);
    free(c_add);
    free(s_add);
    free(c_mul);
    free(s_mul);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}