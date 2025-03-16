#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#define N 5000
#define BLOCK_SIZE 32

__global__ void matrixMultiply(const double *A, const double *B, double *C, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width)
    {
        double sum = 0.0;
        for (int k = 0; k < width; k++)
        {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

void loadMatrix(const char *filename, double *matrix, int size)
{
    std::ifstream file(filename);
    for (int i = 0; i < size * size; ++i)
    {
        file >> matrix[i];
    }
    file.close();
}

void printMatrixSection(const double *matrix, int width, int sectionSize)
{
    for (int i = 0; i < sectionSize; ++i)
    {
        for (int j = 0; j < sectionSize; ++j)
        {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    double *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(double);

    h_A = (double *)malloc(size);
    h_B = (double *)malloc(size);
    h_C = (double *)malloc(size);

    loadMatrix("matrix1.txt", h_A, N);
    loadMatrix("matrix2.txt", h_B, N);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Matrix Multiplication Execution Time: " << milliseconds / 1000.0 << " seconds\n";

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "3x3 section of the result matrix:" << std::endl;
    printMatrixSection(h_C, N, 3);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
