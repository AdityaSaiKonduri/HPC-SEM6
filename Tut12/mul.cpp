#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

const int MATRIX_SIZE = 5000;

void loadMatrix(const std::string& filename, std::vector<double>& matrix, int size) {
    std::ifstream file(filename);
    for (int i = 0; i < size * size; ++i) {
        file >> matrix[i];
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

void printMatrixSection(const std::vector<double>& matrix, int size, int sectionSize) {
    for (int i = 0; i < sectionSize; ++i) {
        for (int j = 0; j < sectionSize; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::vector<double> A(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<double> B(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<double> C(MATRIX_SIZE * MATRIX_SIZE);

    loadMatrix("matrix1.txt", A, MATRIX_SIZE);
    loadMatrix("matrix2.txt", B, MATRIX_SIZE);

    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, MATRIX_SIZE);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Serial Multiplication Execution Time: " << elapsed.count() << " seconds" << std::endl;

    std::cout << "3x3 section of the resultant matrix:" << std::endl;
    printMatrixSection(C, MATRIX_SIZE, 3);

    return 0;
}
