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

void matrixAdd(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int size) {
    for (int i = 0; i < size * size; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    std::vector<double> A(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<double> B(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<double> C(MATRIX_SIZE * MATRIX_SIZE);

    loadMatrix("matrix1.txt", A, MATRIX_SIZE);
    loadMatrix("matrix2.txt", B, MATRIX_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    
    matrixAdd(A, B, C, MATRIX_SIZE);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Serial Addition Execution Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
