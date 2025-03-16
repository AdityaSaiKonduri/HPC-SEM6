#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>

const int MATRIX_SIZE = 10000;
const double MIN_VALUE = 88.0;
const double MAX_VALUE = 888.0;

void generateMatrix(std::vector<double>& matrix, int size, double minValue, double maxValue) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(minValue, maxValue);

    for (int i = 0; i < size * size; ++i) {
        matrix[i] = dis(gen);
    }
}

void saveMatrixToFile(const std::vector<double>& matrix, const std::string& filename, int size) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    std::ostringstream buffer;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            buffer << matrix[i * size + j] << " ";
        }
        buffer << "\n";
    }
    file << buffer.str();
    file.close();
}

int main() {
    std::vector<double> matrix1(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<double> matrix2(MATRIX_SIZE * MATRIX_SIZE);

    generateMatrix(matrix1, MATRIX_SIZE, MIN_VALUE, MAX_VALUE);
    generateMatrix(matrix2, MATRIX_SIZE, MIN_VALUE, MAX_VALUE);

    saveMatrixToFile(matrix1, "matrix1.txt", MATRIX_SIZE);
    saveMatrixToFile(matrix2, "matrix2.txt", MATRIX_SIZE);

    std::cout << "Matrices generated and saved successfully." << std::endl;
    return 0;
}
