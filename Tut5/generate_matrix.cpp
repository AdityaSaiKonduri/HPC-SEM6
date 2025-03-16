#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

const int MATRIX_SIZE = 10000;
const int MIN_VALUE = 1;
const int MAX_VALUE = 100;

int main() {
    std::srand(std::time(0)); // Seed for random number generation

    // Create a 10000 x 10000 matrix
    std::vector<std::vector<int>> matrix(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));

    // Fill the matrix with random values between 1 and 100
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            matrix[i][j] = MIN_VALUE + std::rand() % (MAX_VALUE - MIN_VALUE + 1);
        }
    }

    // Optionally, print the matrix (commented out to avoid excessive output)
    /*
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    */

    return 0;
}