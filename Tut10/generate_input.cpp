#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>  // For setprecision

const size_t NUM_COUNT = 10'000'000; // 10 million numbers

void generate_and_save(const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());  // 64-bit Mersenne Twister
    std::uniform_real_distribution<double> dist(8888.0, 10000.0);

    file << std::scientific << std::setprecision(10);  // Ensures full precision of double

    for (size_t i = 0; i < NUM_COUNT; ++i) {
        file << dist(gen) << "\n";  // Write number with high precision
    }

    file.close();
    std::cout << "Successfully written " << NUM_COUNT << " double values to " << filename << std::endl;
}

int main() {
    generate_and_save("file1.txt");
    generate_and_save("file2.txt");

    return 0;
}
