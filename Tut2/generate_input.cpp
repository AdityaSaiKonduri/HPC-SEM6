#include <iostream>
#include <fstream>
#include <random>

#define N 10000000  // 1 Million

int main() {
    std::ofstream file("input.txt");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1.0, 1000.0);

    for (size_t i = 0; i < N; ++i) {
        file << dis(gen) << "\n";
    }

    file.close();
    std::cout << "Generated input.txt with " << N << " double precision numbers.\n";
    return 0;
}
