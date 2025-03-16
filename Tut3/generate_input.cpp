#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

const size_t N = 1000000;  // 1 Million elements

void generateFile(const string &filename) {
    ofstream file(filename);
    srand(time(0));  // Seed for random numbers

    for (size_t i = 0; i < N; i++) {
        double value = ((double)rand() / RAND_MAX) * 10000000.0000000;  // Random double [0,1000]
        file << value << endl;
    }
    file.close();
}

int main() {
    generateFile("input1.txt");
    generateFile("input2.txt");
    cout << "Input files generated!" << endl;
    return 0;
}
