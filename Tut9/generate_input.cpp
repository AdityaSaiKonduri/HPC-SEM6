#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>  
using namespace std;

int main() {
    const int N = 8000000;  
    ofstream outfile1("file1.txt");
    ofstream outfile2("file2.txt");

    if (!outfile1.is_open() || !outfile2.is_open()) {
        cerr << "Error: Unable to open one or both files for writing." << endl;
        return 1;
    }

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<double> dis(88.0, 8888.0);

    outfile1 << fixed << setprecision(8);
    outfile2 << fixed << setprecision(8);

    for (int i = 0; i < N; ++i) {
        double number = dis(gen);
        outfile1 << number << "\n";
        outfile2 << number << "\n";
    }

    outfile1.close();
    outfile2.close();
    cout << "Generated " << N << " double-precision numbers in file1.txt and file2.txt with at least 6 decimal digits." << endl;
    return 0;
}