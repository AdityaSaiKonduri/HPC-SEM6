#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>
using namespace std;

int main() {
    ifstream infile("input.txt");
    vector<double> numbers;
    double x;

    // Load all numbers into the array
    while (infile >> x) {
        numbers.push_back(x);
    }

    double sum = 0.0;
    
    // Measure time for summation only
    auto start = chrono::high_resolution_clock::now();

    for (double num : numbers) {
        sum += num;
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Serial Sum = " << fixed << setprecision(6) << sum << endl;
    cout << "Serial Time = " << elapsed.count() << " seconds" << endl;
    
    return 0;
}