#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <omp.h>

using namespace std;

const size_t N = 1000000;

void readData(const string &filename, vector<double> &data)
{
    ifstream file(filename);
    double value;
    while (file >> value)
    {
        data.push_back(value);
    }
    file.close();
}

void parallelVectorAddition(const vector<double> &A, const vector<double> &B, vector<double> &C, double &sum)
{
    sum = 0.0;
    #pragma omp parallel for
    for (size_t i = 0; i < A.size(); i++)
    {
        C[i] = A[i] + B[i];
        // sum += C[i];
    }
}

void parallelVectorMultiplication(const vector<double> &A, const vector<double> &B, vector<double> &C, double &prod)
{
    prod = 0.0;
    #pragma omp parallel for
    for (size_t i = 0; i < A.size(); i++)
    {
        C[i] = A[i] * B[i];
        // prod += C[i];
    }
}

// Function to calculate Speedup
double calculateSpeedup(double t1, double tn)
{
    return t1 / tn;
}

// Function to calculate Parallelization Fraction (PF)
double calculateParallelFraction(double t1, double tn, int num_threads)
{
    if (num_threads == 1)
        return 0.0;

    return (1 - (tn / t1)) / (1 - (1.0 / num_threads));
}

int main()
{
    vector<double> A, B, C(N);

    readData("input1.txt", A);
    readData("input2.txt", B);

    vector<int> thread_counts = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    double t1_add = 0.0, t1_mul = 0.0;
    vector<double> times_add, times_mul;
    vector<double> speedup_add, speedup_mul;
    vector<double> pf_add, pf_mul;
    vector<double> sums, prods;

    cout << fixed << setprecision(6);

    for (int num_threads : thread_counts)
    {
        omp_set_num_threads(num_threads);

        double sum_add = 0.0, prod_mul = 0.0;

        double start = omp_get_wtime();
        parallelVectorAddition(A, B, C, sum_add);
        double end = omp_get_wtime();
        double duration_add = (end - start) * 1000;
        times_add.push_back(duration_add);
        sums.push_back(sum_add);

        start = omp_get_wtime();
        parallelVectorMultiplication(A, B, C, prod_mul);
        end = omp_get_wtime();
        double duration_mul = (end - start) * 1000;
        times_mul.push_back(duration_mul);
        prods.push_back(prod_mul);

        if (num_threads == 1)
        {
            t1_add = duration_add;
            t1_mul = duration_mul;
            speedup_add.push_back(1.0);
            speedup_mul.push_back(1.0);
            pf_add.push_back(0.0);
            pf_mul.push_back(0.0);
        }
        else
        {
            speedup_add.push_back(calculateSpeedup(t1_add, duration_add));
            speedup_mul.push_back(calculateSpeedup(t1_mul, duration_mul));

            pf_add.push_back(calculateParallelFraction(t1_add, duration_add, num_threads));
            pf_mul.push_back(calculateParallelFraction(t1_mul, duration_mul, num_threads));
        }
    }

    // Print execution times table
    cout << "\n=== Execution Times (ms) ===\n";
    cout << setw(10) << "Threads" << setw(20) << "Addition Time" << setw(20) << "Multiplication Time\n";
    cout << string(55, '-') << endl;
    for (size_t i = 0; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i]
             << setw(20) << times_add[i]
             << setw(20) << times_mul[i] << endl;
    }

    // Print sums and products
    // cout << "\n=== Sums and Products ===\n";
    // cout << setw(10) << "Threads" << setw(25) << "Sum (Addition)" << setw(25) << "Sum (Multiplication)\n";
    // cout << string(65, '-') << endl;
    // for (size_t i = 0; i < thread_counts.size(); i++)
    // {
    //     cout << setw(10) << thread_counts[i]
    //          << setw(25) << sums[i]
    //          << setw(25) << prods[i] << endl;
    // }

    // Print speedup table
    cout << "\n=== Speedup Table ===\n";
    cout << setw(10) << "Threads" << setw(20) << "Speedup (Addition)" << setw(20) << "Speedup (Multiplication)\n";
    cout << string(55, '-') << endl;
    for (size_t i = 0; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i]
             << setw(20) << speedup_add[i]
             << setw(20) << speedup_mul[i] << endl;
    }

    // Print parallelization fraction table
    cout << "\n=== Parallelization Fraction Table ===\n";
    cout << setw(10) << "Threads" << setw(25) << "P. Fraction (Addition)" << setw(25) << "P. Fraction (Multiplication)\n";
    cout << string(65, '-') << endl;
    for (size_t i = 1; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i]
             << setw(25) << pf_add[i]
             << setw(25) << pf_mul[i] << endl;
    }

    return 0;
}
