#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <omp.h>

using namespace std;

const int SIZE = 1000; // Reduced size for multiplication

void generateMatrix(vector<vector<double>> &matrix)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            matrix[i][j] = static_cast<double>(rand() % 100 + 1);
        }
    }
}

void multiplyMatricesSerial(const vector<vector<double>> &matrix1, const vector<vector<double>> &matrix2, vector<vector<double>> &result)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            result[i][j] = 0.0;
            for (int k = 0; k < SIZE; ++k)
            {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

void multiplyMatricesParallel(const vector<vector<double>> &matrix1, const vector<vector<double>> &matrix2, vector<vector<double>> &result)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            result[i][j] = 0.0;
            for (int k = 0; k < SIZE; ++k)
            {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

double calculateSpeedup(double t1, double tn)
{
    return t1 / tn;
}

double calculateParallelFraction(double t1, double tn, int num_threads)
{
    if (num_threads == 1)
        return 0.0; 
    return (1 - (tn / t1)) / (1 - (1.0 / num_threads));
}

int main()
{
    srand(time(0));

    vector<vector<double>> matrix1(SIZE, vector<double>(SIZE));
    vector<vector<double>> matrix2(SIZE, vector<double>(SIZE));
    vector<vector<double>> result(SIZE, vector<double>(SIZE));

    generateMatrix(matrix1);
    generateMatrix(matrix2);

    vector<int> thread_counts = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    double t1_mult_serial = 0.0;
    double t1_mult_parallel = 0.0;
    vector<double> times_mult_serial;
    vector<double> times_mult_parallel;
    vector<double> speedup_mult;
    vector<double> pf_mult;

    cout << fixed << setprecision(6);

    // Serial multiplication
    double start = omp_get_wtime();
    multiplyMatricesSerial(matrix1, matrix2, result);
    double end = omp_get_wtime();
    t1_mult_serial = (end - start) * 1000;
    times_mult_serial.push_back(t1_mult_serial);

    // Parallel multiplication
    for (int num_threads : thread_counts)
    {
        omp_set_num_threads(num_threads);

        start = omp_get_wtime();
        multiplyMatricesParallel(matrix1, matrix2, result);
        end = omp_get_wtime();
        double duration_mult_parallel = (end - start) * 1000;
        times_mult_parallel.push_back(duration_mult_parallel);

        if (num_threads == 1)
        {
            t1_mult_parallel = duration_mult_parallel;
            speedup_mult.push_back(1.0);
            pf_mult.push_back(0.0);
        }
        else
        {
            speedup_mult.push_back(calculateSpeedup(t1_mult_serial, duration_mult_parallel));
            pf_mult.push_back(calculateParallelFraction(t1_mult_serial, duration_mult_parallel, num_threads));
        }
    }

    // Print execution times table
    cout << "\n=== Execution Times (ms) ===\n";
    cout << setw(10) << "Threads" << setw(25) << "Serial Mult. Time"
         << setw(25) << "Parallel Mult. Time\n";
    cout << string(60, '-') << endl;
    cout << setw(10) << 1 << setw(25) << times_mult_serial[0]
         << setw(25) << times_mult_parallel[0] << endl;

    for (size_t i = 1; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i] << setw(25) << "-"
             << setw(25) << times_mult_parallel[i] << endl;
    }

    // Print speedup table
    cout << "\n=== Speedup Table ===\n";
    cout << setw(10) << "Threads" << setw(30) << "Speedup (Parallel Mult.)\n";
    cout << string(40, '-') << endl;
    for (size_t i = 0; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i] << setw(30) << speedup_mult[i] << endl;
    }

    // Print parallelization fraction table
    cout << "\n=== Parallelization Fraction Table ===\n";
    cout << setw(10) << "Threads" << setw(45) << "P. Fraction (Parallel Mult.)\n";
    cout << string(55, '-') << endl;
    for (size_t i = 0; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i] << setw(45) << pf_mult[i] << endl;
    }

    return 0;
}