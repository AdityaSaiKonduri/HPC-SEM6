#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <omp.h>

using namespace std;

const int SIZE = 10000;

void generateMatrix(vector<vector<double>> &matrix)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            matrix[i][j] = 1.0 + ((double)rand() / RAND_MAX) * 99.0;
        }
    }
}

void addMatricesSerial(const vector<vector<double>> &matrix1, const vector<vector<double>> &matrix2, vector<vector<double>> &result)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

void addMatricesParallel(const vector<vector<double>> &matrix1, const vector<vector<double>> &matrix2, vector<vector<double>> &result)
{
    #pragma omp parallel for
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
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
    double t1_add_serial = 0.0;
    double t1_add_parallel = 0.0;
    vector<double> times_add_serial;
    vector<double> times_add_parallel;
    vector<double> speedup_add;
    vector<double> pf_add;

    cout << fixed << setprecision(6);

    // Serial addition
    double start = omp_get_wtime();
    addMatricesSerial(matrix1, matrix2, result);
    double end = omp_get_wtime();
    t1_add_serial = (end - start) * 1000;
    times_add_serial.push_back(t1_add_serial);

    // Parallel addition
    for (int num_threads : thread_counts)
    {
        omp_set_num_threads(num_threads);

        double start = omp_get_wtime();
        addMatricesParallel(matrix1, matrix2, result);
        double end = omp_get_wtime();
        double duration_add_parallel = (end - start) * 1000;
        times_add_parallel.push_back(duration_add_parallel);

        if (num_threads == 1)
        {
            t1_add_parallel = duration_add_parallel;
            speedup_add.push_back(1.0);
            pf_add.push_back(0.0);
        }
        else
        {
            speedup_add.push_back(calculateSpeedup(t1_add_serial, duration_add_parallel));
            pf_add.push_back(calculateParallelFraction(t1_add_serial, duration_add_parallel, num_threads));
        }
    }

    // Print execution times table
    cout << "\n=== Execution Times (ms) ===\n";
    cout << setw(10) << "Threads" << setw(25) << "Serial Addition Time" << setw(25) << "Parallel Addition Time\n";
    cout << string(55, '-') << endl;
    cout << setw(10) << 1
         << setw(20) << times_add_serial[0]
         << setw(20) << times_add_parallel[0] << endl;
    for (size_t i = 1; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i]
             << setw(20) << "-"
             << setw(20) << times_add_parallel[i] << endl;
    }

    // Print speedup table
    cout << "\n=== Speedup Table ===\n";
    cout << setw(10) << "Threads" << setw(30) << "Speedup (Parallel Addition)\n";
    cout << string(30, '-') << endl;
    for (size_t i = 0; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i]
             << setw(20) << speedup_add[i] << endl;
    }

    // Print parallelization fraction table
    cout << "\n=== Parallelization Fraction Table ===\n";
    cout << setw(10) << "Threads" << setw(45) << "P. Fraction (Parallel Addition)\n";
    cout << string(35, '-') << endl;
    cout << setw(10) << 1
         << setw(25) << 0.0 << endl;
    for (size_t i = 1; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i]
             << setw(25) << pf_add[i] << endl;
    }

    return 0;
}