#include <bits/stdc++.h>
#include <omp.h>

#define N 1000000

using namespace std;

// Function to compute the dot product
double dot_product(vector<double> &A, vector<double> &B)
{
    double sum = 0.0;
    #pragma omp parallel
    {
        double prod = 0.0;
        #pragma omp for
        for (int i = 0; i < N; i++)
        {
            prod = A[i] * B[i];
        }
        #pragma omp critical
        sum += prod;
    }
    return sum;
}

// Function to compute parallelization fraction
double parallelization_fraction(double tp, double t1, double p)
{
    if (t1 == 0.0 || tp == 0.0) return 0.0;
    if (p == 1) return 1;  // Avoid division by zero
    return (1 - (tp / t1)) / (1 - (1 / p));
}

int main()
{
    vector<double> A(N), B(N);

    // Read data from files
    ifstream file1("input1.txt"), file2("input2.txt");
    for (int i = 0; i < N; i++)
    {
        file1 >> A[i];
        file2 >> B[i];
    }
    file1.close();
    file2.close();

    vector<int> thread_counts = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};

    double T1, dot_product_result;
    vector<double> execution_times, speedups, parallel_fractions;

    cout << "=== Execution Time Table (ms) ===\n";
    cout << "Threads   Execution Time\n";
    cout << "-----------------------------\n";

    for (int num_threads : thread_counts)
    {
        omp_set_num_threads(num_threads);

        double start = omp_get_wtime();
        dot_product_result = dot_product(A, B);
        double end = omp_get_wtime();
        double duration = (end - start) * 1000; // Convert to milliseconds

        execution_times.push_back(duration);

        if (num_threads == 1)
        {
            T1 = duration; // Save execution time for 1 thread
        }

        cout << setw(5) << num_threads << "      " << fixed << setprecision(6) << duration << " ms\n";
    }

    cout << "\nDot Product Result: " << fixed << setprecision(6) << dot_product_result << "\n";

    cout << "\n=== Speedup Table ===\n";
    cout << "Threads   Speedup\n";
    cout << "---------------------\n";

    for (size_t i = 0; i < thread_counts.size(); i++)
    {
        double speedup = T1 / execution_times[i];
        speedups.push_back(speedup);
        cout << setw(5) << thread_counts[i] << "      " << fixed << setprecision(6) << speedup << "\n";
    }

    cout << "\n=== Parallelization Fraction Table ===\n";
    cout << "Threads   P. Fraction\n";
    cout << "----------------------\n";

    for (size_t i = 0; i < thread_counts.size(); i++)
    {
        double pf = parallelization_fraction(execution_times[i], T1, thread_counts[i]);
        parallel_fractions.push_back(pf);

        cout << setw(5) << thread_counts[i] << "      " << fixed << setprecision(6) << pf << "\n";
    }

    return 0;
}
