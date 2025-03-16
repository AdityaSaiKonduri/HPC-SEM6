#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <omp.h>

using namespace std;

double sum_reduction(const vector<double> &data)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+ : sum) shared(data)
    for (size_t i = 0; i < data.size(); i++)
    {
        sum += data[i];
    }
    return sum;
}

double critical_section(const vector<double> &data)
{
    double sum = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for
        for (size_t i = 0; i < data.size(); i++)
        {
            local_sum += data[i];
        }
        #pragma omp critical
        {
            sum += local_sum;
        }
    }
    return sum;
}

double parallelization_fraction(double tp, double t1, double p)
{
    if (t1 == 0.0 || tp == 0.0) return 0.0;  // Avoid division by zero
    return (1 - (tp / t1)) / (1 - (1 / p));
}

int main()
{
    ifstream file("input.txt");
    if (!file)
    {
        cerr << "Error: Could not open input.txt" << endl;
        return 1;
    }

    vector<double> data;
    double value;
    while (file >> value)
    {
        data.push_back(value);
    }
    file.close();

    vector<int> thread_counts = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    double t1_reduction = 0.0, t1_critical = 0.0;
    vector<double> times_reduction, times_critical;
    vector<double> speedup_reduction, speedup_critical;
    vector<double> pf_reduction, pf_critical;

    cout << fixed << setprecision(6);

    for (int num_threads : thread_counts)
    {
        omp_set_num_threads(num_threads);

        double start = omp_get_wtime();
        double sum1 = sum_reduction(data);
        double end = omp_get_wtime();
        double duration_reduction = (end - start) * 1000;
        times_reduction.push_back(duration_reduction);

        start = omp_get_wtime();
        double sum2 = critical_section(data);
        end = omp_get_wtime();
        double duration_critical = (end - start) * 1000;
        times_critical.push_back(duration_critical);

        if (num_threads == 1)
        {
            t1_reduction = duration_reduction;
            t1_critical = duration_critical;
            speedup_reduction.push_back(1.0);
            speedup_critical.push_back(1.0);
            pf_reduction.push_back(0.0);
            pf_critical.push_back(0.0);
        }
        else
        {
            double s_reduction = t1_reduction / duration_reduction;
            double s_critical = t1_critical / duration_critical;
            speedup_reduction.push_back(s_reduction);
            speedup_critical.push_back(s_critical);

            double pf_red = parallelization_fraction(duration_reduction, t1_reduction, num_threads);
            double pf_crit = parallelization_fraction(duration_critical, t1_critical, num_threads);
            pf_reduction.push_back(pf_red);
            pf_critical.push_back(pf_crit);
        }

        // Print the sums
        cout << "Threads: " << num_threads << ", Sum (Reduction): " << sum1 << ", Sum (Critical): " << sum2 << endl;
    }

    // Print execution times table
    cout << "\n=== Execution Times (ms) ===\n";
    cout << setw(10) << "Threads" << setw(20) << "Reduction Time" << setw(20) << "Critical Time\n";
    cout << string(55, '-') << endl;
    for (size_t i = 0; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i]
             << setw(20) << times_reduction[i]
             << setw(20) << times_critical[i] << endl;
    }

    // Print speedup table
    cout << "\n=== Speedup Table ===\n";
    cout << setw(10) << "Threads" << setw(20) << "Speedup (Reduction)" << setw(20) << "Speedup (Critical)\n";
    cout << string(55, '-') << endl;
    for (size_t i = 0; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i]
             << setw(20) << speedup_reduction[i]
             << setw(20) << speedup_critical[i] << endl;
    }

    // Print parallelization fraction table
    cout << "\n=== Parallelization Fraction Table ===\n";
    cout << setw(10) << "Threads" << setw(25) << "P. Fraction (Reduction)" << setw(25) << "P. Fraction (Critical)\n";
    cout << string(65, '-') << endl;
    for (size_t i = 1; i < thread_counts.size(); i++)
    {
        cout << setw(10) << thread_counts[i]
             << setw(25) << pf_reduction[i]
             << setw(25) << pf_critical[i] << endl;
    }

    return 0;
}
