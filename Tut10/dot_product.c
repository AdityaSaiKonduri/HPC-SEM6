#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 10000000

int main() {
    FILE *file1, *file2;
    double *vector1, *vector2;
    double dot_product = 0.0;

    // Allocate memory for the vectors
    vector1 = (double *)malloc(SIZE * sizeof(double));
    vector2 = (double *)malloc(SIZE * sizeof(double));

    if (vector1 == NULL || vector2 == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Open the files
    file1 = fopen("file1.txt", "r");
    file2 = fopen("file2.txt", "r");

    if (file1 == NULL || file2 == NULL) {
        printf("Error opening files\n");
        free(vector1);
        free(vector2);
        return 1;
    }

    // Read the numbers from the files
    for (int i = 0; i < SIZE; i++) {
        if (fscanf(file1, "%lf", &vector1[i]) != 1 || fscanf(file2, "%lf", &vector2[i]) != 1) {
            printf("Error reading numbers from files\n");
            fclose(file1);
            fclose(file2);
            free(vector1);
            free(vector2);
            return 1;
        }
    }

    // Close the files
    fclose(file1);
    fclose(file2);

    clock_t start = clock();

    // Compute the dot product
    for (int i = 0; i < SIZE; i++) {
        dot_product += vector1[i] * vector2[i];
    }

    clock_t end = clock();
    // Print the result
    printf("Dot product: %lf\n", dot_product);
    printf("Time taken: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
    // Free the allocated memory
    free(vector1);
    free(vector2);

    return 0;
}