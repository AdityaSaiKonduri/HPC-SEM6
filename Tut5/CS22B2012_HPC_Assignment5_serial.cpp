#include <iostream>
#include <vector>

using namespace std;

void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }
}

vector<vector<int>> addMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<int>> C(rows, vector<int>(cols, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] * B[i][j];
        }
    }

    return C;
}

int main() {
    const float size = 100000;
    vector<vector<int>> A(size, vector<int>(size, 1));
    vector<vector<int>> B(size, vector<int>(size, 2));

    vector<vector<int>> C = addMatrices(A, B);

    cout << "Matrix A:" << endl;
    // printMatrix(A);

    cout << "Matrix B:" << endl;
    // printMatrix(B);

    cout << "Matrix C (A + B):" << endl;
    // printMatrix(C);

    return 0;
}