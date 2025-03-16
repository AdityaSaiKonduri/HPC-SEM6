#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>

using namespace std;

struct GameState {
    vector<vector<int>> grid;
    int gCurr;
    int heuristicScore;
    double logMax;
    double logSecondMax;

    GameState(const vector<vector<int>>& initialGrid) : 
        grid(initialGrid),
        gCurr(0),
        heuristicScore(0),
        logMax(0),
        logSecondMax(0) {}

    // Custom comparison for sorting
    bool operator<(const GameState& other) const {
        if (heuristicScore != other.heuristicScore)
            return heuristicScore > other.heuristicScore;
        if (logMax != other.logMax)
            return logMax > other.logMax;
        return logSecondMax > other.logSecondMax;
    }
};

void randomGenerate(vector<vector<int>>& grid) {
    vector<pair<int, int>> emptyTiles;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (grid[i][j] == 0) {
                emptyTiles.push_back({i, j});
            }
        }
    }

    if (!emptyTiles.empty()) {
        int randomIndex = rand() % emptyTiles.size();
        int x = emptyTiles[randomIndex].first;
        int y = emptyTiles[randomIndex].second;
        grid[x][y] = (rand() % 10 == 0) ? 4 : 2;
    }
}

vector<int> mergeTiles(vector<int> row) {
    vector<int> mergedRow;
    for (size_t i = 0; i < row.size(); i++) {
        if (i + 1 < row.size() && row[i] == row[i + 1] && row[i] != 0) {
            mergedRow.push_back(row[i] * 2);
            i++;
        }
        else if (row[i] != 0) {
            mergedRow.push_back(row[i]);
        }
    }
    while (mergedRow.size() < 4) {
        mergedRow.push_back(0);
    }
    return mergedRow;
}

vector<vector<int>> moveLeft(const vector<vector<int>>& grid) {
    vector<vector<int>> newGrid(4, vector<int>(4, 0));
    for (int i = 0; i < 4; ++i) {
        vector<int> tempRow;
        for (int j = 0; j < 4; ++j) {
            if (grid[i][j] != 0) {
                tempRow.push_back(grid[i][j]);
            }
        }
        tempRow = mergeTiles(tempRow);
        for (int j = 0; j < 4; ++j) {
            newGrid[i][j] = tempRow[j];
        }
    }
    return newGrid;
}

vector<vector<int>> moveRight(const vector<vector<int>>& grid) {
    vector<vector<int>> newGrid(4, vector<int>(4, 0));
    for (int i = 0; i < 4; ++i) {
        vector<int> tempRow;
        for (int j = 3; j >= 0; --j) {
            if (grid[i][j] != 0) {
                tempRow.push_back(grid[i][j]);
            }
        }
        tempRow = mergeTiles(tempRow);
        for (int j = 3, k = 0; j >= 0; --j, ++k) {
            newGrid[i][j] = tempRow[k];
        }
    }
    return newGrid;
}

vector<vector<int>> moveUp(const vector<vector<int>>& grid) {
    vector<vector<int>> newGrid(4, vector<int>(4, 0));
    for (int j = 0; j < 4; j++) {
        vector<int> tempCol;
        for (int i = 0; i < 4; i++) {
            if (grid[i][j] != 0) {
                tempCol.push_back(grid[i][j]);
            }
        }
        tempCol = mergeTiles(tempCol);
        for (int i = 0; i < 4; i++) {
            newGrid[i][j] = tempCol[i];
        }
    }
    return newGrid;
}

vector<vector<int>> moveDown(const vector<vector<int>>& grid) {
    vector<vector<int>> newGrid(4, vector<int>(4, 0));
    for (int j = 0; j < 4; j++) {
        vector<int> tempCol;
        for (int i = 3; i >= 0; i--) {
            if (grid[i][j] != 0) {
                tempCol.push_back(grid[i][j]);
            }
        }
        tempCol = mergeTiles(tempCol);
        for (int i = 3, k = 0; i >= 0; i--, k++) {
            newGrid[i][j] = tempCol[k];
        }
    }
    return newGrid;
}

void displayGrid(const vector<vector<int>>& grid) {
    for (const auto& row : grid) {
        for (int tile : row) {
            cout << tile << "\t";
        }
        cout << endl;
    }
}

vector<GameState> generateNextStates(const GameState& currentState) {
    vector<GameState> nextStates;
    vector<vector<vector<int>> (*)(const vector<vector<int>>&)> moveFunctions = {
        moveLeft, moveRight, moveUp, moveDown
    };

    for (auto moveFunc : moveFunctions) {
        vector<vector<int>> movedGrid = moveFunc(currentState.grid);

        if (movedGrid != currentState.grid) {
            int m = 0, k = 0;
            for (const auto& row : movedGrid) {
                for (int j : row) {
                    if (j > m) {
                        k = m;
                        m = j;
                    }
                    else if (j > k) {
                        k = j;
                    }
                }
            }

            randomGenerate(movedGrid);

            int h = 0;
            for (const auto& row : movedGrid) {
                h += accumulate(row.begin(), row.end(), 0);
            }

            GameState nextState(movedGrid);
            nextState.gCurr = currentState.gCurr + 1;
            nextState.heuristicScore = h;
            nextState.logMax = (m > 0) ? log2(m) : 0;
            nextState.logSecondMax = (k > 0) ? log2(k) : 0;

            nextStates.push_back(nextState);
        }
    }
    return nextStates;
}

tuple<GameState, int> aStarAlgorithm(GameState& initialState) {
    vector<GameState> opened;
    vector<GameState> closed;
    int count = 1;

    opened.push_back(initialState);

    while (!opened.empty()) {
        sort(opened.begin(), opened.end());
        
        GameState currentState = opened.front();
        opened.erase(opened.begin());
        closed.push_back(currentState);

        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                if (currentState.grid[row][col] == 2048) {
                    return make_tuple(currentState, count);
                }
            }
        }

        count++;

        if (count >= 2000) {
            return make_tuple(currentState, count);
        }

        vector<GameState> nextStates = generateNextStates(currentState);
        opened.insert(opened.end(), nextStates.begin(), nextStates.end());
    }

    return make_tuple(initialState, count);
}

int main() {
    srand(time(0));
    vector<vector<int>> initialGrid(4, vector<int>(4, 0));
    randomGenerate(initialGrid);
    randomGenerate(initialGrid);

    GameState initialState(initialGrid);

    cout << "Initial Grid:\n";
    displayGrid(initialState.grid);

    GameState result(initialState);
    int iterations;
    tie(result, iterations) = aStarAlgorithm(initialState);

    cout << "Final Grid:\n";
    displayGrid(result.grid);
    cout << "Number of iterations: " << iterations << endl;

    return 0;
}