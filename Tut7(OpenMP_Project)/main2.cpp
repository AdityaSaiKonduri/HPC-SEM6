#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

const int GRID_SIZE = 4;
const int MAX_NEXT_STATES = 4;

// Helper function to merge tiles in a single array in-place
void mergeTiles(int row[], int size)
{
    int temp[GRID_SIZE] = {0};
    int tempIndex = 0;

    // First collect non-zero elements
    for (int i = 0; i < size; i++)
    {
        if (row[i] != 0)
        {
            temp[tempIndex++] = row[i];
        }
    }

    // Reset original array
    for (int i = 0; i < size; i++)
    {
        row[i] = 0;
    }

    // Merge adjacent equal numbers
    int writeIndex = 0;
    for (int i = 0; i < tempIndex; i++)
    {
        if (i + 1 < tempIndex && temp[i] == temp[i + 1])
        {
            row[writeIndex++] = temp[i] * 2;
            i++;
        }
        else
        {
            row[writeIndex++] = temp[i];
        }
    }
}

void randomGenerate(int grid[GRID_SIZE][GRID_SIZE])
{
    int row = (rand() + 1) % GRID_SIZE;
    int col = (rand() + 1) % GRID_SIZE;
    while (grid[row][col] != 0)
    {
        row = rand() % GRID_SIZE;
        col = rand() % GRID_SIZE;
    }
    grid[row][col] = 2;
}

void moveLeft(int grid[GRID_SIZE][GRID_SIZE])
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        mergeTiles(grid[i], GRID_SIZE);
    }
}

void moveRight(int grid[GRID_SIZE][GRID_SIZE])
{
    // First reverse each row
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE / 2; j++)
        {
            int temp = grid[i][j];
            grid[i][j] = grid[i][GRID_SIZE - 1 - j];
            grid[i][GRID_SIZE - 1 - j] = temp;
        }
    }

    moveLeft(grid);

    // Reverse back
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE / 2; j++)
        {
            int temp = grid[i][j];
            grid[i][j] = grid[i][GRID_SIZE - 1 - j];
            grid[i][GRID_SIZE - 1 - j] = temp;
        }
    }
}

void moveUp(int grid[GRID_SIZE][GRID_SIZE])
{
    // Transpose the grid
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = i + 1; j < GRID_SIZE; j++)
        {
            int temp = grid[i][j];
            grid[i][j] = grid[j][i];
            grid[j][i] = temp;
        }
    }

    moveLeft(grid);

    // Transpose back
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = i + 1; j < GRID_SIZE; j++)
        {
            int temp = grid[i][j];
            grid[i][j] = grid[j][i];
            grid[j][i] = temp;
        }
    }
}

void moveDown(int grid[GRID_SIZE][GRID_SIZE])
{
    // Transpose the grid
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = i + 1; j < GRID_SIZE; j++)
        {
            int temp = grid[i][j];
            grid[i][j] = grid[j][i];
            grid[j][i] = temp;
        }
    }

    moveRight(grid);

    // Transpose back
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = i + 1; j < GRID_SIZE; j++)
        {
            int temp = grid[i][j];
            grid[i][j] = grid[j][i];
            grid[j][i] = temp;
        }
    }
}

void displayGrid(const int grid[GRID_SIZE][GRID_SIZE])
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            cout << grid[i][j] << '\t';
        }
        cout << '\n';
    }
    cout << '\n';
}

struct GameState
{
    int grid[GRID_SIZE][GRID_SIZE];
    int gCurr;
    int heuristicScore;
    double logMax;
    double logSecondMax;

    GameState()
    {
        gCurr = 0;
        heuristicScore = 0;
        logMax = 0;
        logSecondMax = 0;
        for (int i = 0; i < GRID_SIZE; i++)
        {
            for (int j = 0; j < GRID_SIZE; j++)
            {
                grid[i][j] = 0;
            }
        }
    }

    bool operator<(const GameState &other) const
    {
        if (heuristicScore != other.heuristicScore)
        {
            return heuristicScore < other.heuristicScore;
        }
        if (logMax != other.logMax)
        {
            return logMax < other.logMax;
        }
        return logSecondMax < other.logSecondMax;
    }
};

bool equalGrid(const int grid1[GRID_SIZE][GRID_SIZE], const int grid2[GRID_SIZE][GRID_SIZE])
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            if (grid1[i][j] != grid2[i][j])
            {
                return false;
            }
        }
    }
    return true;
}

void copyGrid(int dest[GRID_SIZE][GRID_SIZE], const int src[GRID_SIZE][GRID_SIZE])
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            dest[i][j] = src[i][j];
        }
    }
}

int generateNextStates(const GameState &currentState, GameState nextStates[])
{
    int stateCount = 0;
    void (*moveFunctions[4])(int[GRID_SIZE][GRID_SIZE]) = {moveLeft, moveRight, moveUp, moveDown};

    for (int func = 0; func < 4; func++)
    {
        GameState nextState;
        int tempGrid[GRID_SIZE][GRID_SIZE];
        copyGrid(tempGrid, currentState.grid);
        moveFunctions[func](tempGrid);

        if (!equalGrid(tempGrid, currentState.grid))
        {
            randomGenerate(tempGrid);

            // finding max and second max tile values
            int maxTile = 0, secondMaxTile = 0;
            for (int i = 0; i < GRID_SIZE; i++)
            {
                for (int j = 0; j < GRID_SIZE; j++)
                {
                    if (tempGrid[i][j] > maxTile)
                    {
                        secondMaxTile = maxTile;
                        maxTile = tempGrid[i][j];
                    }
                    else if (tempGrid[i][j] > secondMaxTile && tempGrid[i][j] != maxTile)
                    {
                        secondMaxTile = tempGrid[i][j];
                    }
                }
            }

            // finding heuristic as sum of all tiles
            int heuristic = 0;
            for (int i = 0; i < GRID_SIZE; i++)
            {
                for (int j = 0; j < GRID_SIZE; j++)
                {
                    heuristic += tempGrid[i][j];
                }
            }

            double logMax = (maxTile > 0) ? log2(maxTile) : 0;
            double logSecondMax = (secondMaxTile > 0) ? log2(secondMaxTile) : 0;

            copyGrid(nextState.grid, tempGrid);
            nextState.gCurr = currentState.gCurr + 1;
            nextState.heuristicScore = nextState.gCurr + heuristic;
            nextState.logMax = logMax;
            nextState.logSecondMax = logSecondMax;

            nextStates[stateCount++] = nextState;
        }
    }
    return stateCount;
}

void insertSorted(GameState states[], int &size, const GameState &state)
{
    int i = size - 1;
    while (i >= 0 &&
           (states[i].heuristicScore > state.heuristicScore ||
            (states[i].heuristicScore == state.heuristicScore && states[i].logMax > state.logMax) ||
            (states[i].heuristicScore == state.heuristicScore && states[i].logMax == state.logMax && states[i].logSecondMax > state.logSecondMax)))
    {
        states[i + 1] = states[i];
        i--;
    }
    states[i + 1] = state;
    size++;
}

int getMaxTile(const int grid[GRID_SIZE][GRID_SIZE])
{
    int maxTile = 0;
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            if (grid[i][j] > maxTile)
            {
                maxTile = grid[i][j];
            }
        }
    }
    return maxTile;
}



int main()
{
    int grid[GRID_SIZE][GRID_SIZE] = {0};

    srand(static_cast<unsigned>(time(nullptr)));
    randomGenerate(grid);
    randomGenerate(grid);
    randomGenerate(grid);

    cout << "Initial grid:\n";
    displayGrid(grid);

    GameState initialState;
    copyGrid(initialState.grid, grid);

    GameState finalState;
    int finalCount;
    int arraySize;

    // aStarAlgorithm(initialState, finalState);

    cout << "\nFinal state after " << finalCount << " iterations:\n";
    displayGrid(finalState.grid);

    cout << "\nProgress:\n";

    return 0;
}