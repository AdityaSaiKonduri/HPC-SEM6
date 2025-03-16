#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>

// Define the size of the grid
#define GRID_SIZE 5

// Structure to hold the game state
typedef struct
{
    int grid[GRID_SIZE][GRID_SIZE];
    int gCurr;
    int heuristicScore;
    double logMax;
    double logSecondMax;
} GameState;

// Structure to hold the return values of aStarAlgorithm
typedef struct
{
    GameState gameState;
    double iterations;
} Tuple;

// Function to initialize a GameState
void initializeGameState(GameState *state, int initialGrid[GRID_SIZE][GRID_SIZE])
{
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            state->grid[i][j] = initialGrid[i][j];
        }
    }
    state->gCurr = 0;
    state->heuristicScore = 0;
    state->logMax = 0.0;
    state->logSecondMax = 0.0;
}

// Function to compare two GameStates for sorting
int compareGameStates(const void *a, const void *b)
{
    GameState *stateA = (GameState *)a;
    GameState *stateB = (GameState *)b;
    if (stateA->heuristicScore != stateB->heuristicScore)
        return (stateA->heuristicScore > stateB->heuristicScore) ? -1 : 1;
    if (stateA->logMax != stateB->logMax)
        return (stateA->logMax > stateB->logMax) ? -1 : 1;
    if (stateA->logSecondMax != stateB->logSecondMax)
        return (stateA->logSecondMax > stateB->logSecondMax) ? -1 : 1;
    return 0;
}

// Function to generate a random tile in the grid
void randomGenerate(int grid[GRID_SIZE][GRID_SIZE])
{
    int emptyTiles[GRID_SIZE * GRID_SIZE][2];
    int emptyCount = 0;

#pragma omp parallel
    {
        int localEmptyTiles[GRID_SIZE * GRID_SIZE][2];
        int localEmptyCount = 0;

#pragma omp for collapse(2)
        for (int i = 0; i < GRID_SIZE; i++)
        {
            for (int j = 0; j < GRID_SIZE; j++)
            {
                if (grid[i][j] == 0)
                {
                    localEmptyTiles[localEmptyCount][0] = i;
                    localEmptyTiles[localEmptyCount][1] = j;
                    localEmptyCount++;
                }
            }
        }

        // #pragma omp critical
        {
            memcpy(&emptyTiles[emptyCount], localEmptyTiles, localEmptyCount * sizeof(emptyTiles[0]));
            emptyCount += localEmptyCount;
        }
    }

    if (emptyCount > 0)
    {
        int randomIndex = rand() % emptyCount;
        int x = emptyTiles[randomIndex][0];
        int y = emptyTiles[randomIndex][1];
        grid[x][y] = (rand() % 10 == 0) ? 4 : 2;
    }
}

// Function to merge tiles in a row or column
void mergeTilesHelper(int *line, int size, int mergedLine[GRID_SIZE])
{
    int mergedCount = 0;
    for (int i = 0; i < size; i++)
    {
        if (line[i] == 0)
            continue;
        if (i + 1 < size && line[i] == line[i + 1] && line[i] != 0)
        {
            mergedLine[mergedCount++] = line[i] * 2;
            i++;
        }
        else
        {
            mergedLine[mergedCount++] = line[i];
        }
    }
    while (mergedCount < GRID_SIZE)
    {
        mergedLine[mergedCount++] = 0;
    }
}

// Function to move tiles to the left
void moveLeft(int src[GRID_SIZE][GRID_SIZE], int dest[GRID_SIZE][GRID_SIZE])
{
    // #pragma omp parallel for
    for (int i = 0; i < GRID_SIZE; i++)
    {
        int tempRow[GRID_SIZE];
        int size = 0;
        for (int j = 0; j < GRID_SIZE; j++)
        {
            if (src[i][j] != 0)
            {
                tempRow[size++] = src[i][j];
            }
        }
        int mergedRow[GRID_SIZE];
        mergeTilesHelper(tempRow, size, mergedRow);
        memcpy(&dest[i][0], mergedRow, GRID_SIZE * sizeof(int));
    }
}

// Function to move tiles to the right
void moveRight(int src[GRID_SIZE][GRID_SIZE], int dest[GRID_SIZE][GRID_SIZE])
{
    // #pragma omp parallel for
    for (int i = 0; i < GRID_SIZE; i++)
    {
        int tempRow[GRID_SIZE];
        int size = 0;
        for (int j = GRID_SIZE - 1; j >= 0; j--)
        {
            if (src[i][j] != 0)
            {
                tempRow[size++] = src[i][j];
            }
        }
        int mergedRow[GRID_SIZE];
        mergeTilesHelper(tempRow, size, mergedRow);
        for (int j = GRID_SIZE - 1, k = 0; j >= 0; j--, k++)
        {
            dest[i][j] = mergedRow[k];
        }
    }
}

// Function to move tiles upwards
void moveUp(int src[GRID_SIZE][GRID_SIZE], int dest[GRID_SIZE][GRID_SIZE])
{
    // #pragma omp parallel for
    for (int j = 0; j < GRID_SIZE; j++)
    {
        int tempCol[GRID_SIZE];
        int size = 0;
        for (int i = 0; i < GRID_SIZE; i++)
        {
            if (src[i][j] != 0)
            {
                tempCol[size++] = src[i][j];
            }
        }
        int mergedCol[GRID_SIZE];
        mergeTilesHelper(tempCol, size, mergedCol);
        for (int i = 0; i < GRID_SIZE; i++)
        {
            dest[i][j] = mergedCol[i];
        }
    }
}

// Function to move tiles downwards
void moveDown(int src[GRID_SIZE][GRID_SIZE], int dest[GRID_SIZE][GRID_SIZE])
{
    // #pragma omp parallel for
    for (int j = 0; j < GRID_SIZE; j++)
    {
        int tempCol[GRID_SIZE];
        int size = 0;
        for (int i = GRID_SIZE - 1; i >= 0; i--)
        {
            if (src[i][j] != 0)
            {
                tempCol[size++] = src[i][j];
            }
        }
        int mergedCol[GRID_SIZE];
        mergeTilesHelper(tempCol, size, mergedCol);
        for (int i = GRID_SIZE - 1, k = 0; i >= 0; i--, k++)
        {
            dest[i][j] = mergedCol[k];
        }
    }
}

// Function to generate next states from the current state
int generateNextStates(GameState currentState, GameState *nextStates)
{
    int validMoves = 0;
#pragma omp parallel
    {
        GameState localNextStates[4];
        int localValidMoves = 0;

#pragma omp for
        for (int move = 0; move < 4; move++)
        {
            int movedGrid[GRID_SIZE][GRID_SIZE];
            switch (move)
            {
            case 0:
                moveLeft(currentState.grid, movedGrid);
                break;
            case 1:
                moveRight(currentState.grid, movedGrid);
                break;
            case 2:
                moveUp(currentState.grid, movedGrid);
                break;
            case 3:
                moveDown(currentState.grid, movedGrid);
                break;
            }

            int changed = 0;
            int m = 0, k = 0;
            int h = 0;

#pragma omp parallel for collapse(2) reduction(+ : h) reduction(max : m, k)
            for (int i = 0; i < GRID_SIZE; i++)
            {
                for (int j = 0; j < GRID_SIZE; j++)
                {
                    if (movedGrid[i][j] != currentState.grid[i][j])
                    {
                        changed = 1;
                    }
                    h += movedGrid[i][j];
                    if (movedGrid[i][j] > m)
                    {
                        k = m;
                        m = movedGrid[i][j];
                    }
                    else if (movedGrid[i][j] > k)
                    {
                        k = movedGrid[i][j];
                    }
                }
            }

            if (changed)
            {
                randomGenerate(movedGrid);

                double logM = (m > 0) ? log2((double)m) : 0.0;
                double logK = (k > 0) ? log2((double)k) : 0.0;

                GameState newState;
                initializeGameState(&newState, movedGrid);
                newState.gCurr = currentState.gCurr + 1;
                newState.heuristicScore = h;
                newState.logMax = logM;
                newState.logSecondMax = logK;

                localNextStates[localValidMoves++] = newState;
            }
        }

        #pragma omp critical
        {
            memcpy(&nextStates[validMoves], localNextStates, localValidMoves * sizeof(GameState));
            validMoves += localValidMoves;
        }
    }
    return validMoves;
}

// Function implementing the A* algorithm
Tuple aStarAlgorithm(GameState initialState)
{
    Tuple result;
    GameState *opened = NULL;
    int openedCount = 0;
    int openedCapacity = 10;
    opened = (GameState *)malloc(openedCapacity * sizeof(GameState));

    GameState *closed = NULL;
    int closedCount = 0;
    int closedCapacity = 10;
    closed = (GameState *)malloc(closedCapacity * sizeof(GameState));

    double count = 1;
    opened[openedCount++] = initialState;

    while (openedCount > 0)
    {
        qsort(opened, openedCount, sizeof(GameState), compareGameStates);

        GameState currentState = opened[0];
        memmove(&opened[0], &opened[1], (openedCount - 1) * sizeof(GameState));
        openedCount--;

        if (closedCount >= closedCapacity)
        {
            closedCapacity *= 2;
            closed = (GameState *)realloc(closed, closedCapacity * sizeof(GameState));
        }
        closed[closedCount++] = currentState;

        int goalFound = 0;
        // #pragma omp parallel for collapse(2) reduction(|| : goalFound)
        for (int i = 0; i < GRID_SIZE; i++)
        {
            for (int j = 0; j < GRID_SIZE; j++)
            {
                if (currentState.grid[i][j] == 16384)
                {
                    goalFound = 1;
                }
            }
        }

        if (goalFound)
        {
            result.gameState = currentState;
            result.iterations = count;
            free(opened);
            free(closed);
            return result;
        }

        count++;

        if (count >= 100000)
        {
            result.gameState = currentState;
            result.iterations = count;
            free(opened);
            free(closed);
            return result;
        }

        GameState *nextStates = (GameState *)malloc(4 * sizeof(GameState));
        int numValidMoves = generateNextStates(currentState, nextStates);

        if (openedCount + numValidMoves >= openedCapacity)
        {
            openedCapacity = (openedCount + numValidMoves) * 2;
            opened = (GameState *)realloc(opened, openedCapacity * sizeof(GameState));
        }

        memcpy(&opened[openedCount], nextStates, numValidMoves * sizeof(GameState));
        openedCount += numValidMoves;
        free(nextStates);
    }

    result.gameState = initialState;
    result.iterations = count;
    free(opened);
    free(closed);
    return result;
}

// Function to display the grid
void displayGrid(int grid[GRID_SIZE][GRID_SIZE])
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            printf("%4d ", grid[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double calculateSpeedup(double serialTime, double parallelTime)
{
    return serialTime / parallelTime;
}

double calculateParallelFraction(double t1, double tn, int num_threads)
{
    if (num_threads == 1)
        return 0.0;
    return (1 - (tn / t1)) / (1 - (1.0 / num_threads));
}

int main()
{
    srand(time(NULL));

    int initialGrid[GRID_SIZE][GRID_SIZE] = {0};
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            initialGrid[i][j] = 0;
        }
    }

    randomGenerate(initialGrid);
    randomGenerate(initialGrid);

    GameState initialState;
    initializeGameState(&initialState, initialGrid);

    printf("Initial Grid:\n");
    displayGrid(initialState.grid);

    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    double serialTime = 0.0;
    double parallelTimes[num_thread_counts];
    double speedups[num_thread_counts];
    double parallelFractions[num_thread_counts];

    for (int i = 0; i < num_thread_counts; i++)
    {
        int num_threads = thread_counts[i];
        omp_set_num_threads(num_threads);

        double startTime = omp_get_wtime();
        Tuple result = aStarAlgorithm(initialState);
        double endTime = omp_get_wtime();

        double executionTime = endTime - startTime;
        parallelTimes[i] = executionTime;

        if (num_threads == 1)
        {
            serialTime = executionTime;
        }

        speedups[i] = calculateSpeedup(serialTime, executionTime);
        parallelFractions[i] = calculateParallelFraction(serialTime, executionTime, num_threads);

        printf("Threads: %d\n", num_threads);
        printf("Final Grid:\n");
        // displayGrid(result.gameState.grid);
        printf("Number of iterations: %f\n", result.iterations);
        printf("Execution time: %f seconds\n", executionTime);
        printf("\n");
    }

    // Print Execution Time Table
    printf("\n=== Execution Time Table ===\n");
    printf("%10s %20s\n", "Threads", "Execution Time (s)");
    printf("%10s %20s\n", "-------", "------------------");
    for (int i = 0; i < num_thread_counts; i++)
    {
        printf("%10d %20.6f\n", thread_counts[i], parallelTimes[i]);
    }

    // Print Speedup Table
    printf("\n=== Speedup Table ===\n");
    printf("%10s %20s\n", "Threads", "Speedup");
    printf("%10s %20s\n", "-------", "-------");
    for (int i = 0; i < num_thread_counts; i++)
    {
        printf("%10d %20.6f\n", thread_counts[i], speedups[i]);
    }

    // Print Parallelization Fraction Table
    printf("\n=== Parallelization Fraction Table ===\n");
    printf("%10s %30s\n", "Threads", "Parallelization Fraction");
    printf("%10s %30s\n", "-------", "-------------------------");
    for (int i = 0; i < num_thread_counts; i++)
    {
        printf("%10d %30.6f\n", thread_counts[i], parallelFractions[i]);
    }

    return 0;
}