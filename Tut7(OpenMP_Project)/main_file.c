#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

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
    // Copy the initial grid
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
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            if (grid[i][j] == 0)
            {
                emptyTiles[emptyCount][0] = i;
                emptyTiles[emptyCount][1] = j;
                emptyCount++;
            }
        }
    }

    if (emptyCount > 0)
    {
        int randomIndex = rand() % emptyCount;
        int x = emptyTiles[randomIndex][0];
        int y = emptyTiles[randomIndex][1];

        // 90% chance of 2, 10% chance of 4
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
    // Fill the rest with 0
    while (mergedCount < GRID_SIZE)
    {
        mergedLine[mergedCount++] = 0;
    }
}

// Function to move tiles to the left
void moveLeft(int src[GRID_SIZE][GRID_SIZE], int dest[GRID_SIZE][GRID_SIZE])
{
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
        for (int j = 0; j < GRID_SIZE; j++)
        {
            dest[i][j] = mergedRow[j];
        }
    }
}

// Function to move tiles to the right
void moveRight(int src[GRID_SIZE][GRID_SIZE], int dest[GRID_SIZE][GRID_SIZE])
{
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

// Function to display the grid
void displayGrid(int grid[GRID_SIZE][GRID_SIZE])
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            printf("%d\t", grid[i][j]);
        }
        printf("\n");
    }
}

// Function to generate next states from the current state
int generateNextStates(GameState currentState, GameState *nextStates)
{
    int validMoves = 0;
    // Define move functions to iterate through
    // 0: left, 1: right, 2: up, 3: down
    for (int move = 0; move < 4; move++)
    {
        int movedGrid[GRID_SIZE][GRID_SIZE];
        // Apply the move
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

        // Check if the move actually changed the grid
        int changed = 0;
        for (int i = 0; i < GRID_SIZE && !changed; i++)
        {
            for (int j = 0; j < GRID_SIZE && !changed; j++)
            {
                if (movedGrid[i][j] != currentState.grid[i][j])
                {
                    changed = 1;
                }
            }
        }

        if (changed)
        {
            // Find max and second max tile values
            int m = 0, k = 0;
            for (int i = 0; i < GRID_SIZE; i++)
            {
                for (int j = 0; j < GRID_SIZE; j++)
                {
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

            // Generate a new random tile
            randomGenerate(movedGrid);

            // Calculate heuristic score (sum of all tiles)
            int h = 0;
            for (int i = 0; i < GRID_SIZE; i++)
            {
                for (int j = 0; j < GRID_SIZE; j++)
                {
                    h += movedGrid[i][j];
                }
            }

            // Calculate log values
            double logM = (m > 0) ? log2((double)m) : 0.0;
            double logK = (k > 0) ? log2((double)k) : 0.0;

            // Create new game state
            GameState newState;
            initializeGameState(&newState, movedGrid);
            newState.gCurr = currentState.gCurr + 1;
            newState.heuristicScore = h;
            newState.logMax = logM;
            newState.logSecondMax = logK;

            // Add to nextStates array
            nextStates[validMoves] = newState;
            validMoves++;
        }
    }
    return validMoves;
}

// Function implementing the A* algorithm
Tuple aStarAlgorithm(GameState initialState)
{
    Tuple result;
    // Initialize opened and closed lists
    GameState *opened = NULL;
    int openedCount = 0;
    int openedCapacity = 10;
    opened = (GameState *)malloc(openedCapacity * sizeof(GameState));

    GameState *closed = NULL;
    int closedCount = 0;
    int closedCapacity = 10;
    closed = (GameState *)malloc(closedCapacity * sizeof(GameState));

    double count = 1;

    // Add initial state to opened
    opened[openedCount++] = initialState;

    while (openedCount > 0)
    {
        // Sort opened list
        qsort(opened, openedCount, sizeof(GameState), compareGameStates);

        // Get and remove the first state
        GameState currentState = opened[0];
        // Shift the opened list
        memmove(&opened[0], &opened[1], (openedCount - 1) * sizeof(GameState));
        openedCount--;

        // Add to closed list
        if (closedCount >= closedCapacity)
        {
            closedCapacity *= 2;
            closed = (GameState *)realloc(closed, closedCapacity * sizeof(GameState));
        }
        closed[closedCount++] = currentState;

        // Check for goal state (2048 tile)
        int goalFound = 0;
        for (int i = 0; i < GRID_SIZE && !goalFound; i++)
        {
            for (int j = 0; j < GRID_SIZE && !goalFound; j++)
            {
                if (currentState.grid[i][j] == 16384)
                {
                    result.gameState = currentState;
                    result.iterations = count;
                    free(opened);
                    free(closed);
                    return result;
                }
            }
        }

        count++;

        // Limit search depth
        if (count >= 100000)
        {
            result.gameState = currentState;
            result.iterations = count;
            free(opened);
            free(closed);
            return result;
        }

        // Generate next states
        GameState *nextStates = (GameState *)malloc(4 * sizeof(GameState)); // Maximum 4 moves
        int numValidMoves = generateNextStates(currentState, nextStates);

        for (int i = 0; i < numValidMoves; i++)
        {
            // Add to opened list
            if (openedCount >= openedCapacity)
            {
                openedCapacity *= 2;
                opened = (GameState *)realloc(opened, openedCapacity * sizeof(GameState));
            }
            opened[openedCount++] = nextStates[i];
        }
        free(nextStates);
    }

    // If no solution found
    result.gameState = initialState;
    result.iterations = count;
    free(opened);
    free(closed);
    return result;
}

int main()
{
    srand(time(NULL));
    // Initialize the grid with zeros
    int initialGrid[GRID_SIZE][GRID_SIZE];
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            initialGrid[i][j] = 0;
        }
    }
    // Generate two initial random tiles
    randomGenerate(initialGrid);
    randomGenerate(initialGrid);

    // Create initial game state
    GameState initialState;
    initializeGameState(&initialState, initialGrid);

    printf("Initial Grid:\n");
    displayGrid(initialState.grid);

    // Run A* algorithm
    Tuple result = aStarAlgorithm(initialState);

    printf("Final Grid:\n");
    displayGrid(result.gameState.grid);

    printf("Number of iterations: %f\n", result.iterations);

    return 0;
}