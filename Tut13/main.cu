#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define GRID_SIZE 5
#define TARGET 16384

typedef struct
{
    int grid[GRID_SIZE * GRID_SIZE];
    int gCurr;
    int heuristicScore;
    double logMax;
    double logSecondMax;
} GameState;

__global__ void mergeTilesKernel(int *grid, int direction)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_SIZE)
        return;

    int tempRow[GRID_SIZE];
    int size = 0;

    if (direction == 0)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            int cell = grid[idx * GRID_SIZE + j];
            if (cell != 0)
            {
                tempRow[size++] = cell;
            }
        }
    }
    else if (direction == 1)
    {
        for (int j = GRID_SIZE - 1; j >= 0; j--)
        {
            int cell = grid[idx * GRID_SIZE + j];
            if (cell != 0)
            {
                tempRow[size++] = cell;
            }
        }
    }

    int mergedRow[GRID_SIZE] = {0};
    int mergedCount = 0;

    for (int i = 0; i < size; i++)
    {
        if (i + 1 < size && tempRow[i] == tempRow[i + 1])
        {
            mergedRow[mergedCount++] = tempRow[i] * 2;
            i++;
        }
        else
        {
            mergedRow[mergedCount++] = tempRow[i];
        }
    }

    if (direction == 0)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            grid[idx * GRID_SIZE + j] = j < mergedCount ? mergedRow[j] : 0;
        }
    }
    else if (direction == 1)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            grid[idx * GRID_SIZE + j] = (GRID_SIZE - j - 1) < mergedCount ? mergedRow[GRID_SIZE - j - 1] : 0;
        }
    }
}

__global__ void mergeTilesColumnKernel(int *grid, int direction)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_SIZE)
        return;

    int tempCol[GRID_SIZE];
    int size = 0;

    if (direction == 0)
    {
        for (int i = 0; i < GRID_SIZE; i++)
        {
            int cell = grid[i * GRID_SIZE + idx];
            if (cell != 0)
            {
                tempCol[size++] = cell;
            }
        }
    }
    else if (direction == 1)
    {
        for (int i = GRID_SIZE - 1; i >= 0; i--)
        {
            int cell = grid[i * GRID_SIZE + idx];
            if (cell != 0)
            {
                tempCol[size++] = cell;
            }
        }
    }

    int mergedCol[GRID_SIZE] = {0};
    int mergedCount = 0;

    for (int i = 0; i < size; i++)
    {
        if (i + 1 < size && tempCol[i] == tempCol[i + 1])
        {
            mergedCol[mergedCount++] = tempCol[i] * 2;
            i++;
        }
        else
        {
            mergedCol[mergedCount++] = tempCol[i];
        }
    }

    if (direction == 0)
    {
        for (int i = 0; i < GRID_SIZE; i++)
        {
            grid[i * GRID_SIZE + idx] = i < mergedCount ? mergedCol[i] : 0;
        }
    }
    else if (direction == 1)
    {
        for (int i = 0; i < GRID_SIZE; i++)
        {
            grid[i * GRID_SIZE + idx] = (GRID_SIZE - i - 1) < mergedCount ? mergedCol[GRID_SIZE - i - 1] : 0;
        }
    }
}

__global__ void checkChanges(int *currentGrid, int *movedGrid, int *changed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_SIZE * GRID_SIZE)
        return;

    if (movedGrid[idx] != currentGrid[idx])
    {
        *changed = 1;
    }
}

__global__ void calculateHeuristic(int *grid, int *h, int *m, int *k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_SIZE * GRID_SIZE)
        return;

    int value = grid[idx];
    atomicAdd(h, value);

    if (value > *m)
    {
        atomicExch(k, *m);
        atomicExch(m, value);
    }
    else if (value > *k)
    {
        atomicExch(k, value);
    }
}

void moveLeft(int *grid)
{
    int *d_grid;
    cudaMalloc((void **)&d_grid, sizeof(int) * GRID_SIZE * GRID_SIZE);
    cudaMemcpy(d_grid, grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);

    mergeTilesKernel<<<GRID_SIZE, 1>>>(d_grid, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(grid, d_grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(d_grid);
}

void moveRight(int *grid)
{
    int *d_grid;
    cudaMalloc((void **)&d_grid, sizeof(int) * GRID_SIZE * GRID_SIZE);
    cudaMemcpy(d_grid, grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);

    mergeTilesKernel<<<GRID_SIZE, 1>>>(d_grid, 1);
    cudaDeviceSynchronize();

    cudaMemcpy(grid, d_grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(d_grid);
}

void moveUp(int *grid)
{
    int *d_grid;
    cudaMalloc((void **)&d_grid, sizeof(int) * GRID_SIZE * GRID_SIZE);
    cudaMemcpy(d_grid, grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);

    mergeTilesColumnKernel<<<GRID_SIZE, 1>>>(d_grid, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(grid, d_grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(d_grid);
}

void moveDown(int *grid)
{
    int *d_grid;
    cudaMalloc((void **)&d_grid, sizeof(int) * GRID_SIZE * GRID_SIZE);
    cudaMemcpy(d_grid, grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);

    mergeTilesColumnKernel<<<GRID_SIZE, 1>>>(d_grid, 1);
    cudaDeviceSynchronize();

    cudaMemcpy(grid, d_grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(d_grid);
}

void generateRandomTwo(int *grid)
{
    int emptyCells[GRID_SIZE * GRID_SIZE];
    int emptyCount = 0;
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++)
    {
        if (grid[i] == 0)
        {
            emptyCells[emptyCount] = i;
            emptyCount++;
        }
    }
    if (emptyCount == 0)
    {
        return;
    }
    int randomIndex = rand() % emptyCount;
    grid[emptyCells[randomIndex]] = 2;
}

void initializeGameState(GameState *initialState, int *initialGrid)
{
    initialState->gCurr = 0;
    initialState->heuristicScore = 0;
    initialState->logMax = 0;
    initialState->logSecondMax = 0;
    memcpy(initialState->grid, initialGrid, sizeof(int) * GRID_SIZE * GRID_SIZE);
}

int generateNextStates(GameState currentState, GameState *nextStates)
{
    int validMoves = 0;
    int *d_currentGrid, *d_movedGrid;
    int *h_device, *m_device, *k_device, *changed_device;
    int h, m, k;
    int changed;

    cudaMalloc((void **)&d_currentGrid, sizeof(int) * GRID_SIZE * GRID_SIZE);
    cudaMalloc((void **)&d_movedGrid, sizeof(int) * GRID_SIZE * GRID_SIZE);
    cudaMalloc((void **)&h_device, sizeof(int));
    cudaMalloc((void **)&m_device, sizeof(int));
    cudaMalloc((void **)&k_device, sizeof(int));
    cudaMalloc((void **)&changed_device, sizeof(int));

    cudaMemcpy(d_currentGrid, currentState.grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);

    for (int move = 0; move < 4; move++)
    {
        // Copy the current grid to the moved grid
        cudaMemcpy(d_movedGrid, d_currentGrid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToDevice);
        
        // Reset the changed flag
        changed = 0;
        cudaMemcpy(changed_device, &changed, sizeof(int), cudaMemcpyHostToDevice);
        
        // Apply the move
        int movedGrid[GRID_SIZE * GRID_SIZE];
        cudaMemcpy(movedGrid, d_currentGrid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
        
        switch (move)
        {
        case 0:
            moveLeft(movedGrid);
            break;
        case 1:
            moveRight(movedGrid);
            break;
        case 2:
            moveUp(movedGrid);
            break;
        case 3:
            moveDown(movedGrid);
            break;
        }
        
        // Copy the moved grid back to the device
        cudaMemcpy(d_movedGrid, movedGrid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);
        
        // Check if the grid changed
        checkChanges<<<1, GRID_SIZE * GRID_SIZE>>>(d_currentGrid, d_movedGrid, changed_device);
        cudaDeviceSynchronize();
        cudaMemcpy(&changed, changed_device, sizeof(int), cudaMemcpyDeviceToHost);

        if (changed)
        {
            // Reset the heuristic values
            h = 0;
            m = 0;
            k = 0;
            cudaMemcpy(h_device, &h, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(m_device, &m, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(k_device, &k, sizeof(int), cudaMemcpyHostToDevice);
            
            // Calculate heuristic
            calculateHeuristic<<<1, GRID_SIZE * GRID_SIZE>>>(d_movedGrid, h_device, m_device, k_device);
            cudaDeviceSynchronize();
            cudaMemcpy(&h, h_device, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&m, m_device, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&k, k_device, sizeof(int), cudaMemcpyDeviceToHost);

            // Copy the moved grid back to host
            cudaMemcpy(movedGrid, d_movedGrid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
            
            // Generate random two
            generateRandomTwo(movedGrid);

            // Create new state
            GameState newState;
            memcpy(newState.grid, movedGrid, sizeof(int) * GRID_SIZE * GRID_SIZE);
            newState.gCurr = currentState.gCurr + 1;
            newState.heuristicScore = h;
            newState.logMax = (m > 0) ? log2((double)m) : 0.0;
            newState.logSecondMax = (k > 0) ? log2((double)k) : 0.0;

            nextStates[validMoves++] = newState;
        }
    }

    cudaFree(d_currentGrid);
    cudaFree(d_movedGrid);
    cudaFree(h_device);
    cudaFree(m_device);
    cudaFree(k_device);
    cudaFree(changed_device);

    return validMoves;
}

void printGrid(int *grid)
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            printf("%d ", grid[i * GRID_SIZE + j]);
        }
        printf("\n");
    }
}

int main()
{
    int initialGrid[GRID_SIZE * GRID_SIZE] = {0};
    GameState initialState;
    srand(time(NULL));

    generateRandomTwo(initialGrid);
    generateRandomTwo(initialGrid);

    initializeGameState(&initialState, initialGrid);
    printf("Initial State:\n");
    printGrid(initialState.grid);

    GameState nextStates[4];
    int validMoves = generateNextStates(initialState, nextStates);

    printf("\nValid Moves: %d\n", validMoves);
    for (int i = 0; i < validMoves; i++)
    {
        printf("Next State %d:\n", i + 1);
        printGrid(nextStates[i].grid);
    }

    return 0;
}