# HPC SEM 6 Assignments and Projects

## Tutorials 1 to 7
### OpenMP Parallelization using multi-thread approach
```
gcc -fopenmp -g file.c -o file -lm
```

## Tutorials 8 to 12
### Parallelization of the same problems using CUDA C on GPGPU
#### GPGPU compute capability = 8.6
```
nvcc -arch=sm_86 file.cu -o file
./file - Linux
.\file - Windows
```
