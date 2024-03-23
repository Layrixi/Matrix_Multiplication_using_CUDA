     
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <stdio.h>
#include <iostream>

const int size = 1024;//works only for powers of 2, need to repair later
double results[6];//an array to compare if the results were correct

void printMatrix(float **matrix) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix[i][j] << "  |";
        }
        std::cout << std::endl;
    }
}

void printCudaMatrix(float* matrix)
{
    long int cudaSize = size * size;
    for (int i = 0; i < cudaSIZE; i++)
        std::cout << matrix[i] << "   |";
}


__global__ void mult(float* a, float* b, float* c)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;//col
    int i = threadIdx.y + blockIdx.y * blockDim.y;//row
    float sum=0;
    for (int shift = 0; shift < size; shift++)
        sum += a[i * size + shift] * b[shift * size + j];
    
    c[i*size+j] = sum;
}


cudaError_t CudaMultiplyMatrixes(float **aORG, float **bORG)
{
    float* a = (float*)malloc(size * size * sizeof(float));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            a[i * size + j] = aORG[i][j];
    float* b = (float*)malloc(size * size * sizeof(float));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            b[i * size + j] = bORG[i][j];

    float* c = (float*)malloc(size * size * sizeof(float));

    for (int i = 0; i < size*size; i++)
            c[i] = 0;

    float* cudaMatrixA = 0;
    float* cudaMatrixB = 0;
    float* cudaMatrixC = 0;
    cudaError_t cudaStatus;

    //set gpu
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //cudaMallocs with pitch for variables
    size_t pitch;
    cudaStatus = cudaMallocPitch(&cudaMatrixA, &pitch, size * sizeof(float), size);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    size_t pitch2;
    cudaStatus = cudaMallocPitch(&cudaMatrixB, &pitch2, size * sizeof(float), size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    size_t pitch3;
    cudaStatus = cudaMallocPitch(&cudaMatrixC, &pitch3, size * sizeof(float), size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }



    //copy arrays to GPU
    cudaStatus = cudaMemcpy2D(cudaMatrixA, pitch, a, size * sizeof(float), size * sizeof(float), size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy2D(cudaMatrixB, pitch2, b, size * sizeof(float), size * sizeof(float), size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy2D(cudaMatrixC, pitch3, c, size * sizeof(float), size * sizeof(float), size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //calculate
    dim3 grid(ceilf(size / (float)32), ceilf(size / (float)32), 1);
    dim3 block(32, 32, 1);

    auto start = std::chrono::high_resolution_clock::now();
    mult << < grid,block >> > (cudaMatrixA, cudaMatrixB, cudaMatrixC);
    auto end = std::chrono::high_resolution_clock::now();


    //check for errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, " launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    //gpu->cpu
    cudaStatus = cudaMemcpy2D(c, size*sizeof(float),cudaMatrixC, pitch3, size * sizeof(float), size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    printf("\n\n\n\n\n\n\n\n\n\n\n");
    printCudaMatrix(c);
    for (int i = 0; i < 3; i++)
        results[i+3] = c[i];
    printf("time: %d microS\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
Error:
    cudaFree(cudaMatrixA);
    cudaFree(cudaMatrixB);
    cudaFree(cudaMatrixC);

    return cudaStatus;
}


void multiplyMatrixes(float** a, float** b)
{
    float** c = (float**)malloc(size * sizeof(float*));
    
    for (int i = 0; i < size; ++i) 
        c[i] = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) 
        for (int j = 0; j < size; j++) 
            c[i][j] = 0;

    auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                for (int k = 0; k < size; ++k)
                    c[i][j] += a[i][k] * b[k][j];
    
    auto end = std::chrono::high_resolution_clock::now();
    //printMatrix(c);
    for (int i = 0; i < 3; i++)
        results[i] = c[0][i];
    printf("time: %d microS\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

}





int main()
{
    float **a = (float**)malloc(size*sizeof(float*));
    float **b = (float**)malloc(size*sizeof(float*));
   
    for (int i = 0; i < size; i++) 
    {
        a[i] = (float*)malloc(size * sizeof(float));
        b[i] = (float*)malloc(size * sizeof(float));
    }

    for (int i = 0; i < size; i++) 
        for (int j = 0; j < size; j++) 
        {
            a[i][j] = 0;
            b[i][j] = 0;
        }
    
    
    srand(time(0));
    
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) 
        {
            a[i][j] = static_cast<float>(std::rand()) / RAND_MAX;
            b[i][j] = static_cast<float>(std::rand()) / RAND_MAX;
        }
    
    
    multiplyMatrixes(a, b);
    CudaMultiplyMatrixes(a, b);
    for (int i = 0; i < 3; i++)
        printf("results[%d] = %f\nresults[%d] = %f\n", i, results[i], i+3, results[i+3]);
  
    /* //just a fun code to see the possibilities of CUDA
    int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
        deviceCount = 0;
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
            if (device == 0)
            {
                printf("maxthreads per block %d\n", properties.maxThreadsPerBlock);
                
            }
    }
    */
    return 0;
}

