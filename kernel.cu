#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<time.h>
#include <stdio.h>

void printMatrix(double* mat, int m, int n) {
    printf("\t\t");
    for (int i = 0; i < m * n; i++)
    {
        if (mat[i] < 0) {
            printf("%4.1f\t", mat[i]);
        }
        else {
            printf("%4.2f\t", mat[i]);
        }
        if ((i + 1) % n == 0) {
            printf("\n\t\t");
        }
    }
    printf("\n");
}

void MatrixMultiplication(double* A, double* B, double* C, int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

__global__ void multKernel1(double* c, double* a, double* b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

__global__ void multKernel2(double* c, double* a, double* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int l = blockIdx.x;
    int m = blockIdx.y;
    int n = blockIdx.z;

    printf("Indice del hilo: (%d, %d, %d) | Indice del bloque: (%d, %d, %d) | Calculando el producto: {%f} * {%f}\n", threadIdx.x, threadIdx.y, threadIdx.z, l, m, n, a[i], b[i]);
    c[i] = a[i] * b[i];
}

__global__ void MatrixMultiplicationCuda(double* c, double* a, double* b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    double sum = 0;
    for (int k = 0; k < N; k++) {
        sum += a[i * N + k] * b[k * N + j];
    }
    c[i * N + j] = sum;
}

__global__ void MatrixMultiplicationCuda1(double* c, double* a, double* b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    double sum = 0;
    for (int k = 0; k < N; k++) {
        sum += a[i * N + k] * b[k * N + j];
    }
    c[i * N + j] = sum;
}

cudaError_t multWithCuda1(double* c, double* a, double* b, unsigned int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    multKernel1 << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

cudaError_t multWithCuda2(double* c, double* a, double* b, unsigned int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    multKernel2 << <10, size / 10 >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multKernel2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

cudaError_t multMatrixwithCuda(double* c, double* a, double* b, unsigned int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    int N = size * size;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(size, size);
    MatrixMultiplicationCuda << <1, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatrixMultiplicationCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

cudaError_t multMatrixwithCuda3(double* c, double* a, double* b, unsigned int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    int N = size * size;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(start);

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(size, size);
    for (int i = 0; i < 10; i++) {
        MatrixMultiplicationCuda << <1, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);
    }

    cudaEventRecord(stop);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatrixMultiplicationCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000;
    printf("Tiempo en CUDA: %f\n", seconds / 10);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

cudaError_t multMatrixwithCuda2(double* c, double* a, double* b, unsigned int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    int N = size * size;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //printf("%d\n", size);
    dim3 dimGrid(size/32, size/32);
    dim3 dimBlock(32, 32);
    MatrixMultiplicationCuda1 << <dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatrixMultiplicationCuda1 launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

    return cudaStatus;
}

int funcion1()
{
    const int arraySize = 100;
    double a[arraySize];
    double b[arraySize];
    double c[arraySize];

    for (int i = 0; i < arraySize; i++) {
        a[i] = (double)i;
        b[i] = (double)i * 2;
    }

    // Add vectors in parallel.
    cudaError_t cudaStatus = multWithCuda1(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multWithCuda failed!");
        return 1;
    }

    double prod = 0;
    for (int i = 0; i < arraySize; i++) {
        prod += c[i];
    }

    printf("Producto escalar: %f", prod);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

int funcion2()
{
    const int arraySize = 100;
    double a[arraySize];
    double b[arraySize];
    double c[arraySize];

    for (int i = 0; i < arraySize; i++) {
        a[i] = (double)i;
        b[i] = (double)i * 2;
    }

    // Add vectors in parallel.
    cudaError_t cudaStatus = multWithCuda2(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multWithCuda failed!");
        return 1;
    }

    double prod = 0;
    for (int i = 0; i < arraySize; i++) {
        prod += c[i];
    }

    printf("Producto escalar: %f", prod);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

int funcion3()
{
    const int N = 32;

    double a[N * N];
    double b[N * N];
    double c[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++)
                sum += a[i * N + k] * b[k * N + j];
            c[i * N + j] = sum;
        }
    }

    //printf("Matriz A (suma de su numero de fila mas su numero de columna):\n");
    //printMatrix(a, N, N);
    //printf("Matriz B (resta de su numero de fila menos su numero de columna):\n");
    //printMatrix(b, N, N);
    printf("Resultado de la multiplicacion de A * B (sequencial):\n");
    int j = 0;
    for (int i = 0; i < N * N; i = i + N) {
        printf("%f\n", c[i + j]);
        j++;
    }
    //printMatrix(c, N, N);

    return 0;
}

int funcion4()
{
    const int N = 896;

    double* a = new double[N * N];
    double* b = new double[N * N];
    double* c = new double[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
    }

    clock_t start, stop;
    start = clock();
    for (int i = 0; i < 10; i++) {
        MatrixMultiplication(a, b, c, N);
    }
    stop = clock();

    //printf("Matriz A (suma de su numero de fila mas su numero de columna):\n");
    //printMatrix(a, N, N);
    //printf("Matriz B (resta de su numero de fila menos su numero de columna):\n");
    //printMatrix(b, N, N);
    //printf("Resultado de la multiplicacion de A * B: \n");
    printf("Resultado de la multiplicacion de A * B (sequencial):\n");
    int j = 0;
    for (int i = 0; i < N * N; i = i + N) {
        printf("%f\n", c[i + j]);
        j++;
    }
    //printMatrix(c, N, N);
    printf("Tiempo secuencial: %4.8f segundos\n", (double)(stop - start) / CLOCKS_PER_SEC / 10);

    return 0;
}

int funcion5()
{
    const int N = 3;

    double a[N * N];
    double b[N * N];
    double c[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
    }

    cudaError_t cudaStatus = multMatrixwithCuda(c, a, b, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multWithCuda failed!");
        return 1;
    }

    printf("Matriz A (suma de su numero de fila mas su numero de columna):\n");
    printMatrix(a, N, N);
    printf("Matriz B (resta de su numero de fila menos su numero de columna):\n");
    printMatrix(b, N, N);
    printf("Resultado de la multiplicacion de A * B:\n");
    printMatrix(c, N, N);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

int funcion6()
{
    const int N = 8;

    double a[N * N];
    double b[N * N];
    double c[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
    }

    cudaError_t cudaStatus = multMatrixwithCuda2(c, a, b, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multWithCuda failed!");
        return 1;
    }

    printf("Matriz A (suma de su numero de fila mas su numero de columna):\n");
    printMatrix(a, N, N);
    printf("Matriz B (resta de su numero de fila menos su numero de columna):\n");
    printMatrix(b, N, N);
    printf("Resultado de la multiplicacion de A * B:\n");
    printMatrix(c, N, N);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

int funcion7()
{
    const int N = 896;

    double* a = new double [N * N];
    double* b = new double [N * N];
    double* c = new double [N * N];


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
    }

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cudaStatus = multMatrixwithCuda2(c, a, b, N);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000;
    printf("Tiempo en CUDA: %f segundos\n", seconds / 10);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multMatrixwithCuda2 failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    //printf("Matriz A (suma de su numero de fila mas su numero de columna):\n");
    //printMatrix(a, N, N);
    //printf("Matriz B (resta de su numero de fila menos su numero de columna):\n");
    //printMatrix(b, N, N);
    //printMatrix(c, N, N);
    printf("%s\n", "Resultado de la multiplicacion de A * B(CUDA) :");
    int j = 0;
    for (int i = 0; i < N * N; i = i + N) {
        printf("%f\n", c[i + j]);
        j++;
    }
    //printMatrix(c, N, N);

    return 0;
}

int main()
{
    int error = 0;
    //error = funcion1();
    //error = funcion2();
    //error = funcion3();
    error = funcion4();
    //error = funcion5();
    //error = funcion6();
    error = funcion7();

    return error;
}
