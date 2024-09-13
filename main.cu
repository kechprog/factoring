#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <gmp.h>
#include <cuda.h>
#include <iostream>
#include <ostream>
#include <thread>

#define MAX_THREADS 1024*1023
#define THREADS_PER_BLOCK 1024

__managed__ static uint64_t num[73] = {0};
__managed__ static uint64_t k = 3;

struct Divisor {
    uint64_t p;
    uint64_t pow;
};

void create_num() {
    mpz_t result;
    mpz_init(result);
    
    mpz_ui_pow_ui(result, 2024, 420);  // result = 2024^420
    mpz_sub_ui(result, result, 1);     // result = 2024^420 - 1

    size_t count;
    mpz_export(num, &count, -1, sizeof(uint64_t), 0, 0, result);

    mpz_clear(result);
}

__host__ __device__ uint64_t divs_num(uint64_t divisor) {
    uint64_t rem = 0;
    uint64_t result[73];

    for (int i = 72; i >= 0; i--) {
        unsigned __int128 value = ((unsigned __int128)rem << 64) | num[i];
        result[i] = value / divisor;   // Store the quotient in result
        rem = value % divisor;         // Update the remainder
    }

    return rem;
}

__global__ void check_div(Divisor *results, uint64_t total_threads) {
    uint64_t global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_index >= total_threads) return;

    uint64_t d = results[global_index].p = k + 2 * global_index;
    uint64_t pow = 0;

    while (divs_num(d) == 0) {
        pow += 1;
        d *= results[global_index].p;
    };
    
    results[global_index].pow = pow;
}

void run_test(Divisor *d_results, Divisor *h_results, uint64_t thread_count) {
    uint64_t blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    check_div<<<blocks, THREADS_PER_BLOCK>>>(d_results, thread_count);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize(); 

    cudaMemcpy(h_results, d_results, thread_count * sizeof(Divisor), cudaMemcpyDeviceToHost);
    
}

__global__ void divide_num_by_kernel(uint64_t *num, uint64_t divisor, uint64_t *d_remainder) {
    uint64_t remainder = 0;
    for (int i = 72; i >= 0; i--) {
        unsigned __int128 value = ((unsigned __int128)remainder << 64) | num[i];
        num[i] = value / divisor;   // Store the quotient back in num
        remainder = value % divisor; // Update the remainder
    }

    // Store the final remainder in global memory
    *d_remainder = remainder;
}

void divide_num_by(uint64_t divisor) {
    uint64_t *d_num, *d_remainder;
    uint64_t remainder;

    // Allocate memory for the remainder on the device
    cudaMalloc((void**)&d_remainder, sizeof(uint64_t));

    // Copy num array to device memory
    cudaMalloc((void**)&d_num, 73 * sizeof(uint64_t));
    cudaMemcpy(d_num, num, 73 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Launch the kernel with one block and one thread (sequential execution)
    divide_num_by_kernel<<<1, 1>>>(d_num, divisor, d_remainder);

    // Wait for all threads to finish
    cudaDeviceSynchronize();

    // Copy the modified num array and remainder back to host
    cudaMemcpy(num, d_num, 73 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&remainder, d_remainder, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaFree(d_num);
    cudaFree(d_remainder);
}

int main() {
    create_num();
    uint64_t thread_count;
    Divisor *d_results, *h_results;
    h_results = (Divisor*)malloc(MAX_THREADS * sizeof(Divisor));
    cudaMalloc(&d_results, MAX_THREADS * sizeof(Divisor));

    divide_num_by(3*3*3*3*3);
    divide_num_by(125);
    divide_num_by(49);
    divide_num_by(13);
    divide_num_by(17*17);
    divide_num_by(29);
    divide_num_by(31);
    divide_num_by(37);
    divide_num_by(43);
    divide_num_by(61);
    divide_num_by(71);

    k = 123;
    while (k < UINT64_MAX) {
        thread_count = (k*k - k - 2) / 2;
        thread_count = thread_count > MAX_THREADS ? MAX_THREADS : thread_count;
        run_test(d_results, h_results, thread_count);
        k+= 2*(thread_count + 2);

        for (int i = 0; i < thread_count; i++) {
            if (h_results[i].p == 0 || h_results[i].pow == 0)
                continue;
            std::cout << h_results[i].p << "^" << h_results[i].pow << std::endl;
            for (int j = 0; j < h_results[i].pow; j++)
                divide_num_by(h_results[i].p);
        }
    }

    return 0;
}
