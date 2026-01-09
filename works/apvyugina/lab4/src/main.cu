#include <stdio.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdint.h>
#include <climits>
#include <random>
#include <type_traits>
#include <cuda_runtime.h>
#include <set>
#include "radix_sort.h"

using namespace std;

template<typename T>
vector<T> generateRandomArray(size_t length) {
    vector<T> result;
    result.reserve(length);
    
    random_device rd;
    mt19937 gen(rd());
    
    T min_val, max_val;
    if constexpr(is_same_v<T, uint32_t>) {
        min_val = 0;
        max_val = UINT32_MAX;
    }
    else if constexpr(is_same_v<T, uint64_t>) {
        min_val = 0;
        max_val = UINT64_MAX;
    }
    else if constexpr(is_same_v<T, int32_t>) {
        min_val = INT32_MIN;
        max_val = INT32_MAX;
    }
    else if constexpr(is_same_v<T, int64_t>) {
        min_val = INT64_MIN;
        max_val = INT64_MAX;
    }
    
    uniform_int_distribution<T> dis(min_val, max_val);
    for (size_t i = 0; i < length; i++) {
        result.push_back(dis(gen));
    }
    
    return result;
}

template<typename T>
void testSort(const vector<T>& testData, const char* typeName) {
    int n = testData.size();
    
    printf("Benchmark: %s[%d] ", typeName, n);

    // CPU sort with sort
    vector<T> cpuData = testData;
    auto cpuStart = chrono::high_resolution_clock::now();
    sort(cpuData.begin(), cpuData.end());
    chrono::duration<double> cpuElapsed = chrono::high_resolution_clock::now() - cpuStart;
    double cpuTime = cpuElapsed.count();

    // GPU sort
    T* d_input;
    T* d_output;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));
    
    cudaMemcpy(d_input, testData.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    auto gpuStart = chrono::high_resolution_clock::now();
    if constexpr(is_same_v<T, uint32_t>) {
        radixSort_uint32(d_input, d_output, n);
    }
    else if constexpr(is_same_v<T, int32_t>) {
        radixSort_int32(d_input, d_output, n);
    }
    else if constexpr(is_same_v<T, uint64_t>) {
        radixSort_uint64(d_input, d_output, n);
    }
    else if constexpr(is_same_v<T, int64_t>) {
        radixSort_int64(d_input, d_output, n);
    }

    cudaDeviceSynchronize();
    chrono::duration<double> gpuElapsed = chrono::high_resolution_clock::now() - gpuStart;
    double gpuTime = gpuElapsed.count();
    
    vector<T> gpuResult(n);
    cudaMemcpy(gpuResult.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (cpuData[i] != gpuResult[i]) {
            correct = false;
            break;
        }
    }
    printf("%s\n", correct ? "OK" : "FAILED");
    printf("Time: CPU=%.5fs, GPU=%.5fs | speedup: %.2fx\n\n", cpuTime, gpuTime, (double)cpuTime / gpuTime);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    set<int> arraySizes = {1000, 100000, 5000000, 10000000};
    
    for (const auto& size : arraySizes){
        vector<int32_t> testData32 = generateRandomArray<int32_t>(size);
        testSort<int32_t>(testData32, "int32_t");

        vector<int64_t> testData64 = generateRandomArray<int64_t>(size);
        testSort<int64_t>(testData64, "int64_t");

    }

    int uint_size = 5000000;
    vector<uint32_t> testData32 = generateRandomArray<uint32_t>(uint_size);
    testSort<uint32_t>(testData32, "uint32_t");

    vector<uint64_t> testData64 = generateRandomArray<uint64_t>(uint_size);
    testSort<uint64_t>(testData64, "uint64_t");
    
    return 0;
}

