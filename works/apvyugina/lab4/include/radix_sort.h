#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include <stdint.h>

// Unsigned integer versions
void radixSort_uint32(uint32_t* d_input, uint32_t* d_output, int n);
void radixSort_uint64(uint64_t* d_input, uint64_t* d_output, int n);

// Signed integer versions
void radixSort_int32(int32_t* d_input, int32_t* d_output, int n);
void radixSort_int64(int64_t* d_input, int64_t* d_output, int n);

#endif // RADIX_SORT_H

