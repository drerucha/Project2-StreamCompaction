#pragma once

#ifndef _STREAM_COMPACTION
#define _STREAM_COMPACTION

#include <cuda_runtime.h>


// Constants.
const int BLOCK_SIZE = 128;
const float EPSILON = 0.0001f;
__device__ __constant__ float D_EPSILON = 0.0001f;


// Naive parallel prefix sum that uses global memory exclusively.
void naiveScanGlobalMemoryWrapper( float *h_out, const float *h_in, const int &n );

// Naive parallel prefix sum that uses shared memory.
void naiveScanSharedMemoryWrapper( float *h_out, const float *h_in, const int &n );

// Scatter operation.
void scatterWrapper( float *h_output, const float *h_input, const int &n );

// Full stream compaction.
void streamCompactionWrapper( float **h_output, const float *h_input, const int &n, int &output_length );

#endif