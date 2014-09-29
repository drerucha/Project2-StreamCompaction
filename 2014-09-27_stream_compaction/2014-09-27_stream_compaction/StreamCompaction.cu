#pragma once

#include "StreamCompaction.h"
#include <iostream>


// TODO: Clean this up so there isn't so much duplicated code present.
// For instance, the only real difference between the global and shared memory wrappers for scan is the buffer allocation.



/******************************************/
/*          Function prototypes.          */
/******************************************/


__global__
void naiveScanGlobalMemoryOneBlock( float *g_out, float *g_in, float *g_buffer, int n );

__global__
void computeBlockSumsAndInitializeBuffer( float *g_in, float *g_buffer, float *g_block_sums );

__global__
void naiveScanGlobalMemory( float *g_out, float *g_in, float *g_buffer, float *g_block_sums, int n );

__global__
void naiveScanSharedMemoryOneBlock( float *g_out, float *g_in, int n );

__global__
void computeBlockSums( float *g_in, float *g_block_sums );

__global__
void naiveScanSharedMemory( float *g_out, float *g_in, float *g_block_sums, int n );

__global__
void computeBinaryArray( float *g_out, float *g_in, int n );

__global__
void streamCompaction( float *g_out, float *g_in, float *g_scatter_results, int n );



/*************************************/
/*          KERNEL WRAPPERS          */
/*************************************/


////////////////////////////////////////////////////
// Wrapper to call kernel that performs a naive parallel prefix sum that uses global memory exclusively.
////////////////////////////////////////////////////
void naiveScanGlobalMemoryWrapper( float *h_out, const float *h_in, const int &n )
{
	float *h_block_sums;
	float *d_in, *d_out, *d_buffer, *d_block_sums;

	// Compute number of blocks.
	int num_blocks = ( int )ceil( ( float )n / ( float )BLOCK_SIZE );

	// Compute bytes needed for various array allocations.
	int size = sizeof( float ) * n;
	int buffer_size = size * 2;
	int block_sums_size = sizeof( float ) * num_blocks;

	// Allocate memory for host data.
	h_block_sums = ( float* )malloc( block_sums_size );

	// Allocate memory for device data.
	cudaMalloc( ( void** )&d_in, size );
	cudaMalloc( ( void** )&d_out, size );
	cudaMalloc( ( void** )&d_buffer, buffer_size );
	cudaMalloc( ( void** )&d_block_sums, block_sums_size );
	
	// Copy host input to device input array.
	cudaMemcpy( d_in, h_in, size, cudaMemcpyHostToDevice );

	// CUDA timers start.
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	if ( num_blocks > 1 ) {
		// Initialize h_block_sums values to 0.
		for ( int i = 0; i < num_blocks; ++i ) {
			h_block_sums[i] = 0.0f;
		}

		// Copy h_block_sums to d_block_sums.
		cudaMemcpy( d_block_sums, h_block_sums, block_sums_size, cudaMemcpyHostToDevice );

		// Kernel calls.
		computeBlockSumsAndInitializeBuffer<<< dim3( num_blocks ), BLOCK_SIZE >>>( d_in, d_buffer, d_block_sums );
		naiveScanGlobalMemory<<< dim3( num_blocks ), BLOCK_SIZE >>>( d_out, d_in, d_buffer, d_block_sums, n );
	}
	else {
		// Kernel call.
		naiveScanGlobalMemoryOneBlock<<< 1, n >>>( d_out, d_in, d_buffer, n );
	}

	// CUDA timers end.
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	std::cout << "Naive scan global memory kernel call: " << time << " milliseconds\n" << std::endl;

	// Copy device result to host output array.
	cudaMemcpy( h_out, d_out, size, cudaMemcpyDeviceToHost );

	// Release allocated memory.
	free( h_block_sums );
	cudaFree( d_in );
	cudaFree( d_out );
	cudaFree( d_buffer );
	cudaFree( d_block_sums );
}


////////////////////////////////////////////////////
// Wrapper to call kernel that performs a naive parallel prefix sum that uses shared memory.
////////////////////////////////////////////////////
void naiveScanSharedMemoryWrapper( float *h_out, const float *h_in, const int &n )
{
	float *h_block_sums;
	float *d_in, *d_out, *d_block_sums;

	// Compute number of blocks.
	int num_blocks = ( int )ceil( ( float )n / ( float )BLOCK_SIZE );

	// Compute bytes needed for various array allocations.
	int size = sizeof( float ) * n;
	int buffer_size = size * 2;
	int block_sums_size = sizeof( float ) * num_blocks;

	// Allocate memory for host data.
	h_block_sums = ( float* )malloc( block_sums_size );

	// Allocate memory for device data.
	cudaMalloc( ( void** )&d_in, size );
	cudaMalloc( ( void** )&d_out, size );
	cudaMalloc( ( void** )&d_block_sums, block_sums_size );
	
	// Copy host input to device input array.
	cudaMemcpy( d_in, h_in, size, cudaMemcpyHostToDevice );

	// CUDA timers start.
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	if ( num_blocks > 1 ) {
		// Initialize h_block_sums values to 0.
		for ( int i = 0; i < num_blocks; ++i ) {
			h_block_sums[i] = 0.0f;
		}

		// Copy h_block_sums to d_block_sums.
		cudaMemcpy( d_block_sums, h_block_sums, block_sums_size, cudaMemcpyHostToDevice );

		// Kernel calls.
		computeBlockSums<<< dim3( num_blocks ), BLOCK_SIZE >>>( d_in, d_block_sums );
		naiveScanSharedMemory<<< dim3( num_blocks ), BLOCK_SIZE, sizeof( float ) * BLOCK_SIZE * 2 >>>( d_out, d_in, d_block_sums, n );
	}
	else {
		// Kernel call.
		naiveScanSharedMemoryOneBlock<<< 1, n, buffer_size >>>( d_out, d_in, n );
	}

	// CUDA timers end.
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	std::cout << "Naive scan shared memory kernel call: " << time << " milliseconds\n" << std::endl;

	// Copy device result to host output array.
	cudaMemcpy( h_out, d_out, size, cudaMemcpyDeviceToHost );

	// Release allocated memory.
	free( h_block_sums );
	cudaFree( d_in );
	cudaFree( d_out );
	cudaFree( d_block_sums );
}


////////////////////////////////////////////////////
// Wrapper to call the kernels that perform the scatter operation for stream compaction.
// [ 0, 0, 3, 4, 0, 6, 6, 7, 0, 1 ] => [ 0, 0, 1, 1, 0, 1, 1, 1, 0, 1 ] => [ 0, 0, 0, 1, 2, 2, 3, 4, 5, 5 ]
////////////////////////////////////////////////////
void scatterWrapper( float *h_output, const float *h_input, const int &n )
{
	float *h_binary_array;
	float *d_input, *d_binary_array;

	// Compute number of blocks.
	int num_blocks = ( int )ceil( ( float )n / ( float )BLOCK_SIZE );

	// Compute bytes needed for various array allocations.
	int size = sizeof( float ) * n;

	// Allocate memory for host data.
	h_binary_array = ( float* )malloc( size );

	// Allocate memory for device data.
	cudaMalloc( ( void** )&d_input, size );
	cudaMalloc( ( void** )&d_binary_array, size );
	
	// Copy host input to device input array.
	cudaMemcpy( d_input, h_input, size, cudaMemcpyHostToDevice );

	// Kernel call.
	computeBinaryArray<<< dim3( num_blocks ), BLOCK_SIZE >>>( d_binary_array, d_input, n );

	// Copy device result to host output array.
	cudaMemcpy( h_binary_array, d_binary_array, size, cudaMemcpyDeviceToHost );

	// Perfrom parallel prefix sum on the boolean array.
	naiveScanSharedMemoryWrapper( h_output, h_binary_array, n );

	// Release allocated memory.
	free( h_binary_array );
	cudaFree( d_input );
	cudaFree( d_binary_array );
}


////////////////////////////////////////////////////
// Wrapper to call the kernels that perform full stream compaction.
////////////////////////////////////////////////////
void streamCompactionWrapper( float **h_output, const float *h_input, const int &n, int &output_length )
{
	float *h_scatter_results;
	float *d_input, *d_output, *d_scatter_array;

	// Compute bytes needed for scatter_results and d_input.
	int input_array_size = sizeof( float ) * n;

	// Get results of performing the scatter operation on the input array.
	h_scatter_results = ( float* )malloc( input_array_size );
	scatterWrapper( h_scatter_results, h_input, n );

	// Allocate memory for stream compaction result (h_output).
	output_length = ( int )h_scatter_results[n - 1] + 1;
	int output_array_size = sizeof( float ) * output_length;
	*h_output = ( float* )malloc( output_array_size );

	// Allocate memory for device data.
	cudaMalloc( ( void** )&d_input, input_array_size );
	cudaMalloc( ( void** )&d_output, output_array_size );
	cudaMalloc( ( void** )&d_scatter_array, input_array_size );

	// Copy host input to device input array.
	cudaMemcpy( d_input, h_input, input_array_size, cudaMemcpyHostToDevice );

	// Copy host scatter array to device scatter array.
	cudaMemcpy( d_scatter_array, h_scatter_results, input_array_size, cudaMemcpyHostToDevice );

	// Compute number of blocks.
	int num_blocks = ( int )ceil( ( float )n / ( float )BLOCK_SIZE );

	// Stream compaction kernel call.
	streamCompaction<<< dim3( num_blocks ), BLOCK_SIZE >>>( d_output, d_input, d_scatter_array, n );

	// Copy device result to host output array.
	cudaMemcpy( *h_output, d_output, output_array_size, cudaMemcpyDeviceToHost );

	free( h_scatter_results );
	cudaFree( d_input );
	cudaFree( d_output );
	cudaFree( d_scatter_array );
}



/**********************************************************/
/*          NAIVE SCAN (GLOBAL MEMORY) FUNCTIONS          */
/**********************************************************/


////////////////////////////////////////////////////
// Parallel (GPU) version of a naive exclusive prefix sum.
// This method can only operate on a single block.
// This method utilizes global memory exclusively.
////////////////////////////////////////////////////
__global__
void naiveScanGlobalMemoryOneBlock( float *g_out, float *g_in, float *g_buffer, int n )
{
	int threadId = threadIdx.x;
	int pout = 0;
	int pin = 1;

	// Populate g_buffer.
	// This is exclusive scan, so shift right by one and set first element to 0.
	g_buffer[pout * n + threadId] = ( threadId > 0 ) ? g_in[threadId - 1] : 0;

	// Do not continue until g_buffer is populated.
	__syncthreads();

	for ( int offset = 1; offset < n; offset *= 2 ) {
		// Swap pout and pin.
		pout = 1 - pout;
		pin = 1 - pin;
		
		if ( threadId < offset ) {
			// output[index] = input[index].
			g_buffer[pout * n + threadId] = g_buffer[pin * n + threadId];
		}
		else {
			// output[index] = input[index] + input[index - offset].
			g_buffer[pout * n + threadId] = g_buffer[pin * n + threadId] + g_buffer[pin * n + threadId - offset];
		}

		// All threads must complete this "level" before proceeding.
		__syncthreads();
	}

	// Write output.
	g_out[threadId] = g_buffer[pout * n + threadId];
}


////////////////////////////////////////////////////
// Helper method for parallel (GPU) version of naive exclusive prefix sum.
// This method initializes the global buffer.
// Additionally, this method sums up the elements in each block, and saves those sums to global memory.
////////////////////////////////////////////////////
__global__
void computeBlockSumsAndInitializeBuffer( float *g_in, float *g_buffer, float *g_block_sums )
{
	int block_thread_id = threadIdx.x;
	int global_thread_id = threadIdx.x + ( blockIdx.x * blockDim.x );

	// Populate g_buffer per block.
	// This is exclusive scan, so shift right by one and set first element to 0.
	g_buffer[global_thread_id] = ( block_thread_id > 0 ) ? g_in[global_thread_id - 1] : 0;

	// Compute sum of all elements for every block.
	atomicAdd( &g_block_sums[blockIdx.x], g_in[global_thread_id] );
}


////////////////////////////////////////////////////
// Parallel (GPU) version of a naive exclusive prefix sum.
// This method can operate on any number of blocks of any size.
// This method utilizes global memory exclusively.
////////////////////////////////////////////////////
__global__
void naiveScanGlobalMemory( float *g_out, float *g_in, float *g_buffer, float *g_block_sums, int n )
{
	int block_thread_id = threadIdx.x;
	int global_thread_id = threadIdx.x + ( blockIdx.x * blockDim.x );
	int pout = 0;
	int pin = 1;

	// In cases where blocks do not evenly divide input array, some invalid indices may be present.
	// If invalid index is detected, then return immediately.
	if ( global_thread_id > n - 1 ) {
		return;
	}

	for ( int offset = 1; offset < blockDim.x; offset *= 2 ) {
		// Swap pout and pin.
		pout = 1 - pout;
		pin = 1 - pin;
		
		if ( block_thread_id < offset ) {
			// output[index] = input[index].
			g_buffer[pout * n + global_thread_id] = g_buffer[pin * n + global_thread_id];
		}
		else {
			// output[index] = input[index] + input[index - offset].
			g_buffer[pout * n + global_thread_id] = g_buffer[pin * n + global_thread_id] + g_buffer[pin * n + global_thread_id - offset];
		}

		// All threads must complete this "level" before proceeding.
		__syncthreads();
	}

	// Compute sum of all previous blocks to add to the values in the current block.
	int previous_block_sum = 0;
	for ( int i = blockIdx.x; i > 0; --i ) {
		previous_block_sum += g_block_sums[i - 1];
	}

	// Write output.
	g_out[global_thread_id] = g_buffer[pout * n + global_thread_id] + previous_block_sum;
}



/**********************************************************/
/*          NAIVE SCAN (SHARED MEMORY) FUNCTIONS          */
/**********************************************************/

////////////////////////////////////////////////////
// Parallel (GPU) version of a naive exclusive prefix sum.
// This method can only operate on a single block.
// This method utilizes shared memory.
////////////////////////////////////////////////////
__global__
void naiveScanSharedMemoryOneBlock( float *g_out, float *g_in, int n )
{
	// Shared memory is allocated on kernel invocation.
	// buffer is twice the size of the input array.
	extern __shared__ float buffer[];

	int threadId = threadIdx.x;
	int pout = 0;
	int pin = 1;

	// Populate g_buffer.
	// This is exclusive scan, so shift right by one and set first element to 0.
	buffer[pout * n + threadId] = ( threadId > 0 ) ? g_in[threadId - 1] : 0;

	// Do not continue until buffer is populated.
	__syncthreads();

	for ( int offset = 1; offset < n; offset *= 2 ) {
		// Swap pout and pin.
		pout = 1 - pout;
		pin = 1 - pin;
		
		if ( threadId < offset ) {
			// output[index] = input[index].
			buffer[pout * n + threadId] = buffer[pin * n + threadId];
		}
		else {
			// output[index] = input[index] + input[index - offset].
			buffer[pout * n + threadId] = buffer[pin * n + threadId] + buffer[pin * n + threadId - offset];
		}

		// All threads must complete this "level" before proceeding.
		__syncthreads();
	}

	// Write output.
	g_out[threadId] = buffer[pout * n + threadId];
}


////////////////////////////////////////////////////
// Helper method for parallel (GPU) version of naive exclusive prefix sum.
// This method sums up the elements in each block, and saves those sums to global memory.
////////////////////////////////////////////////////
__global__
void computeBlockSums( float *g_in, float *g_block_sums )
{
	// Compute sum of all elements for every block.
	atomicAdd( &g_block_sums[blockIdx.x], g_in[threadIdx.x + ( blockIdx.x * blockDim.x )] );
}


////////////////////////////////////////////////////
// Parallel (GPU) version of a naive exclusive prefix sum.
// This method can operate on any number of blocks of any size.
// This method utilizes shared memory.
////////////////////////////////////////////////////
__global__
void naiveScanSharedMemory( float *g_out, float *g_in, float *g_block_sums, int n )
{
	// Shared memory is allocated on kernel invocation.
	// buffer is twice the size of the current block.
	extern __shared__ float buffer[];

	int block_thread_id = threadIdx.x;
	int global_thread_id = threadIdx.x + ( blockIdx.x * blockDim.x );
	int block_size = blockDim.x;
	int pout = 0;
	int pin = 1;

	// Populate g_buffer.
	// This is exclusive scan, so shift right by one and set first element to 0.
	buffer[pout * block_size + block_thread_id] = ( block_thread_id > 0 ) ? g_in[global_thread_id - 1] : 0;

	// Do not continue until buffer is populated.
	__syncthreads();

	// In cases where blocks do not evenly divide input array, some invalid indices may be present.
	// If invalid index is detected, then return immediately.
	if ( global_thread_id > n - 1 ) {
		return;
	}

	for ( int offset = 1; offset < blockDim.x; offset *= 2 ) {
		// Swap pout and pin.
		pout = 1 - pout;
		pin = 1 - pin;
		
		if ( block_thread_id < offset ) {
			// output[index] = input[index].
			buffer[pout * block_size + block_thread_id] = buffer[pin * block_size + block_thread_id]; // TODO.
		}
		else {
			// output[index] = input[index] + input[index - offset].
			buffer[pout * block_size + block_thread_id] = buffer[pin * block_size + block_thread_id] + buffer[pin * block_size + block_thread_id - offset]; // TODO.
		}

		// All threads must complete this "level" before proceeding.
		__syncthreads();
	}

	// Compute sum of all previous blocks to add to the values in the current block.
	int previous_block_sum = 0;
	for ( int i = blockIdx.x; i > 0; --i ) {
		previous_block_sum += g_block_sums[i - 1];
	}

	// Write output.
	g_out[global_thread_id] = buffer[pout * block_size + block_thread_id] + previous_block_sum; // TODO.
}



/***************************************/
/*          SCATTER FUNCTIONS          */
/***************************************/


////////////////////////////////////////////////////
// Returns a binary array (0s and 1s) based on an input array.
// [ 0, 0, 3, 4, 0, 6, 6, 7, 0, 1 ] => [ 0, 0, 1, 1, 0, 1, 1, 1, 0, 1 ]
////////////////////////////////////////////////////
__global__
void computeBinaryArray( float *g_out, float *g_in, int n )
{
	int global_thread_id = threadIdx.x + ( blockIdx.x * blockDim.x );

	// In cases where blocks do not evenly divide input array, some invalid indices may be present.
	// If invalid index is detected, then return immediately.
	if ( global_thread_id > n - 1 ) {
		return;
	}

	if ( g_in[global_thread_id] > -D_EPSILON && g_in[global_thread_id] < D_EPSILON ) {
		g_out[global_thread_id] = 0.0f;
	}
	else {
		g_out[global_thread_id] = 1.0f;
	}
}



/*************************************************/
/*          STREAM COMPACTION FUNCTIONS          */
/*************************************************/


////////////////////////////////////////////////////
// Perform stream compaction by looking up output indices in g_scatter_results.
////////////////////////////////////////////////////
__global__
void streamCompaction( float *g_out, float *g_in, float *g_scatter_results, int n )
{
	int global_thread_id = threadIdx.x + ( blockIdx.x * blockDim.x );

	// In cases where blocks do not evenly divide input array, some invalid indices may be present.
	// If invalid index is detected, then return immediately.
	if ( global_thread_id > n - 1 ) {
		return;
	}

	if ( g_in[global_thread_id] < -D_EPSILON || g_in[global_thread_id] > D_EPSILON ) {
		g_out[( int )g_scatter_results[global_thread_id]] = g_in[global_thread_id];
	}
}