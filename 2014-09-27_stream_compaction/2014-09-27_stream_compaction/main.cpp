#pragma once

#include "StreamCompaction.h"
#include <iostream>
#include "CPUTimer.h"


////////////////////////////////////////////////////
// Function prototypes for algorithms.
////////////////////////////////////////////////////

template <typename T>
void serialScan( T *output, const T *input, const int &n );

template <typename T>
void serialScatter( T *output, const T *input, const int &n );


////////////////////////////////////////////////////
// Helper methods.
////////////////////////////////////////////////////

template <typename T>
void printArray( const T *input, const int &n );

template <typename T>
bool testTwoArraysForEquality( const T *arr1, const T *arr2, const int &n );


////////////////////////////////////////////////////
// main()
////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	////////////////////////////////////////////////////
	// Create input array for testing.
	////////////////////////////////////////////////////

	int n = 2500;
	//float input[] = { 3.0f, 4.0f, 6.0f, 7.0f, 9.0f, 10.0f };
	//float input[] = { 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 6.0f, 6.0f, 7.0f, 0.0f, 1.0f };
	float *input = new float[n];
	for ( int i = 0; i < n; ++i ) {
		input[i] = ( float )i + 1.0f;
		//input[i] = ( i % 2 ) ? 0.0f : ( float )i;
	}


	////////////////////////////////////////////////////
	// Serial scan.
	////////////////////////////////////////////////////

	float *serial_scan_result = new float[n];

	CPUTimer *serial_scan_timer = new CPUTimer();
	serialScan( serial_scan_result, input, n );
	serial_scan_timer->stop( "Serial scan time" );


	////////////////////////////////////////////////////
	// Serial scatter.
	////////////////////////////////////////////////////

	float *serial_scatter_result = new float[n];
	serialScatter( serial_scatter_result, input, n );


	////////////////////////////////////////////////////
	// Global memory naive parallel scan.
	////////////////////////////////////////////////////

	float *naive_parallel_scan_result_global = new float[n];
	naiveScanGlobalMemoryWrapper( naive_parallel_scan_result_global, input, n );

	// Validate global memory naive parallel scan result.
	printf( "Global memory naive parallel scan: %s\n\n", ( testTwoArraysForEquality( naive_parallel_scan_result_global, serial_scan_result, n ) ? "OK" : "ERROR" ) );


	////////////////////////////////////////////////////
	// Shared memory naive parallel scan.
	////////////////////////////////////////////////////

	float *naive_parallel_scan_result_shared = new float[n];
	naiveScanSharedMemoryWrapper( naive_parallel_scan_result_shared, input, n );

	// Validate global memory naive parallel scan result.
	printf( "Shared memory naive parallel scan: %s\n\n", ( testTwoArraysForEquality( naive_parallel_scan_result_shared, serial_scan_result, n ) ? "OK" : "ERROR" ) );


	////////////////////////////////////////////////////
	// Parallel scatter operation.
	////////////////////////////////////////////////////

	float *parallel_scatter_result = new float[n];
	scatterWrapper( parallel_scatter_result, input, n );

	// Validate parallel scatter result.
	printf( "Scatter operation: %s\n\n", ( testTwoArraysForEquality( parallel_scatter_result, serial_scatter_result, n ) ? "OK" : "ERROR" ) );


	////////////////////////////////////////////////////
	// Full stream compaction.
	////////////////////////////////////////////////////

	float *stream_compaction_result = NULL;
	int stream_compaction_length = 0;
	streamCompactionWrapper( &stream_compaction_result, input, n, stream_compaction_length );

	// Print stream compaction results.
	//printArray( input, n );
	//printArray( stream_compaction_result, stream_compaction_length );


	////////////////////////////////////////////////////
	// Free allocated memory.
	////////////////////////////////////////////////////

	delete[] serial_scan_result;
	delete[] naive_parallel_scan_result_global;
	delete[] naive_parallel_scan_result_shared;
	delete[] parallel_scatter_result;
	delete[] stream_compaction_result;

	// Prevent output window from closing prematurely.
	std::cin.ignore();

	return 0;
}


////////////////////////////////////////////////////
// Serial (CPU) version of an exclusive prefix sum.
// output: Pointer to a number that represents our output array.
// input: Pointer to a constant number that represents our input array.
// n: Constant reference to an integer that represents the size of our arrays.
////////////////////////////////////////////////////
template <typename T>
void serialScan( T *output, const T *input, const int &n )
{
	// Validate arguments.
	if ( n <= 0 || input == NULL || output == NULL ) {
		return;
	}

	// Perform exclusive prefix sum.
	output[0] = 0;
	for ( int i = 1; i < n; ++i ) {
		output[i] = output[i - 1] + input[i - 1];
	}
}


////////////////////////////////////////////////////
// Serial (CPU) version of the scatter section of stream compaction.
// output: Pointer to a number that represents our output array.
// input: Pointer to a constant number that represents our input array.
// n: Constant reference to an integer that represents the size of our arrays.
////////////////////////////////////////////////////
template <typename T>
void serialScatter( T *output, const T *input, const int &n )
{
	// Convert input array into a boolean array.
	// 0 in boolean array if input is 0.
	// 1 in boolean array if input is not 0.
	T *boolean_array = new T[n];
	for ( int i = 0; i < n; ++i ) {
		if ( input[i] > -EPSILON && input[i] < EPSILON ) {
			boolean_array[i] = false;
		}
		else {
			boolean_array[i] = true;
		}
	}

	// Perform a scan on the newly created boolean array.
	serialScan( output, boolean_array, n );

	// Free allocated memory.
	delete[] boolean_array;
}


////////////////////////////////////////////////////
// Helper method to print the contents of an array.
////////////////////////////////////////////////////
template <typename T>
void printArray( const T *input, const int &n )
{
	for ( int i = 0; i < n; ++i ) {
		std::cout << i << ": " << input[i] << std::endl;
	}
	std::cout << std::endl;
}


////////////////////////////////////////////////////
// Helper method to test if two arrays are equivalent.
////////////////////////////////////////////////////
template <typename T>
bool testTwoArraysForEquality( const T *arr1, const T *arr2, const int &n )
{
	for ( int i = 0; i < n; ++i ) {
		if ( arr1[i] != arr2[i] ) {
			return false;
		}
	}
	return true;
}