#pragma once

#include "StreamCompaction.h"

#include <cuda_runtime.h>
#include <iostream>


////////////////////////////////////////////////////
// Function prototypes.
////////////////////////////////////////////////////

template <typename T>
void serialScan( const T *in, T *out, const int &n );


////////////////////////////////////////////////////
// main()
////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// Test.
	const int n = 6;
	int arr[n] = { 3, 4, 6, 7, 9, 10 };
	int result[n];
	serialScan( arr, result, n );
	for ( int i = 0; i < n; ++i ) {
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
	for ( int i = 0; i < n; ++i ) {
		std::cout << result[i] << " ";
	}
	std::cout << std::endl;

	// Prevent output window from closing prematurely.
	std::cin.ignore();

	return 0;
}


////////////////////////////////////////////////////
// Serial (CPU) version of an exclusive prefix sum.
// in: Pointer to a constant number that represents our input array.
// out: Pointer to a number that represents our output array.
// n: Constant reference to an integer that represents the size of our arrays.
////////////////////////////////////////////////////
template <typename T>
void serialScan( const T *in, T *out, const int &n )
{
	// Validate arguments.
	if ( n <= 0 || in == NULL || out == NULL ) {
		return;
	}

	// Perform exclusive prefix sum.
	out[0] = 0;
	for ( int i = 1; i < n; ++i ) {
		out[i] = out[i - 1] + in[i - 1];
	}
}