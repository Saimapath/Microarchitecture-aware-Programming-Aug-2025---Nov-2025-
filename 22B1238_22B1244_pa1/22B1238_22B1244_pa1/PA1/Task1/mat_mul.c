/*******************************************************************
 * Author: <Name1>, <Name2>
 * Date: <Date>
 * File: mat_mul.c
 * Description: This file contains implementations of matrix multiplication
 *			    algorithms using various optimization techniques.
 *******************************************************************/

// PA 1: Matrix Multiplication

// includes
#include <stdio.h>
#include <stdlib.h>         // for malloc, free, atoi
#include <time.h>           // for time()
#include <chrono>	        // for timing
#include <xmmintrin.h> 		// for SSE
#include <immintrin.h>		// for AVX

#include "helper.h"			// for helper functions

// defines
// NOTE: you can change this value as per your requirement
#define TILE_SIZE	64	// size of the tile for blocking

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void naive_mat_mul(double *A, double *B, double *C, int size) {

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}
}

/**
 * @brief 		Task 1A: Performs matrix multiplication of two matrices using loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void loop_opt_mat_mul(double *A, double *B, double *C, int size){
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			double Aij=A[i * size + j];
			for (int k = 0; k < size; k=k+32) {
				C[i * size + k] += Aij * B[j * size + k];
				C[i * size + k+1] += Aij * B[j * size + k+1];
				C[i * size + k+2] += Aij * B[j * size + k+2];
				C[i * size + k+3] += Aij * B[j * size + k+3];

				C[i * size + k+4] += Aij * B[j * size + k+4];
				C[i * size + k+5] += Aij * B[j * size + k+5];
				C[i * size + k+6] += Aij * B[j * size + k+6];
				C[i * size + k+7] += Aij * B[j * size + k+7];

				C[i * size + k+8] += Aij * B[j * size + k+8];
				C[i * size + k+9] += Aij * B[j * size + k+9];
				C[i * size + k+10] += Aij * B[j * size + k+10];
				C[i * size + k+11] += Aij * B[j * size + k+11];
				C[i * size + k+12] += Aij * B[j * size + k+12];
				C[i * size + k+13] += Aij * B[j * size + k+13];
				C[i * size + k+14] += Aij * B[j * size + k+14];
				C[i * size + k+15] += Aij * B[j * size + k+15];

			}
		}
	}
//-------------------------------------------------------------------------------------------------------------------------------------------

}


/**
 * @brief 		Task 1B: Performs matrix multiplication of two matrices using tiling.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the tile size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------



    for (int i = 0; i < size; i=i+tile_size) {
		for (int j = 0; j < size; j=j+tile_size) {
			for (int k = 0; k < size; k=k+tile_size) {
				for (int it = i; it < i+tile_size; it++) {
					for (int jt = j; jt < j+tile_size; jt++) {
						// double Aij=A[it * tile_size + jt];
						for (int kt = k; kt < k+tile_size; kt++) {
							C[it * size + jt] += A[it * size + kt] * B[kt * size + jt];
						}
					}
				}		
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1C: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
   for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j += 8) {
            // Main SIMD loop for j, processing 8 elements at a time
            if (j + 7 < size) {
                __m512d c_vec = _mm512_setzero_pd();
                
                for (int k = 0; k < size; k++) {
                    __m512d a_val = _mm512_set1_pd(A[i * size + k]);
                    __m512d b_vec = _mm512_loadu_pd(&B[k * size + j]);
                    c_vec = _mm512_fmadd_pd(a_val, b_vec, c_vec);
                }
                
                _mm512_storeu_pd(&C[i * size + j], c_vec);
            } else {
                // Scalar cleanup loop for remaining elements in row i
                for (int jj = j; jj < size; jj++) {
                    double sum = 0.0;
                    for (int k = 0; k < size; k++) {
                        sum += A[i * size + k] * B[k * size + jj];
                    }
                    C[i * size + jj] = sum;
                }
            }
        }
    }
}

/**
 * @brief 		Task 1D: Performs matrix multiplication of two matrices using combination of tiling/SIMD/loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    //tiling+loop reorder
	for (int i = 0; i < size; i=i+tile_size) {
		for (int j = 0; j < size; j=j+tile_size) {
			for (int k = 0; k < size; k=k+tile_size) {
				for (int it = i; it < i+tile_size; it++) {
					for (int jt = j; jt < j+tile_size; jt++) {
						double Aij=A[it * tile_size + jt];
						for (int kt = k; kt < k+tile_size; kt++) {
							C[it * size + kt] += Aij * B[jt * size + kt];
						}
					}
				}		
			}
		}
	}
    
//-------------------------------------------------------------------------------------------------------------------------------------------
	//simd+loop reorder
	// for (int i = 0; i < size; i++) {
    //     for (int k = 0; k < size; k++) {
    //         // Load a single value from matrix A and broadcast it to all 8 lanes
    //         __m512d a_val = _mm512_set1_pd(A[i * size + k]);


    //         // Inner loop for j, processing 8 elements at a time
    //         for (int j = 0; j < size; j += 4) {
    //             // This loop accesses C and B contiguously.
	// 			if(j+3<size){
	// 			__m512d b_vec = _mm512_loadu_pd(&B[k * size + j]);
    //             __m512d c_vec = _mm512_loadu_pd(&C[i * size + j]);

    //             // Perform fused multiply-add on 8 elements
    //             c_vec = _mm512_fmadd_pd(a_val, b_vec, c_vec);

    //             // Store the 8 computed results back to memory
    //             _mm512_storeu_pd(&C[i * size + j], c_vec);
	// 			} else {
	// 				double Aik= A[i*size + k];
	// 				for (int jj = j; jj < size; jj++) {
    //                     C[i * size + jj] +=Aik * B[k * size + jj];
    //             	}
	// 			}
               
    //         }
    //     }
    // }
//-------------------------------------------------------------------------------------------------------------------------------------------
	//simd512+tiling
	// for (int i = 0; i < size; i=i+tile_size) {
	// 	for (int j = 0; j < size; j=j+tile_size) {
	// 		for (int k = 0; k < size; k=k+tile_size) {
				
	// 			for (int it = i; it < i+tile_size; it++) {
	// 				for (int jt = j; jt < j+tile_size; jt += 8) {
	// 					// Main SIMD loop for j, processing 8 elements at a time
	// 					if (jt + 7 < size) {
    //      					__m512d c_vec = _mm512_loadu_pd(&C[it * size + jt]);							
	// 						for (int kt = k; kt < k+tile_size; kt++) {
	// 							__m512d a_val = _mm512_set1_pd(A[it * size + kt]);
	// 							__m512d b_vec = _mm512_loadu_pd(&B[kt * size + jt]);
	// 							c_vec = _mm512_fmadd_pd(a_val, b_vec, c_vec);
	// 						}
							
	// 						_mm512_storeu_pd(&C[it * size + jt], c_vec);
	// 					} else {
	// 						// Scalar cleanup loop for remaining elements in row i
	// 						for (int jj = jt; jj < size; jj++) {
	// 							double sum = 0.0;
	// 							for (int kt = k; kt < k+tile_size; kt++) {
	// 								sum += A[it * size + kt] * B[kt * size + jj];
	// 							}
	// 							C[it * size + jj] += sum;
	// 						}
	// 					}
	// 				}
	// 			}							
	// 		}
	// 	}
	// }
//-------------------------------------------------------------------------------------------------------------------------------------------
	//simd512 + tile  + loop reorder
	// for (int i = 0; i < size; i=i+tile_size) {
	// 	for (int j = 0; j < size; j=j+tile_size) {
	// 		for (int k = 0; k < size; k=k+tile_size) {
				
	// 			for (int it = i; it < i+tile_size; it++) {
	// 				for (int kt = k; kt < k+tile_size; kt++) {
	// 					// Load a single value from matrix A and broadcast it to all 8 lanes
	// 					__m512d a_val = _mm512_set1_pd(A[it * size + kt]);

	// 					// Inner loop for j, processing 8 elements at a time
	// 					for (int jt = j; jt < j+tile_size; jt += 8) {
	// 						// This loop accesses C and B contiguously.
	// 						if(jt+7<j+tile_size){
	// 						__m512d b_vec = _mm512_loadu_pd(&B[kt * size + jt]);
	// 						__m512d c_vec = _mm512_loadu_pd(&C[it * size + jt]);

	// 						// Perform fused multiply-add on 8 elements
	// 						c_vec = _mm512_fmadd_pd(a_val, b_vec, c_vec);

	// 						// Store the 8 computed results back to memory
	// 						_mm512_storeu_pd(&C[it * size + jt], c_vec);
	// 						} else {
	// 							double Aik= A[it*size + kt];
	// 							for (int jj = jt; jj < j+tile_size && jj<size; jj++) {
	// 								C[it * size + jj] +=Aik * B[kt * size + jj];
	// 							}
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}
	// }
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Usage: %s <matrix_dimension>\n", argv[0]);
		return 0;
	}

	else {
		int size = atoi(argv[1]);

		double *A = (double *)malloc(size * size * sizeof(double));
		double *B = (double *)malloc(size * size * sizeof(double));
		double *C = (double *)calloc(size * size, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, size, size);
		initialize_matrix(B, size, size);

		// perform normal matrix multiplication
		auto start = std::chrono::high_resolution_clock::now();
		naive_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

	#ifdef OPTIMIZE_LOOP_OPT
		// Task 1a: perform matrix multiplication with loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		loop_opt_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_loop_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Loop optimized matrix multiplication took %ld ms to execute \n", time_loop_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_loop_mat_mul);
	#endif

	#ifdef OPTIMIZE_TILING
		// Task 1b: perform matrix multiplication with tiling

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		tile_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_tiling_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Tiling matrix multiplication took %ld ms to execute \n", time_tiling_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_tiling_mat_mul);
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 1c: perform matrix multiplication with SIMD instructions 

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		simd_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_simd_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		printf("SIMD matrix multiplication took %ld ms to execute \n", time_simd_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_simd_mat_mul);
	#endif

	#ifdef OPTIMIZE_COMBINED
		// Task 1d: perform matrix multiplication with combination of tiling, SIMD and loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		combination_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_combination = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Combined optimization matrix multiplication took %ld ms to execute \n", time_combination);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_combination);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
