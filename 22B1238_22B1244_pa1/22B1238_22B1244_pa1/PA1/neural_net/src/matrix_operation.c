#include "matrix_operation.h"
#include <immintrin.h>

Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
	for(int i = 0; i < n ; i++) {
		for (int j = 0 ; j< m ; j++) {
			for(int l = 0; l < k; l++) {
				C(i,j) += A(i,l) * B(l,j);
			}
		}
	}
	
	return C;
}

// Loop reordered matrix multiplication (ikj order for better cache locality)
Matrix MatrixOperation::ReorderedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	for(int i = 0; i < n ; i++) {
		for (int l = 0 ; l< k ; l++) {
			double Ail = A(i,l);
			for(int j = 0; j < m; j++) {
				C(i,j) += Ail * B(l,j);
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------


	return C;
}

// Loop unrolled matrix multiplication
Matrix MatrixOperation::UnrolledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);

    const int UNROLL = 16;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for(size_t i = 0; i < n ; i++) {
		for (size_t j = 0 ; j< m ; j++) {
			double sum=0;
			for(size_t l = 0; l < k; l += 8) {
				sum += A(i,l) * B(l,j);
				sum += A(i,l + 1) * B(l + 1,j);
				sum += A(i,l + 2) * B(l + 2,j);
				sum += A(i,l + 3) * B(l + 3,j);
				sum += A(i,l + 4) * B(l + 4,j);
				sum += A(i,l + 5) * B(l + 5,j);
				sum += A(i,l + 6) * B(l + 6,j);
				sum += A(i,l + 7) * B(l + 7,j);
				// sum += A(i,l + 8) * B(l + 8,j);
				// sum += A(i,l + 9) * B(l + 9,j);
				// sum += A(i,l + 10) * B(l + 10,j);
				// sum += A(i,l + 11) * B(l + 11,j);
				// sum += A(i,l + 12) * B(l + 12,j);
				// sum += A(i,l + 13) * B(l + 13,j);
				// sum += A(i,l + 14) * B(l + 14,j);
				// sum += A(i,l + 15) * B(l + 15,j);
				C(i,j) += sum;
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Tiled (blocked) matrix multiplication for cache efficiency
Matrix MatrixOperation::TiledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
    const int T = 128;   // tile size
	int i_max = 0;
	int k_max = 0;
	int j_max = 0;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
	for (int i = 0; i < n; i=i+T) {
		for (int j = 0; j < m; j=j+T) {
			for (int l = 0; l < k; l=l+T) {
				for (int it = i; it < i+T; it++) {
					for (int jt = j; jt < j+T; jt++) {
						// double Aij=A(it, jt);
						for (int kt = l; kt < l+T; kt++) {
							C(it, jt) += A(it, kt) * B(kt, jt);
						}
					}
				}		
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// SIMD vectorized matrix multiplication (using AVX2)
Matrix MatrixOperation::VectorizedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j += 8) {
            if (j + 7 < m) {
                __m512d c_vec = _mm512_setzero_pd();
                for (int l = 0; l < k; l++) {
                    __m512d a_val = _mm512_set1_pd(A(i, l));
                    __m512d b_vec = _mm512_loadu_pd(&B(l, j));
                    c_vec = _mm512_fmadd_pd(a_val, b_vec, c_vec);
                }
                _mm512_storeu_pd(&C(i, j), c_vec);
            } else {
                for (int jj = j; jj < m; jj++) {
                    double sum = 0.0;
                    for (int l = 0; l < k; l++) {
                        sum += A(i, l) * B(l, jj);
                    }
                    C(i, jj) = sum;
                }
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Optimized matrix transpose
Matrix MatrixOperation::Transpose(const Matrix& A) {
	size_t rows = A.getRows();
	size_t cols = A.getCols();
	Matrix result(cols, rows);

	// for (size_t i = 0; i < rows; ++i) {
	// 	for (size_t j = 0; j < cols; ++j) {
	// 		for (size_t ii = 0; i < rows; ++i) {
	// 			for (size_t j = 0; j < cols; ++j) {
	// 				result(j, i) = A(i, j);
			
	// 			}
	// 		}
	// 	}
	// }

	// Optimized transpose using blocking for better cache performance
	// This is a simple implementation, more advanced techniques can be applied
	// Write your code here and comment the above code
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	const int T = 16;
	const int S = 8;

	for (size_t i = 0; i < rows; i += T) {
		for (size_t j = 0; j < cols; j += T) {
			for (size_t it = i; it < i + T && it < rows; it += S) {
				for (size_t jt = j; jt < j + T && jt < cols; jt += S) {
					if (it + S < rows && jt + S < cols) {
						__m512d r0 = _mm512_loadu_pd(&A(it + 0, jt));
						__m512d r1 = _mm512_loadu_pd(&A(it + 1, jt));
						__m512d r2 = _mm512_loadu_pd(&A(it + 2, jt));
						__m512d r3 = _mm512_loadu_pd(&A(it + 3, jt));
						__m512d r4 = _mm512_loadu_pd(&A(it + 4, jt));
						__m512d r5 = _mm512_loadu_pd(&A(it + 5, jt));
						__m512d r6 = _mm512_loadu_pd(&A(it + 6, jt));
						__m512d r7 = _mm512_loadu_pd(&A(it + 7, jt));

						__m512d t0 = _mm512_unpacklo_pd(r0, r1);
						__m512d t1 = _mm512_unpackhi_pd(r0, r1);
						__m512d t2 = _mm512_unpacklo_pd(r2, r3);
						__m512d t3 = _mm512_unpackhi_pd(r2, r3);
						__m512d t4 = _mm512_unpacklo_pd(r4, r5);
						__m512d t5 = _mm512_unpackhi_pd(r4, r5);
						__m512d t6 = _mm512_unpacklo_pd(r6, r7);
						__m512d t7 = _mm512_unpackhi_pd(r6, r7);

						__m512d p0 = _mm512_shuffle_f64x2(t0, t2, 0x44);
						__m512d p1 = _mm512_shuffle_f64x2(t0, t2, 0xEE);
						__m512d p2 = _mm512_shuffle_f64x2(t1, t3, 0x44);
						__m512d p3 = _mm512_shuffle_f64x2(t1, t3, 0xEE);
						__m512d p4 = _mm512_shuffle_f64x2(t4, t6, 0x44);
						__m512d p5 = _mm512_shuffle_f64x2(t4, t6, 0xEE);
						__m512d p6 = _mm512_shuffle_f64x2(t5, t7, 0x44);
						__m512d p7 = _mm512_shuffle_f64x2(t5, t7, 0xEE);

						_mm512_storeu_pd(&result(jt + 0, it), _mm512_shuffle_f64x2(p0, p4, 0x88));
						_mm512_storeu_pd(&result(jt + 1, it), _mm512_shuffle_f64x2(p2, p6, 0x88));
						_mm512_storeu_pd(&result(jt + 2, it), _mm512_shuffle_f64x2(p0, p4, 0xDD));
						_mm512_storeu_pd(&result(jt + 3, it), _mm512_shuffle_f64x2(p2, p6, 0xDD));
						_mm512_storeu_pd(&result(jt + 4, it), _mm512_shuffle_f64x2(p1, p5, 0x88));
						_mm512_storeu_pd(&result(jt + 5, it), _mm512_shuffle_f64x2(p3, p7, 0x88));
						_mm512_storeu_pd(&result(jt + 6, it), _mm512_shuffle_f64x2(p1, p5, 0xDD));
						_mm512_storeu_pd(&result(jt + 7, it), _mm512_shuffle_f64x2(p3, p7, 0xDD));
					} else {
						for (size_t row_idx = it; row_idx < it + S && row_idx < rows; ++row_idx) {
							for (size_t col_idx = jt; col_idx < jt + S && col_idx < cols; ++col_idx) {
								result(col_idx, row_idx) = A(row_idx, col_idx);
							}
						}
					}
				}
			}
		}
	}
//-------------------------------------------------------------------------------------------------------------------------------------------

	
	return result;
}