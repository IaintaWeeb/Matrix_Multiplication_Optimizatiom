
// CS 683 (Autumn 2023)
// PA 1: The Matrix

// includes
#include <stdio.h>
#include <time.h>			// for time-keeping
#include <xmmintrin.h> 		// for intrinsic functions
#include <immintrin.h> 
#include <x86intrin.h>
// gcc -Wall -mavx -o simd_AVX simd_AVX.c

// defines
// NOTE: you can change this value as per your requirement
#define BLOCK_SIZE 20		// size of the block
#define LOOK_AHEAD 40

#define BLOCK_PREFETCH 20
#define Prefetch_jump_i 3
#define Prefetch_jump_j 5
#define Prefetch_jump_k 1
// #define OPTIMIZE_BLOCKING_SIMD

/**
 * @brief 		Generates random numbers between values fMin and fMax.
 * @param 		fMin 	lower range
 * @param 		fMax 	upper range
 * @return 		random floating point number
 */
double fRand(double fMin, double fMax) {

	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

/**
 * @brief 		Initialize a matrix of given dimension with random values.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_matrix(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = fRand(0.0001, 1.0000); // random values between 0 and 1
		}
	}
}

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 */
void normal_mat_mul(double *A, double *B, double *C, int dim) {

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			for (int k = 0; k < dim; k++) {
				C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
			}
		}
	}
	// for(int k=0; k<dim; k++) {
	// 	for(int i=0; i<dim; i++) {
	// 		double r = A[i*dim + k];
	// 		for(int j=0; j<dim; j++){
	// 			C[i*dim + j] += r * B[k*dim + j];
	// 		}
	// 	}
	// }
}
	
/**
 * @brief 		Task 1: Performs matrix multiplication of two matrices using blocking.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the block size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void blocking_mat_mul(double *A, double *B, double *C, int dim, int block_size) {

}

/**
 * @brief 		Task 2: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int dim) {
	// for(int i = 0; i < dim; i++){
	// 	for(int j = 0; j < dim; j++){
	// 		double c0 = 0;
	// 		__m256d vi, vj, vc;
	// 		for(int k = 0; k < dim; k += 4){
	// 			vi = _mm256_load_pd(A + i*dim + k);
	// 			vj = {B[k*dim+j], B[(k+1)*dim+j], B[(k+2)*dim+j], B[(k+3)*dim+j]};
	// 			vc = _mm256_mul_pd(vi, vj);
	// 			vc = _mm256_hadd_pd(vc, vc);
	// 			vc = _mm256_hadd_pd(vc, vc);
	// 			c0 += _mm256_cvtsd_f64(vc);
	// 		}
	// 		C[i*dim+j] = c0;
	// 	}
	// }

	for(int k=0; k<dim; k++){
		for(int i=0; i<dim; i++){
			__m256d _scalar = _mm256_set1_pd(A[i*dim + k]);
			for(int j=0; j<dim; j+=4){
				_mm256_store_pd(C+i*dim+j ,(_mm256_mul_pd(_scalar, 
					_mm256_broadcast_sd(B + k*dim + j)))) ;
			}
		}
	}
}

/**
 * @brief 		Task 3: Performs matrix multiplication of two matrices using software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.+
*/
void prefetch_mat_mul(double *A, double *B, double *C, int dim) {

}

/**
 * @brief 		Bonus Task 1: Performs matrix multiplication of two matrices using blocking along with SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
*/

void blocking_simd_mat_mul(double *A, double *B, double *C, int dim, int block_size) {
	
	for(int ib = 0; ib < dim; ib += block_size){
		for(int jb = 0; jb < dim; jb += block_size){
			for(int kb = 0; kb < dim; kb += block_size){

				for(int i = ib; i < ib + block_size; i++){
					for(int j = jb; j < jb + block_size; j++){
						__m256d sum = _mm256_setzero_pd();

						for(int k = kb; k < kb + block_size; k += 4){
							__m256d a = _mm256_loadu_pd(&A[i*dim + k]);
							__m256d b = {B[k*dim+j], B[(k+1)*dim+j], B[(k+2)*dim+j], B[(k+2)*dim+j]};
							sum = _mm256_hadd_pd(_mm256_mul_pd(a, b), sum);						
						}
						sum = _mm256_hadd_pd(sum, sum);
						sum = _mm256_hadd_pd(sum, sum);
						C[i * dim + j] += _mm256_cvtsd_f64(sum);
					}
				}

			}
		}
	}

}

/**
 * @brief 		Bonus Task 2: Performs matrix multiplication of two matrices using blocking along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
*/
void blocking_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size) {
	//The algorithm prefetching future B column , A row and C block
	int i, j, k, ib, jb, kb;
	for(ib = 0; ib < dim; ib+=block_size ){
		for(jb = 0; jb < dim; jb+=block_size){
			//Prefetching C block
			if(ib*dim+ jb<=(dim*dim)-Prefetch_jump_k*block_size){
				for(int x=0;x<block_size;x++){
					for(int y=0;y<block_size;y+=8) __builtin_prefetch (&C[((ib+x)*dim) + jb + (Prefetch_jump_k*block_size)+y], 1, 0);
				}
			}
			for(kb = 0; kb < dim; kb+=block_size){
				// Prefetching A and B blocks
				if(ib<=dim-(Prefetch_jump_i*block_size)){
					//We should only prefetch A block once as we are going to reuse it again and again
					if(jb==0){	
						for(int x=0;x<block_size;x++){
							for(int y=0;y<block_size;y+=8) __builtin_prefetch (&A[((ib+x+ (Prefetch_jump_i*block_size))*dim) + kb + y], 0, 1);
						}
					}
					//Block B are fetched in columns and hence should be feteched each time
					for(int x=0;x<block_size;x++){
						for(int y=0;y<block_size;y+=8) __builtin_prefetch (&B[((kb+x)*dim) + jb+ (Prefetch_jump_j*block_size )+ y], 0,2);
					}	
				}
							
				for(k = kb; k < kb+block_size; k++ ){ // 0 -> 1
					for(i = ib; i < ib+block_size; i++ ){ // 0 -> 1
						for(j = jb; j < jb+block_size; j++ )
							C[i*dim + j] += A[i*dim + k] * B[k*dim + j]; 
					}
				}
			}
		}
	}
}

/**
 * @brief 		Bonus Task 3: Performs matrix multiplication of two matrices using SIMD instructions along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_prefetch_mat_mul(double *A, double *B, double *C, int dim) {

	for (int i = 0; i < dim; i++) {
		for (int k = 0; k < dim; k++) {
				__builtin_prefetch (&A[i*dim+k+8], 0, 1);
			for (int j = 0; j < LOOK_AHEAD; j+=8) {
				// peeling
				__builtin_prefetch (&B[k*dim+j], 0, 1);
				__builtin_prefetch (&C[i*dim+j], 0, 1);
				//     spatial          temporal          spatial    
				// C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
			}

			__m256d _scalar = _mm256_set1_pd(A[i * dim + k]);

			for (int j = 0; j < dim-LOOK_AHEAD; j+=8) {
				// Unrolling
					// __builtin_prefetch (&A[i*dim+k], 0, 1);
				__builtin_prefetch (&B[k*dim+j+LOOK_AHEAD], 0, 1);
				__builtin_prefetch (&C[i*dim+j+LOOK_AHEAD], 0, 1); 
				_mm256_store_pd(C + i * dim + j ,_mm256_mul_pd(_scalar, _mm256_broadcast_sd(B + k * dim + j)));
				_mm256_store_pd(C + i * dim + j+4 ,_mm256_mul_pd(_scalar, _mm256_broadcast_sd(B + k * dim + j+4)));
			}

			for (int j = dim-LOOK_AHEAD; j < dim; j+=8) {
				_mm256_store_pd(C + i * dim + j ,_mm256_mul_pd(_scalar, _mm256_broadcast_sd(B + k * dim + j)));
				_mm256_store_pd(C + i * dim + j+4 ,_mm256_mul_pd(_scalar, _mm256_broadcast_sd(B + k * dim + j+4)));
			}
		}
	}
}


/**
 * @brief 		Bonus Task 4: Performs matrix multiplication of two matrices using blocking along with SIMD instructions and software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void blocking_simd_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size) {

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
		printf("Pass the matrix dimension as argument :)\n\n");
		return 0;
	}

	else {
		int matrix_dim = atoi(argv[1]);

		// variables definition and initialization
		clock_t t_normal_mult, t_blocking_mult, t_prefetch_mult, t_simd_mult, t_blocking_simd_mult, t_blocking_prefetch_mult, t_simd_prefetch_mult, t_blocking_simd_prefetch_mult;
		double time_normal_mult, time_blocking_mult, time_prefetch_mult, time_simd_mult, time_blocking_simd_mult, time_blocking_prefetch_mult, time_simd_prefetch_mult, time_blocking_simd_prefetch_mult;

		// double *A = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));// __attribute__ ((aligned (64)));
		// double *B = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));// __attribute__ ((aligned (64)));
		// double *C = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));// __attribute__ ((aligned (64));
		double *A = (double *)aligned_alloc(64, matrix_dim * matrix_dim * sizeof(double));
		double *B = (double *)aligned_alloc(64, matrix_dim * matrix_dim * sizeof(double));
		double *C = (double *)aligned_alloc(64, matrix_dim * matrix_dim * sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, matrix_dim, matrix_dim);
		initialize_matrix(B, matrix_dim, matrix_dim);

		// perform normal matrix multiplication
		t_normal_mult = clock();
		normal_mat_mul(A, B, C, matrix_dim);
		t_normal_mult = clock() - t_normal_mult;

		time_normal_mult = ((double)t_normal_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Normal matrix multiplication took %f seconds to execute \n\n", time_normal_mult);

	#ifdef OPTIMIZE_BLOCKING
		// Task 1: perform blocking matrix multiplication

		t_blocking_mult = clock();
		blocking_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_mult = clock() - t_blocking_mult;

		time_blocking_mult = ((double)t_blocking_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking matrix multiplication took %f seconds to execute \n", time_blocking_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_mult);
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 2: perform matrix multiplication with SIMD instructions
		t_simd_mult = clock();
		simd_mat_mul(A, B, C, matrix_dim);
		t_simd_mult = clock() - t_simd_mult;

		time_simd_mult = ((double)t_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD matrix multiplication took %f seconds to execute \n", time_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_mult);
	#endif

	#ifdef OPTIMIZE_PREFETCH
		// Task 3: perform matrix multiplication with prefetching
		t_prefetch_mult = clock();
		prefetch_mat_mul(A, B, C, matrix_dim);
		t_prefetch_mult = clock() - t_prefetch_mult;

		time_prefetch_mult = ((double)t_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Prefetching matrix multiplication took %f seconds to execute \n", time_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_prefetch_mult);
	#endif

	#ifdef OPTIMIZE_BLOCKING_SIMD
		// Bonus Task 1: perform matrix multiplication using blocking along with SIMD instructions
		t_blocking_simd_mult = clock();
		blocking_simd_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_mult = clock() - t_blocking_simd_mult;

		time_blocking_simd_mult = ((double)t_blocking_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD matrix multiplication took %f seconds to execute \n", time_blocking_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_mult);
	#endif

	#ifdef OPTIMIZE_BLOCKING_PREFETCH
		// Bonus Task 2: perform matrix multiplication using blocking along with software prefetching
		t_blocking_prefetch_mult = clock();
		blocking_prefetch_mat_mul(A, B, C, matrix_dim, BLOCK_PREFETCH);
		t_blocking_prefetch_mult = clock() - t_blocking_prefetch_mult;

		time_blocking_prefetch_mult = ((double)t_blocking_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with prefetching matrix multiplication took %f seconds to execute \n", time_blocking_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_prefetch_mult);
	#endif

	#ifdef OPTIMIZE_SIMD_PREFETCH
		// Bonus Task 3: perform matrix multiplication using SIMD instructions along with software prefetching
		t_simd_prefetch_mult = clock();
		simd_prefetch_mat_mul(A, B, C, matrix_dim);
		t_simd_prefetch_mult = clock() - t_simd_prefetch_mult;

		time_simd_prefetch_mult = ((double)t_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD with prefetching matrix multiplication took %f seconds to execute \n", time_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_prefetch_mult);
	#endif

	#ifdef OPTIMIZE_BLOCKING_SIMD_PREFETCH
		// Bonus Task 4: perform matrix multiplication using blocking, SIMD instructions and software prefetching
		t_blocking_simd_prefetch_mult = clock();
		blocking_simd_prefetch_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_prefetch_mult = clock() - t_blocking_simd_prefetch_mult;

		time_blocking_simd_prefetch_mult = ((double)t_blocking_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD and prefetching matrix multiplication took %f seconds to execute \n", time_blocking_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_prefetch_mult);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
