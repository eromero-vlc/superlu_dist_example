#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <sstream>
#include <vector>

/// Return the number of seconds from some start
inline double w_time() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline void check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::stringstream s;
    s << "CUDA error: " << cudaGetErrorName(err) << ": "
      << cudaGetErrorString(err);
    throw std::runtime_error(s.str());
  }
}

inline void check(cusparseStatus_t status) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::string str = "(unknown error code)";
    // clang-format off
    if (status == CUSPARSE_STATUS_NOT_INITIALIZED          ) str = "CUSPARSE_STATUS_NOT_INITIALIZED";
    if (status == CUSPARSE_STATUS_ALLOC_FAILED             ) str = "CUSPARSE_STATUS_ALLOC_FAILED";
    if (status == CUSPARSE_STATUS_INVALID_VALUE            ) str = "CUSPARSE_STATUS_INVALID_VALUE";
    if (status == CUSPARSE_STATUS_ARCH_MISMATCH            ) str = "CUSPARSE_STATUS_ARCH_MISMATCH";
    if (status == CUSPARSE_STATUS_EXECUTION_FAILED         ) str = "CUSPARSE_STATUS_EXECUTION_FAILED";
    if (status == CUSPARSE_STATUS_INTERNAL_ERROR           ) str = "CUSPARSE_STATUS_INTERNAL_ERROR";
    if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED) str = "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    if (status == CUSPARSE_STATUS_NOT_SUPPORTED            ) str = "CUSPARSE_STATUS_NOT_SUPPORTED";
    if (status == CUSPARSE_STATUS_INSUFFICIENT_RESOURCES   ) str = "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    // clang-format on

    std::stringstream ss;
    ss << "cuSparse function returned error " << str;
    throw std::runtime_error(ss.str());
  }
}

// using Z_t = std::complex<double>;
using Z_t = double;
using int_t = int;

struct BSR_Mat_z {
  int_t row_blocks; // number of row blocks
  int_t num_blocks; // total number of nonzero blocks
  int_t block_size; // size of the block
  Z_t *vals;
  int_t *ii;
  int_t *jj;
};

void free(const BSR_Mat_z &A) {
  check(cudaFree(A.vals));
  check(cudaFree(A.ii));
  check(cudaFree(A.jj));
}

BSR_Mat_z zcreate_matrix_qcd(int dim[4], int block_size) {
  /* Generate the dimensions of the matrix */
  int_t m_blocks = (int_t)dim[0] * dim[1] * dim[2] * dim[3];

  /* Compute the number of neighbors */
  int neighbors = 1; /* local nonzero */
  for (int d = 0; d < 4; ++d) {
    if (dim[d] > 1)
      neighbors++;
    if (dim[d] > 2)
      neighbors++;
  }

  /* Create the matrix */
  int_t nnz = m_blocks * block_size * block_size * neighbors;
  std::vector<Z_t> vals(nnz);
  std::vector<int_t> ii(m_blocks + 1);
  std::vector<int_t> jj(m_blocks * neighbors);

  /* NOTE: Assume that the matrix is Hermitian, the format is CSC
           but the L and U resolution are easy to write by rows */

  for (int lt = 0, j = 0; lt < dim[3]; ++lt) {
    for (int lz = 0; lz < dim[2]; ++lz) {
      for (int ly = 0; ly < dim[1]; ++ly) {
        for (int lx = 0; lx < dim[0]; ++lx) {
          // Generate the indices of the neighbors
          auto j0 = jj.begin() + j * neighbors;
          *j0 = j; // set diagonal block
          for (int d = 0, ji = 1; d < 4; ++d) {
            for (int delta = -1;
                 delta < (dim[d] > 2 ? 2 : (dim[d] > 1 ? 0 : -1)); delta += 2) {
              int u[4] = {0, 0, 0, 0};
              u[d] = delta;
              *(j0 + ji) =
                  (lx + u[0] + dim[0]) % dim[0] +                           //
                  (ly + u[1] + dim[1]) % dim[1] * dim[0] +                  //
                  (lz + u[2] + dim[2]) % dim[2] * dim[0] * dim[1] +         //
                  (lt + u[3] + dim[3]) % dim[3] * dim[0] * dim[1] * dim[2]; //
            }
          }
          std::sort(j0, j0 + neighbors);

          // Generate the neighbors nonzeros
          for (int ji = 0; ji < neighbors; ++ji) {
            for (int b = 0; b < block_size; ++b) {
              for (int bj = 0; bj < block_size; ++bj) {
                auto val = (*(j0 + ji) == j ? (b == bj ? 1.0 : 1e-5)
                                            : (b == bj ? 1e-5 : 1e-10));
                vals[(j * neighbors + ji) * block_size * block_size + bj +
                     b * block_size] = (Z_t)val;
              }
            }
          }

          j++;
        }
      }
    }
  }

  for (int_t i = 0; i <= m_blocks; ++i)
    ii[i] = i * neighbors;

  Z_t *vals_d = nullptr;
  int_t *ii_d = nullptr;
  int_t *jj_d = nullptr;
  check(cudaMalloc((void **)&vals_d, sizeof(Z_t) * vals.size()));
  check(cudaMalloc((void **)&ii_d, sizeof(int_t) * ii.size()));
  check(cudaMalloc((void **)&jj_d, sizeof(int_t) * jj.size()));
  check(cudaMemcpy((void *)vals_d, (const void *)vals.data(),
                   sizeof(Z_t) * vals.size(), cudaMemcpyHostToDevice));
  check(cudaMemcpy((void *)ii_d, (const void *)ii.data(),
                   sizeof(int_t) * ii.size(), cudaMemcpyHostToDevice));
  check(cudaMemcpy((void *)jj_d, (const void *)jj.data(),
                   sizeof(int_t) * jj.size(), cudaMemcpyHostToDevice));
  return {m_blocks, (int_t)jj.size(), block_size, vals_d, ii_d, jj_d};
}

void test(cusparseHandle_t handle, const BSR_Mat_z &A, int nrhs, int reps) {
  int mb = A.row_blocks;
  int nnzb = A.num_blocks;
  cusparseMatDescr_t descr_M = 0;
  cusparseMatDescr_t descr_L = 0;
  cusparseMatDescr_t descr_U = 0;
  bsrilu02Info_t info_M = 0;
  bsrsm2Info_t info_L = 0;
  bsrsm2Info_t info_U = 0;
  int pBufferSize_M;
  int pBufferSize_L;
  int pBufferSize_U;
  int pBufferSize;
  void *pBuffer = 0;
  int structural_zero;
  int numerical_zero;
  const double alpha = 1.;
  const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseOperation_t trans_X = CUSPARSE_OPERATION_NON_TRANSPOSE;
  int ldX = A.row_blocks * A.block_size;
  const cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;

  // step 1: create a descriptor which contains
  check(cusparseCreateMatDescr(&descr_M));
  check(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));
  check(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));

  check(cusparseCreateMatDescr(&descr_L));
  check(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
  check(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
  check(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
  check(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT));

  check(cusparseCreateMatDescr(&descr_U));
  check(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
  check(cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL));
  check(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER));
  check(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));

  // step 2: create a empty info structure
  // we need one info for bsrilu02 and two info's for bsrsv2
  check(cusparseCreateBsrilu02Info(&info_M));
  check(cusparseCreateBsrsm2Info(&info_L));
  check(cusparseCreateBsrsm2Info(&info_U));

  // step 3: query how much memory used in bsrilu02 and bsrsv2, and allocate the
  // buffer
  check(cusparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, A.vals,
                                     A.ii, A.jj, A.block_size, info_M,
                                     &pBufferSize_M));
  check(cusparseDbsrsm2_bufferSize(handle, dir, trans_L, trans_X, mb, nrhs,
                                   nnzb, descr_L, A.vals, A.ii, A.jj,
                                   A.block_size, info_L, &pBufferSize_L));
  check(cusparseDbsrsm2_bufferSize(handle, dir, trans_U, trans_X, mb, nrhs,
                                   nnzb, descr_U, A.vals, A.ii, A.jj,
                                   A.block_size, info_U, &pBufferSize_U));

  pBufferSize = std::max(pBufferSize_M, std::max(pBufferSize_L, pBufferSize_U));

  // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
  check(cudaMalloc((void **)&pBuffer, pBufferSize));

  // step 4: perform analysis of incomplete LU factorization on M
  //         perform analysis of triangular solve on L
  //         perform analysis of triangular solve on U
  // The lower(upper) triangular part of M has the same sparsity pattern as
  // L(U), we can do analysis of bsrilu0 and bsrsv2 simultaneously.

  check(cusparseDbsrilu02_analysis(handle, dir, mb, nnzb, descr_M, A.vals, A.ii,
                                   A.jj, A.block_size, info_M, policy_M,
                                   pBuffer));
  auto status = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
  }

  check(cusparseDbsrsm2_analysis(handle, dir, trans_L, trans_X, mb, nrhs, nnzb,
                                 descr_L, A.vals, A.ii, A.jj, A.block_size,
                                 info_L, policy_L, pBuffer));
  check(cusparseDbsrsm2_analysis(handle, dir, trans_U, trans_X, mb, nrhs, nnzb,
                                 descr_U, A.vals, A.ii, A.jj, A.block_size,
                                 info_U, policy_U, pBuffer));

  // step 5: M = L * U
  check(cusparseDbsrilu02(handle, dir, mb, nnzb, descr_M, A.vals, A.ii, A.jj,
                          A.block_size, info_M, policy_M, pBuffer));
  status = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("block U(%d,%d) is not invertible\n", numerical_zero,
           numerical_zero);
  }

  auto m = mb * A.block_size;
  std::vector<Z_t> x(m * nrhs, (Z_t)1);
  Z_t *x_d = nullptr;
  Z_t *y_d = nullptr;
  Z_t *z_d = nullptr;
  check(cudaMalloc((void **)&x_d, sizeof(Z_t) * x.size()));
  check(cudaMalloc((void **)&y_d, sizeof(Z_t) * x.size()));
  check(cudaMalloc((void **)&z_d, sizeof(Z_t) * x.size()));
  check(cudaMemcpy((void *)x_d, (const void *)x.data(), sizeof(Z_t) * x.size(),
                   cudaMemcpyHostToDevice));

  check(cudaDeviceSynchronize());
  auto t0 = w_time();
  for (int i = 0; i < reps; ++i) {
    // step 6: solve L*z = x
    check(cusparseDbsrsm2_solve(handle, dir, trans_L, trans_X, mb, nrhs, nnzb,
                                &alpha, descr_L, A.vals, A.ii, A.jj,
                                A.block_size, info_L, x_d, ldX, z_d, ldX,
                                policy_L, pBuffer));

    // step 7: solve U*y = z
    check(cusparseDbsrsm2_solve(handle, dir, trans_U, trans_X, mb, nrhs, nnzb,
                                &alpha, descr_U, A.vals, A.ii, A.jj,
                                A.block_size, info_U, z_d, ldX, y_d, ldX,
                                policy_U, pBuffer));
  }
  check(cudaDeviceSynchronize());
  double dt = (w_time() - t0) / reps;
  std::cout << "ilu0 solution: " << dt << std::endl;

  // step 6: free resources
  check(cudaFree(pBuffer));
  check(cudaFree(x_d));
  check(cudaFree(y_d));
  check(cudaFree(z_d));
  check(cusparseDestroyMatDescr(descr_M));
  check(cusparseDestroyMatDescr(descr_L));
  check(cusparseDestroyMatDescr(descr_U));
  check(cusparseDestroyBsrilu02Info(info_M));
  check(cusparseDestroyBsrsm2Info(info_L));
  check(cusparseDestroyBsrsm2Info(info_U));
}

int main(int argc, char *argv[]) {
  int dim[4] = {2, 1, 1, 1}; // xyzt
  int block_size = 1;        // nonzero dense block dimension
  int nrep = 10;             // number of repetitions
  int nrhs = 1;

  // Get options
  for (int i = 1; i < argc; ++i) {
    if (strncmp("-dim=", argv[i], 5) == 0) {
      if (sscanf(argv[i] + 5, "%d %d %d %d", &dim[0], &dim[1], &dim[2],
                 &dim[3]) != 4) {
        throw std::runtime_error("-dim= should follow 64 numbers, for instance -dim='2 2 2 2'");
      }
      if (dim[0] < 1 || dim[1] < 1 || dim[2] < 1 || dim[3] < 1) {
        throw std::runtime_error("One of the dimensions is smaller than one");
      }
    } else if (strncmp("-rep=", argv[i], 5) == 0) {
      if (sscanf(argv[i] + 5, "%d", &nrep) != 1) {
        throw std::runtime_error("-rep= should follow a number, for instance -rep=3");
      }
      if (nrep < 1) {
        throw std::runtime_error("The rep should be greater than zero");
      }
    } else if (strncmp("-bs=", argv[i], 4) == 0) {
      if (sscanf(argv[i] + 4, "%d", &block_size) != 1) {
        throw std::runtime_error("-bs= should follow a number, for instance -bs=3");
      }
      if (block_size < 1) {
        throw std::runtime_error("The rep should be greater than zero");
      }
    } else if (strncmp("-n=", argv[i], 3) == 0) {
      if (sscanf(argv[i] + 3, "%d", &nrhs) != 1) {
        throw std::runtime_error("-n= should follow a number, for instance -n=3");
      }
      if (nrhs < 1) {
        throw std::runtime_error("The rhs should be greater than zero");
      }
    } else {
      throw std::runtime_error("Unknown commandline option");
    }
  }

  cusparseHandle_t handle;
  check(cusparseCreate(&handle));
  const auto A = zcreate_matrix_qcd(dim, block_size);
  test(handle, A, nrhs, nrep);

  check(cusparseDestroy(handle));

  return 0;
}
