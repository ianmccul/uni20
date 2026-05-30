#include "Matrix.hpp"

#include <mpi.h>

#include <cassert>
#include <mutex>

#define MATRIX_COUNT_PER_NODE 10000

namespace tensor {

static int currentId = 0;
static std::mutex lock;

Matrix::Matrix(void *ptr, int dim1, int dim2) {
  std::lock_guard<std::mutex> guard(lock);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  impl = std::make_shared<MatrixImpl>();
  impl->id = currentId + mpi_rank * MATRIX_COUNT_PER_NODE;
  impl->dim1 = dim1;
  impl->dim2 = dim2;
  impl->ptr = ptr;
  currentId++;
  assert(currentId < MATRIX_COUNT_PER_NODE);
}

}  // namespace tensor
