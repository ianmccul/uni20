#include "Matrix.hpp"

#include <atomic>

namespace tensor
{

static std::atomic_int currentId = 0;

Matrix::Matrix(void* ptr, int dim1, int dim2)
{
  int const id = currentId.fetch_add(1);

  impl = std::make_shared<MatrixImpl>();
  impl->id = id;
  impl->dim1 = dim1;
  impl->dim2 = dim2;
  impl->ptr = ptr;
}

} // namespace tensor
