#include "Matrix.hpp"

#include <atomic>

namespace tensor
{

static std::atomic_int currentId = 0;

Matrix::Matrix(void* ptr, int dim1, int dim2)
{
  int const id = currentId.fetch_add(1);

  impl = std::make_shared<MatrixImpl>();
  impl->handle = MatrixHandle(id, static_cast<size_t>(dim1), static_cast<size_t>(dim2));
  impl->ptr = ptr;
  impl->hostMemoryKind = HostMemoryKind::Pageable;
}

} // namespace tensor
