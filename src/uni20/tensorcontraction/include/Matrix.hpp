#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>

namespace tensor
{

class CudaDeviceContext;

/// \brief Describes where host storage came from.
enum class HostMemoryKind
{
  Unknown,
  Pageable,
  Pinned
};

/// \brief Portable descriptor for one logical TensorContraction matrix.
struct MatrixHeader
{
    int id = -1;
    int node_id = -1;
    size_t dim1 = 0;
    size_t dim2 = 0;
};

/// \brief Logical matrix identity and shape, independent of concrete storage.
class MatrixHandle {
    MatrixHeader header_{};

  public:
    MatrixHandle() = default;
    explicit MatrixHandle(MatrixHeader header) : header_(header) {}
    MatrixHandle(int id, size_t dim1, size_t dim2, int nodeId = -1) : header_{id, nodeId, dim1, dim2} {}

    [[nodiscard]] MatrixHeader toHeader() const { return header_; }
    [[nodiscard]] int id() const { return header_.id; }
    [[nodiscard]] int nodeId() const { return header_.node_id; }
    [[nodiscard]] size_t rows() const { return header_.dim1; }
    [[nodiscard]] size_t cols() const { return header_.dim2; }
    [[nodiscard]] size_t size() const { return header_.dim1 * header_.dim2; }
    [[nodiscard]] size_t sizeInByte() const { return this->size() * sizeof(double); }
    [[nodiscard]] bool valid() const { return header_.id >= 0; }
    void setNodeId(int id) { header_.node_id = id; }

    bool operator==(MatrixHandle const& other) const { return this->id() == other.id(); }
    bool operator<(MatrixHandle const& other) const { return this->id() < other.id(); }
};

/// \brief Host-resident view of a logical matrix.
class HostMatrixView {
    MatrixHandle handle_{};
    double* ptr_ = nullptr;
    HostMemoryKind memoryKind_ = HostMemoryKind::Unknown;

  public:
    HostMatrixView() = default;
    HostMatrixView(MatrixHandle handle, double* ptr, HostMemoryKind memoryKind)
        : handle_(handle), ptr_(ptr), memoryKind_(memoryKind)
    {}

    [[nodiscard]] MatrixHandle handle() const { return handle_; }
    [[nodiscard]] double* data() const { return ptr_; }
    [[nodiscard]] HostMemoryKind memoryKind() const { return memoryKind_; }
    [[nodiscard]] bool pinned() const { return memoryKind_ == HostMemoryKind::Pinned; }
    [[nodiscard]] bool valid() const { return handle_.valid() && ptr_ != nullptr; }
    [[nodiscard]] size_t rows() const { return handle_.rows(); }
    [[nodiscard]] size_t cols() const { return handle_.cols(); }
    [[nodiscard]] size_t size() const { return handle_.size(); }
    [[nodiscard]] size_t sizeInByte() const { return handle_.sizeInByte(); }

    [[nodiscard]] double* requireData() const
    {
      if (ptr_ == nullptr)
      {
        throw std::logic_error("TensorContraction host matrix view has no host storage");
      }
      return ptr_;
    }
};

/// \brief Device-resident view of a logical matrix.
class DeviceMatrixView {
    MatrixHandle handle_{};
    int deviceId_ = -1;
    double* ptr_ = nullptr;
    CudaDeviceContext* deviceContext_ = nullptr;
    bool contentValid_ = false;

  public:
    DeviceMatrixView() = default;
    DeviceMatrixView(MatrixHandle handle, int deviceId, double* ptr, CudaDeviceContext& deviceContext,
                     bool contentValid)
        : handle_(handle), deviceId_(deviceId), ptr_(ptr), deviceContext_(&deviceContext), contentValid_(contentValid)
    {}

    [[nodiscard]] MatrixHandle handle() const { return handle_; }
    [[nodiscard]] int deviceId() const { return deviceId_; }
    [[nodiscard]] double* data() const { return ptr_; }
    [[nodiscard]] CudaDeviceContext& deviceContext() const { return *deviceContext_; }
    [[nodiscard]] bool contentValid() const { return contentValid_; }
    [[nodiscard]] bool valid() const { return handle_.valid() && ptr_ != nullptr && deviceContext_ != nullptr; }
    [[nodiscard]] size_t rows() const { return handle_.rows(); }
    [[nodiscard]] size_t cols() const { return handle_.cols(); }
    [[nodiscard]] size_t size() const { return handle_.size(); }
    [[nodiscard]] size_t sizeInByte() const { return handle_.sizeInByte(); }
};

struct MatrixImpl
{
    void* ptr = nullptr;
    MatrixHandle handle;
    HostMemoryKind hostMemoryKind = HostMemoryKind::Unknown;
};

class Matrix {
    std::shared_ptr<MatrixImpl> impl;

  public:
    using Header = MatrixHeader; // POD safe to send via MPI as raw bytes.

    Matrix() : impl(std::make_shared<MatrixImpl>()) {}
    Matrix(void* ptr, int dim1, int dim2);
    explicit Matrix(Header h) : impl(std::make_shared<MatrixImpl>(MatrixImpl{nullptr, MatrixHandle(h)})) {}

    Header toHeader() const { return impl->handle.toHeader(); }
    MatrixHandle handle() const { return impl->handle; }

    size_t getFirstDim() const { return impl->handle.rows(); }
    size_t getSecondDim() const { return impl->handle.cols(); }
    double* getPtr() const { return static_cast<double*>(impl->ptr); }
    size_t size() const { return impl->handle.size(); }
    int getId() const { return impl->handle.id(); }
    size_t sizeInByte() const { return size() * sizeof(double); }
    void setPtr(void* otherPtr) { impl->ptr = otherPtr; }
    int getNodeId() const { return impl->handle.nodeId(); }
    void setNodeId(int id) { impl->handle.setNodeId(id); }
    bool hasHostStorage() const { return impl->ptr != nullptr; }
    HostMemoryKind hostMemoryKind() const { return impl->hostMemoryKind; }
    void setHostMemoryKind(HostMemoryKind memoryKind) { impl->hostMemoryKind = memoryKind; }
    HostMatrixView hostView() const { return HostMatrixView(this->handle(), this->getPtr(), this->hostMemoryKind()); }

    bool operator==(const Matrix& other) const { return impl->handle == other.impl->handle; }
    bool operator<(const Matrix& other) const { return impl->handle < other.impl->handle; }
};

} // namespace tensor

namespace std
{
template <> struct hash<tensor::Matrix>
{
    size_t operator()(const tensor::Matrix& m) const { return std::hash<int>()(m.getId()); }
};
} // namespace std
