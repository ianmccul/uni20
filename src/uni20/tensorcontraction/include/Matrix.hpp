#pragma once
#include <cstddef>
#include <functional>
#include <memory>

namespace tensor
{

struct MatrixImpl
{
    void* ptr = nullptr;
    int id = -1;
    int node_id = -1;
    size_t dim1 = 0;
    size_t dim2 = 0;
};

class Matrix {
    std::shared_ptr<MatrixImpl> impl;

  public:
    struct Header
    { // POD — safe to send via MPI as raw bytes
        int id = -1;
        int node_id = -1;
        size_t dim1 = 0;
        size_t dim2 = 0;
    };

    Matrix() : impl(std::make_shared<MatrixImpl>()) {}
    Matrix(void* ptr, int dim1, int dim2);
    explicit Matrix(Header h) : impl(std::make_shared<MatrixImpl>(MatrixImpl{nullptr, h.id, h.node_id, h.dim1, h.dim2}))
    {}

    Header toHeader() const { return {impl->id, impl->node_id, impl->dim1, impl->dim2}; }

    size_t getFirstDim() const { return impl->dim1; }
    size_t getSecondDim() const { return impl->dim2; }
    double* getPtr() const { return static_cast<double*>(impl->ptr); }
    size_t size() const { return impl->dim1 * impl->dim2; }
    int getId() const { return impl->id; }
    size_t sizeInByte() const { return size() * sizeof(double); }
    void setPtr(void* otherPtr) { impl->ptr = otherPtr; }
    int getNodeId() const { return impl->node_id; }
    void setNodeId(int id) { impl->node_id = id; }

    bool operator==(const Matrix& other) const { return impl->id == other.impl->id; }
    bool operator<(const Matrix& other) const { return impl->id < other.impl->id; }
};

} // namespace tensor

namespace std
{
template <> struct hash<tensor::Matrix>
{
    size_t operator()(const tensor::Matrix& m) const { return std::hash<int>()(m.getId()); }
};
} // namespace std
