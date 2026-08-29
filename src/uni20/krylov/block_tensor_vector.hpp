/**
 * \file block_tensor_vector.hpp
 * \ingroup krylov
 * \brief Adapts one fixed BlockTensor structure to the Krylov vector interface.
 */

#pragma once

#include <uni20/krylov/matrix_free.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::krylov
{
namespace detail
{

template <class Tensor> struct IsSparseBlockTensorValue : std::false_type
{};

template <typename Scalar, class DomainType, class CodomainType, BlockTensorStorage Storage>
struct IsSparseBlockTensorValue<BlockTensor<Scalar, DomainType, CodomainType, Storage>>
    : std::bool_constant<SparseBlockStorage<Storage>>
{};

} // namespace detail

/// \brief Krylov vector algebra for one exact owning BlockTensor structure.
/// \details The prototype supplied at construction fixes the vector space:
///          symmetry, domain, codomain, stored-key pattern, scalar type, and
///          storage policy. \c allocate_like reproduces that structure without
///          flattening it. An operator-specific adapter may derive from or
///          contain this class and add \c matvec.
/// \tparam Tensor Owning sparse BlockTensor value type.
template <class Tensor> class BlockTensorVectorOps {
  public:
    using tensor_type = std::remove_cvref_t<Tensor>;
    using scalar_type = typename tensor_type::value_type;
    using real_type = make_real_t<scalar_type>;
    using domain_type = typename tensor_type::domain_type;
    using codomain_type = typename tensor_type::codomain_type;
    using key_type = typename tensor_type::key_type;
    static_assert(detail::IsSparseBlockTensorValue<tensor_type>::value,
                  "BlockTensorVectorOps requires an owning sparse BlockTensor value");
    static_assert(RealOrComplex<scalar_type>, "BlockTensorVectorOps requires a real or complex scalar type");

    /// \brief Freeze the Krylov vector space represented by a prototype tensor.
    explicit BlockTensorVectorOps(tensor_type const& prototype)
        : symmetry_(prototype.symmetry()), domain_(prototype.domain()), codomain_(prototype.codomain()),
          dimension_(block_tensor_dimension(prototype))
    {
      auto const keys = prototype.stored_keys();
      stored_keys_.assign(keys.begin(), keys.end());
    }

    /// \brief Return the number of stored numerical degrees of freedom.
    [[nodiscard]] auto problem_dimension() const noexcept -> std::size_t { return dimension_; }

    /// \brief Validate a vector and return the frozen problem dimension.
    /// \throws std::invalid_argument If the vector does not belong to this
    ///         exact BlockTensor vector space.
    [[nodiscard]] auto vector_dimension(tensor_type const& vector) const -> std::size_t
    {
      this->require_member(vector);
      return dimension_;
    }

    /// \brief Allocate a zero-initialized vector with the frozen structure.
    /// \throws std::invalid_argument If \p vector is not a member of this vector space.
    [[nodiscard]] auto allocate_like(tensor_type const& vector) const -> tensor_type
    {
      this->require_member(vector);
      return tensor_type(symmetry_, domain_, codomain_, stored_keys_);
    }

    /// \brief Copy one vector into another without changing either structure.
    void copy(tensor_type& output, tensor_type const& input) const
    {
      this->require_member(output);
      this->require_member(input);
      uni20::copy(output, input);
    }

    /// \brief Add a scalar multiple of one input to an output.
    void axpy(tensor_type& output, scalar_type factor, tensor_type const& input) const
    {
      this->require_member(output);
      this->require_member(input);
      uni20::axpy(output, factor, input);
    }

    /// \brief Scale one vector in place.
    void scal(tensor_type& vector, scalar_type factor) const
    {
      this->require_member(vector);
      uni20::scale(vector, factor);
    }

    /// \brief Set every stored numerical block to zero.
    void set_zero(tensor_type& vector) const
    {
      this->require_member(vector);
      uni20::set_zero(vector);
    }

    /// \brief Return the conjugate-linear inner product as a host scalar.
    [[nodiscard]] auto inner_product(tensor_type const& lhs, tensor_type const& rhs) const -> scalar_type
    {
      this->require_member(lhs);
      this->require_member(rhs);
      return uni20::inner_product_host(lhs, rhs);
    }

    /// \brief Return the Euclidean norm as a host scalar.
    [[nodiscard]] auto norm(tensor_type const& vector) const -> real_type
    {
      this->require_member(vector);
      return uni20::norm_host(vector);
    }

  private:
    [[nodiscard]] static auto block_tensor_dimension(tensor_type const& tensor) -> std::size_t
    {
      std::size_t dimension = 0;
      for (std::size_t ordinal = 0; ordinal < tensor.stored_block_count(); ++ordinal)
      {
        auto block = tensor.block_by_ordinal(ordinal);
        std::size_t block_dimension;
        if constexpr (DiagonalBlockTensorView<tensor_type>)
        {
          auto descriptor = mdspec_of(block);
          block_dimension = static_cast<std::size_t>(diagonal_components(descriptor).extent(0));
        }
        else
        {
          block_dimension = 1;
          for (std::size_t axis = 0; axis < tensor_type::dense_block_order(); ++axis)
            block_dimension *= static_cast<std::size_t>(block.extent(axis));
        }
        dimension += block_dimension;
      }
      return dimension;
    }

    void require_member(tensor_type const& vector) const
    {
      if (vector.symmetry() != symmetry_ || vector.domain() != domain_ || vector.codomain() != codomain_)
      {
        throw std::invalid_argument("BlockTensor Krylov vector has incompatible symmetry or boundaries");
      }
      if (!std::ranges::equal(vector.stored_keys(), stored_keys_))
      {
        throw std::invalid_argument("BlockTensor Krylov vector has a different stored-key pattern");
      }
    }

    Symmetry symmetry_;
    domain_type domain_;
    codomain_type codomain_;
    std::vector<key_type> stored_keys_;
    std::size_t dimension_ = 0;
};

/// \brief Combine fixed BlockTensor vector algebra with an owned matrix-free apply operation.
/// \details The operation is invoked as `operation(output, input)` only after
///          both vectors have been validated against the exact structure frozen
///          by the prototype. The operation object may retain immutable
///          Hamiltonian or environment state needed by repeated Krylov applies.
/// \tparam Tensor Owning sparse BlockTensor value type.
/// \tparam Operator Output-first matrix-free operation object.
template <class Tensor, class Operator> class BlockTensorMatrixFreeOps : public BlockTensorVectorOps<Tensor> {
  public:
    using base_type = BlockTensorVectorOps<Tensor>;
    using tensor_type = typename base_type::tensor_type;
    using operator_type = std::remove_cvref_t<Operator>;
    static_assert(std::invocable<operator_type&, tensor_type&, tensor_type const&>,
                  "BlockTensor matrix-free operation must be callable as operation(output, input)");
    static_assert(std::same_as<std::invoke_result_t<operator_type&, tensor_type&, tensor_type const&>, void>,
                  "BlockTensor matrix-free operation must return void");

    /// \brief Freeze a vector structure and take ownership of its apply operation.
    BlockTensorMatrixFreeOps(tensor_type const& prototype, operator_type operation)
        : base_type(prototype), operation_(std::move(operation))
    {}

    /// \brief Apply the stored matrix-free operation as `output <- OP*input`.
    /// \throws std::invalid_argument If either vector is outside the frozen space.
    void matvec(tensor_type& output, tensor_type const& input)
    {
      static_cast<void>(this->vector_dimension(output));
      static_cast<void>(this->vector_dimension(input));
      std::invoke(operation_, output, input);
    }

  private:
    operator_type operation_;
};

template <class Tensor, class Operator>
BlockTensorMatrixFreeOps(Tensor const&, Operator) -> BlockTensorMatrixFreeOps<std::remove_cvref_t<Tensor>, Operator>;

} // namespace uni20::krylov
