/**
 * \file block_tensor_concepts.hpp
 * \ingroup symmetry
 * \brief Concepts and associated types for BlockTensor-level views.
 */

#pragma once

#include <uni20/symmetry/symmetry.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Cvref-stripped BlockTensor-level object type.
template <class Tensor> using block_tensor_type_t = std::remove_cvref_t<Tensor>;

/// \brief Scalar value type exposed by a BlockTensor-level object.
template <class Tensor> using block_tensor_value_t = typename block_tensor_type_t<Tensor>::value_type;

/// \brief Logical block-key type exposed by a BlockTensor-level object.
template <class Tensor> using block_tensor_key_t = typename block_tensor_type_t<Tensor>::key_type;

/// \brief Ordered domain type exposed by a BlockTensor-level object.
template <class Tensor> using block_tensor_domain_t = typename block_tensor_type_t<Tensor>::domain_type;

/// \brief Ordered codomain type exposed by a BlockTensor-level object.
template <class Tensor> using block_tensor_codomain_t = typename block_tensor_type_t<Tensor>::codomain_type;

/// \brief Read-only dense-block TensorView exposed by stable block ordinal.
template <class Tensor>
using block_tensor_const_block_t =
    decltype(std::declval<block_tensor_type_t<Tensor> const&>().block_by_ordinal(std::size_t{}));

/// \brief Writable dense-block TensorView exposed by stable block ordinal.
template <class Tensor>
using block_tensor_mutable_block_t =
    decltype(std::declval<std::remove_reference_t<Tensor>&>().block_by_ordinal(std::size_t{}));

/// \brief Tensor-level view of a symmetry-preserving collection of dense blocks.
/// \details An owning `BlockTensor` and a non-owning mapped view may both model
///          this concept. Stored keys are unique and lexicographically ordered;
///          their positions are stable block ordinals for the lifetime of the
///          view. The dense blocks may require execution-domain acquisition.
template <class Tensor>
concept BlockTensorView = requires(block_tensor_type_t<Tensor> const& tensor, std::size_t ordinal) {
  typename block_tensor_type_t<Tensor>::element_type;
  typename block_tensor_type_t<Tensor>::value_type;
  typename block_tensor_type_t<Tensor>::key_type;
  typename block_tensor_type_t<Tensor>::domain_type;
  typename block_tensor_type_t<Tensor>::codomain_type;
  typename block_tensor_type_t<Tensor>::storage_policy;
  typename block_tensor_type_t<Tensor>::backend_selector_type;
  { tensor.symmetry() } -> std::same_as<Symmetry>;
  { tensor.domain() } -> std::same_as<block_tensor_domain_t<Tensor> const&>;
  { tensor.codomain() } -> std::same_as<block_tensor_codomain_t<Tensor> const&>;
  { block_tensor_type_t<Tensor>::order() } -> std::same_as<std::size_t>;
  { block_tensor_type_t<Tensor>::key_coordinate_count() } -> std::same_as<std::size_t>;
  { block_tensor_type_t<Tensor>::dense_block_order() } -> std::same_as<std::size_t>;
  { tensor.stored_block_count() } -> std::same_as<std::size_t>;
  { tensor.stored_keys() } -> std::same_as<std::span<block_tensor_key_t<Tensor> const>>;
  { tensor.block_by_ordinal(ordinal) } -> TensorView;
};

/// \brief BlockTensorView whose stored dense blocks support eventual mutation.
template <class Tensor>
concept MutableBlockTensorView =
    BlockTensorView<Tensor> && requires(std::remove_reference_t<Tensor>& tensor, std::size_t ordinal) {
      { tensor.block_by_ordinal(ordinal) } -> MutableTensorView;
    };

/// \brief BlockTensorView whose read-only dense blocks are immediately accessible.
template <class Tensor>
concept ImmediateBlockTensorView = BlockTensorView<Tensor> && ImmediateTensorView<block_tensor_const_block_t<Tensor>>;

/// \brief MutableBlockTensorView whose writable dense blocks are immediately accessible.
template <class Tensor>
concept MutableImmediateBlockTensorView =
    MutableBlockTensorView<Tensor> && MutableImmediateTensorView<block_tensor_mutable_block_t<Tensor>>;

/// \brief BlockTensorView whose destruction does not destroy its referenced numerical payload.
/// \details This lifetime capability permits a temporary view to be transformed
///          into another borrowed view. Dense-block descriptors materialized
///          from the temporary remain valid after the temporary is destroyed.
///          The ultimate payload owner must still outlive every view derived
///          from it.
template <class Tensor>
concept BorrowedBlockTensorView = BlockTensorView<Tensor> && requires {
  typename block_tensor_type_t<Tensor>::borrowed_block_tensor_view_tag;
};

/// \brief BlockTensorView exposing an independently scheduled read timeline per block.
template <class Tensor>
concept AsyncBlockTensorView =
    BlockTensorView<Tensor> && requires(block_tensor_type_t<Tensor> const& tensor, std::size_t ordinal) {
      tensor.async_block_by_ordinal(ordinal).read();
    };

/// \brief MutableBlockTensorView exposing an independently scheduled write timeline per block.
template <class Tensor>
concept MutableAsyncBlockTensorView =
    AsyncBlockTensorView<Tensor> && MutableBlockTensorView<Tensor> &&
    requires(std::remove_reference_t<Tensor>& tensor, std::size_t ordinal) {
      tensor.async_block_by_ordinal(ordinal).write();
    };

} // namespace uni20
