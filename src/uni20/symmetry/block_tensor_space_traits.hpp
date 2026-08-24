/**
 * \file block_tensor_space_traits.hpp
 * \ingroup symmetry
 * \brief Classifies tensor spaces for block-key and dense-block storage.
 */

#pragma once

#include <uni20/symmetry/block_space.hpp>
#include <uni20/symmetry/dense_space.hpp>
#include <uni20/symmetry/dual_space.hpp>
#include <uni20/symmetry/irregular_space.hpp>
#include <uni20/symmetry/local_space.hpp>
#include <uni20/symmetry/qnum_space.hpp>

#include <concepts>
#include <type_traits>

namespace uni20
{

/// \brief Compile-time storage properties of one BlockTensor boundary space.
/// \details A block coordinate represents a structural choice recorded in a
///          `BlockKey`. A dense axis represents numerical multiplicity stored
///          inside each selected block. These properties are independent.
/// \tparam SpaceType Concrete tensor-space type.
template <class SpaceType> struct BlockTensorSpaceTraits;

/// \brief `LocalSpace` selects one state but stores no within-state multiplicity.
template <> struct BlockTensorSpaceTraits<LocalSpace>
{
    static constexpr bool has_block_coordinate = true;
    static constexpr bool has_dense_axis = false;
};

/// \brief `QNumSpace` carries one fixed irrep and needs no per-block storage axis.
template <> struct BlockTensorSpaceTraits<QNumSpace>
{
    static constexpr bool has_block_coordinate = false;
    static constexpr bool has_dense_axis = false;
};

/// \brief `BlockSpace` selects a sector and stores its degeneracy dimension.
template <> struct BlockTensorSpaceTraits<BlockSpace>
{
    static constexpr bool has_block_coordinate = true;
    static constexpr bool has_dense_axis = true;
};

/// \brief `IrregularSpace` selects a stored block and stores its dimension.
template <> struct BlockTensorSpaceTraits<IrregularSpace>
{
    static constexpr bool has_block_coordinate = true;
    static constexpr bool has_dense_axis = true;
};

/// \brief `DenseSpace` has one structural choice and stores its full extent.
template <> struct BlockTensorSpaceTraits<DenseSpace>
{
    static constexpr bool has_block_coordinate = false;
    static constexpr bool has_dense_axis = true;
};

/// \brief Explicit duality preserves key-coordinate and dense-axis structure.
template <Space SpaceType> struct BlockTensorSpaceTraits<Dual<SpaceType>> : BlockTensorSpaceTraits<SpaceType>
{};

/// \brief Tensor space with an explicit block-coordinate/dense-axis classification.
/// \tparam SpaceType Candidate tensor-space type.
template <class SpaceType>
concept BlockTensorSpace = Space<std::remove_cvref_t<SpaceType>> && requires {
  { BlockTensorSpaceTraits<std::remove_cvref_t<SpaceType>>::has_block_coordinate } -> std::convertible_to<bool>;
  { BlockTensorSpaceTraits<std::remove_cvref_t<SpaceType>>::has_dense_axis } -> std::convertible_to<bool>;
};

static_assert(BlockTensorSpace<LocalSpace>);
static_assert(BlockTensorSpace<QNumSpace>);
static_assert(BlockTensorSpace<BlockSpace>);
static_assert(BlockTensorSpace<IrregularSpace>);
static_assert(BlockTensorSpace<DenseSpace>);

} // namespace uni20
