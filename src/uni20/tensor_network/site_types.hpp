/**
 * \file site_types.hpp
 * \ingroup tensor_network
 * \brief Defines canonical BlockTensor types for matrix-product networks.
 */

#pragma once

#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/dual_space.hpp>

namespace uni20::tensor_network
{

/// \brief One MPS site with the physical factor on the domain side.
template <typename Scalar, class LeftBond, class Physical, class RightBond,
          BlockTensorStorage Storage = SeparateSparseBlockStorage<>>
using MpsSite = BlockTensor<Scalar, Domain<LeftBond, Physical>, Codomain<RightBond>, Storage>;

/// \brief One MPO site with input physical and left auxiliary domain factors.
template <typename Scalar, class LeftAuxiliary, class InputPhysical, class RightAuxiliary, class OutputPhysical,
          BlockTensorStorage Storage = SeparateSparseBlockStorage<>>
using MpoSite =
    BlockTensor<Scalar, Domain<LeftAuxiliary, InputPhysical>, Codomain<RightAuxiliary, OutputPhysical>, Storage>;

/// \brief Left or right MPO environment with `(bra bond, auxiliary, ket bond)` key order.
template <typename Scalar, class BraBond, class Auxiliary, class KetBond,
          BlockTensorStorage Storage = SeparateSparseBlockStorage<>>
using MpoEnvironment = BlockTensor<Scalar, Domain<BraBond, Auxiliary>, Codomain<KetBond>, Storage>;

/// \brief Two adjacent MPS sites combined into one optimization center.
template <typename Scalar, class LeftBond, class LeftPhysical, class RightPhysical, class RightBond,
          BlockTensorStorage Storage = SeparateSparseBlockStorage<>>
using TwoSiteCenter = BlockTensor<Scalar, Domain<LeftBond, LeftPhysical, RightPhysical>, Codomain<RightBond>, Storage>;

/// \brief Charge-preserving operator acting on two adjacent physical spaces.
template <typename Scalar, class LeftPhysical, class RightPhysical,
          BlockTensorStorage Storage = PackedSparseBlockStorage<>>
using TwoSiteLocalOperator = BlockTensor<Scalar, Domain<Dual<LeftPhysical>, Dual<RightPhysical>>,
                                         Codomain<Dual<LeftPhysical>, Dual<RightPhysical>>, Storage>;

/// \brief Rank-zero BlockTensor used for a scalar boundary environment.
template <typename Scalar, BlockTensorStorage Storage = SeparateSparseBlockStorage<>>
using ScalarEnvironment = BlockTensor<Scalar, Domain<>, Codomain<>, Storage>;

} // namespace uni20::tensor_network
