/**
 * \file two_site_split.hpp
 * \ingroup tensor_network
 * \brief Splits a two-site center and installs directional MPS factors.
 */

#pragma once

#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_repartition.hpp>
#include <uni20/symmetry/block_tensor_space_traits.hpp>
#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/tensor_network/finite_chain.hpp>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{

/// \brief Direction in which a two-site split moves the MPS canonical center.
enum class MpsSweepDirection
{
  left_to_right, ///< Absorb singular values into the right site.
  right_to_left  ///< Absorb singular values into the left site.
};

/// \brief Directionally absorbed two-site MPS factors and retained bond data.
/// \tparam FirstSite Owning left MPS site type.
/// \tparam SecondSite Owning right MPS site type.
/// \tparam SingularValues Diagonal BlockTensor over the selected bond.
/// \tparam Truncation Exact selection statistics type.
template <class FirstSite, class SecondSite, class SingularValues, class Truncation> struct TwoSiteMpsSplit
{
    /// \brief Owning left MPS site.
    FirstSite first_site;
    /// \brief Owning right MPS site.
    SecondSite second_site;
    /// \brief Selected singular values over the new internal bond.
    SingularValues singular_values;
    /// \brief Statistics for the selected and discarded singular states.
    Truncation truncation;
    /// \brief Direction used to absorb the singular values.
    MpsSweepDirection direction;
};

/// \brief Bond data retained after installing a two-site MPS split.
/// \tparam SingularValues Diagonal BlockTensor over the installed bond.
/// \tparam Truncation Exact selection statistics type.
template <class SingularValues, class Truncation> struct InstalledMpsBond
{
    /// \brief Selected singular values over the installed internal bond.
    SingularValues singular_values;
    /// \brief Statistics for the selected and discarded singular states.
    Truncation truncation;
    /// \brief Direction used to absorb the singular values.
    MpsSweepDirection direction;
};

namespace detail
{

template <class Space>
concept SplitBondSpace =
    BlockTensorSpace<Space> && BlockTensorSpaceTraits<std::remove_cvref_t<Space>>::has_block_coordinate &&
    BlockTensorSpaceTraits<std::remove_cvref_t<Space>>::has_dense_axis;

template <class Space>
concept SplitPhysicalSpace =
    BlockTensorSpace<Space> && BlockTensorSpaceTraits<std::remove_cvref_t<Space>>::has_block_coordinate &&
    !BlockTensorSpaceTraits<std::remove_cvref_t<Space>>::has_dense_axis;

template <class Storage, class Scalar>
concept MpsSiteStorage = SparseBlockStorage<Storage> && !DiagonalBlockStorage<Storage> &&
                         LocalBlockStorageFor<Storage, std::remove_cv_t<Scalar>, 3, 2> &&
                         !AsyncLocalBlockStorageFor<Storage, std::remove_cv_t<Scalar>, 3, 2>;

template <class Tensor, std::size_t Index>
using split_domain_space_t = typename block_tensor_domain_t<Tensor>::template space_type<Index>;

template <class Tensor, std::size_t Index>
using split_codomain_space_t = typename block_tensor_codomain_t<Tensor>::template space_type<Index>;

template <class Tensor>
concept TwoSiteSvdCenter =
    BlockTensorView<Tensor> && (block_tensor_domain_t<Tensor>::size() == 3) &&
    (block_tensor_codomain_t<Tensor>::size() == 1) && (block_tensor_type_t<Tensor>::key_coordinate_count() == 4) &&
    (block_tensor_type_t<Tensor>::dense_block_order() == 2) && SplitBondSpace<split_domain_space_t<Tensor, 0>> &&
    SplitPhysicalSpace<split_domain_space_t<Tensor, 1>> && SplitPhysicalSpace<split_domain_space_t<Tensor, 2>> &&
    SplitBondSpace<split_codomain_space_t<Tensor, 0>>;

template <SparseBlockStorage OutputStorage, BlockTensorView Source>
auto materialize_block_tensor_copy(Source const& source)
{
  using output_type = BlockTensor<block_tensor_value_t<Source>, block_tensor_domain_t<Source>,
                                  block_tensor_codomain_t<Source>, OutputStorage>;
  using key_type = typename output_type::key_type;
  std::vector<key_type> keys(source.stored_keys().begin(), source.stored_keys().end());
  auto result = [&] {
    if constexpr (requires {
                    output_type(source.symmetry(), source.domain(), source.codomain(), keys,
                                source.allocation_context());
                  })
      return output_type(source.symmetry(), source.domain(), source.codomain(), std::move(keys),
                         source.allocation_context());
    else
      return output_type(source.symmetry(), source.domain(), source.codomain(), std::move(keys));
  }();
  for (std::size_t ordinal = 0; ordinal < result.stored_block_count(); ++ordinal)
  {
    auto output_block = result.block_by_ordinal(ordinal);
    auto input_block = source.block_by_ordinal(ordinal);
    uni20::copy(output_block, input_block);
  }
  return result;
}

} // namespace detail

/// \brief Factorize a canonical two-site MPS center by its internal cut with performance measurements.
/// \details The input boundary is
///          `Domain<left bond, left physical, right physical> -> Codomain<right bond>`.
///          A zero-copy repartition presents the matrix boundary
///          `Domain<left bond, left physical> -> Codomain<right bond, Dual<right physical>>`
///          to the staged block-SVD implementation.
/// \param center Two-site center supported by the selected block-SVD implementation.
/// \param options Dense SVD vector options applied independently by charge.
/// \param measurements Explicit measurement policy or collector.
/// \param batch_event Event identifying the per-charge factorization batch.
/// \return Reusable decomposition whose spectrum may be selected repeatedly.
template <detail::TwoSiteSvdCenter Center, class Measurements, class Event>
  requires uni20::LapackScalar<block_tensor_value_t<Center>> && performance::BatchMeasurementPolicy<Measurements, Event>
[[nodiscard]] auto decompose_two_site_center(Center const& center, linalg::SvdOptions options,
                                             Measurements& measurements, Event batch_event)
{
  auto matrix = repartition<MorphismSide::Domain, BoundaryEnd::Right>(center);
  return block_svd(matrix, options, measurements, batch_event);
}

/// \brief Factorize a canonical two-site MPS center by its internal cut.
/// \details This ordinary overload instantiates no performance measurements.
/// \param center Two-site center supported by the selected block-SVD implementation.
/// \param options Dense SVD vector options applied independently by charge.
/// \return Reusable decomposition whose spectrum may be selected repeatedly.
template <detail::TwoSiteSvdCenter Center>
  requires uni20::LapackScalar<block_tensor_value_t<Center>>
[[nodiscard]] auto decompose_two_site_center(Center const& center, linalg::SvdOptions options = {})
{
  performance::NoMeasurements measurements;
  return decompose_two_site_center(center, options, measurements, nullptr);
}

/// \brief Materialize selected SVD states as a directional pair of owning MPS sites.
/// \details Uni20 reconstructs the repartitioned center as
///          `right_singular_vectors_adjoint * singular_values * left_singular_vectors`.
///          A left-to-right split absorbs the singular values into the second
///          site; a right-to-left split absorbs them into the first site.
/// \tparam OutputStorage Sparse local storage for both returned sites.
/// \param decomposition Reusable decomposition from `decompose_two_site_center()`.
/// \param selection Nonempty selected singular-state set.
/// \param direction Side that receives the singular values.
/// \param options Selected bond label and materialization options.
/// \return Two canonical owning MPS sites, the diagonal selected spectrum, and
///         exact truncation statistics.
/// \throws std::invalid_argument If the selection is empty or the sweep direction is invalid.
template <SparseBlockStorage OutputStorage = SeparateSparseBlockStorage<>, class Decomposition>
  requires detail::MpsSiteStorage<OutputStorage, typename Decomposition::scalar_type>
[[nodiscard]] auto materialize_two_site_mps_split(Decomposition const& decomposition,
                                                  BlockSvdSelection<typename Decomposition::real_type> const& selection,
                                                  MpsSweepDirection direction,
                                                  BlockSvdMaterializationOptions options = {})
{
  static_assert(Decomposition::domain_type::size() == 2, "two-site MPS decomposition requires a two-factor domain");
  static_assert(Decomposition::codomain_type::size() == 2, "two-site MPS decomposition requires a two-factor codomain");
  if (selection.state_ids().empty()) throw std::invalid_argument("two-site MPS split requires a retained state");
  if (direction != MpsSweepDirection::left_to_right && direction != MpsSweepDirection::right_to_left)
    throw std::invalid_argument("two-site MPS split has an invalid sweep direction");

  bool const absorb_left = direction == MpsSweepDirection::left_to_right;
  bool const absorb_right = direction == MpsSweepDirection::right_to_left;
  auto factors = uni20::detail::materialize_svd_with_absorption(decomposition, selection, std::move(options),
                                                                absorb_left, absorb_right);
  if (direction == MpsSweepDirection::left_to_right)
  {
    auto first_site = detail::materialize_block_tensor_copy<OutputStorage>(factors.right_singular_vectors_adjoint);
    auto second_view = repartition<MorphismSide::Codomain, BoundaryEnd::Right>(factors.left_singular_vectors);
    auto second_site = detail::materialize_block_tensor_copy<OutputStorage>(second_view);
    return TwoSiteMpsSplit{.first_site = std::move(first_site),
                           .second_site = std::move(second_site),
                           .singular_values = std::move(factors.singular_values),
                           .truncation = std::move(factors.truncation),
                           .direction = direction};
  }

  auto first_site = detail::materialize_block_tensor_copy<OutputStorage>(factors.right_singular_vectors_adjoint);
  auto second_view = repartition<MorphismSide::Codomain, BoundaryEnd::Right>(factors.left_singular_vectors);
  auto second_site = detail::materialize_block_tensor_copy<OutputStorage>(second_view);
  return TwoSiteMpsSplit{.first_site = std::move(first_site),
                         .second_site = std::move(second_site),
                         .singular_values = std::move(factors.singular_values),
                         .truncation = std::move(factors.truncation),
                         .direction = direction};
}

/// \brief Materialize and install a selected two-site split in a finite MPS.
/// \details Both replacement sites are constructed before `FiniteMps::replace_pair()`
///          validates and changes the chain. The returned diagonal tensor records
///          the selected Schmidt spectrum even though its values have been
///          absorbed into one installed site.
/// \param mps Finite MPS receiving the replacement pair.
/// \param first_index Index of the left site in the pair.
/// \param decomposition Reusable two-site decomposition.
/// \param selection Nonempty selected singular-state set.
/// \param direction Sweep direction controlling singular-value absorption.
/// \param options Selected internal-bond label.
/// \return Installed bond spectrum and truncation statistics.
template <typename Scalar, Space Bond, Space Physical, BlockTensorStorage SiteStorage, class Decomposition>
  requires std::same_as<Bond, BlockSpace> && detail::MpsSiteStorage<SiteStorage, Scalar>
[[nodiscard]] auto replace_two_site_from_svd(FiniteMps<Scalar, Bond, Physical, SiteStorage>& mps,
                                             std::size_t first_index, Decomposition const& decomposition,
                                             BlockSvdSelection<typename Decomposition::real_type> const& selection,
                                             MpsSweepDirection direction, BlockSvdMaterializationOptions options = {})
{
  auto split = materialize_two_site_mps_split<SiteStorage>(decomposition, selection, direction, std::move(options));
  mps.replace_pair(first_index, std::move(split.first_site), std::move(split.second_site));
  return InstalledMpsBond{.singular_values = std::move(split.singular_values),
                          .truncation = std::move(split.truncation),
                          .direction = direction};
}

} // namespace uni20::tensor_network
