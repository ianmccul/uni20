/**
 * \file environment.hpp
 * \ingroup tensor_network
 * \brief Constructs and updates symmetry-preserving MPO environments.
 */

#pragma once

#include <uni20/async/debug_scheduler.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/linalg/ops/contract.hpp>
#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/symmetry/block_tensor_space_traits.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/tensor.hpp>
#include <uni20/tensor/transform.hpp>
#include <uni20/tensor_network/site_types.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{
namespace detail
{

template <class Tensor, std::size_t Index>
using environment_domain_space_t = typename block_tensor_domain_t<Tensor>::template space_type<Index>;

template <class Tensor, std::size_t Index>
using environment_codomain_space_t = typename block_tensor_codomain_t<Tensor>::template space_type<Index>;

template <class Space>
concept EnvironmentBondSpace =
    BlockTensorSpace<Space> && BlockTensorSpaceTraits<std::remove_cvref_t<Space>>::has_block_coordinate &&
    BlockTensorSpaceTraits<std::remove_cvref_t<Space>>::has_dense_axis;

template <class Space>
concept EnvironmentDiscreteSpace =
    BlockTensorSpace<Space> && BlockTensorSpaceTraits<std::remove_cvref_t<Space>>::has_block_coordinate &&
    !BlockTensorSpaceTraits<std::remove_cvref_t<Space>>::has_dense_axis;

template <class Tensor>
concept ImmediateHostBlockTensorView =
    ImmediateBlockTensorView<Tensor> &&
    HostAccessibleMdspan<immediate_tensor_mdspan_t<block_tensor_const_block_t<Tensor>>>;

template <class Storage, class Value>
using environment_storage_block_t = typename Storage::template storage_t<Value, 3, 2>::mutable_block_type;

template <class Storage, class Value>
concept EnvironmentStorageFor =
    SparseBlockStorage<Storage> && !DiagonalBlockStorage<Storage> && LocalBlockStorageFor<Storage, Value, 3, 2> &&
    !AsyncLocalBlockStorageFor<Storage, Value, 3, 2>;

template <class Tensor>
concept EnvironmentMatrixView =
    BlockTensorView<Tensor> && (block_tensor_domain_t<Tensor>::size() == 2) &&
    (block_tensor_codomain_t<Tensor>::size() == 1) && (block_tensor_type_t<Tensor>::key_coordinate_count() == 3) &&
    (block_tensor_type_t<Tensor>::dense_block_order() == 2) &&
    EnvironmentBondSpace<environment_domain_space_t<Tensor, 0>> &&
    EnvironmentDiscreteSpace<environment_domain_space_t<Tensor, 1>> &&
    EnvironmentBondSpace<environment_codomain_space_t<Tensor, 0>>;

template <class Tensor>
concept EnvironmentMpoView =
    ImmediateHostBlockTensorView<Tensor> && (block_tensor_domain_t<Tensor>::size() == 2) &&
    (block_tensor_codomain_t<Tensor>::size() == 2) && (block_tensor_type_t<Tensor>::key_coordinate_count() == 4) &&
    (block_tensor_type_t<Tensor>::dense_block_order() == 0) &&
    EnvironmentDiscreteSpace<environment_domain_space_t<Tensor, 0>> &&
    EnvironmentDiscreteSpace<environment_domain_space_t<Tensor, 1>> &&
    EnvironmentDiscreteSpace<environment_codomain_space_t<Tensor, 0>> &&
    EnvironmentDiscreteSpace<environment_codomain_space_t<Tensor, 1>>;

template <class Environment, class BraSite, class Mpo, class KetSite>
concept CompatibleEnvironmentUpdateScalars =
    std::same_as<block_tensor_value_t<Environment>, block_tensor_value_t<BraSite>> &&
    std::same_as<block_tensor_value_t<Environment>, block_tensor_value_t<Mpo>> &&
    std::same_as<block_tensor_value_t<Environment>, block_tensor_value_t<KetSite>>;

template <class Environment, class BraSite, class Mpo, class KetSite>
concept CompatibleLeftEnvironmentUpdateSpaces =
    std::same_as<environment_domain_space_t<Environment, 0>, environment_domain_space_t<BraSite, 0>> &&
    std::same_as<environment_domain_space_t<Environment, 1>, environment_domain_space_t<Mpo, 0>> &&
    std::same_as<environment_codomain_space_t<Environment, 0>, environment_domain_space_t<KetSite, 0>> &&
    std::same_as<environment_domain_space_t<BraSite, 1>, environment_codomain_space_t<Mpo, 1>> &&
    std::same_as<environment_domain_space_t<KetSite, 1>, environment_domain_space_t<Mpo, 1>>;

template <class Environment, class BraSite, class Mpo, class KetSite>
concept CompatibleRightEnvironmentUpdateSpaces =
    std::same_as<environment_domain_space_t<Environment, 0>, environment_codomain_space_t<BraSite, 0>> &&
    std::same_as<environment_domain_space_t<Environment, 1>, environment_codomain_space_t<Mpo, 0>> &&
    std::same_as<environment_codomain_space_t<Environment, 0>, environment_codomain_space_t<KetSite, 0>> &&
    std::same_as<environment_domain_space_t<BraSite, 1>, environment_codomain_space_t<Mpo, 1>> &&
    std::same_as<environment_domain_space_t<KetSite, 1>, environment_domain_space_t<Mpo, 1>>;

enum class EnvironmentUpdateDirection
{
  left,
  right,
};

struct EnvironmentUpdateTerm
{
    BlockKey<3> output_key;
    std::size_t output_ordinal = 0;
    std::size_t environment_ordinal;
    std::size_t bra_site_ordinal;
    std::size_t mpo_ordinal;
    std::size_t ket_site_ordinal;
};

struct EnvironmentUpdatePlan
{
    std::vector<BlockKey<3>> output_keys;
    std::vector<EnvironmentUpdateTerm> terms;
    std::vector<std::size_t> group_offsets;
};

template <EnvironmentUpdateDirection Direction, EnvironmentMatrixView Environment, EnvironmentMatrixView BraSite,
          EnvironmentMpoView Mpo, EnvironmentMatrixView KetSite>
auto make_environment_update_plan(Environment const& environment, BraSite const& bra_site, Mpo const& mpo,
                                  KetSite const& ket_site) -> EnvironmentUpdatePlan
{
  EnvironmentUpdatePlan result;
  for (std::size_t environment_ordinal = 0; environment_ordinal < environment.stored_block_count();
       ++environment_ordinal)
  {
    auto const& environment_key = environment.stored_keys()[environment_ordinal];
    for (std::size_t mpo_ordinal = 0; mpo_ordinal < mpo.stored_block_count(); ++mpo_ordinal)
    {
      auto const& mpo_key = mpo.stored_keys()[mpo_ordinal];
      constexpr std::size_t mpo_auxiliary = Direction == EnvironmentUpdateDirection::left ? 0 : 2;
      if (environment_key.coordinate(1) != mpo_key.coordinate(mpo_auxiliary)) continue;

      for (std::size_t bra_ordinal = 0; bra_ordinal < bra_site.stored_block_count(); ++bra_ordinal)
      {
        auto const& bra_key = bra_site.stored_keys()[bra_ordinal];
        constexpr std::size_t site_bond = Direction == EnvironmentUpdateDirection::left ? 0 : 2;
        if (environment_key.coordinate(0) != bra_key.coordinate(site_bond) ||
            mpo_key.coordinate(3) != bra_key.coordinate(1))
          continue;

        for (std::size_t ket_ordinal = 0; ket_ordinal < ket_site.stored_block_count(); ++ket_ordinal)
        {
          auto const& ket_key = ket_site.stored_keys()[ket_ordinal];
          if (environment_key.coordinate(2) != ket_key.coordinate(site_bond) ||
              mpo_key.coordinate(1) != ket_key.coordinate(1))
            continue;

          BlockKey<3> const output_key =
              Direction == EnvironmentUpdateDirection::left
                  ? BlockKey<3>{{bra_key.coordinate(2), mpo_key.coordinate(2), ket_key.coordinate(2)}}
                  : BlockKey<3>{{bra_key.coordinate(0), mpo_key.coordinate(0), ket_key.coordinate(0)}};
          result.terms.push_back({.output_key = output_key,
                                  .environment_ordinal = environment_ordinal,
                                  .bra_site_ordinal = bra_ordinal,
                                  .mpo_ordinal = mpo_ordinal,
                                  .ket_site_ordinal = ket_ordinal});
        }
      }
    }
  }

  std::ranges::stable_sort(result.terms, {}, &EnvironmentUpdateTerm::output_key);
  for (auto& term : result.terms)
  {
    if (result.output_keys.empty() || result.output_keys.back() != term.output_key)
      result.output_keys.push_back(term.output_key);
    term.output_ordinal = result.output_keys.size() - 1;
  }
  result.group_offsets.assign(result.output_keys.size() + 1, 0);
  for (auto const& term : result.terms)
    ++result.group_offsets[term.output_ordinal + 1];
  std::partial_sum(result.group_offsets.begin(), result.group_offsets.end(), result.group_offsets.begin());
  return result;
}

template <class Function> void execute_environment_groups(SerialBlockExecution, std::size_t size, Function&& function)
{
  for (std::size_t index = 0; index < size; ++index)
    function(index);
}

template <class Function>
void execute_environment_groups(SchedulerBatchBlockExecution, std::size_t size, Function&& function)
{
  async::execute_batch(size, std::forward<Function>(function));
}

template <class LeafStorage, class Scalar, TensorView OutputBlock>
[[nodiscard]] auto make_environment_intermediate(OutputBlock const& output_block, std::size_t rows, std::size_t columns)
{
  using intermediate_type = Tensor<Scalar, 2, LeafStorage, ColumnMajor>;
  auto descriptor = mdspec_of(output_block);
  if constexpr (requires {
                  descriptor.data_descriptor().buffer().resources();
                  intermediate_type(descriptor.data_descriptor().buffer().resources(), rows, columns);
                })
  {
    return intermediate_type(descriptor.data_descriptor().buffer().resources(), rows, columns);
  }
  else
  {
    return intermediate_type(rows, columns);
  }
}

template <EnvironmentUpdateDirection Direction, MutableBlockTensorView Output, EnvironmentMatrixView Environment,
          EnvironmentMatrixView BraSite, EnvironmentMpoView Mpo, EnvironmentMatrixView KetSite>
void execute_environment_update(Output& output, EnvironmentUpdatePlan const& plan, Environment const& environment,
                                BraSite const& bra_site, Mpo const& mpo, KetSite const& ket_site)
{
  using scalar_type = block_tensor_value_t<Output>;
  using leaf_storage_policy = typename block_tensor_type_t<Output>::storage_policy::leaf_storage_policy;

  execute_environment_groups(
      typename std::remove_cvref_t<Output>::storage_policy::block_execution_policy{}, output.stored_block_count(),
      [&](std::size_t output_ordinal) {
        auto output_block = output.block_by_ordinal(output_ordinal);
        std::size_t const first = plan.group_offsets[output_ordinal];
        std::size_t const last = plan.group_offsets[output_ordinal + 1];
        for (std::size_t index = first; index < last; ++index)
        {
          auto const& term = plan.terms[index];
          auto const environment_block = environment.block_by_ordinal(term.environment_ordinal);
          auto const bra_block = bra_site.block_by_ordinal(term.bra_site_ordinal);
          auto const mpo_block = mpo.block_by_ordinal(term.mpo_ordinal);
          auto const ket_block = ket_site.block_by_ordinal(term.ket_site_ordinal);
          auto conjugated_bra = uni20::conj(bra_block);
          scalar_type const coefficient = mpo_block[];
          scalar_type const beta = index == first ? scalar_type{} : scalar_type{1};

          if constexpr (Direction == EnvironmentUpdateDirection::left)
          {
            constexpr std::array<std::pair<std::size_t, std::size_t>, 1> first_axes{std::pair{1U, 0U}};
            constexpr std::array<std::pair<std::size_t, std::size_t>, 1> second_axes{std::pair{0U, 0U}};
            auto intermediate = make_environment_intermediate<leaf_storage_policy, scalar_type>(
                output_block, static_cast<std::size_t>(environment_block.extent(0)),
                static_cast<std::size_t>(ket_block.extent(1)));
            linalg::contract(intermediate, scalar_type{1}, environment_block, ket_block, first_axes, scalar_type{});
            linalg::contract(output_block, coefficient, conjugated_bra, intermediate, second_axes, beta);
          }
          else
          {
            constexpr std::array<std::pair<std::size_t, std::size_t>, 1> first_axes{std::pair{1U, 0U}};
            constexpr std::array<std::pair<std::size_t, std::size_t>, 1> second_axes{std::pair{1U, 1U}};
            auto intermediate = make_environment_intermediate<leaf_storage_policy, scalar_type>(
                output_block, static_cast<std::size_t>(bra_block.extent(0)),
                static_cast<std::size_t>(environment_block.extent(1)));
            linalg::contract(intermediate, scalar_type{1}, conjugated_bra, environment_block, first_axes,
                             scalar_type{});
            linalg::contract(output_block, coefficient, intermediate, ket_block, second_axes, beta);
          }
        }
      });
}

template <EnvironmentUpdateDirection Direction, EnvironmentMatrixView Environment, EnvironmentMatrixView BraSite,
          EnvironmentMpoView Mpo, EnvironmentMatrixView KetSite>
void validate_environment_update(Environment const& environment, BraSite const& bra_site, Mpo const& mpo,
                                 KetSite const& ket_site)
{
  if (environment.symmetry() != bra_site.symmetry() || environment.symmetry() != mpo.symmetry() ||
      environment.symmetry() != ket_site.symmetry())
    throw std::invalid_argument("MPO environment update operands require one symmetry");

  bool compatible = false;
  if constexpr (Direction == EnvironmentUpdateDirection::left)
  {
    compatible = environment.domain().template space<0>() == bra_site.domain().template space<0>() &&
                 environment.domain().template space<1>() == mpo.domain().template space<0>() &&
                 environment.codomain().template space<0>() == ket_site.domain().template space<0>() &&
                 bra_site.domain().template space<1>() == mpo.codomain().template space<1>() &&
                 ket_site.domain().template space<1>() == mpo.domain().template space<1>();
  }
  else
  {
    compatible = environment.domain().template space<0>() == bra_site.codomain().template space<0>() &&
                 environment.domain().template space<1>() == mpo.codomain().template space<0>() &&
                 environment.codomain().template space<0>() == ket_site.codomain().template space<0>() &&
                 bra_site.domain().template space<1>() == mpo.codomain().template space<1>() &&
                 ket_site.domain().template space<1>() == mpo.domain().template space<1>();
  }
  if (!compatible)
    throw std::invalid_argument("MPO environment update has incompatible environment, site, or MPO spaces");
}

} // namespace detail

/// \brief Construct an identity MPO environment at one scalar auxiliary state.
/// \details One identity matrix block is stored for every bond sector. The
///          environment key order is `(bra bond, auxiliary, ket bond)`.
/// \tparam Value Scalar element type.
/// \tparam Storage Sparse local storage selected for the result.
/// \param bond Exact bra and ket bond space.
/// \param auxiliary MPO boundary auxiliary space.
/// \param auxiliary_index Identity-charge auxiliary state to activate.
/// \throws std::invalid_argument If symmetries differ, the index is invalid,
///         or the selected auxiliary state is not the identity charge.
template <typename Value, SparseBlockStorage Storage = SeparateSparseBlockStorage<>>
  requires Scalar<Value> && detail::EnvironmentStorageFor<Storage, Value>
[[nodiscard]] auto make_identity_mpo_environment(BlockSpace const& bond, LocalSpace const& auxiliary,
                                                 std::size_t auxiliary_index)
    -> MpoEnvironment<Value, BlockSpace, LocalSpace, BlockSpace, Storage>
{
  if (bond.symmetry() != auxiliary.symmetry())
    throw std::invalid_argument("identity MPO environment spaces require one symmetry");
  if (auxiliary_index >= auxiliary.size())
    throw std::invalid_argument("identity MPO environment auxiliary index is out of range");
  if (!is_identity(auxiliary[auxiliary_index]))
    throw std::invalid_argument("identity MPO environment requires an identity-charge auxiliary state");

  using result_type = MpoEnvironment<Value, BlockSpace, LocalSpace, BlockSpace, Storage>;
  using key_type = typename result_type::key_type;
  std::vector<key_type> keys;
  keys.reserve(bond.size());
  for (std::size_t sector = 0; sector < bond.size(); ++sector)
    keys.push_back(key_type{{sector, auxiliary_index, sector}});

  result_type result(bond.symmetry(), Domain{bond, auxiliary}, Codomain{bond}, std::move(keys));
  for (std::size_t ordinal = 0; ordinal < result.stored_block_count(); ++ordinal)
  {
    auto block = result.block_by_ordinal(ordinal);
    ColumnMajorTensor<Value, 2> identity(block.extent(0), block.extent(1));
    auto identity_span = identity.mdspan();
    for (index_type column = 0; column < identity.extent(1); ++column)
    {
      for (index_type row = 0; row < identity.extent(0); ++row)
        identity_span[row, column] = row == column ? Value{1} : Value{};
    }
    uni20::copy(block, identity);
  }
  return result;
}

/// \brief Extend a left MPO environment across distinct bra and ket MPS sites.
/// \details Computes each reachable output block as a sparse sum of
///          `coefficient * conj(bra)^T * environment * ket` contributions.
/// \tparam OutputStorage Sparse local storage selected for the result.
/// \param environment Environment immediately to the left of the site.
/// \param bra_site Bra MPS site whose left bond meets the environment.
/// \param mpo MPO site whose left auxiliary meets the environment.
/// \param ket_site Ket MPS site whose left bond meets the environment.
template <SparseBlockStorage OutputStorage = SeparateSparseBlockStorage<>, detail::EnvironmentMatrixView Environment,
          detail::EnvironmentMatrixView BraSite, detail::EnvironmentMpoView Mpo, detail::EnvironmentMatrixView KetSite>
  requires detail::CompatibleEnvironmentUpdateScalars<Environment, BraSite, Mpo, KetSite> &&
           detail::CompatibleLeftEnvironmentUpdateSpaces<Environment, BraSite, Mpo, KetSite> &&
           detail::EnvironmentStorageFor<OutputStorage, block_tensor_value_t<Environment>>
[[nodiscard]] auto extend_left_environment(Environment const& environment, BraSite const& bra_site, Mpo const& mpo,
                                           KetSite const& ket_site)
{
  using bra_bond_type = typename block_tensor_codomain_t<BraSite>::template space_type<0>;
  using auxiliary_type = typename block_tensor_codomain_t<Mpo>::template space_type<0>;
  using ket_bond_type = typename block_tensor_codomain_t<KetSite>::template space_type<0>;
  using scalar_type = block_tensor_value_t<Environment>;
  using result_type = MpoEnvironment<scalar_type, bra_bond_type, auxiliary_type, ket_bond_type, OutputStorage>;

  detail::validate_environment_update<detail::EnvironmentUpdateDirection::left>(environment, bra_site, mpo, ket_site);
  auto plan = detail::make_environment_update_plan<detail::EnvironmentUpdateDirection::left>(environment, bra_site, mpo,
                                                                                             ket_site);
  result_type result(environment.symmetry(),
                     Domain{bra_site.codomain().template space<0>(), mpo.codomain().template space<0>()},
                     Codomain{ket_site.codomain().template space<0>()}, std::move(plan.output_keys));
  detail::execute_environment_update<detail::EnvironmentUpdateDirection::left>(result, plan, environment, bra_site, mpo,
                                                                               ket_site);
  return result;
}

/// \brief Extend a left MPO environment using one site as both bra and ket.
/// \param environment Environment immediately to the left of the site.
/// \param site MPS site used as both bra and ket.
/// \param mpo MPO site whose left auxiliary meets the environment.
template <SparseBlockStorage OutputStorage = SeparateSparseBlockStorage<>, detail::EnvironmentMatrixView Environment,
          detail::EnvironmentMatrixView Site, detail::EnvironmentMpoView Mpo>
[[nodiscard]] auto extend_left_environment(Environment const& environment, Site const& site, Mpo const& mpo)
{
  return extend_left_environment<OutputStorage>(environment, site, mpo, site);
}

/// \brief Extend a right MPO environment across distinct bra and ket MPS sites.
/// \details Computes each reachable output block as a sparse sum of
///          `coefficient * conj(bra) * environment * ket^T` contributions.
/// \tparam OutputStorage Sparse local storage selected for the result.
/// \param environment Environment immediately to the right of the site.
/// \param bra_site Bra MPS site whose right bond meets the environment.
/// \param mpo MPO site whose right auxiliary meets the environment.
/// \param ket_site Ket MPS site whose right bond meets the environment.
template <SparseBlockStorage OutputStorage = SeparateSparseBlockStorage<>, detail::EnvironmentMatrixView Environment,
          detail::EnvironmentMatrixView BraSite, detail::EnvironmentMpoView Mpo, detail::EnvironmentMatrixView KetSite>
  requires detail::CompatibleEnvironmentUpdateScalars<Environment, BraSite, Mpo, KetSite> &&
           detail::CompatibleRightEnvironmentUpdateSpaces<Environment, BraSite, Mpo, KetSite> &&
           detail::EnvironmentStorageFor<OutputStorage, block_tensor_value_t<Environment>>
[[nodiscard]] auto extend_right_environment(Environment const& environment, BraSite const& bra_site, Mpo const& mpo,
                                            KetSite const& ket_site)
{
  using bra_bond_type = typename block_tensor_domain_t<BraSite>::template space_type<0>;
  using auxiliary_type = typename block_tensor_domain_t<Mpo>::template space_type<0>;
  using ket_bond_type = typename block_tensor_domain_t<KetSite>::template space_type<0>;
  using scalar_type = block_tensor_value_t<Environment>;
  using result_type = MpoEnvironment<scalar_type, bra_bond_type, auxiliary_type, ket_bond_type, OutputStorage>;

  detail::validate_environment_update<detail::EnvironmentUpdateDirection::right>(environment, bra_site, mpo, ket_site);
  auto plan = detail::make_environment_update_plan<detail::EnvironmentUpdateDirection::right>(environment, bra_site,
                                                                                              mpo, ket_site);
  result_type result(environment.symmetry(),
                     Domain{bra_site.domain().template space<0>(), mpo.domain().template space<0>()},
                     Codomain{ket_site.domain().template space<0>()}, std::move(plan.output_keys));
  detail::execute_environment_update<detail::EnvironmentUpdateDirection::right>(result, plan, environment, bra_site,
                                                                                mpo, ket_site);
  return result;
}

/// \brief Extend a right MPO environment using one site as both bra and ket.
/// \param environment Environment immediately to the right of the site.
/// \param site MPS site used as both bra and ket.
/// \param mpo MPO site whose right auxiliary meets the environment.
template <SparseBlockStorage OutputStorage = SeparateSparseBlockStorage<>, detail::EnvironmentMatrixView Environment,
          detail::EnvironmentMatrixView Site, detail::EnvironmentMpoView Mpo>
[[nodiscard]] auto extend_right_environment(Environment const& environment, Site const& site, Mpo const& mpo)
{
  return extend_right_environment<OutputStorage>(environment, site, mpo, site);
}

} // namespace uni20::tensor_network
