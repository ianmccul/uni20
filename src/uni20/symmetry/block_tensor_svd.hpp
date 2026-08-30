/**
 * \file block_tensor_svd.hpp
 * \ingroup symmetry
 * \brief Staged host singular value decompositions of bosonic Abelian BlockTensor values.
 */

#pragma once

#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/performance_measurements.hpp>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>
#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/tensor/tensor.hpp>
#include <uni20/tensor/transform.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Stable identity of one paired or side-specific block-SVD state.
/// \details The index is local to one charge-sector factorization and does not
///          depend on the state's position in the globally sorted spectrum.
struct BlockSvdStateId
{
    QNum sector;
    std::size_t index;

    /// \brief Compare sector and local index for exact equality.
    auto operator==(BlockSvdStateId const&) const -> bool = default;
};

/// \brief One charge-labelled entry in a globally sorted singular spectrum.
template <uni20::Real Real> struct BlockSvdState
{
    BlockSvdStateId id;
    Real singular_value;
};

template <uni20::Real Real> class BlockSvdSelection;

template <uni20::Real Real>
[[nodiscard]] auto make_svd_selection(std::span<BlockSvdState<Real> const> spectrum,
                                      std::span<BlockSvdStateId const> requested) -> BlockSvdSelection<Real>;

template <uni20::Real Real>
[[nodiscard]] auto select_svd_states(std::span<BlockSvdState<Real> const> spectrum,
                                     linalg::SvdTruncationPolicy<Real> const& policy = {}) -> BlockSvdSelection<Real>;

/// \brief Arbitrary paired-state selection from a block-SVD spectrum.
template <uni20::Real Real> class BlockSvdSelection {
  public:
    /// \brief Return selected state identities in global spectrum order.
    auto state_ids() const noexcept -> std::span<BlockSvdStateId const> { return state_ids_; }

    /// \brief Return statistics for this exact state set.
    auto truncation() const noexcept -> linalg::SvdTruncationInfo<Real> const& { return truncation_; }

  private:
    BlockSvdSelection(std::vector<BlockSvdStateId> state_ids, linalg::SvdTruncationInfo<Real> truncation)
        : state_ids_(std::move(state_ids)), truncation_(std::move(truncation))
    {}

    template <uni20::Real OtherReal>
    friend auto make_svd_selection(std::span<BlockSvdState<OtherReal> const> spectrum,
                                   std::span<BlockSvdStateId const> requested) -> BlockSvdSelection<OtherReal>;

    template <uni20::Real OtherReal>
    friend auto select_svd_states(std::span<BlockSvdState<OtherReal> const> spectrum,
                                  linalg::SvdTruncationPolicy<OtherReal> const& policy) -> BlockSvdSelection<OtherReal>;

    std::vector<BlockSvdStateId> state_ids_;
    linalg::SvdTruncationInfo<Real> truncation_;
};

/// \brief Selection of singular vectors which need not form paired triplets.
class BlockSvdVectorSelection {
  public:
    /// \brief Construct from charge-sector vector identities.
    explicit BlockSvdVectorSelection(std::vector<BlockSvdStateId> state_ids) : state_ids_(std::move(state_ids)) {}

    /// \brief Return selected vector identities in canonical sector order.
    auto state_ids() const noexcept -> std::span<BlockSvdStateId const> { return state_ids_; }

  private:
    std::vector<BlockSvdStateId> state_ids_;
};

/// \brief Options controlling one selected block-SVD materialization.
struct BlockSvdMaterializationOptions
{
    std::string bond_label = {};
};

/// \brief Materialized paired factors and statistics for one SVD selection.
template <class LeftTensor, class SingularValues, class RightAdjointTensor, class TruncationInfo>
struct BlockSvdMaterialization
{
    LeftTensor left_singular_vectors;
    SingularValues singular_values;
    RightAdjointTensor right_singular_vectors_adjoint;
    TruncationInfo truncation;
};

namespace detail
{

template <class Boundary> struct SvdBoundaryShape;

template <template <class...> class Boundary, class... Spaces>
struct SvdBoundaryShape<Boundary<Spaces...>> : BoundaryBlockShape<Boundary<Spaces...>>
{};

template <std::size_t KeyCoordinateCount, std::size_t DenseOrder> struct SvdBoundaryFragment
{
    QNum charge;
    std::array<std::size_t, KeyCoordinateCount> coordinates{};
    std::array<std::size_t, DenseOrder> extents{};
    std::size_t flattened_extent = 1;
    std::size_t sector_offset = 0;
};

template <class Fragment> struct SvdBoundarySector
{
    QNum charge = {};
    std::vector<Fragment> fragments = {};
    std::size_t flattened_extent = 0;
};

template <std::size_t I = 0, class Boundary, class Fragment>
void enumerate_svd_boundary_fragments(Boundary const& boundary, QNum charge, Fragment fragment, std::size_t key_axis,
                                      std::size_t dense_axis, std::vector<Fragment>& fragments)
{
  if constexpr (I == Boundary::size())
  {
    fragment.charge = charge;
    fragment.flattened_extent = 1;
    for (std::size_t extent : fragment.extents)
      fragment.flattened_extent *= extent;
    fragments.push_back(std::move(fragment));
  }
  else
  {
    auto const& space = boundary.template space<I>();
    using space_type = std::remove_cvref_t<decltype(space)>;
    constexpr bool has_coordinate = BlockTensorSpaceTraits<space_type>::has_block_coordinate;
    constexpr bool has_dense_axis = BlockTensorSpaceTraits<space_type>::has_dense_axis;
    std::size_t const coordinate_count = has_coordinate ? factor_size(space) : 1;

    for (std::size_t coordinate = 0; coordinate < coordinate_count; ++coordinate)
    {
      Fragment next = fragment;
      if constexpr (has_coordinate) next.coordinates[key_axis] = coordinate;
      if constexpr (has_dense_axis) next.extents[dense_axis] = factor_extent(space, coordinate);
      enumerate_svd_boundary_fragments<I + 1>(boundary, charge + factor_qnum(space, coordinate), std::move(next),
                                              key_axis + (has_coordinate ? 1 : 0),
                                              dense_axis + (has_dense_axis ? 1 : 0), fragments);
    }
  }
}

template <class Boundary> auto make_svd_boundary_fragments(Symmetry symmetry, Boundary const& boundary)
{
  using shape = SvdBoundaryShape<Boundary>;
  using fragment_type = SvdBoundaryFragment<shape::key_coordinate_count, shape::dense_block_order>;
  std::vector<fragment_type> fragments;
  enumerate_svd_boundary_fragments(boundary, QNum::identity(symmetry), fragment_type{}, 0, 0, fragments);
  return fragments;
}

template <class Fragment>
auto group_svd_boundary_fragments(std::vector<Fragment> fragments) -> std::vector<SvdBoundarySector<Fragment>>
{
  std::map<std::uint64_t, SvdBoundarySector<Fragment>> grouped;
  for (auto& fragment : fragments)
  {
    auto const code = fragment.charge.raw_code();
    auto [position, inserted] = grouped.try_emplace(code, SvdBoundarySector<Fragment>{.charge = fragment.charge});
    static_cast<void>(inserted);
    fragment.sector_offset = position->second.flattened_extent;
    position->second.flattened_extent += fragment.flattened_extent;
    position->second.fragments.push_back(std::move(fragment));
  }

  std::vector<SvdBoundarySector<Fragment>> result;
  result.reserve(grouped.size());
  for (auto& [code, sector] : grouped)
  {
    static_cast<void>(code);
    result.push_back(std::move(sector));
  }
  return result;
}

template <std::size_t First, std::size_t Second>
auto concatenate_coordinates(std::array<std::size_t, First> const& first, std::array<std::size_t, Second> const& second)
    -> std::array<std::size_t, First + Second>
{
  std::array<std::size_t, First + Second> result{};
  std::ranges::copy(first, result.begin());
  std::ranges::copy(second, result.begin() + static_cast<std::ptrdiff_t>(First));
  return result;
}

template <std::size_t Rank>
auto column_major_indices(std::size_t linear, std::array<std::size_t, Rank> const& extents)
    -> std::array<uni20::index_type, Rank>
{
  std::array<uni20::index_type, Rank> result{};
  for (std::size_t axis = 0; axis < Rank; ++axis)
  {
    result[axis] = static_cast<uni20::index_type>(linear % extents[axis]);
    linear /= extents[axis];
  }
  return result;
}

template <class Block, std::size_t Rank>
decltype(auto) block_element(Block&& block, std::array<uni20::index_type, Rank> const& indices)
{
  return [&]<std::size_t... I>(std::index_sequence<I...>) -> decltype(auto) {
    return std::forward<Block>(block)[indices[I]...];
  }(std::make_index_sequence<Rank>{});
}

template <std::size_t First, std::size_t Second>
auto concatenate_indices(std::array<uni20::index_type, First> const& first,
                         std::array<uni20::index_type, Second> const& second)
    -> std::array<uni20::index_type, First + Second>
{
  std::array<uni20::index_type, First + Second> result{};
  std::ranges::copy(first, result.begin());
  std::ranges::copy(second, result.begin() + static_cast<std::ptrdiff_t>(First));
  return result;
}

template <uni20::Real Real> class SvdNonnegativeSum {
  public:
    void add(Real value)
    {
      Real const next = sum_ + value;
      if (sum_ >= value)
        correction_ += (sum_ - next) + value;
      else
        correction_ += (value - next) + sum_;
      sum_ = next;
    }

    auto value() const -> Real { return sum_ + correction_; }

  private:
    Real sum_{};
    Real correction_{};
};

inline auto same_svd_state_id(BlockSvdStateId const& lhs, BlockSvdStateId const& rhs) -> bool { return lhs == rhs; }

template <uni20::Real Real>
auto canonical_svd_selection(std::span<BlockSvdState<Real> const> spectrum, std::span<BlockSvdStateId const> requested)
    -> std::pair<std::vector<BlockSvdStateId>, std::vector<unsigned char>>
{
  std::vector<unsigned char> selected(spectrum.size(), 0);
  for (BlockSvdStateId const& id : requested)
  {
    auto const found =
        std::ranges::find_if(spectrum, [&](auto const& state) { return same_svd_state_id(state.id, id); });
    if (found == spectrum.end()) throw std::invalid_argument("block-SVD selection contains an unknown state");
    auto const ordinal = static_cast<std::size_t>(found - spectrum.begin());
    if (selected[ordinal]) throw std::invalid_argument("block-SVD selection contains a repeated state");
    selected[ordinal] = 1;
  }

  std::vector<BlockSvdStateId> canonical;
  canonical.reserve(requested.size());
  for (std::size_t ordinal = 0; ordinal < spectrum.size(); ++ordinal)
  {
    if (selected[ordinal]) canonical.push_back(spectrum[ordinal].id);
  }
  return {std::move(canonical), std::move(selected)};
}

template <uni20::Real Real>
auto summarize_svd_selection(std::span<BlockSvdState<Real> const> spectrum, std::span<unsigned char const> selected)
    -> linalg::SvdTruncationInfo<Real>
{
  Real scale{};
  if (!spectrum.empty()) scale = spectrum.front().singular_value;

  SvdNonnegativeSum<Real> total;
  SvdNonnegativeSum<Real> discarded;
  std::optional<Real> smallest_retained;
  std::optional<Real> largest_discarded;
  std::size_t retained_rank = 0;
  for (std::size_t ordinal = 0; ordinal < spectrum.size(); ++ordinal)
  {
    Real const value = spectrum[ordinal].singular_value;
    if (!uni20::isfinite(value) || value < Real{})
    {
      throw std::runtime_error("block-SVD spectrum contains an invalid singular value");
    }
    Real const square = scale == Real{} ? Real{} : (value / scale) * (value / scale);
    total.add(square);
    if (selected[ordinal])
    {
      ++retained_rank;
      if (!smallest_retained || value < *smallest_retained) smallest_retained = value;
    }
    else if (!largest_discarded || value > *largest_discarded)
    {
      largest_discarded = value;
    }
    if (!selected[ordinal]) discarded.add(square);
  }

  Real const total_scaled = total.value();
  return {.available_rank = spectrum.size(),
          .retained_rank = retained_rank,
          .original_squared_norm = scale * scale * total_scaled,
          .discarded_weight = total_scaled == Real{} ? Real{} : discarded.value() / total_scaled,
          .smallest_retained_singular_value = smallest_retained,
          .largest_discarded_singular_value = largest_discarded};
}

template <class Selection>
concept BlockSvdStateSelection = requires(Selection const& selection) {
  { selection.state_ids() } -> std::same_as<std::span<BlockSvdStateId const>>;
};

template <class Function> void execute_svd_sector_batch(SerialBlockExecution, std::size_t size, Function&& function)
{
  for (std::size_t index = 0; index < size; ++index)
    function(index);
}

template <class Function>
void execute_svd_sector_batch(SchedulerBatchBlockExecution, std::size_t size, Function&& function)
{
  async::execute_batch(size, std::forward<Function>(function));
}

} // namespace detail

/// \brief Intermediate per-sector block SVD retaining factors and logical basis maps.
/// \tparam Scalar Dense provider scalar type.
/// \tparam DomainType Original tensor domain.
/// \tparam CodomainType Original tensor codomain.
/// \tparam BlockExecutionPolicy Execution policy inherited from the factorized tensor view.
template <uni20::LapackScalar Scalar, class DomainType, class CodomainType, class BlockExecutionPolicy>
class BlockSvdDecomposition {
  public:
    using scalar_type = Scalar;
    using real_type = uni20::make_real_t<scalar_type>;
    using domain_type = DomainType;
    using codomain_type = CodomainType;
    using block_execution_policy = BlockExecutionPolicy;
    using domain_shape = detail::SvdBoundaryShape<domain_type>;
    using codomain_shape = detail::SvdBoundaryShape<codomain_type>;
    using domain_fragment_type =
        detail::SvdBoundaryFragment<domain_shape::key_coordinate_count, domain_shape::dense_block_order>;
    using codomain_fragment_type =
        detail::SvdBoundaryFragment<codomain_shape::key_coordinate_count, codomain_shape::dense_block_order>;
    using source_key_type = BlockKey<domain_shape::key_coordinate_count + codomain_shape::key_coordinate_count>;
    using matrix_type = ColumnMajorTensor<scalar_type, 2>;
    using singular_value_tensor_type = ColumnMajorTensor<real_type, 1>;

    static_assert(domain_type::size() + 1 <= 4, "block-SVD right factor exceeds the first BlockTensor order limit");
    static_assert(codomain_type::size() + 1 <= 4, "block-SVD left factor exceeds the first BlockTensor order limit");

    /// \brief Provider result and logical fragment maps for one charge sector.
    struct Sector
    {
        QNum charge;
        std::vector<domain_fragment_type> domain_fragments;
        std::vector<codomain_fragment_type> codomain_fragments;
        std::vector<source_key_type> stored_source_keys;
        matrix_type left_singular_vectors;
        singular_value_tensor_type singular_values;
        matrix_type right_singular_vectors_adjoint;
    };

    /// \brief Construct a completed decomposition from sector provider results.
    BlockSvdDecomposition(Symmetry symmetry, domain_type domain, codomain_type codomain, linalg::SvdOptions options,
                          std::vector<Sector> sectors)
        : symmetry_(symmetry), domain_(std::move(domain)), codomain_(std::move(codomain)), options_(options),
          sectors_(std::move(sectors))
    {
      for (Sector const& sector : sectors_)
      {
        for (std::size_t index = 0; index < static_cast<std::size_t>(sector.singular_values.extent(0)); ++index)
        {
          spectrum_.push_back({.id = {.sector = sector.charge, .index = index},
                               .singular_value = sector.singular_values[static_cast<uni20::index_type>(index)]});
        }
      }
      std::ranges::sort(spectrum_, [](auto const& lhs, auto const& rhs) {
        if (lhs.singular_value != rhs.singular_value) return lhs.singular_value > rhs.singular_value;
        if (lhs.id.sector.raw_code() != rhs.id.sector.raw_code())
          return lhs.id.sector.raw_code() < rhs.id.sector.raw_code();
        return lhs.id.index < rhs.id.index;
      });
    }

    /// \brief Return the symmetry shared by the input and every sector.
    auto symmetry() const noexcept -> Symmetry { return symmetry_; }

    /// \brief Return the original domain boundary.
    auto domain() const noexcept -> domain_type const& { return domain_; }

    /// \brief Return the original codomain boundary.
    auto codomain() const noexcept -> codomain_type const& { return codomain_; }

    /// \brief Return exact-SVD vector extent options used by every sector.
    auto options() const noexcept -> linalg::SvdOptions { return options_; }

    /// \brief Return charge sectors in canonical charge order.
    auto sectors() const noexcept -> std::span<Sector const> { return sectors_; }

    /// \brief Return paired singular states in deterministic descending order.
    auto spectrum() const noexcept -> std::span<BlockSvdState<real_type> const> { return spectrum_; }

    /// \brief Return unpaired full left vectors in the dimensional-excess null space.
    auto left_null_space() const -> BlockSvdVectorSelection
    {
      std::vector<BlockSvdStateId> result;
      for (Sector const& sector : sectors_)
      {
        auto const rank = static_cast<std::size_t>(sector.singular_values.extent(0));
        auto const extent = static_cast<std::size_t>(sector.left_singular_vectors.extent(1));
        for (std::size_t index = rank; index < extent; ++index)
          result.push_back({.sector = sector.charge, .index = index});
      }
      return BlockSvdVectorSelection{std::move(result)};
    }

    /// \brief Return unpaired full right vectors in the dimensional-excess null space.
    auto right_null_space() const -> BlockSvdVectorSelection
    {
      std::vector<BlockSvdStateId> result;
      for (Sector const& sector : sectors_)
      {
        auto const rank = static_cast<std::size_t>(sector.singular_values.extent(0));
        auto const extent = static_cast<std::size_t>(sector.right_singular_vectors_adjoint.extent(0));
        for (std::size_t index = rank; index < extent; ++index)
          result.push_back({.sector = sector.charge, .index = index});
      }
      return BlockSvdVectorSelection{std::move(result)};
    }

  private:
    Symmetry symmetry_;
    domain_type domain_;
    codomain_type codomain_;
    linalg::SvdOptions options_;
    std::vector<Sector> sectors_;
    std::vector<BlockSvdState<real_type>> spectrum_;
};

/// \brief Factorize an immediate BlockTensorView independently by conserved charge with performance measurements.
/// \details Missing stored blocks are assembled as zero submatrices inside
///          their symmetry sector. The returned intermediate retains provider
///          factors, logical fragment maps, and the source block execution
///          policy for repeatable materialization. Scheduler-batch storage
///          assembles and factorizes independent sectors concurrently, with
///          estimated-expensive sectors submitted first. LAPACK execution
///          within one sector remains an ordinary synchronous provider call.
/// \tparam Tensor Immediate BlockTensor-level input view.
/// \tparam Measurements Compile-time-selected performance measurement policy.
/// \tparam Event Event identifier used for the sector batch.
/// \param tensor Immediate tensor to factorize.
/// \param options Dense provider vector-extent options.
/// \param measurements Explicit measurement policy or collector.
/// \param batch_event Event identifying the sector-factorization batch.
/// \return Reusable per-charge decomposition and globally ordered spectrum.
template <ImmediateBlockTensorView Tensor, class Measurements, class Event>
  requires uni20::LapackScalar<block_tensor_value_t<Tensor>> && performance::BatchMeasurementPolicy<Measurements, Event>
[[nodiscard]] auto block_svd(Tensor const& tensor, linalg::SvdOptions options, Measurements& measurements,
                             Event batch_event)
    -> BlockSvdDecomposition<block_tensor_value_t<Tensor>, block_tensor_domain_t<Tensor>,
                             block_tensor_codomain_t<Tensor>,
                             typename block_tensor_type_t<Tensor>::storage_policy::block_execution_policy>
{
  using scalar_type = block_tensor_value_t<Tensor>;
  using block_execution_policy = typename block_tensor_type_t<Tensor>::storage_policy::block_execution_policy;
  using decomposition_type = BlockSvdDecomposition<scalar_type, block_tensor_domain_t<Tensor>,
                                                   block_tensor_codomain_t<Tensor>, block_execution_policy>;
  using sector_type = typename decomposition_type::Sector;
  auto domain_sectors =
      detail::group_svd_boundary_fragments(detail::make_svd_boundary_fragments(tensor.symmetry(), tensor.domain()));
  auto codomain_sectors =
      detail::group_svd_boundary_fragments(detail::make_svd_boundary_fragments(tensor.symmetry(), tensor.codomain()));

  using domain_sector_type = typename decltype(domain_sectors)::value_type;
  using codomain_sector_type = typename decltype(codomain_sectors)::value_type;
  struct SectorInput
  {
      domain_sector_type domain;
      codomain_sector_type codomain;
  };

  std::vector<SectorInput> sector_inputs;
  std::size_t domain_sector = 0;
  std::size_t codomain_sector = 0;
  while (domain_sector < domain_sectors.size() || codomain_sector < codomain_sectors.size())
  {
    bool const has_domain = domain_sector < domain_sectors.size();
    bool const has_codomain = codomain_sector < codomain_sectors.size();
    if (has_domain && (!has_codomain || domain_sectors[domain_sector].charge.raw_code() <
                                            codomain_sectors[codomain_sector].charge.raw_code()))
    {
      if (options.right == linalg::SvdVectorExtent::Full)
      {
        auto& domain = domain_sectors[domain_sector];
        QNum const charge = domain.charge;
        sector_inputs.push_back({.domain = std::move(domain), .codomain = codomain_sector_type{.charge = charge}});
      }
      ++domain_sector;
      continue;
    }
    if (has_codomain && (!has_domain || codomain_sectors[codomain_sector].charge.raw_code() <
                                            domain_sectors[domain_sector].charge.raw_code()))
    {
      if (options.left == linalg::SvdVectorExtent::Full)
      {
        auto& codomain = codomain_sectors[codomain_sector];
        QNum const charge = codomain.charge;
        sector_inputs.push_back({.domain = domain_sector_type{.charge = charge}, .codomain = std::move(codomain)});
      }
      ++codomain_sector;
      continue;
    }

    sector_inputs.push_back(
        {.domain = std::move(domain_sectors[domain_sector]), .codomain = std::move(codomain_sectors[codomain_sector])});
    ++domain_sector;
    ++codomain_sector;
  }

  std::vector<std::optional<sector_type>> sector_results(sector_inputs.size());
  auto factorize_sector = [&](std::size_t sector) {
    auto& domain = sector_inputs[sector].domain;
    auto& codomain = sector_inputs[sector].codomain;
    ColumnMajorTensor<scalar_type, 2> matrix(static_cast<uni20::index_type>(codomain.flattened_extent),
                                             static_cast<uni20::index_type>(domain.flattened_extent));
    uni20::fill(matrix, scalar_type{});
    std::vector<typename decomposition_type::source_key_type> stored_source_keys;
    for (auto const& domain_fragment : domain.fragments)
    {
      for (auto const& codomain_fragment : codomain.fragments)
      {
        auto const key_coordinates =
            detail::concatenate_coordinates(domain_fragment.coordinates, codomain_fragment.coordinates);
        typename decomposition_type::source_key_type const key{key_coordinates};
        auto const found = std::ranges::lower_bound(tensor.stored_keys(), key);
        if (found == tensor.stored_keys().end() || *found != key) continue;
        auto const ordinal = static_cast<std::size_t>(found - tensor.stored_keys().begin());
        auto block = tensor.block_by_ordinal(ordinal);
        stored_source_keys.push_back(key);

        for (std::size_t column = 0; column < domain_fragment.flattened_extent; ++column)
        {
          auto const domain_indices = detail::column_major_indices(column, domain_fragment.extents);
          for (std::size_t row = 0; row < codomain_fragment.flattened_extent; ++row)
          {
            auto const codomain_indices = detail::column_major_indices(row, codomain_fragment.extents);
            auto const block_indices = detail::concatenate_indices(domain_indices, codomain_indices);
            matrix[static_cast<uni20::index_type>(codomain_fragment.sector_offset + row),
                   static_cast<uni20::index_type>(domain_fragment.sector_offset + column)] =
                detail::block_element(block, block_indices);
          }
        }
      }
    }

    auto exact = linalg::svd(std::as_const(matrix), options);
    sector_results[sector].emplace(
        sector_type{.charge = domain.charge,
                    .domain_fragments = std::move(domain.fragments),
                    .codomain_fragments = std::move(codomain.fragments),
                    .stored_source_keys = std::move(stored_source_keys),
                    .left_singular_vectors = std::move(exact.left_singular_vectors),
                    .singular_values = std::move(exact.singular_values),
                    .right_singular_vectors_adjoint = std::move(exact.right_singular_vectors_adjoint)});
  };

  std::vector<std::size_t> sector_order(sector_inputs.size());
  std::iota(sector_order.begin(), sector_order.end(), 0);
  auto estimated_cost = [&](std::size_t sector) {
    std::size_t const rows = sector_inputs[sector].codomain.flattened_extent;
    std::size_t const columns = sector_inputs[sector].domain.flattened_extent;
    return rows * columns * std::min(rows, columns);
  };
  std::ranges::stable_sort(sector_order,
                           [&](std::size_t lhs, std::size_t rhs) { return estimated_cost(lhs) > estimated_cost(rhs); });

  performance::measure_batch(
      measurements, batch_event, sector_order.size(),
      [&](auto& function) {
        detail::execute_svd_sector_batch(block_execution_policy{}, sector_order.size(), function);
      },
      [&](std::size_t task) { factorize_sector(sector_order[task]); });

  std::vector<sector_type> sectors;
  sectors.reserve(sector_results.size());
  for (auto& sector : sector_results)
    sectors.push_back(std::move(sector).value());

  return decomposition_type{tensor.symmetry(), tensor.domain(), tensor.codomain(), options, std::move(sectors)};
}

/// \brief Factorize an immediate BlockTensorView independently by conserved charge.
/// \details This ordinary overload instantiates no performance measurements.
/// \tparam Tensor Immediate BlockTensor-level input view.
/// \param tensor Immediate tensor to factorize.
/// \param options Dense provider vector-extent options.
/// \return Reusable per-charge decomposition and globally ordered spectrum.
template <ImmediateBlockTensorView Tensor>
  requires uni20::LapackScalar<block_tensor_value_t<Tensor>>
[[nodiscard]] auto block_svd(Tensor const& tensor, linalg::SvdOptions options = {})
{
  performance::NoMeasurements measurements;
  return block_svd(tensor, options, measurements, nullptr);
}

/// \brief Build an arbitrary paired-state selection and its exact statistics.
template <uni20::Real Real>
[[nodiscard]] auto make_svd_selection(std::span<BlockSvdState<Real> const> spectrum,
                                      std::span<BlockSvdStateId const> requested) -> BlockSvdSelection<Real>
{
  auto [canonical, selected] = detail::canonical_svd_selection(spectrum, requested);
  auto statistics = detail::summarize_svd_selection(spectrum, std::span<unsigned char const>{selected});
  return BlockSvdSelection<Real>{std::move(canonical), std::move(statistics)};
}

/// \brief Apply the standard truncation policy to a globally sorted block-SVD spectrum.
template <uni20::Real Real>
[[nodiscard]] auto select_svd_states(std::span<BlockSvdState<Real> const> spectrum,
                                     linalg::SvdTruncationPolicy<Real> const& policy) -> BlockSvdSelection<Real>
{
  std::vector<Real> values;
  values.reserve(spectrum.size());
  for (auto const& state : spectrum)
    values.push_back(state.singular_value);
  auto statistics = linalg::select_svd_truncation(std::span<Real const>{values}, policy);

  std::vector<BlockSvdStateId> selected;
  selected.reserve(statistics.retained_rank);
  for (std::size_t ordinal = 0; ordinal < statistics.retained_rank; ++ordinal)
    selected.push_back(spectrum[ordinal].id);
  return BlockSvdSelection<Real>{std::move(selected), std::move(statistics)};
}

/// \brief Select every paired state not present in an existing selection.
template <uni20::Real Real>
[[nodiscard]] auto complement_svd_selection(std::span<BlockSvdState<Real> const> spectrum,
                                            BlockSvdSelection<Real> const& selection) -> BlockSvdSelection<Real>
{
  auto [canonical, selected] = detail::canonical_svd_selection(spectrum, selection.state_ids());
  static_cast<void>(canonical);
  std::vector<BlockSvdStateId> complement;
  complement.reserve(spectrum.size() - selection.state_ids().size());
  for (std::size_t ordinal = 0; ordinal < spectrum.size(); ++ordinal)
  {
    if (!selected[ordinal]) complement.push_back(spectrum[ordinal].id);
  }
  return make_svd_selection(spectrum, std::span<BlockSvdStateId const>{complement});
}

namespace detail
{

template <class Decomposition> struct BlockSvdSelectionPlan
{
    BlockSpace bond_space;
    std::vector<std::vector<std::size_t>> sector_indices;
    std::vector<std::optional<std::size_t>> bond_coordinates;
};

template <class Decomposition, class Extent>
auto make_block_svd_selection_plan(Decomposition const& decomposition, std::span<BlockSvdStateId const> state_ids,
                                   std::string bond_label, Extent&& extent) -> BlockSvdSelectionPlan<Decomposition>
{
  std::vector<std::vector<std::size_t>> sector_indices(decomposition.sectors().size());
  for (BlockSvdStateId const& id : state_ids)
  {
    auto const found =
        std::ranges::find_if(decomposition.sectors(), [&](auto const& sector) { return sector.charge == id.sector; });
    if (found == decomposition.sectors().end())
      throw std::invalid_argument("block-SVD materialization contains an unknown charge sector");
    auto const sector = static_cast<std::size_t>(found - decomposition.sectors().begin());
    if (id.index >= extent(*found)) throw std::invalid_argument("block-SVD state index is out of range");
    sector_indices[sector].push_back(id.index);
  }

  std::vector<BlockSector> bond_sectors;
  std::vector<std::optional<std::size_t>> bond_coordinates(decomposition.sectors().size());
  for (std::size_t sector = 0; sector < decomposition.sectors().size(); ++sector)
  {
    auto& indices = sector_indices[sector];
    std::ranges::sort(indices);
    if (std::ranges::adjacent_find(indices) != indices.end())
      throw std::invalid_argument("block-SVD materialization contains a repeated state");
    if (indices.empty()) continue;
    bond_coordinates[sector] = bond_sectors.size();
    bond_sectors.push_back({.q = decomposition.sectors()[sector].charge, .dim = indices.size()});
  }

  return {.bond_space = BlockSpace(decomposition.symmetry(), bond_sectors, std::move(bond_label)),
          .sector_indices = std::move(sector_indices),
          .bond_coordinates = std::move(bond_coordinates)};
}

template <class Decomposition>
auto make_left_singular_vectors(Decomposition const& decomposition, BlockSvdSelectionPlan<Decomposition> const& plan)
{
  using scalar_type = typename Decomposition::scalar_type;
  using output_type =
      BlockTensor<scalar_type, Domain<BlockSpace>, typename Decomposition::codomain_type, PackedSparseBlockStorage<>>;
  using key_type = typename output_type::key_type;
  constexpr std::size_t codomain_key_count = Decomposition::codomain_shape::key_coordinate_count;
  std::vector<key_type> keys;
  for (std::size_t sector = 0; sector < decomposition.sectors().size(); ++sector)
  {
    if (!plan.bond_coordinates[sector]) continue;
    for (auto const& fragment : decomposition.sectors()[sector].codomain_fragments)
    {
      std::array<std::size_t, 1 + codomain_key_count> coordinates{};
      coordinates[0] = *plan.bond_coordinates[sector];
      std::ranges::copy(fragment.coordinates, coordinates.begin() + 1);
      keys.emplace_back(coordinates);
    }
  }

  output_type result(decomposition.symmetry(), Domain{plan.bond_space}, decomposition.codomain(), std::move(keys));
  for (std::size_t sector = 0; sector < decomposition.sectors().size(); ++sector)
  {
    if (!plan.bond_coordinates[sector]) continue;
    auto const& source = decomposition.sectors()[sector];
    for (auto const& fragment : source.codomain_fragments)
    {
      std::array<std::size_t, 1 + codomain_key_count> coordinates{};
      coordinates[0] = *plan.bond_coordinates[sector];
      std::ranges::copy(fragment.coordinates, coordinates.begin() + 1);
      auto block = result.block(key_type{coordinates});
      for (std::size_t selected_index = 0; selected_index < plan.sector_indices[sector].size(); ++selected_index)
      {
        for (std::size_t row = 0; row < fragment.flattened_extent; ++row)
        {
          auto const codomain_indices = column_major_indices(row, fragment.extents);
          std::array<uni20::index_type, 1> const bond_index{static_cast<uni20::index_type>(selected_index)};
          auto const output_indices = concatenate_indices(bond_index, codomain_indices);
          block_element(block, output_indices) =
              source.left_singular_vectors[static_cast<uni20::index_type>(fragment.sector_offset + row),
                                           static_cast<uni20::index_type>(plan.sector_indices[sector][selected_index])];
        }
      }
    }
  }
  return result;
}

template <class Decomposition>
auto make_right_singular_vectors_adjoint(Decomposition const& decomposition,
                                         BlockSvdSelectionPlan<Decomposition> const& plan)
{
  using scalar_type = typename Decomposition::scalar_type;
  using output_type =
      BlockTensor<scalar_type, typename Decomposition::domain_type, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using key_type = typename output_type::key_type;
  constexpr std::size_t domain_key_count = Decomposition::domain_shape::key_coordinate_count;

  std::vector<key_type> keys;
  for (std::size_t sector = 0; sector < decomposition.sectors().size(); ++sector)
  {
    if (!plan.bond_coordinates[sector]) continue;
    for (auto const& fragment : decomposition.sectors()[sector].domain_fragments)
    {
      std::array<std::size_t, domain_key_count + 1> coordinates{};
      std::ranges::copy(fragment.coordinates, coordinates.begin());
      coordinates[domain_key_count] = *plan.bond_coordinates[sector];
      keys.emplace_back(coordinates);
    }
  }

  output_type result(decomposition.symmetry(), decomposition.domain(), Codomain{plan.bond_space}, std::move(keys));
  for (std::size_t sector = 0; sector < decomposition.sectors().size(); ++sector)
  {
    if (!plan.bond_coordinates[sector]) continue;
    auto const& source = decomposition.sectors()[sector];
    for (auto const& fragment : source.domain_fragments)
    {
      std::array<std::size_t, domain_key_count + 1> coordinates{};
      std::ranges::copy(fragment.coordinates, coordinates.begin());
      coordinates[domain_key_count] = *plan.bond_coordinates[sector];
      auto block = result.block(key_type{coordinates});
      for (std::size_t selected_index = 0; selected_index < plan.sector_indices[sector].size(); ++selected_index)
      {
        for (std::size_t column = 0; column < fragment.flattened_extent; ++column)
        {
          auto const domain_indices = column_major_indices(column, fragment.extents);
          std::array<uni20::index_type, 1> const bond_index{static_cast<uni20::index_type>(selected_index)};
          auto const output_indices = concatenate_indices(domain_indices, bond_index);
          block_element(block, output_indices) =
              source.right_singular_vectors_adjoint[static_cast<uni20::index_type>(
                                                        plan.sector_indices[sector][selected_index]),
                                                    static_cast<uni20::index_type>(fragment.sector_offset + column)];
        }
      }
    }
  }
  return result;
}

template <class Decomposition>
auto make_block_singular_values(Decomposition const& decomposition, BlockSvdSelectionPlan<Decomposition> const& plan)
{
  using real_type = typename Decomposition::real_type;
  using output_type = BlockTensor<real_type, Domain<BlockSpace>, Codomain<BlockSpace>, PackedDiagonalBlockStorage<>>;
  using key_type = typename output_type::key_type;
  std::vector<key_type> keys;
  keys.reserve(plan.bond_space.size());
  for (std::size_t sector = 0; sector < decomposition.sectors().size(); ++sector)
  {
    if (!plan.bond_coordinates[sector]) continue;
    keys.emplace_back(std::array<std::size_t, 2>{*plan.bond_coordinates[sector], *plan.bond_coordinates[sector]});
  }

  output_type result(decomposition.symmetry(), Domain{plan.bond_space}, Codomain{plan.bond_space}, std::move(keys));
  for (std::size_t sector = 0; sector < decomposition.sectors().size(); ++sector)
  {
    if (!plan.bond_coordinates[sector]) continue;
    key_type const key{std::array<std::size_t, 2>{*plan.bond_coordinates[sector], *plan.bond_coordinates[sector]}};
    auto values = result.diagonal_values(key);
    for (std::size_t index = 0; index < plan.sector_indices[sector].size(); ++index)
    {
      values[index] = decomposition.sectors()[sector]
                          .singular_values[static_cast<uni20::index_type>(plan.sector_indices[sector][index])];
    }
  }
  return result;
}

} // namespace detail

/// \brief Materialize paired left, singular-value, and right-adjoint factors.
/// \details Selection identities are canonicalized against this decomposition.
///          Returned truncation statistics are derived from this decomposition's
///          spectrum, even when the selection was created from a compatible
///          decomposition with different singular values.
template <class Decomposition>
[[nodiscard]] auto materialize_svd(Decomposition const& decomposition,
                                   BlockSvdSelection<typename Decomposition::real_type> const& selection,
                                   BlockSvdMaterializationOptions options = {})
{
  auto [state_ids, selected] = detail::canonical_svd_selection(decomposition.spectrum(), selection.state_ids());
  auto truncation = detail::summarize_svd_selection(decomposition.spectrum(), std::span<unsigned char const>{selected});
  auto plan = detail::make_block_svd_selection_plan(
      decomposition, state_ids, std::move(options.bond_label),
      [](auto const& sector) { return static_cast<std::size_t>(sector.singular_values.extent(0)); });
  auto left = detail::make_left_singular_vectors(decomposition, plan);
  auto values = detail::make_block_singular_values(decomposition, plan);
  auto right = detail::make_right_singular_vectors_adjoint(decomposition, plan);
  return BlockSvdMaterialization<decltype(left), decltype(values), decltype(right),
                                 linalg::SvdTruncationInfo<typename Decomposition::real_type>>{
      .left_singular_vectors = std::move(left),
      .singular_values = std::move(values),
      .right_singular_vectors_adjoint = std::move(right),
      .truncation = std::move(truncation)};
}

/// \brief Materialize selected left singular vectors, including full null-space vectors.
template <class Decomposition, detail::BlockSvdStateSelection Selection>
[[nodiscard]] auto materialize_left_singular_vectors(Decomposition const& decomposition, Selection const& selection,
                                                     BlockSvdMaterializationOptions options = {})
{
  auto plan = detail::make_block_svd_selection_plan(
      decomposition, selection.state_ids(), std::move(options.bond_label),
      [](auto const& sector) { return static_cast<std::size_t>(sector.left_singular_vectors.extent(1)); });
  return detail::make_left_singular_vectors(decomposition, plan);
}

/// \brief Materialize selected right singular vectors adjoint, including full null-space vectors.
template <class Decomposition, detail::BlockSvdStateSelection Selection>
[[nodiscard]] auto materialize_right_singular_vectors_adjoint(Decomposition const& decomposition,
                                                              Selection const& selection,
                                                              BlockSvdMaterializationOptions options = {})
{
  auto plan = detail::make_block_svd_selection_plan(
      decomposition, selection.state_ids(), std::move(options.bond_label),
      [](auto const& sector) { return static_cast<std::size_t>(sector.right_singular_vectors_adjoint.extent(0)); });
  return detail::make_right_singular_vectors_adjoint(decomposition, plan);
}

} // namespace uni20
