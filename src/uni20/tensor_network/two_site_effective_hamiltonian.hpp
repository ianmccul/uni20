/**
 * \file two_site_effective_hamiltonian.hpp
 * \ingroup tensor_network
 * \brief Defines the first host two-site effective-Hamiltonian apply objects.
 */

#pragma once

#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/ops/contract.hpp>
#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/symmetry/block_tensor_repartition.hpp>
#include <uni20/tensor/tensor.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{
namespace detail
{

template <class Tensor>
concept TwoSiteCenterView = ImmediateBlockTensorView<Tensor> && (block_tensor_domain_t<Tensor>::size() == 3) &&
                            (block_tensor_codomain_t<Tensor>::size() == 1);

template <class Tensor>
concept TwoSiteOperatorView = ImmediateBlockTensorView<Tensor> && (block_tensor_domain_t<Tensor>::size() == 2) &&
                              (block_tensor_codomain_t<Tensor>::size() == 2);

template <class Output, class Input>
concept CompatibleTwoSiteCenters = TwoSiteCenterView<Output> && TwoSiteCenterView<Input> &&
                                   std::same_as<block_tensor_value_t<Output>, block_tensor_value_t<Input>> &&
                                   std::same_as<block_tensor_key_t<Output>, block_tensor_key_t<Input>> &&
                                   std::same_as<block_tensor_domain_t<Output>, block_tensor_domain_t<Input>> &&
                                   std::same_as<block_tensor_codomain_t<Output>, block_tensor_codomain_t<Input>>;

template <class Tensor>
concept MpoEffectiveCenter =
    ImmediateBlockTensorView<Tensor> && (block_tensor_domain_t<Tensor>::size() == 3) &&
    (block_tensor_codomain_t<Tensor>::size() == 1) && (block_tensor_type_t<Tensor>::key_coordinate_count() == 4) &&
    (block_tensor_type_t<Tensor>::dense_block_order() == 2);

template <class Tensor>
concept MpoEffectiveEnvironment =
    ImmediateBlockTensorView<Tensor> && (block_tensor_domain_t<Tensor>::size() == 2) &&
    (block_tensor_codomain_t<Tensor>::size() == 1) && (block_tensor_type_t<Tensor>::key_coordinate_count() == 3) &&
    (block_tensor_type_t<Tensor>::dense_block_order() == 2);

template <class Tensor>
concept MpoEffectiveSite =
    ImmediateBlockTensorView<Tensor> && (block_tensor_domain_t<Tensor>::size() == 2) &&
    (block_tensor_codomain_t<Tensor>::size() == 2) && (block_tensor_type_t<Tensor>::key_coordinate_count() == 4) &&
    (block_tensor_type_t<Tensor>::dense_block_order() == 0);

template <class Center, class LeftEnvironment, class FirstMpo, class SecondMpo, class RightEnvironment>
concept CompatibleMpoEffectiveScalars =
    std::same_as<block_tensor_value_t<Center>, block_tensor_value_t<LeftEnvironment>> &&
    std::same_as<block_tensor_value_t<Center>, block_tensor_value_t<FirstMpo>> &&
    std::same_as<block_tensor_value_t<Center>, block_tensor_value_t<SecondMpo>> &&
    std::same_as<block_tensor_value_t<Center>, block_tensor_value_t<RightEnvironment>>;

template <class Function> void execute_effective_groups(SerialBlockExecution, std::size_t size, Function&& function)
{
  for (std::size_t index = 0; index < size; ++index)
    function(index);
}

template <class Function>
void execute_effective_groups(SchedulerBatchBlockExecution, std::size_t size, Function&& function)
{
  async::execute_batch(size, std::forward<Function>(function));
}

} // namespace detail

/// \brief Apply a local two-site operator to a fixed BlockTensor center.
/// \details The center has boundary `Domain<left bond, left physical,
///          right physical> -> Codomain<right bond>`. The implementation bends
///          both physical factors into the codomain, contracts them with the
///          operator in one adjacent group, and restores the canonical center
///          boundary through zero-copy mapped views. The operator is retained
///          by value for repeated matrix-free Krylov application. When that
///          value is a borrowed view, its ultimate payload owner must outlive
///          this operation object.
/// \tparam Hamiltonian Immediate host BlockTensor view describing the local operator.
template <detail::TwoSiteOperatorView Hamiltonian> class LocalTwoSiteEffectiveHamiltonian {
  public:
    using hamiltonian_type = std::remove_cvref_t<Hamiltonian>;

    /// \brief Retain an immutable local two-site Hamiltonian object.
    explicit LocalTwoSiteEffectiveHamiltonian(hamiltonian_type hamiltonian) : hamiltonian_(std::move(hamiltonian)) {}

    /// \brief Overwrite a fixed center with the Hamiltonian application.
    /// \tparam Output Compatible mutable center view.
    /// \tparam Input Compatible read-only center view.
    /// \param output Fixed-structure destination center.
    /// \param input Source center.
    /// \pre Output numerical storage does not overlap input or Hamiltonian storage.
    template <MutableImmediateBlockTensorView Output, detail::TwoSiteCenterView Input>
      requires detail::CompatibleTwoSiteCenters<Output, Input>
    void operator()(Output& output, Input const& input) const
    {
      auto operator_input = permute<0, 1, 3, 2>(repartition<MorphismSide::Domain, BoundaryEnd::Right>(
          repartition<MorphismSide::Domain, BoundaryEnd::Right>(input)));
      auto applied = contract_adjacent<2>(operator_input, hamiltonian_);
      auto canonical_output = repartition<MorphismSide::Codomain, BoundaryEnd::Right>(
          repartition<MorphismSide::Codomain, BoundaryEnd::Right>(permute<0, 1, 3, 2>(applied)));
      copy(output, canonical_output);
    }

    /// \brief Return the retained local Hamiltonian.
    [[nodiscard]] auto hamiltonian() const noexcept -> hamiltonian_type const& { return hamiltonian_; }

  private:
    hamiltonian_type hamiltonian_;
};

template <class Hamiltonian>
LocalTwoSiteEffectiveHamiltonian(Hamiltonian) -> LocalTwoSiteEffectiveHamiltonian<std::remove_cvref_t<Hamiltonian>>;

/// \brief Matrix-free two-site Hamiltonian compiled from environments and MPO sites.
/// \details Construction joins the stored logical keys of the left environment,
///          two adjacent MPO sites, and right environment into an immutable
///          sparse R/A/B/C term plan for one fixed center structure. Each apply
///          groups terms by output block and evaluates `A * B * transpose(C)`
///          through ordinary dense tensor contraction dispatch. No whole-center
///          dense projection or high-rank BlockTensor intermediate is formed.
///          The four fixed operands are retained by value. If any retained
///          operand is a borrowed view, its ultimate payload owner must outlive
///          this operation object.
/// \tparam Center Fixed two-site center structure used to compile block ordinals.
/// \tparam LeftEnvironment Left `(bra bond, auxiliary, ket bond)` environment.
/// \tparam FirstMpo Left MPO site with `(left auxiliary, ket, right auxiliary, bra)` key order.
/// \tparam SecondMpo Right MPO site with the same key convention.
/// \tparam RightEnvironment Right `(bra bond, auxiliary, ket bond)` environment.
template <detail::MpoEffectiveCenter Center, detail::MpoEffectiveEnvironment LeftEnvironment,
          detail::MpoEffectiveSite FirstMpo, detail::MpoEffectiveSite SecondMpo,
          detail::MpoEffectiveEnvironment RightEnvironment>
  requires detail::CompatibleMpoEffectiveScalars<Center, LeftEnvironment, FirstMpo, SecondMpo, RightEnvironment>
class TwoSiteEffectiveHamiltonian {
  public:
    using center_type = std::remove_cvref_t<Center>;
    using left_environment_type = std::remove_cvref_t<LeftEnvironment>;
    using first_mpo_type = std::remove_cvref_t<FirstMpo>;
    using second_mpo_type = std::remove_cvref_t<SecondMpo>;
    using right_environment_type = std::remove_cvref_t<RightEnvironment>;
    using scalar_type = block_tensor_value_t<center_type>;
    using domain_type = block_tensor_domain_t<center_type>;
    using codomain_type = block_tensor_codomain_t<center_type>;
    using key_type = block_tensor_key_t<center_type>;

  private:
    static_assert(std::same_as<typename left_environment_type::codomain_type::template space_type<0>,
                               typename domain_type::template space_type<0>>);
    static_assert(std::same_as<typename left_environment_type::domain_type::template space_type<0>,
                               typename domain_type::template space_type<0>>);
    static_assert(std::same_as<typename left_environment_type::domain_type::template space_type<1>,
                               typename first_mpo_type::domain_type::template space_type<0>>);
    static_assert(std::same_as<typename first_mpo_type::domain_type::template space_type<1>,
                               typename domain_type::template space_type<1>>);
    static_assert(std::same_as<typename first_mpo_type::codomain_type::template space_type<1>,
                               typename domain_type::template space_type<1>>);
    static_assert(std::same_as<typename first_mpo_type::codomain_type::template space_type<0>,
                               typename second_mpo_type::domain_type::template space_type<0>>);
    static_assert(std::same_as<typename second_mpo_type::domain_type::template space_type<1>,
                               typename domain_type::template space_type<2>>);
    static_assert(std::same_as<typename second_mpo_type::codomain_type::template space_type<1>,
                               typename domain_type::template space_type<2>>);
    static_assert(std::same_as<typename second_mpo_type::codomain_type::template space_type<0>,
                               typename right_environment_type::domain_type::template space_type<1>>);
    static_assert(std::same_as<typename right_environment_type::domain_type::template space_type<0>,
                               typename codomain_type::template space_type<0>>);
    static_assert(std::same_as<typename right_environment_type::codomain_type::template space_type<0>,
                               typename codomain_type::template space_type<0>>);

    struct Term
    {
        std::size_t output_ordinal;
        std::size_t input_ordinal;
        std::size_t left_environment_ordinal;
        std::size_t first_mpo_ordinal;
        std::size_t second_mpo_ordinal;
        std::size_t right_environment_ordinal;
    };

  public:
    /// \brief Compile one fixed-center sparse effective-Hamiltonian plan.
    /// \param prototype Center value whose exact boundary and stored keys define the vector space.
    /// \param left_environment Left environment retained by value.
    /// \param first_mpo Left MPO site retained by value.
    /// \param second_mpo Right MPO site retained by value.
    /// \param right_environment Right environment retained by value.
    /// \throws std::invalid_argument If spaces are incompatible or the center
    ///         stored pattern is not closed under the represented Hamiltonian.
    TwoSiteEffectiveHamiltonian(center_type const& prototype, left_environment_type left_environment,
                                first_mpo_type first_mpo, second_mpo_type second_mpo,
                                right_environment_type right_environment)
        : symmetry_(prototype.symmetry()), domain_(prototype.domain()), codomain_(prototype.codomain()),
          stored_keys_(prototype.stored_keys().begin(), prototype.stored_keys().end()),
          left_environment_(std::move(left_environment)), first_mpo_(std::move(first_mpo)),
          second_mpo_(std::move(second_mpo)), right_environment_(std::move(right_environment))
    {
      this->validate_operand_spaces(prototype);
      terms_ = this->make_terms();
      group_offsets_ = this->make_group_offsets();
    }

    /// \brief Overwrite a compatible fixed center with one planned Hamiltonian apply.
    /// \details Output blocks are independent batch items when the output
    ///          storage selects scheduler-batch execution. Contributions to one
    ///          output block remain serial and use beta zero then one.
    /// \pre Distinct input and output views do not overlap numerical storage.
    template <MutableImmediateBlockTensorView Output, detail::MpoEffectiveCenter Input>
      requires detail::CompatibleTwoSiteCenters<Output, Input> &&
               std::same_as<block_tensor_value_t<Output>, scalar_type> &&
               std::same_as<block_tensor_key_t<Output>, key_type> &&
               std::same_as<block_tensor_domain_t<Output>, domain_type> &&
               std::same_as<block_tensor_codomain_t<Output>, codomain_type>
    void operator()(Output& output, Input const& input) const
    {
      this->require_center(output);
      this->require_center(input);
      if constexpr (std::same_as<std::remove_cvref_t<Output>, std::remove_cvref_t<Input>>)
      {
        if (std::addressof(output) == std::addressof(input))
        {
          throw std::invalid_argument("two-site effective Hamiltonian output must not alias its input");
        }
      }

      detail::execute_effective_groups(
          typename std::remove_cvref_t<Output>::storage_policy::block_execution_policy{}, output.stored_block_count(),
          [&](std::size_t output_ordinal) { this->execute_group(output, input, output_ordinal); });
    }

    /// \brief Return the number of compiled logical R/A/B/C contributions.
    [[nodiscard]] auto term_count() const noexcept -> std::size_t { return terms_.size(); }

  private:
    void validate_operand_spaces(center_type const& prototype) const
    {
      if (left_environment_.symmetry() != symmetry_ || first_mpo_.symmetry() != symmetry_ ||
          second_mpo_.symmetry() != symmetry_ || right_environment_.symmetry() != symmetry_)
      {
        throw std::invalid_argument("two-site effective Hamiltonian operands require one symmetry");
      }

      if (left_environment_.codomain().template space<0>() != prototype.domain().template space<0>() ||
          left_environment_.domain().template space<0>() != prototype.domain().template space<0>() ||
          left_environment_.domain().template space<1>() != first_mpo_.domain().template space<0>() ||
          first_mpo_.domain().template space<1>() != prototype.domain().template space<1>() ||
          first_mpo_.codomain().template space<1>() != prototype.domain().template space<1>() ||
          first_mpo_.codomain().template space<0>() != second_mpo_.domain().template space<0>() ||
          second_mpo_.domain().template space<1>() != prototype.domain().template space<2>() ||
          second_mpo_.codomain().template space<1>() != prototype.domain().template space<2>() ||
          second_mpo_.codomain().template space<0>() != right_environment_.domain().template space<1>() ||
          right_environment_.domain().template space<0>() != prototype.codomain().template space<0>() ||
          right_environment_.codomain().template space<0>() != prototype.codomain().template space<0>())
      {
        throw std::invalid_argument(
            "two-site effective Hamiltonian has incompatible center, MPO, or environment spaces");
      }
    }

    [[nodiscard]] auto find_center_ordinal(key_type const& key) const -> std::optional<std::size_t>
    {
      auto const found = std::ranges::lower_bound(stored_keys_, key);
      if (found == stored_keys_.end() || *found != key) return std::nullopt;
      return static_cast<std::size_t>(found - stored_keys_.begin());
    }

    [[nodiscard]] auto make_terms() const -> std::vector<Term>
    {
      std::vector<Term> result;
      for (std::size_t left = 0; left < left_environment_.stored_block_count(); ++left)
      {
        auto const& left_key = left_environment_.stored_keys()[left];
        for (std::size_t first = 0; first < first_mpo_.stored_block_count(); ++first)
        {
          auto const& first_key = first_mpo_.stored_keys()[first];
          if (left_key.coordinate(1) != first_key.coordinate(0)) continue;
          for (std::size_t second = 0; second < second_mpo_.stored_block_count(); ++second)
          {
            auto const& second_key = second_mpo_.stored_keys()[second];
            if (first_key.coordinate(2) != second_key.coordinate(0)) continue;
            for (std::size_t right = 0; right < right_environment_.stored_block_count(); ++right)
            {
              auto const& right_key = right_environment_.stored_keys()[right];
              if (second_key.coordinate(2) != right_key.coordinate(1)) continue;

              key_type const input_key{
                  {left_key.coordinate(2), first_key.coordinate(1), second_key.coordinate(1), right_key.coordinate(2)}};
              auto const input = this->find_center_ordinal(input_key);
              if (!input) continue;
              key_type const output_key{
                  {left_key.coordinate(0), first_key.coordinate(3), second_key.coordinate(3), right_key.coordinate(0)}};
              auto const output = this->find_center_ordinal(output_key);
              if (!output)
              {
                throw std::invalid_argument(
                    "two-site center stored pattern is not closed under its effective Hamiltonian");
              }
              result.push_back({*output, *input, left, first, second, right});
            }
          }
        }
      }
      std::ranges::stable_sort(result, {}, &Term::output_ordinal);
      return result;
    }

    [[nodiscard]] auto make_group_offsets() const -> std::vector<std::size_t>
    {
      std::vector<std::size_t> result(stored_keys_.size() + 1, 0);
      for (auto const& term : terms_)
        ++result[term.output_ordinal + 1];
      std::partial_sum(result.begin(), result.end(), result.begin());
      return result;
    }

    template <BlockTensorView Tensor> void require_center(Tensor const& center) const
    {
      if (center.symmetry() != symmetry_ || center.domain() != domain_ || center.codomain() != codomain_ ||
          !std::ranges::equal(center.stored_keys(), stored_keys_))
      {
        throw std::invalid_argument("two-site effective Hamiltonian requires its compiled center structure");
      }
    }

    template <MutableImmediateBlockTensorView Output, detail::MpoEffectiveCenter Input>
    void execute_group(Output& output, Input const& input, std::size_t output_ordinal) const
    {
      auto output_block = output.block_by_ordinal(output_ordinal);
      std::size_t const first = group_offsets_[output_ordinal];
      std::size_t const last = group_offsets_[output_ordinal + 1];
      if (first == last)
      {
        uni20::transform_inplace(output_block, linalg::scale{scalar_type{}});
        return;
      }

      constexpr std::array<std::pair<std::size_t, std::size_t>, 1> left_dimensions{std::pair{1U, 0U}};
      constexpr std::array<std::pair<std::size_t, std::size_t>, 1> right_dimensions{std::pair{1U, 1U}};
      for (std::size_t index = first; index < last; ++index)
      {
        auto const& term = terms_[index];
        auto const left_block = left_environment_.block_by_ordinal(term.left_environment_ordinal);
        auto const input_block = input.block_by_ordinal(term.input_ordinal);
        auto const right_block = right_environment_.block_by_ordinal(term.right_environment_ordinal);
        auto const first_mpo_block = first_mpo_.block_by_ordinal(term.first_mpo_ordinal);
        auto const second_mpo_block = second_mpo_.block_by_ordinal(term.second_mpo_ordinal);
        scalar_type const coefficient = first_mpo_block[] * second_mpo_block[];

        ColumnMajorTensor<scalar_type, 2> intermediate(left_block.extent(0), input_block.extent(1));
        linalg::contract(intermediate, scalar_type{1}, left_block, input_block, left_dimensions, scalar_type{0});
        scalar_type const beta = index == first ? scalar_type{0} : scalar_type{1};
        linalg::contract(output_block, coefficient, intermediate, right_block, right_dimensions, beta);
      }
    }

    Symmetry symmetry_;
    domain_type domain_;
    codomain_type codomain_;
    std::vector<key_type> stored_keys_;
    left_environment_type left_environment_;
    first_mpo_type first_mpo_;
    second_mpo_type second_mpo_;
    right_environment_type right_environment_;
    std::vector<Term> terms_;
    std::vector<std::size_t> group_offsets_;
};

template <class Center, class LeftEnvironment, class FirstMpo, class SecondMpo, class RightEnvironment>
TwoSiteEffectiveHamiltonian(Center const&, LeftEnvironment, FirstMpo, SecondMpo, RightEnvironment)
    -> TwoSiteEffectiveHamiltonian<std::remove_cvref_t<Center>, std::remove_cvref_t<LeftEnvironment>,
                                   std::remove_cvref_t<FirstMpo>, std::remove_cvref_t<SecondMpo>,
                                   std::remove_cvref_t<RightEnvironment>>;

} // namespace uni20::tensor_network
