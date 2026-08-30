/**
 * \file two_site_effective_hamiltonian.hpp
 * \ingroup tensor_network
 * \brief Defines the first host two-site effective-Hamiltonian apply objects.
 */

#pragma once

#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/symmetry/block_tensor_repartition.hpp>
#include <uni20/tensor_network/rabc_contraction.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <map>
#include <memory>
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
///          sparse R/A/B/C term plan for one fixed center structure and
///          snapshots their scalar coefficients into an immutable sparse
///          `f(r,a,b,c)` tensor. Construction also selects and prepares the
///          current R/A/B/C backend so repeated applications reuse its schedule
///          and intermediate workspace.
///          No whole-center dense projection or high-rank BlockTensor
///          intermediate is formed. The two environments are retained by
///          value. If either is a borrowed view, its ultimate payload owner
///          must outlive this operation object.
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
    using plan_type = RabcContractionPlan<scalar_type, key_type, block_tensor_key_t<left_environment_type>, key_type,
                                          block_tensor_key_t<right_environment_type>>;

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

  public:
    /// \brief Compile one fixed-center sparse effective-Hamiltonian plan.
    /// \param prototype Center value whose exact boundary and stored keys define the vector space.
    /// \param left_environment Left environment retained by value.
    /// \param first_mpo Left MPO site whose coefficients are compiled into `f`.
    /// \param second_mpo Right MPO site whose coefficients are compiled into `f`.
    /// \param right_environment Right environment retained by value.
    /// \throws std::invalid_argument If spaces are incompatible or the center
    ///         stored pattern is not closed under the represented Hamiltonian.
    TwoSiteEffectiveHamiltonian(center_type const& prototype, left_environment_type left_environment,
                                first_mpo_type first_mpo, second_mpo_type second_mpo,
                                right_environment_type right_environment)
        : symmetry_(prototype.symmetry()), domain_(prototype.domain()), codomain_(prototype.codomain()),
          stored_keys_(prototype.stored_keys().begin(), prototype.stored_keys().end()),
          left_environment_(std::move(left_environment)), right_environment_(std::move(right_environment))
    {
      this->validate_operand_spaces(prototype, first_mpo, second_mpo);
      auto plan = this->make_plan(first_mpo, second_mpo);
      prepared_rabc_.emplace(
          prepare_rabc_contract(prototype, std::move(plan), left_environment_, prototype, right_environment_));
    }

    /// \brief Overwrite a compatible fixed center with one planned Hamiltonian apply.
    /// \details The sparse R/A/B/C kernel backend owns contraction ordering,
    ///          intermediate reuse, batching, placement, and communication.
    /// \pre Distinct input and output views do not overlap numerical storage.
    template <MutableImmediateBlockTensorView Output, detail::MpoEffectiveCenter Input>
      requires detail::CompatibleTwoSiteCenters<Output, Input> &&
               std::same_as<block_tensor_value_t<Output>, scalar_type> &&
               std::same_as<block_tensor_key_t<Output>, key_type> &&
               std::same_as<block_tensor_domain_t<Output>, domain_type> &&
               std::same_as<block_tensor_codomain_t<Output>, codomain_type>
    void operator()(Output& output, Input const& input)
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

      (*prepared_rabc_)(output, left_environment_, input, right_environment_);
    }

    /// \brief Return the number of compiled logical R/A/B/C contributions.
    [[nodiscard]] auto term_count() const noexcept -> std::size_t { return prepared_rabc_->plan().term_count(); }

    /// \brief Return the immutable sparse coefficient plan used by the prepared backend.
    [[nodiscard]] auto plan() const noexcept -> plan_type const&
    {
      return prepared_rabc_->plan();
    }

    /// \brief Return the number of retained host right-first intermediate blocks.
    [[nodiscard]] auto prepared_intermediate_count() const noexcept -> std::size_t
    {
      return prepared_rabc_->intermediate_count();
    }

  private:
    void validate_operand_spaces(center_type const& prototype, first_mpo_type const& first_mpo,
                                 second_mpo_type const& second_mpo) const
    {
      if (left_environment_.symmetry() != symmetry_ || first_mpo.symmetry() != symmetry_ ||
          second_mpo.symmetry() != symmetry_ || right_environment_.symmetry() != symmetry_)
      {
        throw std::invalid_argument("two-site effective Hamiltonian operands require one symmetry");
      }

      if (left_environment_.codomain().template space<0>() != prototype.domain().template space<0>() ||
          left_environment_.domain().template space<0>() != prototype.domain().template space<0>() ||
          left_environment_.domain().template space<1>() != first_mpo.domain().template space<0>() ||
          first_mpo.domain().template space<1>() != prototype.domain().template space<1>() ||
          first_mpo.codomain().template space<1>() != prototype.domain().template space<1>() ||
          first_mpo.codomain().template space<0>() != second_mpo.domain().template space<0>() ||
          second_mpo.domain().template space<1>() != prototype.domain().template space<2>() ||
          second_mpo.codomain().template space<1>() != prototype.domain().template space<2>() ||
          second_mpo.codomain().template space<0>() != right_environment_.domain().template space<1>() ||
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

    [[nodiscard]] auto make_plan(first_mpo_type const& first_mpo, second_mpo_type const& second_mpo) const
        -> plan_type
    {
      std::map<std::size_t, std::vector<std::size_t>> left_by_input_bond;
      for (std::size_t ordinal = 0; ordinal < left_environment_.stored_block_count(); ++ordinal)
        left_by_input_bond[left_environment_.stored_keys()[ordinal].coordinate(2)].push_back(ordinal);

      using coordinate_pair = std::array<std::size_t, 2>;
      std::map<coordinate_pair, std::vector<std::size_t>> first_by_left_auxiliary_and_input;
      for (std::size_t ordinal = 0; ordinal < first_mpo.stored_block_count(); ++ordinal)
      {
        auto const& key = first_mpo.stored_keys()[ordinal];
        first_by_left_auxiliary_and_input[{key.coordinate(0), key.coordinate(1)}].push_back(ordinal);
      }
      std::map<coordinate_pair, std::vector<std::size_t>> second_by_left_auxiliary_and_input;
      for (std::size_t ordinal = 0; ordinal < second_mpo.stored_block_count(); ++ordinal)
      {
        auto const& key = second_mpo.stored_keys()[ordinal];
        second_by_left_auxiliary_and_input[{key.coordinate(0), key.coordinate(1)}].push_back(ordinal);
      }
      std::map<coordinate_pair, std::vector<std::size_t>> right_by_auxiliary_and_input_bond;
      for (std::size_t ordinal = 0; ordinal < right_environment_.stored_block_count(); ++ordinal)
      {
        auto const& key = right_environment_.stored_keys()[ordinal];
        right_by_auxiliary_and_input_bond[{key.coordinate(1), key.coordinate(2)}].push_back(ordinal);
      }

      std::vector<RabcTerm<scalar_type>> terms;
      for (std::size_t input = 0; input < stored_keys_.size(); ++input)
      {
        auto const& input_key = stored_keys_[input];
        auto const left_match = left_by_input_bond.find(input_key.coordinate(0));
        if (left_match == left_by_input_bond.end()) continue;
        for (std::size_t const left : left_match->second)
        {
          auto const& left_key = left_environment_.stored_keys()[left];
          auto const first_match =
              first_by_left_auxiliary_and_input.find({left_key.coordinate(1), input_key.coordinate(1)});
          if (first_match == first_by_left_auxiliary_and_input.end()) continue;
          for (std::size_t const first : first_match->second)
          {
            auto const& first_key = first_mpo.stored_keys()[first];
            auto const second_match =
                second_by_left_auxiliary_and_input.find({first_key.coordinate(2), input_key.coordinate(2)});
            if (second_match == second_by_left_auxiliary_and_input.end()) continue;
            for (std::size_t const second : second_match->second)
            {
              auto const& second_key = second_mpo.stored_keys()[second];
              auto const right_match =
                  right_by_auxiliary_and_input_bond.find({second_key.coordinate(2), input_key.coordinate(3)});
              if (right_match == right_by_auxiliary_and_input_bond.end()) continue;
              scalar_type const coefficient =
                  first_mpo.block_by_ordinal(first)[] * second_mpo.block_by_ordinal(second)[];
              if (coefficient == scalar_type{}) continue;
              for (std::size_t const right : right_match->second)
              {
                auto const& right_key = right_environment_.stored_keys()[right];
                key_type const output_key{{left_key.coordinate(0), first_key.coordinate(3), second_key.coordinate(3),
                                           right_key.coordinate(0)}};
                auto const output = this->find_center_ordinal(output_key);
                if (!output)
                {
                  throw std::invalid_argument(
                      "two-site center stored pattern is not closed under its effective Hamiltonian");
                }
                terms.push_back({.r_key_index = *output,
                                 .a_key_index = left,
                                 .b_key_index = input,
                                 .c_key_index = right,
                                 .coefficient = coefficient});
              }
            }
          }
        }
      }
      using a_key_type = typename plan_type::a_key_type;
      using c_key_type = typename plan_type::c_key_type;
      return plan_type(stored_keys_,
                       std::vector<a_key_type>(left_environment_.stored_keys().begin(),
                                               left_environment_.stored_keys().end()),
                       stored_keys_,
                       std::vector<c_key_type>(right_environment_.stored_keys().begin(),
                                               right_environment_.stored_keys().end()),
                       std::move(terms));
    }

    template <BlockTensorView Tensor> void require_center(Tensor const& center) const
    {
      if (center.symmetry() != symmetry_ || center.domain() != domain_ || center.codomain() != codomain_ ||
          !std::ranges::equal(center.stored_keys(), stored_keys_))
      {
        throw std::invalid_argument("two-site effective Hamiltonian requires its compiled center structure");
      }
    }

    Symmetry symmetry_;
    domain_type domain_;
    codomain_type codomain_;
    std::vector<key_type> stored_keys_;
    left_environment_type left_environment_;
    right_environment_type right_environment_;
    using prepared_rabc_type = decltype(prepare_rabc_contract(
        std::declval<center_type const&>(), std::declval<plan_type>(),
        std::declval<left_environment_type const&>(), std::declval<center_type const&>(),
        std::declval<right_environment_type const&>()));
    std::optional<prepared_rabc_type> prepared_rabc_;
};

template <class Center, class LeftEnvironment, class FirstMpo, class SecondMpo, class RightEnvironment>
TwoSiteEffectiveHamiltonian(Center const&, LeftEnvironment, FirstMpo, SecondMpo, RightEnvironment)
    -> TwoSiteEffectiveHamiltonian<std::remove_cvref_t<Center>, std::remove_cvref_t<LeftEnvironment>,
                                   std::remove_cvref_t<FirstMpo>, std::remove_cvref_t<SecondMpo>,
                                   std::remove_cvref_t<RightEnvironment>>;

/// \brief Compile a two-site effective Hamiltonian borrowing environment payloads.
/// \details The returned operation retains identity mapped views by value, so
///          keys, boundaries, and block descriptors are not borrowed from
///          temporary view objects. Environment payload owners must outlive
///          the returned operation. MPO coefficients are snapshotted into the
///          sparse `f` plan and the MPO payloads are not retained.
/// \param prototype Center value whose exact boundary and stored keys define the vector space.
/// \param left_environment Left environment payload owner.
/// \param first_mpo Left MPO site compiled into the sparse coefficient plan.
/// \param second_mpo Right MPO site compiled into the sparse coefficient plan.
/// \param right_environment Right environment payload owner.
/// \return Compiled operation containing zero-copy environment views and an owned `f` plan.
template <detail::MpoEffectiveCenter Center, detail::MpoEffectiveEnvironment LeftEnvironment,
          detail::MpoEffectiveSite FirstMpo, detail::MpoEffectiveSite SecondMpo,
          detail::MpoEffectiveEnvironment RightEnvironment>
  requires detail::CompatibleMpoEffectiveScalars<Center, LeftEnvironment, FirstMpo, SecondMpo, RightEnvironment>
[[nodiscard]] auto make_two_site_effective_hamiltonian(Center const& prototype, LeftEnvironment const& left_environment,
                                                       FirstMpo const& first_mpo, SecondMpo const& second_mpo,
                                                       RightEnvironment const& right_environment)
{
  return TwoSiteEffectiveHamiltonian(prototype, as_block_tensor_view(left_environment), as_block_tensor_view(first_mpo),
                                     as_block_tensor_view(second_mpo), as_block_tensor_view(right_environment));
}

} // namespace uni20::tensor_network
