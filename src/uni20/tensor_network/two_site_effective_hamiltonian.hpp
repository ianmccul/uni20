/**
 * \file two_site_effective_hamiltonian.hpp
 * \ingroup tensor_network
 * \brief Defines the first host two-site effective-Hamiltonian apply object.
 */

#pragma once

#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/symmetry/block_tensor_repartition.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

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
template <detail::TwoSiteOperatorView Hamiltonian> class TwoSiteEffectiveHamiltonian {
  public:
    using hamiltonian_type = std::remove_cvref_t<Hamiltonian>;

    /// \brief Retain an immutable local two-site Hamiltonian object.
    explicit TwoSiteEffectiveHamiltonian(hamiltonian_type hamiltonian) : hamiltonian_(std::move(hamiltonian)) {}

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
TwoSiteEffectiveHamiltonian(Hamiltonian) -> TwoSiteEffectiveHamiltonian<std::remove_cvref_t<Hamiltonian>>;

} // namespace uni20::tensor_network
