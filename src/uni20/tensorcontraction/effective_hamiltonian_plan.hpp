#pragma once

#include <uni20/tensorcontraction/matrix_family.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace uni20::tensorcontraction
{

class EffectiveHamiltonianPlan {
  public:
    struct Term
    {
        std::size_t r = 0;
        std::size_t a = 0;
        std::size_t b = 0;
        std::size_t c = 0;
        double coefficient = 0.0;
    };

    EffectiveHamiltonianPlan(MatrixFamily r_mats, MatrixFamily a_mats, MatrixFamily b_mats, MatrixFamily c_mats,
                             std::span<Term const> terms);
    EffectiveHamiltonianPlan(EffectiveHamiltonianPlan&&) noexcept;
    EffectiveHamiltonianPlan& operator=(EffectiveHamiltonianPlan&&) noexcept;
    EffectiveHamiltonianPlan(EffectiveHamiltonianPlan const&) = delete;
    EffectiveHamiltonianPlan& operator=(EffectiveHamiltonianPlan const&) = delete;
    ~EffectiveHamiltonianPlan();

    [[nodiscard]] std::size_t term_count() const noexcept;
    [[nodiscard]] bool compiled() const noexcept;
    [[nodiscard]] std::span<double const> r_values(std::size_t index) const;

    void compile();
    void apply();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace uni20::tensorcontraction
