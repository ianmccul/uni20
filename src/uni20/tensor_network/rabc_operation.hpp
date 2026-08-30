/**
 * \file rabc_operation.hpp
 * \ingroup tensor_network
 * \brief Operation tag for sparse R/A/B/C contraction dispatch.
 */

#pragma once

#include <string_view>

namespace uni20::tensor_network
{

/// \brief Fixed-output sparse R/A/B/C contraction operation tag.
/// \details Computes `R_r = sum(f(r,a,b,c) A_a B_b transpose(C_c))`
///          without replacing the output block structure.
struct rabc_contract_op
{
    static constexpr std::string_view name = "rabc_contract";
};

} // namespace uni20::tensor_network
