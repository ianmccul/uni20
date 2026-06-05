#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"
#include "Utils.h"

#include <fmt/core.h>

#include <compare>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction
{

enum class VariableFamily
{
  Middle,
  Right,
};

struct EffectiveHamiltonianOperator::Impl
{
    MatrixFamily r_mats;
    MatrixFamily a_mats;
    MatrixFamily b_mats;
    MatrixFamily c_mats;
    std::vector<MatrixFamily::Block> input_blocks;
    std::vector<MatrixFamily::Block> output_blocks;
    std::vector<Term> terms;
    VariableFamily variable_family = VariableFamily::Right;
    std::unique_ptr<tensor::Swapper> swapper;
    std::unique_ptr<tensor::Arranger> arranger;
    bool is_compiled = false;

    Impl(MatrixFamily a, MatrixFamily b, std::span<MatrixFamily::Block const> input,
         std::span<MatrixFamily::Block const> output, std::span<Term const> input_terms)
        : r_mats(output), a_mats(std::move(a)), b_mats(std::move(b)), c_mats(input),
          input_blocks(input.begin(), input.end()), output_blocks(output.begin(), output.end()),
          terms(input_terms.begin(), input_terms.end()), variable_family(VariableFamily::Right)
    {
      initialize_runtime();
    }

    Impl(VariableFamily variable, MatrixFamily a, MatrixFamily c, std::span<MatrixFamily::Block const> input,
         std::span<MatrixFamily::Block const> output, std::span<Term const> input_terms)
        : r_mats(output), a_mats(std::move(a)), b_mats(input), c_mats(std::move(c)),
          input_blocks(input.begin(), input.end()), output_blocks(output.begin(), output.end()),
          terms(input_terms.begin(), input_terms.end()), variable_family(variable)
    {
      initialize_runtime();
    }

    void initialize_runtime();
    [[nodiscard]] bool host_backend() const { return arranger == nullptr; }
};

namespace
{

int checked_index(std::size_t value)
{
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    throw std::length_error("TensorContraction term index must fit in int");
  }
  return static_cast<int>(value);
}

void validate_index(std::size_t index, std::size_t size, char family)
{
  if (index >= size)
  {
    throw std::out_of_range(std::string("TensorContraction term references missing ") + family + " block");
  }
}

std::vector<tensor::TermTy> convert_terms(std::span<EffectiveHamiltonianOperator::Term const> terms)
{
  std::vector<tensor::TermTy> converted;
  converted.reserve(terms.size());
  for (auto const& term : terms)
  {
    converted.emplace_back(checked_index(term.r), checked_index(term.a), checked_index(term.b), checked_index(term.c),
                           term.coefficient);
  }
  return converted;
}

void validate_term_shapes(MatrixFamily const& r_mats, MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                          MatrixFamily const& c_mats, std::span<EffectiveHamiltonianOperator::Term const> terms)
{
  for (auto const& term : terms)
  {
    validate_index(term.r, r_mats.size(), 'R');
    validate_index(term.a, a_mats.size(), 'A');
    validate_index(term.b, b_mats.size(), 'B');
    validate_index(term.c, c_mats.size(), 'C');

    auto const r = r_mats.block(term.r);
    auto const a = a_mats.block(term.a);
    auto const b = b_mats.block(term.b);
    auto const c = c_mats.block(term.c);

    if (b.cols != c.rows)
    {
      throw std::invalid_argument("TensorContraction term has incompatible B/C dimensions");
    }
    if (a.cols != b.rows)
    {
      throw std::invalid_argument("TensorContraction term has incompatible A/(B*C) dimensions");
    }
    if (r.rows != a.rows || r.cols != c.cols)
    {
      throw std::invalid_argument("TensorContraction term result dimensions do not match R block");
    }
  }
}

void validate_family_shape(MatrixFamily const& actual, std::span<MatrixFamily::Block const> expected,
                           char const* family_name)
{
  if (actual.blocks().size() != expected.size())
  {
    throw std::invalid_argument(std::string("TensorContraction ") + family_name + " vector has the wrong block count");
  }
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    if (actual.block(i) != expected[i])
    {
      throw std::invalid_argument(std::string("TensorContraction ") + family_name +
                                  " vector has incompatible block shapes");
    }
  }
}

bool use_host_effective_hamiltonian_backend()
{
  auto const* backend = std::getenv("UNI20_TENSORCONTRACTION_BACKEND");
  if (backend == nullptr)
  {
    return false;
  }
  return std::string(backend) == "host" || std::string(backend) == "cpu";
}

bool use_legacy_arranger_rabc_planner()
{
  auto const* planner = std::getenv("UNI20_TENSORCONTRACTION_RABC_PLANNER");
  if (planner == nullptr)
  {
    return false;
  }
  auto const value = std::string(planner);
  return value == "arranger" || value == "legacy";
}

void host_apply(MatrixFamily const& r_mats, MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                MatrixFamily const& c_mats, std::span<EffectiveHamiltonianOperator::Term const> terms,
                MatrixFamily& out)
{
  out.fill(0.0);
  for (auto const& term : terms)
  {
    auto const r_block = r_mats.block(term.r);
    auto const a_block = a_mats.block(term.a);
    auto const b_block = b_mats.block(term.b);
    auto const c_block = c_mats.block(term.c);
    auto const a = a_mats.values(term.a);
    auto const b = b_mats.values(term.b);
    auto const c = c_mats.values(term.c);
    auto r = out.values(term.r);

    // This is the host mirror of TensorContraction's A * B * C path.  It is
    // intentionally simple and exists so small MPS unit tests do not need to
    // construct CUDA/NCCL runtimes and their large virtual-address mappings.
    std::vector<double> bc(b_block.rows * c_block.cols, 0.0);
    for (std::size_t row = 0; row < b_block.rows; ++row)
    {
      for (std::size_t inner = 0; inner < b_block.cols; ++inner)
      {
        auto const b_value = b[row * b_block.cols + inner];
        for (std::size_t col = 0; col < c_block.cols; ++col)
        {
          bc[row * c_block.cols + col] += b_value * c[inner * c_block.cols + col];
        }
      }
    }

    for (std::size_t row = 0; row < r_block.rows; ++row)
    {
      for (std::size_t inner = 0; inner < a_block.cols; ++inner)
      {
        auto const a_value = a[row * a_block.cols + inner];
        for (std::size_t col = 0; col < r_block.cols; ++col)
        {
          r[row * r_block.cols + col] += term.coefficient * a_value * bc[inner * c_block.cols + col];
        }
      }
    }
  }
}

auto output_device_for(tensor::Swapper& swapper, tensor::Matrix r_mat) -> int
{
  auto [device_id, buffer] = swapper.getPreStoreBufferOrNone(r_mat);
  if (buffer != nullptr)
  {
    return device_id;
  }
  constexpr int fallback_device = 0;
  swapper.registerGpuAllocation(r_mat, fallback_device);
  return fallback_device;
}

auto require_buffer_on(tensor::Swapper& swapper, tensor::Matrix mat,
                       int device_id) -> std::shared_ptr<tensor::GpuBuffer>
{
  auto buffer = swapper.getGpuBufferOrNone(mat, device_id);
  if (buffer != nullptr)
  {
    return buffer;
  }
  swapper.registerGpuAllocation(mat, device_id);
  buffer = swapper.getGpuBufferOrNone(mat, device_id);
  if (buffer == nullptr)
  {
    throw std::logic_error("TensorContraction deterministic RABC executor failed to allocate an output buffer");
  }
  return buffer;
}

void zero_device_matrix(tensor::Swapper& swapper, tensor::Matrix mat, int device_id)
{
  auto buffer = require_buffer_on(swapper, mat, device_id);
  auto access = swapper.createAccessPlan({}, {buffer}, device_id);
  CUDA_CALL(cudaMemsetAsync(buffer->getPtr(), 0, mat.sizeInByte(), access.stream()));
}

void gemm_device_matrix(tensor::Swapper& swapper, tensor::Matrix result, tensor::Matrix lhs, tensor::Matrix rhs,
                        double alpha, double beta, int device_id)
{
  auto lhs_buffer = swapper.ensureLocalCopy(lhs, device_id);
  auto rhs_buffer = swapper.ensureLocalCopy(rhs, device_id);
  auto result_buffer = require_buffer_on(swapper, result, device_id);
  auto access = swapper.createBlasAccessPlan({lhs_buffer, rhs_buffer}, {result_buffer}, device_id);

  CUBLAS_CALL(cublasDgemm(access.handle(), CUBLAS_OP_N, CUBLAS_OP_N, rhs.getSecondDim(), lhs.getFirstDim(),
                          lhs.getSecondDim(), &alpha, rhs_buffer->getPtr(), rhs.getSecondDim(), lhs_buffer->getPtr(),
                          lhs.getSecondDim(), &beta, result_buffer->getPtr(), result.getSecondDim()));
}

struct RightFirstIntermediateKey
{
    int device_id = 0;
    int b = 0;
    int c = 0;

    auto operator<=>(RightFirstIntermediateKey const&) const = default;
};

struct RightFirstIntermediate
{
    tensor::Matrix matrix;
};

void deterministic_right_first_apply(std::span<tensor::Matrix const> r_mats, std::span<tensor::Matrix const> a_mats,
                                     std::span<tensor::Matrix const> b_mats, std::span<tensor::Matrix const> c_mats,
                                     std::span<EffectiveHamiltonianOperator::Term const> terms,
                                     tensor::Swapper& swapper)
{
  std::vector<int> r_devices(r_mats.size(), 0);
  std::vector<bool> r_written(r_mats.size(), false);
  for (std::size_t r = 0; r < r_mats.size(); ++r)
  {
    r_devices[r] = output_device_for(swapper, r_mats[r]);
  }

  std::map<RightFirstIntermediateKey, RightFirstIntermediate> intermediates;
  for (auto const& term : terms)
  {
    auto const target_device = r_devices.at(term.r);
    auto const& b_mat = b_mats[term.b];
    auto const& c_mat = c_mats[term.c];
    RightFirstIntermediateKey const key{
        .device_id = target_device, .b = checked_index(term.b), .c = checked_index(term.c)};
    auto [intermediate_it, inserted] = intermediates.try_emplace(key);
    if (inserted)
    {
      auto intermediate =
          tensor::Matrix(nullptr, checked_index(b_mat.getFirstDim()), checked_index(c_mat.getSecondDim()));
      swapper.registerGpuAllocation(intermediate, target_device);
      gemm_device_matrix(swapper, intermediate, b_mat, c_mat, 1.0, 0.0, target_device);
      intermediate_it->second.matrix = intermediate;
    }

    // This is the initial deterministic planner: right-first only.  The seam is
    // deliberately narrow so a later cost model can choose left-first when the
    // left basis is smaller or when communication costs favor it.
    double const beta = r_written[static_cast<std::size_t>(term.r)] ? 1.0 : 0.0;
    gemm_device_matrix(swapper, r_mats[term.r], a_mats[term.a], intermediate_it->second.matrix, term.coefficient, beta,
                       target_device);
    r_written[static_cast<std::size_t>(term.r)] = true;
  }

  for (std::size_t r = 0; r < r_mats.size(); ++r)
  {
    if (!r_written[r])
    {
      fmt::print(stderr,
                 "[TENSORCONTRACTION][RABC_WARNING] Output R block id={} shape={}x{} device={} received no "
                 "Hamiltonian terms; zeroing it explicitly.\n",
                 r_mats[r].getId(), r_mats[r].getFirstDim(), r_mats[r].getSecondDim(), r_devices[r]);
      zero_device_matrix(swapper, r_mats[r], r_devices[r]);
    }
  }

  for (auto const& [_, intermediate] : intermediates)
  {
    swapper.clear(intermediate.matrix);
  }
}

} // namespace

void EffectiveHamiltonianOperator::Impl::initialize_runtime()
{
  if (use_host_effective_hamiltonian_backend())
  {
    return;
  }
  swapper = std::make_unique<tensor::Swapper>();
  arranger = std::make_unique<tensor::Arranger>(*swapper);
}

EffectiveHamiltonianOperator::EffectiveHamiltonianOperator(MatrixFamily a_mats, MatrixFamily b_mats,
                                                           std::span<MatrixFamily::Block const> input_blocks,
                                                           std::span<MatrixFamily::Block const> output_blocks,
                                                           std::span<Term const> terms)
    : impl_(nullptr)
{
  MatrixFamily r_mats(output_blocks);
  MatrixFamily c_mats(input_blocks);
  validate_term_shapes(r_mats, a_mats, b_mats, c_mats, terms);
  impl_ = std::make_unique<Impl>(std::move(a_mats), std::move(b_mats), input_blocks, output_blocks, terms);
}

EffectiveHamiltonianOperator::EffectiveHamiltonianOperator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

auto EffectiveHamiltonianOperator::variable_middle(MatrixFamily a_mats, MatrixFamily c_mats,
                                                   std::span<MatrixFamily::Block const> input_blocks,
                                                   std::span<MatrixFamily::Block const> output_blocks,
                                                   std::span<Term const> terms) -> EffectiveHamiltonianOperator
{
  MatrixFamily r_mats(output_blocks);
  MatrixFamily b_mats(input_blocks);
  validate_term_shapes(r_mats, a_mats, b_mats, c_mats, terms);
  return EffectiveHamiltonianOperator(std::make_unique<Impl>(VariableFamily::Middle, std::move(a_mats),
                                                             std::move(c_mats), input_blocks, output_blocks, terms));
}

EffectiveHamiltonianOperator::EffectiveHamiltonianOperator(EffectiveHamiltonianOperator&&) noexcept = default;
EffectiveHamiltonianOperator&
EffectiveHamiltonianOperator::operator=(EffectiveHamiltonianOperator&&) noexcept = default;
EffectiveHamiltonianOperator::~EffectiveHamiltonianOperator() = default;

std::size_t EffectiveHamiltonianOperator::term_count() const noexcept { return impl_->terms.size(); }

bool EffectiveHamiltonianOperator::compiled() const noexcept { return impl_->is_compiled; }

MatrixFamily EffectiveHamiltonianOperator::make_input_vector() const { return MatrixFamily(impl_->input_blocks); }

MatrixFamily EffectiveHamiltonianOperator::make_output_vector() const { return MatrixFamily(impl_->output_blocks); }

void EffectiveHamiltonianOperator::compile()
{
  if (impl_->is_compiled)
  {
    return;
  }

  validate_term_shapes(impl_->r_mats, impl_->a_mats, impl_->b_mats, impl_->c_mats, impl_->terms);
  if (impl_->host_backend())
  {
    impl_->is_compiled = true;
    return;
  }

  auto terms = convert_terms(impl_->terms);
  auto const& r = raw_matrices(impl_->r_mats);
  auto const& a = raw_matrices(impl_->a_mats);
  auto const& b = raw_matrices(impl_->b_mats);
  auto const& c = raw_matrices(impl_->c_mats);

  impl_->arranger->resetWork();
  impl_->arranger->analyzeComputation(r, a, b, c, terms);
  impl_->arranger->compileWorklists(r, a, b, c, /*syncResultsToHost=*/true);
  impl_->is_compiled = true;
}

void EffectiveHamiltonianOperator::apply(MatrixFamily const& x, MatrixFamily& y)
{
  validate_family_shape(x, impl_->input_blocks, "input");
  validate_family_shape(y, impl_->output_blocks, "output");

  if (!impl_->is_compiled)
  {
    compile();
  }

  if (impl_->variable_family == VariableFamily::Middle)
  {
    impl_->b_mats.assign(x);
  }
  else
  {
    impl_->c_mats.assign(x);
  }
  impl_->r_mats.fill(0.0);
  if (impl_->host_backend())
  {
    host_apply(impl_->r_mats, impl_->a_mats, impl_->b_mats, impl_->c_mats, impl_->terms, impl_->r_mats);
  }
  else
  {
    impl_->arranger->doContraction(raw_matrices(impl_->r_mats), raw_matrices(impl_->a_mats),
                                   raw_matrices(impl_->b_mats), raw_matrices(impl_->c_mats));
  }
  y.assign(impl_->r_mats);
}

void EffectiveHamiltonianOperator::apply_resident(MatrixFamily const& x, MatrixFamily& y, VectorAlgebraEngine& algebra)
{
  validate_family_shape(x, impl_->input_blocks, "input");
  validate_family_shape(y, impl_->output_blocks, "output");

  if (impl_->host_backend() || algebra.uses_host_backend())
  {
    this->apply(x, y);
    return;
  }

  auto& arranger = algebra.resident_arranger();
  auto terms = convert_terms(impl_->terms);
  auto const& r = raw_matrices(y);
  auto const& a = raw_matrices(impl_->a_mats);
  auto const& b = impl_->variable_family == VariableFamily::Middle ? raw_matrices(x) : raw_matrices(impl_->b_mats);
  auto const& c = impl_->variable_family == VariableFamily::Middle ? raw_matrices(impl_->c_mats) : raw_matrices(x);

  validate_term_shapes(y, impl_->a_mats, impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
                       impl_->variable_family == VariableFamily::Middle ? impl_->c_mats : x, impl_->terms);

  // Static environments are host-authored but should not be refreshed during
  // every Krylov matvec.  The active Lanczos input/output vectors are already
  // resident in this same runtime.
  arranger.localizeCoalescedForLinearAlgebra(a, impl_->a_mats.coalesced_values(), /*uploadFromHost=*/true,
                                             /*refreshExisting=*/false);
  if (impl_->variable_family == VariableFamily::Middle)
  {
    arranger.localizeCoalescedForLinearAlgebra(c, impl_->c_mats.coalesced_values(), /*uploadFromHost=*/true,
                                               /*refreshExisting=*/false);
  }
  else
  {
    arranger.localizeCoalescedForLinearAlgebra(b, impl_->b_mats.coalesced_values(), /*uploadFromHost=*/true,
                                               /*refreshExisting=*/false);
  }
  arranger.localizeCoalescedForLinearAlgebra(raw_matrices(x), x.coalesced_values(), /*uploadFromHost=*/false);
  arranger.localizeCoalescedForLinearAlgebra(raw_matrices(y), y.coalesced_values(), /*uploadFromHost=*/false);

  if (!use_legacy_arranger_rabc_planner())
  {
    auto& swapper = arranger.residentSwapper();
    deterministic_right_first_apply(r, a, b, c, impl_->terms, swapper);
    return;
  }

  arranger.resetWork();
  arranger.analyzeComputation(r, a, b, c, terms);
  arranger.compileWorklists(r, a, b, c, /*syncResultsToHost=*/false);
  arranger.doContraction(r, a, b, c);
}

} // namespace uni20::tensorcontraction
