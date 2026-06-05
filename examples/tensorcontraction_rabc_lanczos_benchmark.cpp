#include <uni20/tensorcontraction/lanczos.hpp>
#include <uni20/tensorcontraction/rabc_lanczos_fixture.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <fmt/core.h>

#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

namespace utc = uni20::tensorcontraction;

namespace
{

void ensure_mpi_initialized()
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0)
  {
    return;
  }

  MPI_Init(nullptr, nullptr);
  std::atexit([] {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0)
    {
      MPI_Finalize();
    }
  });
}

auto optional_env_string(char const* name) -> std::optional<std::string>
{
  char const* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0')
  {
    return std::nullopt;
  }
  return std::string(raw);
}

auto env_int(char const* name, int fallback) -> int
{
  auto const text = optional_env_string(name);
  if (!text.has_value())
  {
    return fallback;
  }

  std::size_t consumed = 0;
  auto const value = std::stoi(*text, &consumed);
  if (consumed != text->size())
  {
    throw std::invalid_argument(std::string("invalid integer value for ") + name + ": " + *text);
  }
  return value;
}

auto env_bool(char const* name, bool fallback) -> bool
{
  auto const text = optional_env_string(name);
  if (!text.has_value())
  {
    return fallback;
  }
  return *text == "1" || *text == "true" || *text == "on" || *text == "yes";
}

auto fixture_path(int argc, char** argv) -> std::string
{
  if (argc > 1)
  {
    return argv[1];
  }
  if (auto path = optional_env_string("UNI20_RABC_FIXTURE"); path.has_value())
  {
    return *path;
  }
  throw std::invalid_argument("usage: tensorcontraction_rabc_lanczos_benchmark <fixture-path>");
}

auto stop_reason_name(utc::LanczosStopReason reason) -> char const*
{
  switch (reason)
  {
    case utc::LanczosStopReason::Converged:
      return "converged";
    case utc::LanczosStopReason::MaxIterations:
      return "max_iterations";
    case utc::LanczosStopReason::InvariantSubspace:
      return "invariant_subspace";
    case utc::LanczosStopReason::LossOfOrthogonality:
      return "loss_of_orthogonality";
  }
  return "unknown";
}

auto clone_family(utc::MatrixFamily const& source) -> utc::MatrixFamily
{
  utc::MatrixFamily clone(source.blocks());
  clone.assign(source);
  return clone;
}

class BenchFile {
  public:
    BenchFile()
    {
      auto path = optional_env_string("MP_BENCHFILE");
      if (!path.has_value())
      {
        return;
      }

      file_ = std::fopen(path->c_str(), "w");
      if (file_ != nullptr)
      {
        fmt::print(file_, "#Repeat #WallS #Energy #Residual #Iter #StopReason #MatvecS #MatvecN #ReductionsS "
                          "#OrthoS #RitzDiagS #RitzVectorS #ResidualVectorS #ResidualNormS #FinishS\n");
      }
    }

    BenchFile(BenchFile const&) = delete;
    BenchFile& operator=(BenchFile const&) = delete;

    ~BenchFile()
    {
      if (file_ != nullptr)
      {
        std::fclose(file_);
      }
    }

    void write(int repeat, double wall_seconds, utc::LanczosResult const& result)
    {
      if (file_ == nullptr)
      {
        return;
      }

      auto const& timing = result.timings;
      fmt::print(file_,
                 "{} {:.9g} {:.16g} {:.9g} {} {} {:.9g} {} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} "
                 "{:.9g}\n",
                 repeat, wall_seconds, result.eigenvalue, result.residual_norm, result.iterations,
                 stop_reason_name(result.stop_reason), timing.matvec.wall_seconds, timing.matvec_count,
                 timing.reductions.wall_seconds, timing.orthogonalization.wall_seconds,
                 timing.ritz_diagonalization.wall_seconds, timing.ritz_vector.wall_seconds,
                 timing.residual_vector.wall_seconds, timing.residual_norm.wall_seconds, timing.finish.wall_seconds);
    }

  private:
    std::FILE* file_ = nullptr;
};

} // namespace

auto main(int argc, char** argv) -> int
{
  try
  {
    ensure_mpi_initialized();

    auto fixture = utc::read_rabc_lanczos_fixture(fixture_path(argc, argv));
    auto op = fixture.make_operator();
    utc::VectorAlgebraEngine algebra;
    if (algebra.uses_host_backend())
    {
      throw std::runtime_error("R/A/B/C Lanczos benchmark requires the resident CUDA/MPI TensorContraction backend");
    }

    auto initial = clone_family(fixture.input_vector);
    algebra.upload(initial);
    algebra.set_host_synchronization(false);

    auto const max_iterations = env_int("UNI20_RABC_LANCZOS_ITERS", 24);
    auto const min_iterations = env_int("UNI20_RABC_LANCZOS_MIN_ITERS", max_iterations);
    auto const repeats = env_int("UNI20_RABC_REPEATS", 1);
    auto const warmup = env_bool("UNI20_RABC_WARMUP", false);
    utc::LanczosOptions options{
        .max_iterations = max_iterations,
        .min_iterations = min_iterations,
        .tolerance = 1.0e-12,
    };

    auto apply = [&](utc::MatrixFamily const& x, utc::MatrixFamily& y) { op.apply_resident(x, y, algebra); };
    auto run_once = [&] {
      auto guess = utc::make_like(initial);
      struct GuessResidentRelease
      {
          utc::VectorAlgebraEngine& algebra;
          utc::MatrixFamily const& guess;
          ~GuessResidentRelease() { algebra.release(guess); }
      } release_guess{algebra, guess};
      algebra.copy(initial, guess);
      return utc::lanczos_lowest_with_engine(guess, apply, algebra, options);
    };

    if (warmup)
    {
      static_cast<void>(run_once());
    }

    BenchFile bench;
    fmt::print("R/A/B/C Lanczos fixture benchmark\n");
    fmt::print("blocks={} terms={} max_iterations={} min_iterations={} repeats={}\n", fixture.input_vector.size(),
               fixture.terms.size(), max_iterations, min_iterations, repeats);

    for (int repeat = 0; repeat < repeats; ++repeat)
    {
      auto const start = std::chrono::steady_clock::now();
      auto result = run_once();
      auto const stop = std::chrono::steady_clock::now();
      auto const wall_seconds = std::chrono::duration<double>(stop - start).count();
      fmt::print("repeat={} wall={:.6g}s energy={:.16g} residual={:.9g} iter={} stop={} matvec={:.6g}s n={}\n", repeat,
                 wall_seconds, result.eigenvalue, result.residual_norm, result.iterations,
                 stop_reason_name(result.stop_reason), result.timings.matvec.wall_seconds, result.timings.matvec_count);
      bench.write(repeat, wall_seconds, result);
    }
  }
  catch (std::exception const& ex)
  {
    fmt::print(stderr, "R/A/B/C Lanczos fixture benchmark failed: {}\n", ex.what());
    return 1;
  }

  return 0;
}
