#include <uni20/linalg/backends/cpu/dense_matrix.hpp>
#include <uni20/linalg/backends/cpu/matrix_exponential.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uni20::linalg::backends::cpu
{
namespace detail
{

constexpr std::array<double, 5> kThetaBounds{
    1.495585217958292e-2, // theta_3
    2.539398330063230e-1, // theta_5
    9.504178996162932e-1, // theta_7
    2.097847961257068,    // theta_9
    5.371920351148152     // theta_13
};

template <typename Scalar, typename Integer> constexpr Scalar exact_integer(Integer value)
{
  return static_cast<Scalar>(value);
}

template <typename T> auto adl_abs(T const& value)
{
  using std::abs;
  return abs(value);
}

template <typename T> auto adl_sqrt(T const& value)
{
  using std::sqrt;
  return sqrt(value);
}

template <typename T> auto adl_log2(T const& value)
{
  using std::log2;
  return log2(value);
}

template <typename T, typename Exponent> auto adl_pow(T const& value, Exponent const& exponent)
{
  using std::pow;
  return pow(value, exponent);
}

template <typename T> auto adl_sin(T const& value)
{
  using std::sin;
  return sin(value);
}

template <typename T> bool adl_isfinite(T const& value)
{
  using std::isfinite;
  return isfinite(value);
}

template <typename T> auto adl_ceil(T const& value)
{
  using std::ceil;
  return ceil(value);
}

template <typename T> auto adl_ldexp(T const& value, int exponent)
{
  using std::ldexp;
  return ldexp(value, exponent);
}

template <typename Scalar>
DenseMatrix<Scalar>
linear_combination(std::size_t rows, std::size_t cols,
                   std::initializer_list<std::pair<DenseMatrix<Scalar> const*, Scalar>> const& terms)
{
  DenseMatrix<Scalar> result(rows, cols);
  for (auto const& term : terms)
  {
    DenseMatrix<Scalar> const& mat = *term.first;
    Scalar const coefficient = term.second;
    for (std::size_t i = 0; i < rows; ++i)
    {
      for (std::size_t j = 0; j < cols; ++j)
      {
        result[i, j] += coefficient * mat[i, j];
      }
    }
  }
  return result;
}

template <typename Scalar> DenseMatrix<Scalar> solve_pade(DenseMatrix<Scalar> const& U, DenseMatrix<Scalar> const& V)
{
  DenseMatrix<Scalar> numerator = add(V, U);
  DenseMatrix<Scalar> denominator = subtract(V, U);
  return solve_linear_system(std::move(denominator), std::move(numerator));
}

template <typename Scalar> DenseMatrix<Scalar> pade3(DenseMatrix<Scalar> const& A)
{
  std::array<Scalar, 4> const b{exact_integer<Scalar>(120), exact_integer<Scalar>(60), exact_integer<Scalar>(12),
                                exact_integer<Scalar>(1)};
  std::size_t const n = A.rows();
  DenseMatrix<Scalar> const identity = make_identity<Scalar>(n);
  DenseMatrix<Scalar> const A2 = multiply(A, A);

  DenseMatrix<Scalar> const tmp = linear_combination<Scalar>(n, n, {{&A2, b[3]}, {&identity, b[1]}});
  DenseMatrix<Scalar> const U = multiply(A, tmp);

  DenseMatrix<Scalar> const V = linear_combination<Scalar>(n, n, {{&A2, b[2]}, {&identity, b[0]}});

  return solve_pade(U, V);
}

template <typename Scalar> DenseMatrix<Scalar> pade5(DenseMatrix<Scalar> const& A)
{
  std::array<Scalar, 6> const b{exact_integer<Scalar>(30240), exact_integer<Scalar>(15120), exact_integer<Scalar>(3360),
                                exact_integer<Scalar>(420),   exact_integer<Scalar>(30),    exact_integer<Scalar>(1)};
  std::size_t const n = A.rows();
  DenseMatrix<Scalar> const identity = make_identity<Scalar>(n);
  DenseMatrix<Scalar> const A2 = multiply(A, A);
  DenseMatrix<Scalar> const A4 = multiply(A2, A2);

  DenseMatrix<Scalar> const tmp = linear_combination<Scalar>(n, n, {{&A4, b[5]}, {&A2, b[3]}, {&identity, b[1]}});
  DenseMatrix<Scalar> const U = multiply(A, tmp);

  DenseMatrix<Scalar> const V = linear_combination<Scalar>(n, n, {{&A4, b[4]}, {&A2, b[2]}, {&identity, b[0]}});

  return solve_pade(U, V);
}

template <typename Scalar> DenseMatrix<Scalar> pade7(DenseMatrix<Scalar> const& A)
{
  std::array<Scalar, 8> const b{exact_integer<Scalar>(17297280), exact_integer<Scalar>(8648640),
                                exact_integer<Scalar>(1995840),  exact_integer<Scalar>(277200),
                                exact_integer<Scalar>(25200),    exact_integer<Scalar>(1512),
                                exact_integer<Scalar>(56),       exact_integer<Scalar>(1)};
  std::size_t const n = A.rows();
  DenseMatrix<Scalar> const identity = make_identity<Scalar>(n);
  DenseMatrix<Scalar> const A2 = multiply(A, A);
  DenseMatrix<Scalar> const A4 = multiply(A2, A2);
  DenseMatrix<Scalar> const A6 = multiply(A4, A2);

  DenseMatrix<Scalar> const tmp =
      linear_combination<Scalar>(n, n, {{&A6, b[7]}, {&A4, b[5]}, {&A2, b[3]}, {&identity, b[1]}});
  DenseMatrix<Scalar> const U = multiply(A, tmp);

  DenseMatrix<Scalar> const V =
      linear_combination<Scalar>(n, n, {{&A6, b[6]}, {&A4, b[4]}, {&A2, b[2]}, {&identity, b[0]}});

  return solve_pade(U, V);
}

template <typename Scalar> DenseMatrix<Scalar> pade9(DenseMatrix<Scalar> const& A)
{
  std::array<Scalar, 10> const b{exact_integer<Scalar>(17643225600LL),
                                 exact_integer<Scalar>(8821612800LL),
                                 exact_integer<Scalar>(2075673600),
                                 exact_integer<Scalar>(302702400),
                                 exact_integer<Scalar>(30270240),
                                 exact_integer<Scalar>(2162160),
                                 exact_integer<Scalar>(110880),
                                 exact_integer<Scalar>(3960),
                                 exact_integer<Scalar>(90),
                                 exact_integer<Scalar>(1)};
  std::size_t const n = A.rows();
  DenseMatrix<Scalar> const identity = make_identity<Scalar>(n);
  DenseMatrix<Scalar> const A2 = multiply(A, A);
  DenseMatrix<Scalar> const A4 = multiply(A2, A2);
  DenseMatrix<Scalar> const A6 = multiply(A4, A2);
  DenseMatrix<Scalar> const A8 = multiply(A6, A2);

  DenseMatrix<Scalar> const tmp =
      linear_combination<Scalar>(n, n, {{&A8, b[9]}, {&A6, b[7]}, {&A4, b[5]}, {&A2, b[3]}, {&identity, b[1]}});
  DenseMatrix<Scalar> const U = multiply(A, tmp);

  DenseMatrix<Scalar> const V =
      linear_combination<Scalar>(n, n, {{&A8, b[8]}, {&A6, b[6]}, {&A4, b[4]}, {&A2, b[2]}, {&identity, b[0]}});

  return solve_pade(U, V);
}

template <typename Scalar>
DenseMatrix<Scalar> pade13(DenseMatrix<Scalar> const& A, DenseMatrix<Scalar> const& A2, DenseMatrix<Scalar> const& A4,
                           DenseMatrix<Scalar> const& A6)
{
  std::array<Scalar, 14> const b{exact_integer<Scalar>(64764752532480000LL),
                                 exact_integer<Scalar>(32382376266240000LL),
                                 exact_integer<Scalar>(7771770303897600LL),
                                 exact_integer<Scalar>(1187353796428800LL),
                                 exact_integer<Scalar>(129060195264000LL),
                                 exact_integer<Scalar>(10559470521600LL),
                                 exact_integer<Scalar>(670442572800LL),
                                 exact_integer<Scalar>(33522128640LL),
                                 exact_integer<Scalar>(1323241920),
                                 exact_integer<Scalar>(40840800),
                                 exact_integer<Scalar>(960960),
                                 exact_integer<Scalar>(16380),
                                 exact_integer<Scalar>(182),
                                 exact_integer<Scalar>(1)};

  std::size_t const n = A.rows();
  DenseMatrix<Scalar> const identity = make_identity<Scalar>(n);

  DenseMatrix<Scalar> const first = linear_combination<Scalar>(n, n, {{&A6, b[13]}, {&A4, b[11]}, {&A2, b[9]}});
  DenseMatrix<Scalar> tmp = multiply(A6, first);
  DenseMatrix<Scalar> const second =
      linear_combination<Scalar>(n, n, {{&A6, b[7]}, {&A4, b[5]}, {&A2, b[3]}, {&identity, b[1]}});
  tmp = add(tmp, second);
  DenseMatrix<Scalar> const U = multiply(A, tmp);

  DenseMatrix<Scalar> const third = linear_combination<Scalar>(n, n, {{&A6, b[12]}, {&A4, b[10]}, {&A2, b[8]}});
  DenseMatrix<Scalar> V = multiply(A6, third);
  DenseMatrix<Scalar> const fourth =
      linear_combination<Scalar>(n, n, {{&A6, b[6]}, {&A4, b[4]}, {&A2, b[2]}, {&identity, b[0]}});
  V = add(V, fourth);

  return solve_pade(U, V);
}

template <typename Scalar> int compute_scaling_exponent(DenseMatrix<Scalar> const& A4, DenseMatrix<Scalar> const& A6)
{
  using AccumReal = uni20::accumulation_real_t<Scalar>;

  AccumReal const norm4 = matrix_one_norm(A4);
  AccumReal const norm6 = matrix_one_norm(A6);
  if (!adl_isfinite(norm4) || !adl_isfinite(norm6))
  {
    throw std::overflow_error("matrix powers overflowed while computing matrix_exponential scaling");
  }

  AccumReal const d4 = adl_pow(norm4, AccumReal{0.25});
  AccumReal const d6 = adl_pow(norm6, AccumReal{1} / AccumReal{6});
  AccumReal const eta = std::max(d4, d6);
  if (eta == AccumReal{})
  {
    return 0;
  }

  AccumReal const ratio = eta / static_cast<AccumReal>(kThetaBounds.back());
  if (ratio <= AccumReal{1})
  {
    return 0;
  }

  AccumReal const exponent = adl_log2(ratio);
  if (exponent <= AccumReal{})
  {
    return 0;
  }
  return static_cast<int>(adl_ceil(exponent));
}

template <typename Scalar> uni20::accumulation_real_t<Scalar> entry_abs_log2_upper_bound(Scalar const& value)
{
  using AccumReal = uni20::accumulation_real_t<Scalar>;

  if constexpr (uni20::Complex<Scalar>)
  {
    AccumReal const real = adl_abs(static_cast<AccumReal>(value.real()));
    AccumReal const imag = adl_abs(static_cast<AccumReal>(value.imag()));
    if (!adl_isfinite(real) || !adl_isfinite(imag))
    {
      throw std::overflow_error("matrix_exponential requires finite matrix entries");
    }

    AccumReal const component = std::max(real, imag);
    if (component == AccumReal{})
    {
      return -uni20::numeric_limits<AccumReal>::infinity();
    }

    AccumReal const real_scaled = real / component;
    AccumReal const imag_scaled = imag / component;
    AccumReal const factor = adl_sqrt(real_scaled * real_scaled + imag_scaled * imag_scaled);
    return adl_log2(component) + adl_log2(factor);
  }
  else
  {
    AccumReal const magnitude = adl_abs(static_cast<AccumReal>(value));
    if (!adl_isfinite(magnitude))
    {
      throw std::overflow_error("matrix_exponential requires finite matrix entries");
    }
    if (magnitude == AccumReal{})
    {
      return -uni20::numeric_limits<AccumReal>::infinity();
    }
    return adl_log2(magnitude);
  }
}

template <typename Scalar>
uni20::accumulation_real_t<Scalar> matrix_max_abs_log2_upper_bound(DenseMatrix<Scalar> const& mat)
{
  using AccumReal = uni20::accumulation_real_t<Scalar>;

  AccumReal result = -uni20::numeric_limits<AccumReal>::infinity();
  for (std::size_t i = 0; i < mat.size(); ++i)
  {
    result = std::max(result, entry_abs_log2_upper_bound(mat.data()[i]));
  }
  return result;
}

template <typename Scalar> int compute_prescaling_exponent(DenseMatrix<Scalar> const& A)
{
  using Real = uni20::make_real_t<Scalar>;
  using AccumReal = uni20::accumulation_real_t<Scalar>;

  AccumReal const max_entry_log2 = matrix_max_abs_log2_upper_bound(A);
  if (max_entry_log2 == -uni20::numeric_limits<AccumReal>::infinity())
  {
    return 0;
  }

  AccumReal const dimension = std::max(AccumReal{1}, static_cast<AccumReal>(A.rows()));
  AccumReal const safe_power_log2 =
      AccumReal{-1} + (adl_log2(static_cast<AccumReal>(uni20::numeric_limits<Real>::max())) / AccumReal{6}) -
      adl_log2(dimension);
  AccumReal const exponent = max_entry_log2 - safe_power_log2;
  if (exponent <= AccumReal{})
  {
    return 0;
  }
  return static_cast<int>(adl_ceil(exponent));
}

template <typename Scalar> bool has_real_entries(DenseMatrix<Scalar> const& A)
{
  if constexpr (!uni20::Complex<Scalar>)
  {
    return true;
  }
  else
  {
    using Real = uni20::make_real_t<Scalar>;
    for (std::size_t i = 0; i < A.size(); ++i)
    {
      if (A.data()[i].imag() != Real{})
      {
        return false;
      }
    }
    return true;
  }
}

template <typename OutputScalar, typename InputScalar, typename Coefficient>
DenseMatrix<OutputScalar> scale_to_output(DenseMatrix<InputScalar> const& matrix, Coefficient const& coefficient)
{
  DenseMatrix<OutputScalar> result(matrix.rows(), matrix.cols());
  for (std::size_t i = 0; i < matrix.size(); ++i)
  {
    result.data()[i] = static_cast<OutputScalar>(matrix.data()[i]) * static_cast<OutputScalar>(coefficient);
  }
  return result;
}

template <typename Scalar> uni20::make_real_t<Scalar> real_value(Scalar const& value)
{
  if constexpr (uni20::Complex<Scalar>)
  {
    return value.real();
  }
  else
  {
    return value;
  }
}

template <typename Scalar> void validate_finite_entries(DenseMatrix<Scalar> const& A)
{
  using AccumReal = uni20::accumulation_real_t<Scalar>;

  for (std::size_t i = 0; i < A.size(); ++i)
  {
    if constexpr (uni20::Complex<Scalar>)
    {
      if (!adl_isfinite(static_cast<AccumReal>(A.data()[i].real())) ||
          !adl_isfinite(static_cast<AccumReal>(A.data()[i].imag())))
      {
        throw std::overflow_error("matrix_exponential requires finite matrix entries");
      }
    }
    else
    {
      if (!adl_isfinite(static_cast<AccumReal>(A.data()[i])))
      {
        throw std::overflow_error("matrix_exponential requires finite matrix entries");
      }
    }
  }
}

template <typename Scalar>
bool try_real_skew_symmetric_3x3_matrix_exponential(DenseMatrix<Scalar> const& A, DenseMatrix<Scalar>& result)
{
  using Real = uni20::make_real_t<Scalar>;
  using AccumReal = uni20::accumulation_real_t<Scalar>;

  if (A.rows() != 3 || A.cols() != 3 || !has_real_entries(A))
  {
    return false;
  }

  if (A[0, 0] != Scalar{} || A[1, 1] != Scalar{} || A[2, 2] != Scalar{} || A[0, 1] != -A[1, 0] || A[0, 2] != -A[2, 0] ||
      A[1, 2] != -A[2, 1])
  {
    return false;
  }

  Real const x = -real_value(A[1, 2]);
  Real const y = real_value(A[0, 2]);
  Real const z = -real_value(A[0, 1]);
  AccumReal const max_component = std::max(
      {adl_abs(static_cast<AccumReal>(x)), adl_abs(static_cast<AccumReal>(y)), adl_abs(static_cast<AccumReal>(z))});
  if (max_component == AccumReal{})
  {
    result = make_identity<Scalar>(3);
    return true;
  }
  if (!adl_isfinite(max_component))
  {
    throw std::overflow_error("matrix_exponential requires finite matrix entries");
  }

  AccumReal const x_scaled = static_cast<AccumReal>(x) / max_component;
  AccumReal const y_scaled = static_cast<AccumReal>(y) / max_component;
  AccumReal const z_scaled = static_cast<AccumReal>(z) / max_component;
  AccumReal const norm_scaled = adl_sqrt(x_scaled * x_scaled + y_scaled * y_scaled + z_scaled * z_scaled);
  AccumReal const theta = max_component * norm_scaled;
  if (!adl_isfinite(theta))
  {
    return false;
  }

  Real const ux = static_cast<Real>(x_scaled / norm_scaled);
  Real const uy = static_cast<Real>(y_scaled / norm_scaled);
  Real const uz = static_cast<Real>(z_scaled / norm_scaled);
  DenseMatrix<Scalar> K(3, 3);
  K[0, 1] = Scalar(-uz);
  K[0, 2] = Scalar(uy);
  K[1, 0] = Scalar(uz);
  K[1, 2] = Scalar(-ux);
  K[2, 0] = Scalar(-uy);
  K[2, 1] = Scalar(ux);

  AccumReal const half_theta = theta / AccumReal{2};
  AccumReal const sin_half_theta = adl_sin(half_theta);
  Real const sine = static_cast<Real>(adl_sin(theta));
  Real const one_minus_cosine = static_cast<Real>(AccumReal{2} * sin_half_theta * sin_half_theta);
  DenseMatrix<Scalar> const K2 = multiply(K, K);
  result = add(add(make_identity<Scalar>(3), scale(K, sine)), scale(K2, one_minus_cosine));
  return true;
}

template <uni20::RealOrComplex Scalar> DenseMatrix<Scalar> matrix_exponential_scaled(DenseMatrix<Scalar> A)
{
  using Real = uni20::make_real_t<Scalar>;

  if (A.rows() != A.cols())
  {
    throw std::invalid_argument("matrix_exponential requires a square matrix");
  }

  if (A.rows() == 0)
  {
    return DenseMatrix<Scalar>();
  }

  std::size_t const n = A.rows();
  detail::validate_finite_entries(A);

  uni20::accumulation_real_t<Scalar> const normA = matrix_one_norm(A);
  if (normA == uni20::accumulation_real_t<Scalar>{})
  {
    return make_identity<Scalar>(n);
  }
  DenseMatrix<Scalar> skew_symmetric_3x3_result;
  if (detail::try_real_skew_symmetric_3x3_matrix_exponential(A, skew_symmetric_3x3_result))
  {
    return skew_symmetric_3x3_result;
  }
  if (!adl_isfinite(normA))
  {
    throw std::overflow_error("matrix_exponential requires a finite matrix norm");
  }

  if (normA <= detail::kThetaBounds[0])
  {
    return detail::pade3(A);
  }
  if (normA <= detail::kThetaBounds[1])
  {
    return detail::pade5(A);
  }
  if (normA <= detail::kThetaBounds[2])
  {
    return detail::pade7(A);
  }
  if (normA <= detail::kThetaBounds[3])
  {
    return detail::pade9(A);
  }

  int const prescaling = detail::compute_prescaling_exponent(A);
  if (prescaling != 0)
  {
    Real const prescale = adl_ldexp(Real{1}, -prescaling);
    A = scale(std::move(A), prescale);
  }

  DenseMatrix<Scalar> A2 = multiply(A, A);
  DenseMatrix<Scalar> A4 = multiply(A2, A2);
  DenseMatrix<Scalar> A6 = multiply(A4, A2);

  int const additional_scaling = detail::compute_scaling_exponent(A4, A6);
  int const s = prescaling + additional_scaling;

  DenseMatrix<Scalar> A2_scaled;
  DenseMatrix<Scalar> A4_scaled;
  DenseMatrix<Scalar> A6_scaled;
  if (additional_scaling == 0)
  {
    A2_scaled = std::move(A2);
    A4_scaled = std::move(A4);
    A6_scaled = std::move(A6);
  }
  else
  {
    Real const scale_real = adl_ldexp(Real{1}, -additional_scaling);
    Real const scale_sq = scale_real * scale_real;
    Real const scale_pow4 = scale_sq * scale_sq;
    Real const scale_pow6 = scale_pow4 * scale_sq;
    A = scale(std::move(A), scale_real);
    A2_scaled = scale(std::move(A2), scale_sq);
    A4_scaled = scale(std::move(A4), scale_pow4);
    A6_scaled = scale(std::move(A6), scale_pow6);
  }

  DenseMatrix<Scalar> result = detail::pade13(A, A2_scaled, A4_scaled, A6_scaled);
  for (int k = 0; k < s; ++k)
  {
    result = multiply(result, result);
  }

  return result;
}

} // namespace detail

template <uni20::RealOrComplex Scalar>
DenseMatrix<Scalar> matrix_exponential(DenseMatrix<Scalar> const& matrix, uni20::make_real_t<Scalar> t)
{
  return detail::matrix_exponential_scaled(detail::scale_to_output<Scalar>(matrix, t));
}

template <uni20::RealOrComplex Scalar>
DenseMatrix<uni20::complex<uni20::make_real_t<Scalar>>> matrix_exponential(DenseMatrix<Scalar> const& matrix,
                                                                           uni20::complex<uni20::make_real_t<Scalar>> t)
{
  using OutputScalar = uni20::complex<uni20::make_real_t<Scalar>>;
  return detail::matrix_exponential_scaled(detail::scale_to_output<OutputScalar>(matrix, t));
}

template DenseMatrix<float> matrix_exponential(DenseMatrix<float> const&, float);
template DenseMatrix<double> matrix_exponential(DenseMatrix<double> const&, double);
template DenseMatrix<long double> matrix_exponential(DenseMatrix<long double> const&, long double);
template DenseMatrix<uni20::complex<float>> matrix_exponential(DenseMatrix<uni20::complex<float>> const&, float);
template DenseMatrix<uni20::complex<double>> matrix_exponential(DenseMatrix<uni20::complex<double>> const&, double);
template DenseMatrix<uni20::complex<long double>> matrix_exponential(DenseMatrix<uni20::complex<long double>> const&,
                                                                     long double);

template DenseMatrix<uni20::complex<float>> matrix_exponential(DenseMatrix<float> const&, uni20::complex<float>);
template DenseMatrix<uni20::complex<double>> matrix_exponential(DenseMatrix<double> const&, uni20::complex<double>);
template DenseMatrix<uni20::complex<long double>> matrix_exponential(DenseMatrix<long double> const&,
                                                                     uni20::complex<long double>);
template DenseMatrix<uni20::complex<float>> matrix_exponential(DenseMatrix<uni20::complex<float>> const&,
                                                               uni20::complex<float>);
template DenseMatrix<uni20::complex<double>> matrix_exponential(DenseMatrix<uni20::complex<double>> const&,
                                                                uni20::complex<double>);
template DenseMatrix<uni20::complex<long double>> matrix_exponential(DenseMatrix<uni20::complex<long double>> const&,
                                                                     uni20::complex<long double>);

#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
template DenseMatrix<mplapack_binary128_t> matrix_exponential(DenseMatrix<mplapack_binary128_t> const&,
                                                              mplapack_binary128_t);
template DenseMatrix<uni20::complex<mplapack_binary128_t>>
matrix_exponential(DenseMatrix<uni20::complex<mplapack_binary128_t>> const&, mplapack_binary128_t);
template DenseMatrix<uni20::complex<mplapack_binary128_t>> matrix_exponential(DenseMatrix<mplapack_binary128_t> const&,
                                                                              uni20::complex<mplapack_binary128_t>);
template DenseMatrix<uni20::complex<mplapack_binary128_t>>
matrix_exponential(DenseMatrix<uni20::complex<mplapack_binary128_t>> const&, uni20::complex<mplapack_binary128_t>);
#endif

} // namespace uni20::linalg::backends::cpu
