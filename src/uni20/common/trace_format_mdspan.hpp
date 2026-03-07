#include <mdspan>

namespace trace
{

template <typename MDS>
concept MdspanLike = requires(MDS m) {
                       typename MDS::element_type;
                       m.rank();
                       m.extent(0);
                       m(0); // implies operator() works
                     };

// Recursively visit mdspan, producing nested string containers
template <typename MDS>
auto mdspan_to_strings(const MDS& mds, const FormattingOptions& opts, std::size_t dim = 0,
                       std::vector<std::size_t> index_prefix = {}) -> std::vector<std::string>
{
  std::vector<std::string> result;
  std::size_t extent = mds.extent(dim);

  for (std::size_t i = 0; i < extent; ++i)
  {
    auto new_prefix = index_prefix;
    new_prefix.push_back(i);

    if (dim + 1 == mds.rank())
    {
      // Base case: this is a 1D slice
      std::vector<std::string> line;
      for (std::size_t j = 0; j < mds.extent(dim + 1); ++j)
      {
        auto full_index = new_prefix;
        full_index.push_back(j);

        // Convert index vector to parameter pack for operator()
        auto val = [&]<std::size_t... Is>(std::index_sequence<Is...>) { return mds(full_index[Is]...); }
        (std::make_index_sequence<sizeof...(decltype(new_prefix))>{});
        line.push_back(formatValue(val, opts));
      }
      result.push_back(formatContainerToString(line));
    }
    else
    {
      result.push_back(formatContainerToString(mdspan_to_strings(mds, opts, dim + 1, new_prefix)));
    }
  }

  return result;
}

template <MdspanLike MDS> std::string formatValue(const MDS& mds, const FormattingOptions& opts)
{
  return formatContainerToString(mdspan_to_strings(mds, opts));
}

} // namespace trace

namespace trace
{

template <std::size_t R, typename MDS>
auto mdspan_to_strings(const MDS& mds, const FormattingOptions& opts, std::size_t dim = 0,
                       std::array<std::size_t, R> index_prefix = {}) -> std::vector<std::string>
{
  std::vector<std::string> result;
  std::size_t extent = mds.extent(dim);

  for (std::size_t i = 0; i < extent; ++i)
  {
    auto next_prefix = index_prefix;
    next_prefix[dim] = i;

    if (dim + 1 == R)
    {
      // Last dimension — format a single "row"
      std::vector<std::string> line;
      for (std::size_t j = 0; j < mds.extent(R - 1); ++j)
      {
        next_prefix[R - 1] = j;
        auto val = [&]<std::size_t... Is>(std::index_sequence<Is...>) { return mds(next_prefix[Is]...); }
        (std::make_index_sequence<R>{});
        line.push_back(formatValue(val, opts));
      }
      result.push_back(formatContainerToString(line));
    }
    else
    {
      result.push_back(formatContainerToString(mdspan_to_strings<R>(mds, opts, dim + 1, next_prefix)));
    }
  }

  return result;
}

template <typename MDS>
std::string formatValue(const MDS& mds, const FormattingOptions& opts)
  requires MdspanLike<MDS>
{
  constexpr std::size_t R = MDS::rank(); // must be constexpr for std::array
  return formatContainerToString(mdspan_to_strings<R>(mds, opts));
}

template <std::size_t Rank> auto make_full_slice_tuple()
{
  return []<std::size_t... Is>(std::index_sequence<Is...>)
  {
    return std::make_tuple((void(Is), stdex::full_extent)...);
  }
  (std::make_index_sequence<Rank>{});
}

template <typename MDS> auto submdspan_first(const MDS& mds, std::size_t i)
{
  constexpr std::size_t rank = MDS::rank();
  auto slice_tuple = make_full_slice_tuple<rank - 1>();
  return std::apply([&](auto... rest) { return stdex::submdspan(mds, i, rest...); }, slice_tuple);
}

template <typename MDS> auto format_mdspan_to_strings(const MDS& mds)
{
  using idx_t = typename MDS::index_type;
  constexpr std::size_t rank = MDS::rank();

  if constexpr (rank == 0)
  {
    return std::vector<std::string>{formatValue(mds(), trace::formatting_options)};
  }
  else if constexpr (rank == 1)
  {
    std::vector<std::string> out;
    for (idx_t i = 0; i < mds.extent(0); ++i)
    {
      out.push_back(formatValue(mds(i), trace::formatting_options));
    }
    return out;
  }
  else
  {
    std::vector<decltype(format_mdspan_to_strings(submdspan_first(mds, 0)))> out;
    for (idx_t i = 0; i < mds.extent(0); ++i)
    {
      out.push_back(format_mdspan_to_strings(submdspan_first(mds, i)));
    }
    return out;
  }
}

✅ The solution you’re hinting at : a wrapper smart_submdspan :

    template <typename MDS, typename... Fixed>
    auto
    smart_submdspan(const MDS& mds, Fixed... fixed)
{
  constexpr std::size_t total_rank = MDS::rank();
  constexpr std::size_t num_fixed = sizeof...(fixed);
  static_assert(num_fixed <= total_rank);

  auto rest = []<std::size_t... Is>(std::index_sequence<Is...>)
  {
    return std::make_tuple((void(Is), stdex::full_extent)...);
  }
  (std::make_index_sequence<total_rank - num_fixed>{});

  return std::apply([&](auto... rest_extents) { return stdex::submdspan(mds, fixed..., rest_extents...); }, rest);
}

Now you can call :

    smart_submdspan(mds, i);
smart_submdspan(mds, i, j); // etc.

And it will automatically fill in the full_extent tail for you. Much better ergonomics.
🌟 Want it to be even more magical?

If you alias smart_submdspan to operator() in a lightweight wrapper around mdspan, you get slicing that feels like Python:

auto view = mdspan_wrapper(mds);
view(3);       // Fix dimension 0
view(3, 1);    // Fix dims 0 and 1
view(_, _, 5); // Full extent for dims 0,1 and fix dim 2
