#pragma once

#include "mdspan.hpp"

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <ranges>
#include <type_traits>

// implementation of trace functions.
// TODO: the formatting part, uncluding formatValue, could be moved to a public header

// trace namespace: Global settings for our trace facility.
namespace trace
{

namespace detail
{

template <typename T, std::integral U, typename... Extra>
inline std::int64_t get_ulps(T const& a, T const& b, U ulps, Extra&&... extra)
{
  return static_cast<std::int64_t>(ulps);
}

template <typename T> inline std::int64_t get_ulps(T const& a, T const& b) { return 4; }

template <typename T, typename... Extra> inline std::int64_t get_ulps(T const& a, T const& b, Extra&&...) { return 4; }

} // namespace detail

struct FormattingOptions;

/// \brief get the FormattingOptions object for a given module; use "" as the module name (or default parameter) for
/// the global format
inline FormattingOptions& get_formatting_options(const std::string& module = "");

/// \brief Configuration and formatting options for a trace module.
struct FormattingOptions
{
    //--- Precision settings ---------------------------------------------------

    /// Floating-point precision for formatting float32 values.
    int fp_precision_float32 = 6;

    /// Floating-point precision for formatting float64 values.
    int fp_precision_float64 = 15;

    /// Floating-point precision for formatting configured float128 values.
    int fp_precision_float128 = 36;

    //--- Output layout ---------------------------------------------------------

    /// Maximum width (in characters) before switching to multi-line.
    int terminal_width = terminal::columns();

    //--- Color configuration --------------------------------------------------

    /// Enables/disables/auto color for this module (from UNI20_COLOR or programmatic override).
    struct ColorOptionTraits
    {
        enum Enum
        {
          yes,
          no,
          autocolor
        };
        static constexpr Enum Default = autocolor;
        static constexpr const char* StaticName = "Color options (yes/no/auto)";
        static constexpr std::array<const char*, 3> Names = {"yes", "no", "auto"};
    };

    using ColorOptions = NamedEnumeration<ColorOptionTraits>;
    ColorOptions color;

    /// Whether to actually emit color sequences.
    bool showColor = terminal::is_a_terminal(stderr);

    //--- Global flags ----------------------------------------------------------

    /// Abort on error if true. This is static, since it only makes sense globally
    inline static bool errorsAbort = true;

    /// Prefix each trace with a timestamp (from UNI20_TRACE_TIMESTAMP or module override).
    bool timestamp = false;

    /// Controls thread-id prefixing (from UNI20_TRACE_THREAD_ID or module override).
    /// Values: yes / no / auto.
    struct ThreadIdOptionTraits
    {
        enum Enum
        {
          yes,
          no,
          auto_detect
        };
        static constexpr Enum Default = auto_detect;
        static constexpr const char* StaticName = "Thread-id options (yes/no/auto)";
        static constexpr std::array<const char*, 3> Names = {"yes", "no", "auto"};
    };

    using ThreadIdOptions = NamedEnumeration<ThreadIdOptionTraits>;
    ThreadIdOptions threadId = ThreadIdOptions();

    //--- Output sink -----------------------------------------------------------

    /// File to which this module writes (stderr by default, or UNI20_TRACEFILE).
    FILE* outputStream = stderr;

    // The output sink is a function that takes a string and returns void
    // Default version writes to stderr
    using Sink = std::function<void(std::string)>;

    /// Function that actually emits strings (defaults to fputs to stderr).
    Sink sink = [](std::string s) {
      std::fputs(s.c_str(), stderr);
      std::fflush(stderr);
    };

    /// Shared presentation rendering policy used by trace formatting.
    uni20::presentation::output_policy renderPolicy = uni20::presentation::terminal_policy(outputStream);

    /// Presentation tensor-art policy used for mdspan and tensor-like values.
    uni20::presentation::mdspan_format_options mdspanFormatPolicy;

    /// Presentation preview policy used for mdspan and tensor-like trace values.
    uni20::presentation::mdspan_preview_options mdspanPreviewPolicy;

    //--- Style map -------------------------------------------------------------

    /// Holds per-kind styles (keys like "TRACE", "TRACE_LINE", etc).
    mutable std::map<std::string, terminal::TerminalStyle> Styles;

  private:
    static ColorOptions color_option_from_policy(uni20::presentation::color_mode mode)
    {
      using CO = ColorOptions::Enum;
      switch (mode)
      {
        case uni20::presentation::color_mode::always:
          return CO::yes;
        case uni20::presentation::color_mode::never:
          return CO::no;
        case uni20::presentation::color_mode::automatic:
          return CO::autocolor;
      }
      return CO::autocolor;
    }

  public:
    //--- Constructors ---------------------------------------------------------

    /// \brief Default constructor.
    /// \details Seeds all built-in style keys and reads the global trace environment.
    FormattingOptions()
    {
      color = color_option_from_policy(renderPolicy.color);
      updateShowColor();

      // Default style definitions
      static constexpr std::pair<std::string_view, std::string_view> kDefaults[] = {{"TRACE", "Cyan"},
                                                                                    {"DEBUG_TRACE", "Green"},
                                                                                    {"TRACE_EXPR", "Blue"},
                                                                                    {"TRACE_VALUE", ""},
                                                                                    {"TRACE_MODULE", "Cyan;Bold"},
                                                                                    {"TRACE_FILENAME", "Red"},
                                                                                    {"TRACE_LINE", "Bold"},
                                                                                    {"TRACE_STRING", "Cyan"},
                                                                                    {"CHECK", "Red"},
                                                                                    {"DEBUG_CHECK", "Red"},
                                                                                    {"PRECONDITION", "Red"},
                                                                                    {"DEBUG_PRECONDITION", "Red"},
                                                                                    {"PANIC", "Red"},
                                                                                    {"ERROR", "Red"},
                                                                                    {"TIMESTAMP", "LightGray"},
                                                                                    {"THREAD_ID", "LightMagenta"}};

      for (auto [kind, def] : kDefaults)
      {
        std::string env = std::string("UNI20_COLOR_") + kind;
        Styles[std::string(kind)] = terminal::getenv_or_default<terminal::TerminalStyle>(env, def);
      }

      // Global sink override
      if (auto* path = std::getenv("UNI20_TRACEFILE"))
      {
        FILE* out = nullptr;

        if (std::strcmp(path, "-") == 0 || std::strcmp(path, "stdout") == 0)
        {
          out = stdout;
        }
        else if (std::strcmp(path, "stderr") == 0)
        {
          out = stderr;
        }
        else if (std::strlen(path) > 0)
        {
          bool append = false;
          if (path[0] == '+')
          {
            ++path;
            append = true;
          }
          out = std::fopen(path, append ? "a" : "w");
        }

        if (out)
        {
          this->set_output_stream(out);
        }
      }

      // Precision
      fp_precision_float32 = terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT32", 6);
      fp_precision_float64 = terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT64", 15);
      fp_precision_float128 = terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT128", 36);

      // Global flags
      timestamp = terminal::getenv_or_default<terminal::toggle>("UNI20_TRACE_TIMESTAMP", true);
      threadId = parse_thread_id_option_from_env("UNI20_TRACE_THREAD_ID", threadId);
    }

    /// \brief Module‐specific constructor
    ///
    /// Delegates to the default ctor then applies only trace module overrides:
    ///   - UNI20_TRACEFILE_MODULE_<MODULE>
    ///   - UNI20_TRACE_TIMESTAMP_MODULE_<MODULE>
    ///   - UNI20_TRACE_THREAD_ID_MODULE_<MODULE>
    FormattingOptions(std::string_view module)
    {
      *this = trace::get_formatting_options(""); // inherit all global settings

      std::string mod{module};

      // Style "TRACE" is special: the inherited version comes from "TRACE_MODULE", as a default for all modules
      Styles["TRACE"] = Styles["TRACE_MODULE"];
      // Override with _MODULE_XXXX versions from the environment, if they exist
      static constexpr std::string_view kDefaults[] = {"TRACE",        "DEBUG_TRACE",    "TRACE_EXPR",
                                                       "TRACE_VALUE",  "TRACE_FILENAME", "TRACE_LINE",
                                                       "TRACE_STRING", "TIMESTAMP",      "THREAD_ID"};

      for (auto kind : kDefaults)
      {
        std::string env = fmt::format("UNI20_COLOR_{}_MODULE_{}", kind, module);
        terminal::TerminalStyle def = Styles[std::string(kind)];
        Styles[std::string(kind)] = terminal::getenv_or_default<terminal::TerminalStyle>(env, def);
      }

      // Module sink override via UNI20_TRACEFILE_MODULE_<MODULE>
      if (auto* path = std::getenv(("UNI20_TRACEFILE_MODULE_" + mod).c_str()))
      {
        FILE* out = nullptr;

        if (std::strcmp(path, "-") == 0 || std::strcmp(path, "stdout") == 0)
        {
          out = stdout;
        }
        else if (std::strcmp(path, "stderr") == 0)
        {
          out = stderr;
        }
        else if (std::strlen(path) > 0)
        {
          bool append = false;
          if (path[0] == '+')
          {
            ++path;
            append = true;
          }
          out = std::fopen(path, append ? "a" : "w");
        }

        if (out)
        {
          this->set_output_stream(out);
        }
      }

      // Precision overrides
      fp_precision_float32 =
          terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT32_MODULE_" + mod, fp_precision_float32);
      fp_precision_float64 =
          terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT64_MODULE_" + mod, fp_precision_float64);
      fp_precision_float128 =
          terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT128_MODULE_" + mod, fp_precision_float128);

      // flags overrides
      timestamp = terminal::getenv_or_default<terminal::toggle>("UNI20_TRACE_TIMESTAMP_MODULE_" + mod, timestamp);
      threadId = parse_thread_id_option_from_env("UNI20_TRACE_THREAD_ID_MODULE_" + mod, threadId);
    }

    //--- Public Interface ------------------------------------------------------

    /// Set a custom sink function for this module.
    void set_sink(Sink s)
    {
      sink = std::move(s);
      outputStream = nullptr;
      updateShowColor();
    }

    /// Change the output FILE* for this module.
    void set_output_stream(FILE* f)
    {
      outputStream = f;
      sink = [f](std::string s) {
        std::fputs(s.c_str(), f);
        std::fflush(f);
      };
      updateShowColor();
    }

    /// Enable or disable color output for this module.
    void set_color_output(ColorOptions c)
    {
      color = c;
      updateShowColor();
    }

    /// Enable, disable, or auto-detect color output using the shared presentation color mode.
    void set_color_output(uni20::presentation::color_mode c)
    {
      color = color_option_from_policy(c);
      updateShowColor();
    }

    /// Query whether color should be used in this module.
    bool should_show_color() const { return showColor; }

    /// \brief Return the shared presentation output policy used by this trace module.
    /// \return Mutable presentation policy.
    uni20::presentation::output_policy& presentation_policy() { return renderPolicy; }

    /// \brief Return the shared presentation output policy used by this trace module.
    /// \return Immutable presentation policy.
    uni20::presentation::output_policy const& presentation_policy() const { return renderPolicy; }

    /// \brief Return the shared mdspan/tensor presentation policy used by this trace module.
    /// \return Mutable mdspan formatting policy.
    uni20::presentation::mdspan_format_options& mdspan_format_policy() { return mdspanFormatPolicy; }

    /// \brief Return the shared mdspan/tensor presentation policy used by this trace module.
    /// \return Immutable mdspan formatting policy.
    uni20::presentation::mdspan_format_options const& mdspan_format_policy() const { return mdspanFormatPolicy; }

    /// \brief Return the shared mdspan/tensor preview policy used by this trace module.
    /// \return Mutable mdspan preview policy.
    uni20::presentation::mdspan_preview_options& mdspan_preview_policy() { return mdspanPreviewPolicy; }

    /// \brief Return the shared mdspan/tensor preview policy used by this trace module.
    /// \return Immutable mdspan preview policy.
    uni20::presentation::mdspan_preview_options const& mdspan_preview_policy() const { return mdspanPreviewPolicy; }

    /// \brief Build numeric formatting controls from trace precision settings.
    /// \return Numeric presentation options preserving trace fixed-point behavior.
    uni20::presentation::numeric_format_options numeric_format_policy() const
    {
      auto policy = mdspanFormatPolicy.numeric;
      policy.float32_precision = fp_precision_float32;
      policy.float64_precision = fp_precision_float64;
      policy.float128_precision = fp_precision_float128;
      policy.notation = uni20::presentation::real_notation::fixed;
      return policy;
    }

    /// Enable or disable abort-on-error for this module.
    static void set_errors_abort(bool b) { errorsAbort = b; }

    /// Query the abort-on-error setting.
    static bool errors_abort() { return errorsAbort; }

    /// Get or compute the terminal style for a given kind in this module.
    // terminal::TerminalStyle get_module_terminal_style(const std::string& kind, const std::string& module) const
    // {
    //   std::string key = kind + "_MODULE_" + module;
    //   if (!Styles.count(key))
    //   {
    //     std::string env_mod = "UNI20_COLOR_" + kind + "_MODULE_" + module;
    //     std::string env_glob = "UNI20_COLOR_" + kind;
    //     Styles[key] = terminal::getenv_or_default<terminal::TerminalStyle>(
    //         env_mod, terminal::getenv_or_default<terminal::TerminalStyle>(env_glob, Styles[kind]));
    //   }
    //   return Styles[key];
    // }

    /// \brief Format text using the "global" style (no module) for the given kind.
    std::string format_style(const std::string& str, const std::string& kind) const
    {
      // empty module name = global/default styles
      if (Styles.find(kind) == Styles.end())
      {
        fmt::print(stderr, "UNEXPECTED: unknown format style: {}\n", kind);
        return str;
      }

      uni20::presentation::styled_text text;
      this->append_style(text, str, kind);
      return this->render(text);
    }

    /// \brief Format a semantic glyph using this module's style for the given kind.
    /// \param glyph Semantic presentation glyph.
    /// \param kind Style-map key to apply.
    /// \return Rendered glyph string.
    std::string format_glyph(uni20::presentation::semantic_glyph glyph, std::string const& kind) const
    {
      if (Styles.find(kind) == Styles.end())
      {
        fmt::print(stderr, "UNEXPECTED: unknown format style: {}\n", kind);
        return uni20::presentation::render_glyph(glyph, renderPolicy);
      }

      uni20::presentation::styled_text text;
      this->append_glyph(text, glyph, kind);
      return this->render(text);
    }

    /// \brief Append styled text without rendering it immediately.
    /// \param text Presentation document to append to.
    /// \param str Text payload.
    /// \param kind Style-map key to apply.
    void append_style(uni20::presentation::styled_text& text, std::string_view str, std::string const& kind) const
    {
      auto const it = Styles.find(kind);
      if (it == Styles.end())
      {
        fmt::print(stderr, "UNEXPECTED: unknown format style: {}\n", kind);
        text.append(str);
        return;
      }
      text.append(str, it->second);
    }

    /// \brief Append a styled semantic glyph without rendering it immediately.
    /// \param text Presentation document to append to.
    /// \param glyph Semantic presentation glyph.
    /// \param kind Style-map key to apply.
    void append_glyph(uni20::presentation::styled_text& text, uni20::presentation::semantic_glyph glyph,
                      std::string const& kind) const
    {
      auto const it = Styles.find(kind);
      if (it == Styles.end())
      {
        fmt::print(stderr, "UNEXPECTED: unknown format style: {}\n", kind);
        text.append(glyph);
        return;
      }
      text.append(glyph, it->second);
    }

    /// \brief Return the terminal style associated with a trace style-map key.
    /// \param kind Style-map key such as `TRACE` or `TRACE_LINE`.
    /// \return Configured style, or an empty style when the key is unknown.
    [[nodiscard]] terminal::TerminalStyle presentation_style(std::string_view kind) const
    {
      auto const it = Styles.find(std::string(kind));
      return it == Styles.end() ? terminal::TerminalStyle{} : it->second;
    }

    /// \brief Render a presentation document through this trace module's policy.
    /// \param text Presentation document to render.
    /// \return Rendered string.
    std::string render(uni20::presentation::styled_text const& text) const
    {
      return uni20::presentation::render(text, renderPolicy);
    }

  private:
    static ThreadIdOptions parse_thread_id_option_from_string(std::string_view value, ThreadIdOptions fallback)
    {
      using Mode = ThreadIdOptions::Enum;

      if (iequals(value, "auto")) return Mode::auto_detect;

      if (iequals(value, "yes") || iequals(value, "true") || iequals(value, "on") || iequals(value, "1"))
        return Mode::yes;

      if (iequals(value, "no") || iequals(value, "false") || iequals(value, "off") || iequals(value, "0"))
        return Mode::no;

      return fallback;
    }

    static ThreadIdOptions parse_thread_id_option_from_env(std::string const& var, ThreadIdOptions fallback)
    {
      if (auto const* raw = std::getenv(var.c_str()))
      {
        return parse_thread_id_option_from_string(raw, fallback);
      }
      return fallback;
    }

    /// \brief Update showColor based on the `color` setting and current outputStream.
    void updateShowColor()
    {
      using CO = ColorOptions::Enum;
      renderPolicy.output_stream = outputStream;
      if (color == CO::yes)
        renderPolicy.color = uni20::presentation::color_mode::always;
      else if (color == CO::no)
        renderPolicy.color = uni20::presentation::color_mode::never;
      else /* auto */
        renderPolicy.color = uni20::presentation::color_mode::automatic;

      showColor = uni20::presentation::should_emit_color(renderPolicy);
    }

    friend FormattingOptions& get_formatting_options(const std::string& module);
};

/// Returns the FormattingOptions for a module, or the “default” when called with no args.
/// Empty module name ⇒ use the no-arg ctor (global defaults only).
inline FormattingOptions& get_formatting_options(const std::string& module)
{
  static std::recursive_mutex mtx;
  static std::unordered_map<std::string, FormattingOptions> table;
  std::lock_guard lock(mtx);

  auto it = table.find(module);
  if (it != table.end()) return it->second;

  // else construct new

  if (module.empty())
  {
    // Use default-constructed instance with no recursion
    auto [it, _] = table.try_emplace("", FormattingOptions());
    return it->second;
  }
  else
  {
    // Copy from global instance
    auto [it, _] = table.try_emplace(module, module);
    return it->second;
  }
}

// Concept for a type that has an fmt::formatter specialization.
template <typename T>
concept HasFmtFormatter = fmt::has_formatter<std::remove_cvref_t<T>, fmt::format_context>::value;

// Concept for a type that is formattable via std::format (C++20 and later).
template <typename T, typename CharT = char>
concept HasStdFormatter = std::formattable<std::remove_cvref_t<T>, CharT>;

// Formatted output of containers, if they look like a range and have no fmt formatter.
template <typename T>
concept Container = std::ranges::forward_range<T> && (!HasFmtFormatter<T>);

// formatValue: Converts a value to a string using fmt::format.
// The generic version works for most types.

/// \brief Format a non-container, non-Uni20-scalar type with an fmt formatter.
/// \tparam T Type with an `fmt::formatter` specialization.
/// \param value Value to format.
/// \param opts Formatting options (unused by this overload).
/// \return The string produced by `fmt::format("{}", value)`.
template <typename T>
std::string formatValue(const T& value, FormattingOptions const& opts)
  requires(!Container<T> && HasFmtFormatter<T> && !uni20::RealOrComplex<T>)
{
  return fmt::format("{}", value);
}

template <typename T>
std::string formatValue(const T& value, FormattingOptions const& /*opts*/)
  requires(!Container<T> && !HasFmtFormatter<T> && HasStdFormatter<T> && !uni20::RealOrComplex<T>)
{
  return std::format("{}", value);
}

/// \brief Format any Uni20 real scalar using its configured trace precision.
/// \tparam T Uni20 real scalar type, including configured extension types.
/// \param value Real value to format.
/// \param opts Trace numeric formatting options.
/// \return Formatted real value.
template <uni20::Real T> inline std::string formatValue(T value, FormattingOptions const& opts)
{
  return uni20::presentation::format_real(value, opts.numeric_format_policy());
}

/// \brief Format any Uni20 complex scalar using its real type's trace precision.
/// \tparam T Uni20 complex scalar type.
/// \param value Complex value to format.
/// \param opts Trace numeric formatting options.
/// \return Formatted complex value.
template <uni20::Complex T>
  requires uni20::Real<uni20::make_real_t<T>>
inline std::string formatValue(T const& value, FormattingOptions const& opts)
{
  return uni20::presentation::format_complex(value, opts.numeric_format_policy());
}

// Overload for any coroutine_handle type (primary template)
template <typename Promise>
inline std::string formatValue(const std::coroutine_handle<Promise>& h, FormattingOptions const& /*opts*/)
{
  // Print the address as a pointer (may be null)
  return fmt::format("coroutine_handle<{}> @ {:p}", uni20::demangle::demangle(typeid(h.promise()).name()),
                     reinterpret_cast<const void*>(h.address()));
}

// Overload for std::coroutine_handle<>
inline std::string formatValue(const std::coroutine_handle<>& h, FormattingOptions const& /*opts*/)
{
  return fmt::format("coroutine_handle<> @ {:p}", reinterpret_cast<const void*>(h.address()));
}

/// \brief Format each element of a container by recursively calling `formatValue`.
/// \tparam ContainerType  Any container with `std::begin`/`std::end`.
/// \param c               The container whose elements to format.
/// \param opts            Formatting options forwarded to each element call.
/// \returns               A `std::vector<std::string>` of the formatted elements.
template <Container ContainerType>
auto formatValue(const ContainerType& c,
                 FormattingOptions const& opts) -> std::vector<decltype(formatValue(*std::begin(c), opts))>
{
  std::vector<decltype(formatValue(*std::begin(c), opts))> result;
  result.reserve(std::ranges::distance(c));
  for (auto const& elem : c)
  {
    result.push_back(formatValue(elem, opts));
  }
  return result;
}

/// \brief Format a C‐string (null‐terminated) as a normal string.
/// \param s    Pointer to a null‐terminated character array.
/// \param opts Formatting options (unused here).
/// \returns    The contents of the string, or "(null)" if `s==nullptr`.
inline std::string formatValue(std::string_view s, FormattingOptions const& /*opts*/)
{
  if (s.empty()) return std::string("(null)");
  return fmt::format("{}", s);
}

/// \brief Format a C‐string (null‐terminated) as a normal string.
/// \param s    Pointer to a null‐terminated character array.
/// \param opts Formatting options (unused here).
/// \returns    The contents of the string, or "(null)" if `s==nullptr`.
inline std::string formatValue(const char* s, FormattingOptions const& /*opts*/)
{
  if (!s) return std::string("(null)");
  return fmt::format("{}", s);
}

/// \brief Format a mutable C‐string.  Delegates to the `const char*` overload.
/// \param s    Pointer to a null‐terminated character array.
/// \param opts Formatting options.
/// \returns    The contents of the string, or "(null)" if `s==nullptr`.
inline std::string formatValue(char* s, FormattingOptions const& opts)
{
  return formatValue(static_cast<const char*>(s), opts);
}

/// \brief Format any non‐character pointer by showing its pointee type and address.
/// \tparam U  The pointee type.
/// \param ptr Pointer to format.
/// \param opts Formatting options (unused here).
/// \requires `U` is not `char` or `const char`.
/// \returns A string like `"MyType* @ 0x7fffdeadbeef"`.
template <typename U>
inline std::string formatValue(U* ptr, FormattingOptions const& /*opts*/)
  requires(!std::is_same_v<U, char> && !std::is_same_v<U, const char>)
{
  return fmt::format("{}* @ {:p}", uni20::demangle::demangle(typeid(U).name()), fmt::ptr(ptr));
}

template <typename T>
concept HasPresentationMdspanView = (!uni20::presentation::mdspan_like<T>) && requires(T const& value) {
  value.mdspan();
} && uni20::presentation::mdspan_like<decltype(std::declval<T const&>().mdspan())>;

enum class trace_format_context
{
  normal,
  fatal
};

namespace detail
{
inline bool trace_dump_enabled()
{
  if (auto const* value = std::getenv("UNI20_TRACE_DUMP"))
  {
    std::string_view mode(value);
    if (iequals(mode, "never") || iequals(mode, "no") || iequals(mode, "off") || iequals(mode, "false") ||
        iequals(mode, "0"))
    {
      return false;
    }
  }
  return true;
}

inline std::string sanitize_dump_label(std::string_view label)
{
  std::string out;
  out.reserve(std::min<std::size_t>(label.size(), 80));
  for (auto const ch : label)
  {
    auto const c = static_cast<unsigned char>(ch);
    if (std::isalnum(c) || ch == '_' || ch == '-' || ch == '.')
    {
      out.push_back(ch);
    }
    else if (!out.empty() && out.back() != '_')
    {
      out.push_back('_');
    }
    if (out.size() >= 80) break;
  }
  if (out.empty()) out = "value";
  return out;
}

inline std::filesystem::path trace_dump_directory()
{
  if (auto const* dir = std::getenv("UNI20_TRACE_DUMP_DIR"); dir != nullptr && std::strlen(dir) > 0)
  {
    return std::filesystem::path(dir);
  }
  return std::filesystem::temp_directory_path() / "uni20-trace";
}

inline std::filesystem::path next_trace_dump_path(std::string_view label)
{
  static std::atomic<std::uint64_t> counter = 0;
  auto dir = trace_dump_directory();
  std::filesystem::create_directories(dir);
  auto const now = std::chrono::system_clock::now().time_since_epoch();
  auto const stamp = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
  auto const id = counter.fetch_add(1, std::memory_order_relaxed);
  return dir / fmt::format("uni20_trace_{}_{}_{}.txt", stamp, id, sanitize_dump_label(label));
}

template <typename MDS, std::size_t Rank>
std::string dump_index_string(std::array<typename std::remove_cvref_t<MDS>::index_type, Rank> const& indices)
{
  if constexpr (Rank == 0)
  {
    return "[]";
  }
  else
  {
    std::string out = "[";
    for (std::size_t i = 0; i < Rank; ++i)
    {
      if (i > 0) out += ", ";
      out += std::to_string(static_cast<std::size_t>(indices[i]));
    }
    out += "]";
    return out;
  }
}

inline std::string dump_shape_string(std::vector<std::size_t> const& extents)
{
  std::string out = "shape=(";
  for (std::size_t i = 0; i < extents.size(); ++i)
  {
    if (i > 0) out += ", ";
    out += std::to_string(extents[i]);
  }
  out += ")";
  return out;
}

template <typename MDS, typename ElementFormatter, std::size_t Rank>
void write_mdspan_dump_recursive(std::ofstream& out, MDS const& mds,
                                 std::array<typename std::remove_cvref_t<MDS>::index_type, Rank>& indices,
                                 std::size_t axis, uni20::presentation::output_policy const& policy,
                                 ElementFormatter& formatter)
{
  if (axis == Rank)
  {
    out << dump_index_string<MDS>(indices) << '\t'
        << uni20::presentation::render_text(std::invoke(formatter, mds[indices]), policy) << '\n';
    return;
  }

  auto const extent = static_cast<std::size_t>(mds.extent(axis));
  for (std::size_t i = 0; i < extent; ++i)
  {
    indices[axis] = static_cast<typename std::remove_cvref_t<MDS>::index_type>(i);
    write_mdspan_dump_recursive(out, mds, indices, axis + 1, policy, formatter);
  }
}

template <uni20::presentation::mdspan_like MDS, typename ElementFormatter>
std::string write_mdspan_dump(MDS const& mds, FormattingOptions const& opts, std::string_view label,
                              std::vector<std::size_t> const& extents, std::size_t element_count,
                              ElementFormatter&& formatter)
{
  if (!trace_dump_enabled()) return "full data dump disabled by UNI20_TRACE_DUMP";

  try
  {
    auto path = next_trace_dump_path(label);
    std::ofstream out(path);
    if (!out) return "full data dump failed: could not open " + path.string();

    auto dump_policy = opts.presentation_policy();
    dump_policy.color = uni20::presentation::color_mode::never;
    dump_policy.charset = uni20::presentation::text_charset::utf8;
    dump_policy.wrap_width = std::nullopt;

    out << "# Uni20 trace mdspan dump\n";
    out << "# " << dump_shape_string(extents) << '\n';
    out << "# elements=" << element_count << '\n';
    out << "# index\tvalue\n";

    constexpr std::size_t rank = std::remove_cvref_t<MDS>::rank();
    using index_type = typename std::remove_cvref_t<MDS>::index_type;
    auto element_formatter = std::forward<ElementFormatter>(formatter);
    std::array<index_type, rank> indices{};
    if constexpr (rank == 0)
    {
      out << "[]\t" << uni20::presentation::render_text(std::invoke(element_formatter, mds[indices]), dump_policy)
          << '\n';
    }
    else
    {
      write_mdspan_dump_recursive(out, mds, indices, 0, dump_policy, element_formatter);
    }

    return "full data: " + path.string();
  }
  catch (std::exception const& ex)
  {
    return std::string("full data dump failed: ") + ex.what();
  }
}
} // namespace detail

template <uni20::presentation::mdspan_like MDS>
inline std::string formatMdspanForTrace(MDS const& mds, FormattingOptions const& opts, int value_width,
                                        trace_format_context context, std::string_view label)
{
  auto preview_options = opts.mdspan_preview_policy();
  preview_options.format = opts.mdspan_format_policy();
  preview_options.format.numeric = opts.numeric_format_policy();

  auto policy = opts.presentation_policy();
  if (value_width > 0)
  {
    policy.wrap_width = static_cast<std::size_t>(value_width);
  }

  auto formatter = [&opts](auto const& value) { return formatValue(value, opts); };
  auto preview = uni20::presentation::format_mdspan_preview(mds, policy, formatter, preview_options);
  if (context == trace_format_context::fatal && preview.elided)
  {
    preview.text += "\n";
    preview.text += detail::write_mdspan_dump(mds, opts, label, preview.extents, preview.element_count, formatter);
  }
  return preview.text;
}

/// \brief Format an mdspan-like value as presentation-layer tensor art.
/// \tparam MDS Mdspan-like object type.
/// \param mds Mdspan-like object to render.
/// \param opts Trace formatting options, including presentation policy and scalar precision.
/// \return Display-cell-aligned matrix or higher-order tensor art.
template <uni20::presentation::mdspan_like MDS>
inline std::string formatValue(MDS const& mds, FormattingOptions const& opts)
{
  return formatMdspanForTrace(mds, opts, opts.terminal_width, trace_format_context::normal, {});
}

/// \brief Format tensor/view-like values through their mdspan view.
/// \tparam T Object type exposing `mdspan()`.
/// \param value Tensor or tensor view object to render.
/// \param opts Trace formatting options.
/// \return Display-cell-aligned matrix or higher-order tensor art.
template <HasPresentationMdspanView T> inline std::string formatValue(T const& value, FormattingOptions const& opts)
{
  return formatValue(value.mdspan(), opts);
}

template <uni20::presentation::mdspan_like MDS>
inline std::string formatValueForContext(MDS const& mds, FormattingOptions const& opts, int value_width,
                                         trace_format_context context, std::string_view label)
{
  return formatMdspanForTrace(mds, opts, value_width, context, label);
}

template <HasPresentationMdspanView T>
inline std::string formatValueForContext(T const& value, FormattingOptions const& opts, int value_width,
                                         trace_format_context context, std::string_view label)
{
  return formatMdspanForTrace(value.mdspan(), opts, value_width, context, label);
}

template <typename T>
auto formatValueForContext(T const& value, FormattingOptions const& opts, int /*value_width*/,
                           trace_format_context /*context*/, std::string_view /*label*/)
{
  return formatValue(value, opts);
}

// Helper function to trim leading and trailing whitespace.
inline std::string trim(const std::string& s)
{
  // Find the first non-space character.
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
    ++start;
  if (start == s.size())
  {
    return "";
  }
  // Find the last non-space character.
  size_t end = s.size() - 1;
  while (end > start && std::isspace(static_cast<unsigned char>(s[end])))
    --end;
  return s.substr(start, end - start + 1);
}

namespace detail
{
struct TraceNameValue
{
    std::string name;
    std::string value;
};
} // namespace detail

/// \brief special case for detail::TraceNameValue to return the pre-formatted name,value pair
inline detail::TraceNameValue formatValue(detail::TraceNameValue const& nv, FormattingOptions const& opts)
{
  return nv;
}

// parseNames: Splits the stringified parameter list into tokens.
// Each token is paired with a boolean flag that is true if a top-level
// string or character literal was encountered.
inline std::vector<std::pair<std::string, bool>> parseNames(std::string_view s)
{
  std::vector<std::pair<std::string, bool>> tokens;
  std::string current;
  bool tokenHasTopLevelLiteral = false;

  // Counters for grouping symbols.
  int parenCount = 0, squareCount = 0, curlyCount = 0;

  // Flags for being inside a literal.
  bool inDoubleQuote = false, inSingleQuote = false;

  for (size_t i = 0; i < s.size(); ++i)
  {
    char c = s[i];

    // Inside a double-quoted string literal.
    if (inDoubleQuote)
    {
      current.push_back(c);
      if (c == '\\')
      {
        // Copy the escaped character.
        if (i + 1 < s.size())
        {
          current.push_back(s[i + 1]);
          ++i;
        }
      }
      else if (c == '"')
      {
        inDoubleQuote = false;
      }
      continue;
    }

    // Inside a single-quoted character literal.
    if (inSingleQuote)
    {
      current.push_back(c);
      if (c == '\\')
      {
        if (i + 1 < s.size())
        {
          current.push_back(s[i + 1]);
          ++i;
        }
      }
      else if (c == '\'')
      {
        inSingleQuote = false;
      }
      continue;
    }

    // At top-level, if we see a literal-start, mark the token.
    if ((c == '"' || c == '\'') && parenCount == 0 && squareCount == 0 && curlyCount == 0)
    {
      tokenHasTopLevelLiteral = true;
      if (c == '"')
        inDoubleQuote = true;
      else
        inSingleQuote = true;
      current.push_back(c);
      continue;
    }

    // Split at commas if we are not inside any grouping.
    if (c == ',' && parenCount == 0 && squareCount == 0 && curlyCount == 0)
    {
      tokens.push_back({trim(current), tokenHasTopLevelLiteral});
      current.clear();
      tokenHasTopLevelLiteral = false;
      continue; // Skip adding the comma.
    }

    // Update grouping counters.
    if (c == '(')
      ++parenCount;
    else if (c == ')')
    {
      if (parenCount > 0) --parenCount;
    }
    else if (c == '[')
      ++squareCount;
    else if (c == ']')
    {
      if (squareCount > 0) --squareCount;
    }
    else if (c == '{')
      ++curlyCount;
    else if (c == '}')
    {
      if (curlyCount > 0) --curlyCount;
    }

    current.push_back(c);
  }

  // Push any remaining text as the final token.
  if (!current.empty()) tokens.push_back({trim(current), tokenHasTopLevelLiteral});

  return tokens;
}

// returns true if str spans more than one line
inline bool isMultiline(const std::string& str) { return std::find(str.begin(), str.end(), '\n') != str.end(); }

template <typename T> bool isMultiline(const std::vector<T>& str)
{
  for (const auto& s : str)
  {
    if (isMultiline(s)) return true;
  }
  return false;
}

inline std::size_t displayWidth(std::string_view str, uni20::presentation::output_policy const& policy)
{
  return uni20::presentation::display_width(str, policy);
}

inline std::size_t displayWidth(std::string_view str, FormattingOptions const& opts)
{
  return displayWidth(str, opts.presentation_policy());
}

// get the maximum display width of a line for a multi-line string
inline std::size_t getMaxLineWidth(const std::string& str, uni20::presentation::output_policy const& policy)
{
  std::size_t maxWidth = 0;
  std::size_t lineStart = 0;
  while (lineStart <= str.size())
  {
    auto const lineEnd = str.find('\n', lineStart);
    auto const line = lineEnd == std::string::npos ? std::string_view(str).substr(lineStart)
                                                   : std::string_view(str).substr(lineStart, lineEnd - lineStart);
    maxWidth = std::max(maxWidth, displayWidth(line, policy));
    if (lineEnd == std::string::npos) break;
    lineStart = lineEnd + 1;
  }
  return maxWidth;
}

inline int getMaxLineWidth(const std::string& str)
{
  return static_cast<int>(getMaxLineWidth(str, uni20::presentation::plain_policy()));
}

// recursively get the maximum width of a string or nested vectors of strings.
inline size_t maxWidth(const std::string& str, uni20::presentation::output_policy const& policy)
{
  return getMaxLineWidth(str, policy);
}

inline size_t maxWidth(const std::string& str) { return maxWidth(str, uni20::presentation::plain_policy()); }

template <typename T> size_t maxWidth(const std::vector<T>& elems, uni20::presentation::output_policy const& policy)
{
  size_t max_width = 0;
  for (const auto& e : elems)
    max_width = std::max(max_width, maxWidth(e, policy));
  return max_width;
}

template <typename T> size_t maxWidth(const std::vector<T>& elems)
{
  return maxWidth(elems, uni20::presentation::plain_policy());
}

inline bool canFormatAsPresentationMatrix(const std::vector<std::vector<std::string>>& elems)
{
  if (elems.empty()) return false;

  auto const cols = elems.front().size();
  if (cols == 0) return false;

  for (auto const& row : elems)
  {
    if (row.size() != cols) return false;
    if (isMultiline(row)) return false;
  }
  return true;
}

inline std::string formatRectangularContainerAsPresentationMatrix(const std::vector<std::vector<std::string>>& elems,
                                                                  FormattingOptions const& opts)
{
  std::vector<std::string> flat;
  flat.reserve(elems.size() * elems.front().size());
  for (auto const& row : elems)
  {
    flat.insert(flat.end(), row.begin(), row.end());
  }

  using extents_type = stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>;
  auto matrix = stdex::mdspan<std::string const, extents_type, stdex::layout_right>(flat.data(), elems.size(),
                                                                                    elems.front().size());
  auto options = opts.mdspan_format_policy();
  options.numeric = opts.numeric_format_policy();
  return uni20::presentation::format_mdspan(
      matrix, opts.presentation_policy(), [](std::string const& value) { return value; }, options);
}

inline std::string formatContainerToStringImpl(const std::string& elem,
                                               uni20::presentation::output_policy const& policy, size_t max_width = 0)
{
  return uni20::presentation::pad_left(elem, max_width, policy);
}

inline std::string formatContainerToString(const std::string& elem, size_t max_width = 0)
{
  return formatContainerToStringImpl(elem, uni20::presentation::plain_policy(), max_width);
}

inline std::string formatContainerToString(const std::string& elem, FormattingOptions const& opts, size_t max_width = 0)
{
  return formatContainerToStringImpl(elem, opts.presentation_policy(), max_width);
}

inline std::string formatContainerToStringImpl(const std::vector<std::string>& elems,
                                               uni20::presentation::output_policy const& policy, size_t max_width = 0)
{
  std::string inlineStr;

  // The multiline case; put line breaks between each element
  if (isMultiline(elems))
  {
    inlineStr = "[\n";
    for (int i = 0; i < std::ssize(elems); ++i)
    {
      if (i > 0) inlineStr += ",\n  ";
      inlineStr += uni20::presentation::indent_text(elems[i], 2, policy, false);
    }
    inlineStr += "\n]";
  }
  else
  {
    // Get the maximum width of the components
    max_width = std::max(max_width, maxWidth(elems, policy));
    inlineStr = "[ ";
    for (int i = 0; i < std::ssize(elems); ++i)
    {
      if (i > 0) inlineStr += ", ";
      inlineStr += formatContainerToStringImpl(elems[i], policy, max_width);
    }
    inlineStr += " ]";
  }
  return inlineStr;
}

inline std::string formatContainerToString(const std::vector<std::string>& elems, size_t max_width = 0)
{
  return formatContainerToStringImpl(elems, uni20::presentation::plain_policy(), max_width);
}

inline std::string formatContainerToString(const std::vector<std::string>& elems, FormattingOptions const& opts,
                                           size_t max_width = 0)
{
  return formatContainerToStringImpl(elems, opts.presentation_policy(), max_width);
}

// This probably works fine for arbitrary nesting
inline std::string formatContainerToStringImpl(const std::vector<std::vector<std::string>>& elems,
                                               uni20::presentation::output_policy const& policy, size_t max_width = 0)
{
  std::string inlineStr = "[ ";
  if (isMultiline(elems))
  {
    for (int i = 0; i < std::ssize(elems); ++i)
    {
      if (i > 0) inlineStr += ",\n  ";
      inlineStr += uni20::presentation::indent_text(formatContainerToStringImpl(elems[i], policy), 2, policy, false);
    }
    inlineStr += "\n]";
  }
  else
  {
    // Get the maximum width of the components
    max_width = std::max(max_width, maxWidth(elems, policy));
    for (size_t i = 0; i < elems.size(); ++i)
    {
      if (i > 0) inlineStr += ",\n  ";
      inlineStr += formatContainerToStringImpl(elems[i], policy, max_width);
    }
    inlineStr += " ]";
  }
  return inlineStr;
}

inline std::string formatContainerToString(const std::vector<std::vector<std::string>>& elems, size_t max_width = 0)
{
  return formatContainerToStringImpl(elems, uni20::presentation::plain_policy(), max_width);
}

inline std::string formatContainerToString(const std::vector<std::vector<std::string>>& elems,
                                           FormattingOptions const& opts, size_t max_width = 0)
{
  if (canFormatAsPresentationMatrix(elems))
  {
    return formatRectangularContainerAsPresentationMatrix(elems, opts);
  }
  return formatContainerToStringImpl(elems, opts.presentation_policy(), max_width);
}

// For non-container (plain string) output.
inline uni20::presentation::styled_text formatItemText(const std::pair<std::string, bool>& name,
                                                       const std::string& value, const FormattingOptions& opts,
                                                       int available_width)
{
  uni20::presentation::styled_text text;

  // If the name is a string literal, then just return the value separately
  if (name.second)
  {
    opts.append_style(text, value, "TRACE_STRING");
    return text;
  }

  // If we'd spill over onto another line, then insert a line break
  auto const item_width = displayWidth(name.first, opts) + 3 + getMaxLineWidth(value, opts.presentation_policy());
  if (isMultiline(value) || (available_width > 0 && item_width > static_cast<std::size_t>(available_width)))
  {
    auto const indent = displayWidth(name.first, opts) + 3;
    text.append("\n");
    opts.append_style(text, name.first, "TRACE_EXPR");
    text.append(" = ");
    opts.append_style(text, uni20::presentation::indent_text(value, indent, opts.presentation_policy(), false),
                      "TRACE_VALUE");
    return text;
  }
  opts.append_style(text, name.first, "TRACE_EXPR");
  text.append(" = ");
  opts.append_style(text, value, "TRACE_VALUE");
  return text;
}

inline std::string formatItemString(const std::pair<std::string, bool>& name, const std::string& value,
                                    const FormattingOptions& opts, int available_width)
{
  return opts.render(formatItemText(name, value, opts, available_width));
}

inline uni20::presentation::styled_text formatItemText(const std::pair<std::string, bool>& name,
                                                       detail::TraceNameValue const& value,
                                                       const FormattingOptions& opts, int available_width)
{
  uni20::presentation::styled_text text;

  // If we'd spill over onto another line, then insert a line break
  auto const item_width = displayWidth(value.name, opts) + 3 + getMaxLineWidth(value.value, opts.presentation_policy());
  if (isMultiline(value.value) || (available_width > 0 && item_width > static_cast<std::size_t>(available_width)))
  {
    auto const indent = displayWidth(value.name, opts) + 3;
    text.append("\n");
    opts.append_style(text, value.name, "TRACE_EXPR");
    text.append(" = ");
    opts.append_style(text, uni20::presentation::indent_text(value.value, indent, opts.presentation_policy(), false),
                      "TRACE_VALUE");
    return text;
  }
  opts.append_style(text, value.name, "TRACE_EXPR");
  text.append(" = ");
  opts.append_style(text, value.value, "TRACE_VALUE");
  return text;
}

inline std::string formatItemString(const std::pair<std::string, bool>& name, detail::TraceNameValue const& value,
                                    const FormattingOptions& opts, int available_width)
{
  return opts.render(formatItemText(name, value, opts, available_width));
}

// Containers (can be nested)
template <typename T>
uni20::presentation::styled_text formatItemText(const std::pair<std::string, bool>& name, const std::vector<T>& values,
                                                const FormattingOptions& opts, int available_width)
{
  std::string formatted = formatContainerToString(values, opts);
  return formatItemText(name, formatted, opts, available_width);
}

template <typename T>
std::string formatItemString(const std::pair<std::string, bool>& name, const std::vector<T>& values,
                             const FormattingOptions& opts, int available_width)
{
  return opts.render(formatItemText(name, values, opts, available_width));
}

// formatTrace: Recursively formats all trace items into one string.

inline uni20::presentation::styled_text
formatParametersText(std::vector<std::pair<std::string, bool>>::const_iterator b, const FormattingOptions& opts)
{
  return {};
}

inline uni20::presentation::styled_text
formatParametersTextForContext(std::vector<std::pair<std::string, bool>>::const_iterator b,
                               const FormattingOptions& opts, trace_format_context context)
{
  return {};
}

inline std::string formatParameters(std::vector<std::pair<std::string, bool>>::const_iterator b,
                                    const FormattingOptions& opts)
{
  return opts.render(formatParametersText(b, opts));
}

template <typename T, typename... Ts>
uni20::presentation::styled_text
formatParametersTextForContext(std::vector<std::pair<std::string, bool>>::const_iterator b,
                               const FormattingOptions& opts, trace_format_context context, const T& first,
                               const Ts&... rest)
{
  auto value_width = opts.terminal_width;
  if (value_width > 0 && !b->second)
  {
    auto const indent = displayWidth(b->first, opts) + 3;
    if (indent < static_cast<std::size_t>(value_width))
    {
      value_width -= static_cast<int>(indent);
    }
  }

  auto result =
      formatItemText(*b, formatValueForContext(first, opts, value_width, context, b->first), opts, opts.terminal_width);

  ++b;
  if constexpr (sizeof...(rest) > 0)
  {
    result.append(", ");
    result.append(formatParametersTextForContext(b, opts, context, rest...));
  }
  return result;
}

template <typename T, typename... Ts>
uni20::presentation::styled_text formatParametersText(std::vector<std::pair<std::string, bool>>::const_iterator b,
                                                      const FormattingOptions& opts, const T& first, const Ts&... rest)
{
  return formatParametersTextForContext(b, opts, trace_format_context::normal, first, rest...);
}

template <typename T, typename... Ts>
std::string formatParameters(std::vector<std::pair<std::string, bool>>::const_iterator b, const FormattingOptions& opts,
                             const T& first, const Ts&... rest)
{
  return opts.render(formatParametersText(b, opts, first, rest...));
}

template <typename... Args>
uni20::presentation::styled_text formatParameterListText(const char* exprList, const FormattingOptions& opts,
                                                         const Args&... args)
{
  auto names = parseNames(exprList);
  return formatParametersText(names.begin(), opts, args...);
}

template <typename... Args>
uni20::presentation::styled_text formatParameterListTextForContext(const char* exprList, const FormattingOptions& opts,
                                                                   trace_format_context context, const Args&... args)
{
  auto names = parseNames(exprList);
  return formatParametersTextForContext(names.begin(), opts, context, args...);
}

template <typename... Args>
std::string formatParameterList(const char* exprList, const FormattingOptions& opts, const Args&... args)
{
  return opts.render(formatParameterListText(exprList, opts, args...));
}

/// \brief Print formatted string via the default (empty-module) sink.
template <typename... Args> void print(fmt::format_string<Args...> fmt_str, Args&&... args)
{
  auto& opts = get_formatting_options(); // empty-module defaults
  uni20::presentation::styled_text text;
  text.append(fmt::format(fmt_str, std::forward<Args>(args)...));
  opts.sink(opts.render(text));
}

inline std::string format_timestamp()
{
  auto const now = std::chrono::system_clock::now();
  auto const whole_seconds = std::chrono::floor<std::chrono::seconds>(now);
  auto const nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now - whole_seconds);
  auto const local_time = fmt::localtime(std::chrono::system_clock::to_time_t(whole_seconds));
  return fmt::format("{:%F %T}.{:09}", local_time, nanos.count());
}

namespace detail
{
inline std::thread::id const startup_thread_id = std::this_thread::get_id();
} // namespace detail

inline bool should_show_thread_id(FormattingOptions const& opts)
{
  using Mode = FormattingOptions::ThreadIdOptions::Enum;

  if (opts.threadId == Mode::yes) return true;
  if (opts.threadId == Mode::no) return false;
  return std::this_thread::get_id() != detail::startup_thread_id;
}

inline uni20::presentation::styled_text format_trace_prefix_text(FormattingOptions const& opts)
{
  uni20::presentation::styled_text prefix;
  bool has_prefix = false;

  if (opts.timestamp)
  {
    opts.append_style(prefix, format_timestamp(), "TIMESTAMP");
    has_prefix = true;
  }

  if (should_show_thread_id(opts))
  {
    if (has_prefix) prefix.append(" ");
    auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::string th = fmt::format("[TID {:>8x}]", id);
    opts.append_style(prefix, th, "THREAD_ID");
    has_prefix = true;
  }

  if (has_prefix) prefix.append(" ");
  return prefix;
}

inline std::string format_trace_prefix(FormattingOptions const& opts)
{
  return opts.render(format_trace_prefix_text(opts));
}

inline void append_trace_location(uni20::presentation::styled_text& text, FormattingOptions const& opts,
                                  std::string_view label, std::string_view style_kind, const char* file, int line)
{
  opts.append_style(text, label, std::string(style_kind));
  text.append(" at ");
  opts.append_style(text, file, "TRACE_FILENAME");
  opts.append_style(text, fmt::format(":{}", line), "TRACE_LINE");
}

inline void append_trace_module_location(uni20::presentation::styled_text& text, FormattingOptions const& opts,
                                         std::string_view label, std::string_view style_kind, const char* module,
                                         const char* file, int line)
{
  opts.append_style(text, label, std::string(style_kind));
  text.append(" in module ");
  opts.append_style(text, module, std::string(style_kind));
  text.append(" at ");
  opts.append_style(text, file, "TRACE_FILENAME");
  opts.append_style(text, fmt::format(":{}", line), "TRACE_LINE");
}

inline void append_parameter_separator(uni20::presentation::styled_text& text, FormattingOptions const& opts,
                                       std::string_view style_kind, bool start_newline)
{
  if (start_newline)
  {
    text.append("\n ");
  }
  else
  {
    text.append(" ");
  }
  opts.append_glyph(text, uni20::presentation::semantic_glyph::arrow_right, std::string(style_kind));
  text.append(" ");
}

inline void append_optional_parameters(uni20::presentation::styled_text& text, FormattingOptions const& opts,
                                       uni20::presentation::styled_text const& parameters, std::string_view style_kind,
                                       bool start_newline = false)
{
  if (parameters.empty()) return;
  append_parameter_separator(text, opts, style_kind, start_newline);
  text.append(parameters);
}

//-----------------------------------------------------------------------------
// Non-module TRACE
//-----------------------------------------------------------------------------
template <typename... Args> void TraceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListText(exprList, opts, args...);
  auto text = format_trace_prefix_text(opts);
  append_trace_location(text, opts, "TRACE", "TRACE", file, line);
  append_optional_parameters(text, opts, parameters, "TRACE");
  text.append("\n");
  opts.sink(opts.render(text));
}

//-----------------------------------------------------------------------------
// TRACE_ONCE
//-----------------------------------------------------------------------------
template <typename... Args> void TraceOnceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListText(exprList, opts, args...);
  auto text = format_trace_prefix_text(opts);
  append_trace_location(text, opts, "TRACE_ONCE", "TRACE", file, line);
  append_optional_parameters(text, opts, parameters, "TRACE");
  text.append("\n");
  opts.sink(opts.render(text));
}

//-----------------------------------------------------------------------------
// Module-aware TRACE
//-----------------------------------------------------------------------------
template <typename... Args>
void TraceModuleCall(const char* module, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options(module);

  auto parameters = formatParameterListText(exprList, opts, args...);
  auto text = format_trace_prefix_text(opts);
  append_trace_module_location(text, opts, "TRACE", "TRACE", module, file, line);
  append_optional_parameters(text, opts, parameters, "TRACE");
  text.append("\n");
  opts.sink(opts.render(text));
}

//-----------------------------------------------------------------------------
// Non-module DEBUG_TRACE
//-----------------------------------------------------------------------------
template <typename... Args> void DebugTraceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListText(exprList, opts, args...);
  auto text = format_trace_prefix_text(opts);
  append_trace_location(text, opts, "DEBUG_TRACE", "DEBUG_TRACE", file, line);
  append_optional_parameters(text, opts, parameters, "DEBUG_TRACE");
  text.append("\n");
  opts.sink(opts.render(text));
}

//-----------------------------------------------------------------------------
// DEBUG_TRACE_ONCE
//-----------------------------------------------------------------------------
template <typename... Args>
void DebugTraceOnceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListText(exprList, opts, args...);
  auto text = format_trace_prefix_text(opts);
  append_trace_location(text, opts, "DEBUG_TRACE_ONCE", "DEBUG_TRACE", file, line);
  append_optional_parameters(text, opts, parameters, "DEBUG_TRACE");
  text.append("\n");
  opts.sink(opts.render(text));
}

//-----------------------------------------------------------------------------
// Module-aware DEBUG_TRACE
//-----------------------------------------------------------------------------
template <typename... Args>
void DebugTraceModuleCall(const char* module, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options(module);

  auto parameters = formatParameterListText(exprList, opts, args...);
  auto text = format_trace_prefix_text(opts);
  append_trace_module_location(text, opts, "DEBUG_TRACE", "DEBUG_TRACE", module, file, line);
  append_optional_parameters(text, opts, parameters, "DEBUG_TRACE");
  text.append("\n");
  opts.sink(opts.render(text));
}

namespace detail
{
inline uni20::presentation::output_policy diagnostic_presentation_policy(FormattingOptions const& opts)
{
  auto policy = opts.presentation_policy();
  if (!policy.wrap_width.has_value() && opts.terminal_width > 0)
  {
    policy.wrap_width = static_cast<std::size_t>(opts.terminal_width);
  }
  return policy;
}

#if TRACE_HAS_STACKTRACE
inline uni20::presentation::stacktrace_format_options stacktrace_presentation_options(FormattingOptions const& opts)
{
  return uni20::presentation::stacktrace_format_options{
      .indent = 2,
      .tree_style = opts.presentation_style("TRACE_LINE"),
      .index_style = opts.presentation_style("TRACE_LINE"),
      .description_style = opts.presentation_style("TRACE"),
      .source_file_style = opts.presentation_style("TRACE_FILENAME"),
      .source_line_style = opts.presentation_style("TRACE_LINE"),
  };
}
#endif

inline void emit_stacktrace(FormattingOptions& opts, std::string_view style_kind, std::size_t skip_frames)
{
#if TRACE_HAS_STACKTRACE
  uni20::presentation::styled_text heading;
  opts.append_style(heading, "Stacktrace:", std::string(style_kind));
  heading.append("\n");
  opts.sink(opts.render(heading));
  auto const policy = diagnostic_presentation_policy(opts);
  opts.sink(opts.render(uni20::presentation::format_stacktrace(std::stacktrace::current(skip_frames), policy,
                                                               stacktrace_presentation_options(opts))));
#else
  uni20::presentation::styled_text warning;
  opts.append_glyph(warning, uni20::presentation::semantic_glyph::warning, std::string(style_kind));
  warning.append(" ");
  opts.append_style(warning, "WARNING: std::stacktrace is unavailable in this build; stacktrace omitted.",
                    std::string(style_kind));
  warning.append("\n");
  opts.sink(opts.render(warning));
#endif
}

[[noreturn]] inline void abort_with_stacktrace(FormattingOptions& opts, std::string const& message,
                                               std::string_view style_kind, std::size_t skip_frames)
{
  opts.sink(message);
  emit_stacktrace(opts, style_kind, skip_frames);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

[[noreturn]] inline void abort_with_stacktrace(FormattingOptions& opts, uni20::presentation::styled_text const& message,
                                               std::string_view style_kind, std::size_t skip_frames)
{
  opts.sink(opts.render(message));
  emit_stacktrace(opts, style_kind, skip_frames);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

[[noreturn]] inline void abort_after_diagnostic(FormattingOptions& opts,
                                                uni20::presentation::styled_text const& message)
{
  opts.sink(opts.render(message));
  std::fflush(nullptr);
  std::abort();
}

template <typename... Args>
inline void emit_trace_line(FormattingOptions& opts, std::string_view label, std::string_view style_kind,
                            const char* exprList, const char* file, int line, const Args&... args)
{
  auto parameters = formatParameterListText(exprList, opts, args...);
  auto text = format_trace_prefix_text(opts);
  append_trace_location(text, opts, label, style_kind, file, line);
  append_optional_parameters(text, opts, parameters, style_kind);
  text.append("\n");
  opts.sink(opts.render(text));
}

template <typename... Args>
inline void emit_trace_line_module(FormattingOptions& opts, std::string_view label, std::string_view style_kind,
                                   const char* module, const char* exprList, const char* file, int line,
                                   const Args&... args)
{
  auto parameters = formatParameterListText(exprList, opts, args...);
  auto text = format_trace_prefix_text(opts);
  append_trace_module_location(text, opts, label, style_kind, module, file, line);
  append_optional_parameters(text, opts, parameters, style_kind);
  text.append("\n");
  opts.sink(opts.render(text));
}

inline uni20::presentation::styled_text make_diagnostic_header(FormattingOptions const& opts, std::string_view label,
                                                               std::string_view style_kind, const char* file, int line)
{
  uni20::presentation::styled_text text;
  auto const glyph = [&] {
    if (style_kind == "ERROR") return uni20::presentation::semantic_glyph::failure;
    if (style_kind == "CHECK" || style_kind == "DEBUG_CHECK" || style_kind == "PRECONDITION" ||
        style_kind == "DEBUG_PRECONDITION" || style_kind == "PANIC")
    {
      return uni20::presentation::semantic_glyph::fatal;
    }
    return uni20::presentation::semantic_glyph::warning;
  }();
  opts.append_glyph(text, glyph, std::string(style_kind));
  text.append(" ");
  append_trace_location(text, opts, label, style_kind, file, line);
  return text;
}

inline void append_condition_false(uni20::presentation::styled_text& text, FormattingOptions const& opts,
                                   const char* cond)
{
  text.append("\n");
  opts.append_style(text, cond, "TRACE_EXPR");
  text.append(" is ");
  opts.append_style(text, "false", "TRACE_VALUE");
  text.append("!");
}

inline void append_not_equal(uni20::presentation::styled_text& text, FormattingOptions const& opts, const char* a,
                             const char* b)
{
  text.append("\n");
  opts.append_style(text, a, "TRACE_EXPR");
  text.append(" is not equal to ");
  opts.append_style(text, b, "TRACE_EXPR");
  text.append("!");
}

inline void append_not_approx_equal(uni20::presentation::styled_text& text, FormattingOptions const& opts,
                                    const char* a, const char* b, std::int64_t ulps)
{
  text.append("\n");
  opts.append_style(text, a, "TRACE_EXPR");
  text.append(" is not approx-equal to ");
  opts.append_style(text, b, "TRACE_EXPR");
  text.append(" (to ");
  opts.append_style(text, fmt::format("{}", ulps), "TRACE_EXPR");
  text.append(" ULP)!");
}

inline void append_diagnostic_parameters(uni20::presentation::styled_text& text, FormattingOptions const& opts,
                                         uni20::presentation::styled_text const& parameters,
                                         std::string_view style_kind)
{
  append_optional_parameters(text, opts, parameters, style_kind, true);
  text.append("\n");
}
} // namespace detail

template <typename... Args> void TraceStackCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();
  detail::emit_trace_line(opts, "TRACE_STACK", "TRACE", exprList, file, line, args...);
  detail::emit_stacktrace(opts, "TRACE", 2);
}

template <typename... Args>
void TraceStackOnceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();
  detail::emit_trace_line(opts, "TRACE_ONCE_STACK", "TRACE", exprList, file, line, args...);
  detail::emit_stacktrace(opts, "TRACE", 2);
}

template <typename... Args>
void TraceModuleStackCall(const char* module, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options(module);
  detail::emit_trace_line_module(opts, "TRACE_MODULE_STACK", "TRACE", module, exprList, file, line, args...);
  detail::emit_stacktrace(opts, "TRACE", 2);
}

template <typename... Args>
void DebugTraceStackCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();
  detail::emit_trace_line(opts, "DEBUG_TRACE_STACK", "DEBUG_TRACE", exprList, file, line, args...);
  detail::emit_stacktrace(opts, "DEBUG_TRACE", 2);
}

template <typename... Args>
void DebugTraceStackOnceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();
  detail::emit_trace_line(opts, "DEBUG_TRACE_ONCE_STACK", "DEBUG_TRACE", exprList, file, line, args...);
  detail::emit_stacktrace(opts, "DEBUG_TRACE", 2);
}

template <typename... Args>
void DebugTraceModuleStackCall(const char* module, const char* exprList, const char* file, int line,
                               const Args&... args)
{
  auto& opts = get_formatting_options(module);
  detail::emit_trace_line_module(opts, "DEBUG_TRACE_MODULE_STACK", "DEBUG_TRACE", module, exprList, file, line,
                                 args...);
  detail::emit_stacktrace(opts, "DEBUG_TRACE", 2);
}

template <typename... Args>
[[noreturn]] void CheckCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "CHECK", "CHECK", file, line);
  detail::append_condition_false(text, opts, cond);
  detail::append_diagnostic_parameters(text, opts, parameters, "CHECK");
  detail::abort_with_stacktrace(opts, text, "CHECK", 2);
}

//------------------------------------------------------------------------------
// DEBUG_CHECK
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void DebugCheckCall(const char* cond, const char* exprList, const char* file, int line,
                                 const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "DEBUG_CHECK", "DEBUG_CHECK", file, line);
  detail::append_condition_false(text, opts, cond);
  detail::append_diagnostic_parameters(text, opts, parameters, "DEBUG_CHECK");
  detail::abort_with_stacktrace(opts, text, "DEBUG_CHECK", 2);
}

//------------------------------------------------------------------------------
// CHECK_EQUAL
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void CheckEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line,
                                 const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "CHECK_EQUAL", "CHECK", file, line);
  detail::append_not_equal(text, opts, a, b);
  detail::append_diagnostic_parameters(text, opts, parameters, "CHECK");
  detail::abort_with_stacktrace(opts, text, "CHECK", 2);
}

//------------------------------------------------------------------------------
// DEBUG_CHECK_EQUAL
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void DebugCheckEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line,
                                      const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "DEBUG_CHECK_EQUAL", "DEBUG_CHECK", file, line);
  detail::append_not_equal(text, opts, a, b);
  detail::append_diagnostic_parameters(text, opts, parameters, "DEBUG_CHECK");
  detail::abort_with_stacktrace(opts, text, "DEBUG_CHECK", 2);
}

//------------------------------------------------------------------------------
// CHECK_FLOATING_EQ
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void CheckFloatingEqCall(const char* a, const char* b, std::int64_t ulps, const char* exprList,
                                      const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "CHECK_FLOATING_EQ", "CHECK", file, line);
  detail::append_not_approx_equal(text, opts, a, b, ulps);
  detail::append_diagnostic_parameters(text, opts, parameters, "CHECK");
  detail::abort_with_stacktrace(opts, text, "CHECK", 2);
}

//------------------------------------------------------------------------------
// DEBUG_CHECK_FLOATING_EQ
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void DebugCheckFloatingEqCall(const char* a, const char* b, std::int64_t ulps, const char* exprList,
                                           const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "DEBUG_CHECK_FLOATING_EQ", "DEBUG_CHECK", file, line);
  detail::append_not_approx_equal(text, opts, a, b, ulps);
  detail::append_diagnostic_parameters(text, opts, parameters, "DEBUG_CHECK");
  detail::abort_with_stacktrace(opts, text, "DEBUG_CHECK", 2);
}

//------------------------------------------------------------------------------
// PRECONDITION
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void PreconditionCall(const char* cond, const char* exprList, const char* file, int line,
                                   const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "PRECONDITION", "PRECONDITION", file, line);
  detail::append_condition_false(text, opts, cond);
  detail::append_diagnostic_parameters(text, opts, parameters, "PRECONDITION");
  detail::abort_with_stacktrace(opts, text, "PRECONDITION", 2);
}

//------------------------------------------------------------------------------
// DEBUG_PRECONDITION
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void DebugPreconditionCall(const char* cond, const char* exprList, const char* file, int line,
                                        const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "DEBUG_PRECONDITION", "DEBUG_PRECONDITION", file, line);
  detail::append_condition_false(text, opts, cond);
  detail::append_diagnostic_parameters(text, opts, parameters, "DEBUG_PRECONDITION");
  detail::abort_with_stacktrace(opts, text, "DEBUG_PRECONDITION", 2);
}

//------------------------------------------------------------------------------
// PRECONDITION_EQUAL
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void PreconditionEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line,
                                        const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "PRECONDITION_EQUAL", "PRECONDITION", file, line);
  detail::append_not_equal(text, opts, a, b);
  detail::append_diagnostic_parameters(text, opts, parameters, "PRECONDITION");
  detail::abort_with_stacktrace(opts, text, "PRECONDITION", 2);
}

//------------------------------------------------------------------------------
// DEBUG_PRECONDITION_EQUAL
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void DebugPreconditionEqualCall(const char* a, const char* b, const char* exprList, const char* file,
                                             int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "DEBUG_PRECONDITION_EQUAL", "DEBUG_PRECONDITION", file, line);
  detail::append_not_equal(text, opts, a, b);
  detail::append_diagnostic_parameters(text, opts, parameters, "DEBUG_PRECONDITION");
  detail::abort_with_stacktrace(opts, text, "DEBUG_PRECONDITION", 2);
}

//------------------------------------------------------------------------------
// PRECONDITION_FLOATING_EQ
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void PreconditionFloatingEqCall(const char* a, const char* b, std::int64_t ulps, const char* exprList,
                                             const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "PRECONDITION_FLOATING_EQ", "PRECONDITION", file, line);
  detail::append_not_approx_equal(text, opts, a, b, ulps);
  detail::append_diagnostic_parameters(text, opts, parameters, "PRECONDITION");
  detail::abort_with_stacktrace(opts, text, "PRECONDITION", 2);
}

//------------------------------------------------------------------------------
// DEBUG_PRECONDITION_FLOATING_EQ
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void DebugPreconditionFloatingEqCall(const char* a, const char* b, std::int64_t ulps, const char* exprList,
                                                  const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "DEBUG_PRECONDITION_FLOATING_EQ", "DEBUG_PRECONDITION", file, line);
  detail::append_not_approx_equal(text, opts, a, b, ulps);
  detail::append_diagnostic_parameters(text, opts, parameters, "DEBUG_PRECONDITION");
  detail::abort_with_stacktrace(opts, text, "DEBUG_PRECONDITION", 2);
}

//------------------------------------------------------------------------------
// PANIC
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void PanicCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto parameters = formatParameterListTextForContext(exprList, opts, trace_format_context::fatal, args...);
  auto text = detail::make_diagnostic_header(opts, "PANIC", "PANIC", file, line);
  append_optional_parameters(text, opts, parameters, "PANIC");
  text.append("\n");
  detail::abort_with_stacktrace(opts, text, "PANIC", 2);
}

//------------------------------------------------------------------------------
// ERROR
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void ErrorCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto const context = opts.errors_abort() ? trace_format_context::fatal : trace_format_context::normal;
  auto parameters = formatParameterListTextForContext(exprList, opts, context, args...);
  auto text = detail::make_diagnostic_header(opts, "ERROR", "ERROR", file, line);
  append_optional_parameters(text, opts, parameters, "ERROR");
  text.append("\n");

  if (opts.errors_abort())
  {
    detail::abort_with_stacktrace(opts, text, "ERROR", 2);
  }
  throw std::runtime_error(opts.render(text));
}

//------------------------------------------------------------------------------
// ERROR_IF
//------------------------------------------------------------------------------
template <typename... Args>
[[noreturn]] void ErrorIfCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  auto const context = opts.errors_abort() ? trace_format_context::fatal : trace_format_context::normal;
  auto parameters = formatParameterListTextForContext(exprList, opts, context, args...);
  auto text = detail::make_diagnostic_header(opts, "ERROR", "ERROR", file, line);
  append_optional_parameters(text, opts, parameters, "ERROR");
  detail::append_condition_false(text, opts, cond);
  text.append("\n");

  if (opts.errors_abort())
  {
    detail::abort_with_stacktrace(opts, text, "ERROR", 2);
  }
  throw std::runtime_error(opts.render(text));
}

namespace detail
{
template <class Error>
concept HasDiagnosticReport = requires(Error const& error) {
  { diagnostic_report(error) } -> std::same_as<uni20::presentation::report_builder>;
};

inline void append_diagnostic_context(uni20::presentation::styled_text& text, FormattingOptions const& opts,
                                      uni20::diagnostic_error const& error,
                                      uni20::presentation::output_policy const& policy)
{
  if (error.source_location().has_value())
  {
    auto const& location = *error.source_location();
    opts.append_style(text, "Source location:", "ERROR");
    text.append("\n  ");
    opts.append_style(text, location.file_name(), "TRACE_FILENAME");
    opts.append_style(text, fmt::format(":{}", location.line()), "TRACE_LINE");
    text.append("\n");
  }

#if TRACE_HAS_STACKTRACE
  if (error.stacktrace().has_value())
  {
    opts.append_style(text, "Stacktrace:", "ERROR");
    text.append("\n");
    text.append(
        uni20::presentation::format_stacktrace(*error.stacktrace(), policy, stacktrace_presentation_options(opts)));
  }
#endif
}

template <class Error>
  requires std::derived_from<std::remove_cvref_t<Error>, uni20::diagnostic_error> && HasDiagnosticReport<Error>
uni20::presentation::styled_text format_diagnostic_text(FormattingOptions const& opts, Error const& error,
                                                        uni20::presentation::output_policy const& policy)
{
  auto text = uni20::presentation::render_report(diagnostic_report(error), policy);
  append_diagnostic_context(text, opts, error, policy);
  return text;
}

template <class Error>
uni20::presentation::styled_text raised_error_text(FormattingOptions const& opts, Error const& error,
                                                   std::source_location where)
{
  if constexpr (HasDiagnosticReport<Error>)
  {
    if constexpr (std::derived_from<std::remove_cvref_t<Error>, uni20::diagnostic_error>)
    {
      return format_diagnostic_text(opts, error, diagnostic_presentation_policy(opts));
    }
    else
    {
      return uni20::presentation::render_report(diagnostic_report(error), opts.presentation_policy());
    }
  }
  else
  {
    auto text = make_diagnostic_header(opts, "ERROR", "ERROR", where.file_name(), static_cast<int>(where.line()));
    text.append("\n");
    opts.append_style(text, error.what(), "TRACE_VALUE");
    text.append("\n");
    return text;
  }
}
} // namespace detail

/// \brief Format a structured diagnostic using its ADL presentation report and attached context.
/// \tparam Error Concrete diagnostic exception type with a `diagnostic_report` customization.
/// \param error Error whose report, source location, and stacktrace are formatted.
/// \param policy Presentation policy controlling report layout and stacktrace wrapping.
/// \return Styled diagnostic document ready for terminal or plain rendering.
template <class Error>
  requires std::derived_from<std::remove_cvref_t<Error>, uni20::diagnostic_error> && detail::HasDiagnosticReport<Error>
uni20::presentation::styled_text format_diagnostic(Error const& error, uni20::presentation::output_policy const& policy)
{
  return detail::format_diagnostic_text(get_formatting_options(), error, policy);
}

/// \brief Format a structured diagnostic using the active trace presentation policy.
/// \tparam Error Concrete diagnostic exception type with a `diagnostic_report` customization.
/// \param error Error whose report, source location, and stacktrace are formatted.
/// \return Styled diagnostic document wrapped to the configured terminal width.
template <class Error>
  requires std::derived_from<std::remove_cvref_t<Error>, uni20::diagnostic_error> && detail::HasDiagnosticReport<Error>
uni20::presentation::styled_text format_diagnostic(Error const& error)
{
  auto& opts = get_formatting_options();
  return detail::format_diagnostic_text(opts, error, detail::diagnostic_presentation_policy(opts));
}

/// \brief Raise a concrete exception through Uni20's configured error boundary.
/// \details Throw mode preserves the concrete exception type. Abort mode renders
///          its presentation report, when available, and emits a stacktrace.
/// \tparam Exception Type derived from `std::exception`.
/// \param exception Exception object to enrich and raise.
/// \param where Source location at which the error is raised.
template <class Exception>
  requires std::derived_from<std::remove_cvref_t<Exception>, std::exception>
[[noreturn]] void raise(Exception&& exception, std::source_location where = std::source_location::current())
{
  using error_type = std::remove_cvref_t<Exception>;
  error_type error(std::forward<Exception>(exception));

  if constexpr (std::derived_from<error_type, uni20::diagnostic_error>)
  {
    error.set_source_location(where);
#if TRACE_HAS_STACKTRACE
    error.set_stacktrace(std::stacktrace::current(1));
#endif
  }

  auto& opts = get_formatting_options();
  if (opts.errors_abort())
  {
    auto text = detail::raised_error_text(opts, error, where);
#if TRACE_HAS_STACKTRACE
    if constexpr (std::derived_from<error_type, uni20::diagnostic_error> && detail::HasDiagnosticReport<error_type>)
    {
      if (error.stacktrace().has_value())
      {
        detail::abort_after_diagnostic(opts, text);
      }
    }
#endif
    detail::abort_with_stacktrace(opts, text, "ERROR", 2);
  }
  throw error;
}

} // namespace trace
/// \brief Format a non-container, non-floating-point type using `std::format` when only a
///        standard formatter is available.
/// \tparam T             Any type that lacks an fmt::formatter but satisfies `std::formattable`.
/// \param value         The value to format.
/// \returns             The string produced by `std::format("{}", value)`.
