#pragma once

// A class to represent an enumeration that is iterable and has a string name
// associated with each item.
//
// To use, define a class or struct that has 4 members:
// Enum        - must be an enumeration type; enumeration constants must be contiguous and starting from 0.
// Default     - a constexpr value of type Enum, that is the value of a default-constructed NamedEnumeration
// StaticName  - must be a static constexpr of type const char* that gives a desciption of the enumeration.
// Names       - must be a static constexpr std::array of strings, of exactly the same size as Enum.
//

// An example:
// struct MyEnumTraits
// {
//    enum Enum { Some, Enumeration, Elements };
//    inline static constexpr Enum Default = Enumeration;
//    inline static constexpr const char* StaticName = "the example enumeration";
//    inline static constexpr std::array<const char*, 3> Names = { "some", "enumeration", "elements" };
// };
//
// When constructing a NamedEnumeration from a string, the name is not case sensitive.
//

#include "string_util.hpp"

#include <array>
#include <exception>
#include <fmt/format.h>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// An enumeration that supports iteration, as well as
template <typename Traits> class NamedEnumeration : public Traits {
  public:
    using Enum = typename Traits::Enum;
    using Traits::StaticName;
    inline static constexpr std::size_t N = Traits::Names.size();
    inline static constexpr Enum DEFAULT = Traits::Default;
    inline static constexpr Enum BEGIN = static_cast<Enum>(0);
    inline static constexpr Enum END = static_cast<Enum>(N);

    NamedEnumeration() : e(DEFAULT) {}

    NamedEnumeration(Enum a) : e(a) {}

    explicit NamedEnumeration(std::string_view Name);

    // Enable iteration (including range-based for loop) over the available algorithms
    NamedEnumeration begin() const { return BEGIN; }
    NamedEnumeration end() const { return END; }

    static constexpr std::size_t size() { return N; }

    bool operator==(const NamedEnumeration& Other) const { return e == Other.e; }
    bool operator!=(const NamedEnumeration& Other) const { return e != Other.e; }
    bool operator==(Enum a) const { return e == a; }
    bool operator!=(Enum a) const { return e != a; }
    NamedEnumeration& operator++()
    {
      e = static_cast<Enum>(e + 1);
      return *this;
    }
    NamedEnumeration& operator--()
    {
      e = static_cast<Enum>(e - 1);
      return *this;
    }
    const NamedEnumeration& operator*() const { return *this; }

    // returns a comma-separated list of the enumeration names
    static std::string ListAll();

    // returns an array of the enumeration items
    static std::vector<std::string> EnumerateAll();

    std::string Name() const { return Traits::Names[e]; }

  private:
    Enum e;
};

template <typename Traits> std::string NamedEnumeration<Traits>::ListAll()
{
  std::string Result;
  bool first = true;
  for (auto a : NamedEnumeration())
  {
    if (!first) Result += ", ";
    Result += a.Name();
    first = false;
  }
  return Result;
}

template <typename Traits> std::vector<std::string> NamedEnumeration<Traits>::EnumerateAll()
{
  std::vector<std::string> Result;
  Result.reserve(NamedEnumeration::size());
  for (auto a : NamedEnumeration())
  {
    Result.push_back(a.Name());
  }
  return Result;
}

template <typename Traits> NamedEnumeration<Traits>::NamedEnumeration(std::string_view Name)
{
  for (auto a : NamedEnumeration())
  {
    if (iequals(a.Name(), Name))
    {
      e = a.e;
      return;
    }
  }
  using namespace std::literals::string_literals;
  std::string ErrorStr = "Unknown initializer for "s + StaticName + "; choices are " + this->ListAll() + '.';
  throw std::runtime_error(ErrorStr);
}

template <typename Traits, typename CharT> struct std::formatter<NamedEnumeration<Traits>, CharT>
{
    constexpr auto parse(std::basic_format_parse_context<CharT>& ctx) { return ctx.begin(); }

    template <typename FormatContext> auto format(const NamedEnumeration<Traits>& e, FormatContext& ctx) const
    {
      return std::format_to(ctx.out(), "{}", e.Name());
    }
};
