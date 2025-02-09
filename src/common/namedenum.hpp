#pragma once

// A class to represent an enumeration that is iterable and has a string name
// associated with each item.
//
// To use, define a class or struct that has 4 members:
// Enum        - must be an enumeration type
// Default     - a constexpr value of type Enum, that is the value of a default-constructed NamedEnumeration
// StaticName  - must be a static constexpr of type char const* that gives a desciption of the enumeration.
// Names       - must be a static constexpr array of strings, of exactly the same size as Enum.
//
// Note that although Names is static constexpr, it must be instantiated in a .cpp file.
//

// An example:
// struct MyEnumTraits
// {
//    enum Enum { Some, Enumeration, Elements };
//    static constexpr Enum Default = Enumeration;
//    static constexpr const char* StaticName = "the example enumeration";
//    static constexpr std::array<const char*, 3> Names = { "some", "enumeration", "elements" };
// };
//
// When constructing a NamedEnumeration from a string, the name is not case sensitive.
//

#include "string_util.hpp"

#include <array>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// An enumeration that supports iteration, as well as
template <typename Traits> class NamedEnumeration : public Traits
{
  public:
    using Enum = typename Traits::Enum;
    using Traits::StaticName;
    static constexpr std::size_t N = Traits::Names.size();
    static constexpr Enum DEFAULT = Traits::Default;
    static constexpr Enum BEGIN = static_cast<Enum>(0);
    static constexpr Enum END = static_cast<Enum>(N);

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
