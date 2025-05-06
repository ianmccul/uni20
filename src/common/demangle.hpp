
#pragma once

#include <cstdlib>
#include <string>
#include <typeinfo>

#if defined(__GNUG__)
#include <cxxabi.h>
#elif defined(_MSC_VER)
#include <DbgHelp.h>
#include <windows.h>
// link against Dbghelp.lib
#pragma comment(lib, "Dbghelp.lib")
#endif

namespace uni20::demangle
{

/// \brief Turn a compiler‐mangled name into a human‐readable one.
/// \param name  The result of typeid(...).name().
/// \returns     A demangled string if supported; otherwise returns `name`.
inline std::string demangle(const char* name)
{
#if defined(__GNUG__)
  int status = 0;
  char* dem = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string out = (status == 0 && dem) ? dem : name;
  std::free(dem);
  return out;
#elif defined(_MSC_VER)
  char buffer[1024] = {};
  // UNDNAME_COMPLETE gives full undecoration
  if (UnDecorateSymbolName(name, buffer, sizeof(buffer), UNDNAME_COMPLETE))
    return std::string(buffer);
  else
    return name;
#else
  return name;
#endif
}

} // namespace uni20::demangle
