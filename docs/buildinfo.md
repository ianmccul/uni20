# Uni20 Build Information

Uni20 records CMake build metadata in a generated C++ header. The same generated data backs the C++ API, the Python binding, and the terminal example.

## C++ API

Include `<uni20/buildinfo.hpp>` and query the current build:

```cpp
#include <uni20/buildinfo.hpp>

auto const info = uni20::build_info::current();
```

`current()` returns a lightweight view with string fields for the generator, build type, system, and compiler, plus two spans:

- `info.build_options`: `UNI20_*` cache options and dependency settings.
- `info.detected_environment`: detected `UNI20_DETECTED_*` entries.

Each entry has:

```cpp
std::string_view key;
std::string_view value;
std::string_view help;
```

## Python API

The Python binding exposes the same data as a dictionary:

```bash
export PYTHONPATH="$(pwd)/build/bindings/python:${PYTHONPATH}"
python3 -c "import pprint, uni20; pprint.pp(uni20.buildinfo())"
```

The dictionary has scalar fields such as `generator`, `build_type`, and `cxx_compiler_id`, plus `build_options` and `detected_environment` sub-dictionaries. Each sub-dictionary entry has a `value` field and, when available, a `help` field.

Use `buildinfo_pretty()` when a formatted string is more convenient than the raw dictionary:

```bash
python3 -c "import uni20; print(uni20.buildinfo_pretty())"
```

## Pretty-Print Example

Build and run the C++ example:

```bash
cmake --build build --target buildinfo_example
./build/examples/buildinfo_example
```

The example uses the common presentation layer for color, glyphs, display-cell width handling, and wrapping. It does not define build-info-specific environment variables. Existing general controls apply:

- `UNI20_GLYPHS`: selects `emoji`, `unicode`, or `ascii` semantic glyphs.
- `UNI20_CHARSET`: selects `utf8`, `escape`, or `replace` text rendering.
- `UNI20_COLOR`: selects `auto`, forced color, or disabled color.
- `NO_COLOR`: disables automatic ANSI color output when set to a non-empty value.
- `COLUMNS`: overrides terminal width detection for wrapping when set to a positive integer.
