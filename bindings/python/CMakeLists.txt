cmake_minimum_required(VERSION 3.18)
project(uni20_python LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG stable
)
FetchContent_MakeAvailable(pybind11)

# Option 2 (Alternative): If you already have pybind11 installed system-wide,
# you could comment out the FetchContent block above and use:
# find_package(pybind11 CONFIG REQUIRED)

# Create the Python module using pybind11
# This command creates a Python extension module (shared library) from your source file.
pybind11_add_module(uni20_python MODULE
  uni20_python.cpp
)

# Link against the uni20 library (assumed to be defined in src/CMakeLists.txt)
target_link_libraries(uni20_python PRIVATE uni20)

# Optionally, specify the C++ standard if needed.
set_target_properties(uni20_python PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

# You can also add additional include directories if your bindings need access
# to headers outside of the default include paths. For example:
# target_include_directories(uni20_python PRIVATE ${CMAKE_SOURCE_DIR}/src)
