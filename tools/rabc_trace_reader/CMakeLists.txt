# Minimal, self-contained build for the R/A/B/C term-trace reader.
#
# Provides a header-only INTERFACE library `rabc_trace` (which pulls in
# nlohmann/json) and an example executable. To use from another project:
#
#   add_subdirectory(rabc_trace_reader)
#   target_link_libraries(your_target PRIVATE rabc_trace)

cmake_minimum_required(VERSION 3.16)
project(rabc_trace_reader CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Prefer an installed nlohmann/json; otherwise fetch the header-only release.
find_package(nlohmann_json 3.2 QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(rabc_trace INTERFACE)
target_include_directories(rabc_trace INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rabc_trace INTERFACE nlohmann_json::nlohmann_json)
target_compile_features(rabc_trace INTERFACE cxx_std_17)

option(RABC_TRACE_BUILD_EXAMPLE "Build the term-trace reader example" ON)
if(RABC_TRACE_BUILD_EXAMPLE)
  add_executable(rabc_trace_example example.cpp)
  target_link_libraries(rabc_trace_example PRIVATE rabc_trace)
endif()
