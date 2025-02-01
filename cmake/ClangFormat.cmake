# ClangFormat.cmake
# This file sets up a custom target for clang-format if clang-format is available.

find_program(CLANG_FORMAT_EXE clang-format)
if(CLANG_FORMAT_EXE)
  message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")

  # List all source files that you want to format.
  file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.hpp
    ${CMAKE_SOURCE_DIR}/tests/*.cpp
    ${CMAKE_SOURCE_DIR}/bindings/python/*.cpp
  )

  # Create a custom target that formats all these source files.
  add_custom_target(
    clang_format
    COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_CXX_SOURCE_FILES}
    COMMENT "Running clang-format on all source files"
  )
else()
  message(WARNING "clang-format not found. Code formatting target will be unavailable.")
endif()
