if(NOT UNI20_BUILD_PYTHON)
  message(STATUS "Skipping Python binding tests because UNI20_BUILD_PYTHON=OFF")
  return()
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)

set(PYTHON_BINDINGS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bindings/python")
set(PYTHON_GREETING_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_greet.py")
set(PYTHON_BUILDINFO_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_buildinfo.py")

add_test(
  NAME python.bindings.greet
  COMMAND ${Python_EXECUTABLE} "${PYTHON_GREETING_TEST}" "${PYTHON_BINDINGS_OUTPUT_DIR}"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_test(
  NAME python.bindings.buildinfo
  COMMAND ${Python_EXECUTABLE} "${PYTHON_BUILDINFO_TEST}" "${PYTHON_BINDINGS_OUTPUT_DIR}"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
