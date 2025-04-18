# tests/CMakeLists.txt

enable_testing()
include(GoogleTest)

# Add subdirectories that contribute source files
add_subdirectory(common)
add_subdirectory(level1)

# Retrieve accumulated combined test inputs
get_property(UNI20_COMBINED_TEST_SRCS GLOBAL PROPERTY UNI20_COMBINED_TEST_SRCS)
get_property(UNI20_COMBINED_TEST_LIBS GLOBAL PROPERTY UNI20_COMBINED_TEST_LIBS)

# Optionally build the combined test executable
if(UNI20_BUILD_COMBINED_TESTS AND UNI20_COMBINED_TEST_SRCS)
  add_executable(uni20_tests
    ${UNI20_COMBINED_TEST_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp
  )
  target_link_libraries(uni20_tests PRIVATE ${UNI20_COMBINED_TEST_LIBS} gtest_main)
  gtest_discover_tests(uni20_tests)
elseif(UNI20_BUILD_COMBINED_TESTS)
  message(WARNING "UNI20_BUILD_COMBINED_TESTS is ON, but no test sources were registered.")
endif()
