# Clear any previously accumulated test source lists
set_property(GLOBAL PROPERTY UNI20_COMBINED_TEST_SRCS "")
set_property(GLOBAL PROPERTY UNI20_COMBINED_TEST_LIBS "")

function(add_test_module name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES LIBS)
  cmake_parse_arguments(TESTMOD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Qualify sources with full paths for use in the combined executable
  set(ABS_SOURCES "")
  foreach(src IN LISTS TESTMOD_SOURCES)
    list(APPEND ABS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
  endforeach()

  add_executable(uni20_${name}_tests ${ABS_SOURCES})
  target_link_libraries(uni20_${name}_tests PRIVATE ${TESTMOD_LIBS} GTest::gtest_main)
  target_compile_definitions(uni20_${name}_tests PRIVATE UNIT_TEST)
  gtest_discover_tests(uni20_${name}_tests)

  if(UNI20_BUILD_COMBINED_TESTS)
    set_property(GLOBAL APPEND PROPERTY UNI20_COMBINED_TEST_SRCS ${ABS_SOURCES})
    set_property(GLOBAL APPEND PROPERTY UNI20_COMBINED_TEST_LIBS ${TESTMOD_LIBS})
  endif()
endfunction()
