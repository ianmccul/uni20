add_library(uni20_common STATIC
#    common.cpp
    common.hpp
)
target_link_libraries(uni20_common PRIVATE uni20_deps)
target_include_directories(uni20_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
