add_library(uni20_linalg STATIC
    backends/cpu/matrix_exponential.cpp
)

target_include_directories(uni20_linalg
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(uni20_linalg
    PUBLIC
        uni20_core
        uni20_common
)
