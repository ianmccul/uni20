add_library(uni20_expokit STATIC
    adaptive_exponential.cpp
)

target_include_directories(uni20_expokit
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(uni20_expokit
    PUBLIC
        uni20_deps
        uni20_linalg
)
