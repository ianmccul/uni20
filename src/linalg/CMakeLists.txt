add_library(uni20_linalg INTERFACE)

target_include_directories(uni20_linalg
    INTERFACE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(uni20_linalg
    INTERFACE
        uni20_core
        uni20_common
)
