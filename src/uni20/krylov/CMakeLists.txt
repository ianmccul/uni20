add_library(uni20_krylov INTERFACE)

target_include_directories(uni20_krylov
    INTERFACE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(uni20_krylov
    INTERFACE
        mdspan
        uni20_deps
        uni20_linalg
)
