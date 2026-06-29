add_library(uni20_krylov INTERFACE)

find_package(LAPACK REQUIRED)

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

if(TARGET LAPACK::LAPACK)
  target_link_libraries(uni20_krylov INTERFACE LAPACK::LAPACK)
elseif(LAPACK_LIBRARIES)
  target_link_libraries(uni20_krylov INTERFACE ${LAPACK_LIBRARIES})
endif()
