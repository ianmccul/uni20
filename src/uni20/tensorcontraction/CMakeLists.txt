# src/uni20/tensorcontraction/CMakeLists.txt

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(LAPACK QUIET)
find_package(PkgConfig QUIET)

if(NOT TARGET NCCL::NCCL)
  if(PkgConfig_FOUND)
    pkg_check_modules(NCCL QUIET IMPORTED_TARGET nccl)
  endif()

  if(TARGET PkgConfig::NCCL)
    add_library(NCCL::NCCL INTERFACE IMPORTED)
    target_link_libraries(NCCL::NCCL INTERFACE PkgConfig::NCCL)
  else()
    set(_nccl_include_hints ${CUDAToolkit_INCLUDE_DIRS})
    set(_nccl_library_hints ${CUDAToolkit_LIBRARY_DIR})

    foreach(root IN ITEMS NCCL_ROOT CUDA_HOME)
      if(DEFINED ENV{${root}} AND NOT "$ENV{${root}}" STREQUAL "")
        list(APPEND _nccl_include_hints "$ENV{${root}}" "$ENV{${root}}/include")
        list(APPEND _nccl_library_hints "$ENV{${root}}" "$ENV{${root}}/lib"
                                        "$ENV{${root}}/lib64")
      endif()
    endforeach()

    find_path(UNI20_TENSORCONTRACTION_NCCL_INCLUDE_DIR
              NAMES nccl.h
              HINTS ${_nccl_include_hints}
              PATH_SUFFIXES include)
    find_library(UNI20_TENSORCONTRACTION_NCCL_LIBRARY
                 NAMES nccl
                 HINTS ${_nccl_library_hints}
                 PATH_SUFFIXES lib lib64)

    if(NOT UNI20_TENSORCONTRACTION_NCCL_INCLUDE_DIR
       OR NOT UNI20_TENSORCONTRACTION_NCCL_LIBRARY)
      message(FATAL_ERROR
              "NCCL not found. Install NCCL with pkg-config metadata or set NCCL_ROOT.")
    endif()

    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(
      NCCL::NCCL
      PROPERTIES
        IMPORTED_LOCATION "${UNI20_TENSORCONTRACTION_NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${UNI20_TENSORCONTRACTION_NCCL_INCLUDE_DIR}")
  endif()
endif()

add_library(uni20_tensorcontraction_runtime STATIC
  src/effective_hamiltonian_operator.cpp
  src/effective_hamiltonian_plan.cpp
  src/matrix_family.cpp
  src/vector_algebra.cpp
  src/Arranger.cpp
  src/Calculator.cpp
  src/Matrix.cpp
  src/MatrixAllocator.cpp
  src/Swapper.cpp
)

target_include_directories(uni20_tensorcontraction_runtime
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(uni20_tensorcontraction_runtime
  PUBLIC
    uni20_common
    uni20_core
    CUDA::cudart
    CUDA::cublas
    MPI::MPI_CXX
    NCCL::NCCL
)

if(LAPACK_FOUND)
  target_link_libraries(uni20_tensorcontraction_runtime PUBLIC ${LAPACK_LIBRARIES})
  target_compile_definitions(uni20_tensorcontraction_runtime PUBLIC UNI20_TENSORCONTRACTION_HAS_LAPACK=1)
endif()

target_compile_features(uni20_tensorcontraction_runtime PUBLIC cxx_std_20)
relax_warnings(uni20_tensorcontraction_runtime)

add_library(uni20::tensorcontraction_runtime ALIAS uni20_tensorcontraction_runtime)
