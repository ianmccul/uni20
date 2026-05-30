# tests/tensorcontraction/CMakeLists.txt

add_executable(uni20_tensorcontraction_tests
  test_main.cpp
  test_data_file_plan.cpp
  test_effective_hamiltonian_operator.cpp
  test_effective_hamiltonian_plan.cpp
  test_matrix_family.cpp
  test_vector_algebra.cpp
)

target_link_libraries(uni20_tensorcontraction_tests
  PRIVATE
    uni20_common
    uni20_core
    uni20_tensorcontraction_runtime
    GTest::gtest
)

target_compile_definitions(uni20_tensorcontraction_tests PRIVATE UNIT_TEST)
target_compile_definitions(uni20_tensorcontraction_tests
  PRIVATE
    UNI20_TENSORCONTRACTION_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

gtest_discover_tests(uni20_tensorcontraction_tests)

if(MPIEXEC_EXECUTABLE)
  add_test(
    NAME TensorContractionMPI.EffectiveHamiltonianPlanCompile
    COMMAND
      "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "2"
      "bash" "-lc"
      "export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec \"$@\""
      "uni20-tensorcontraction-mpi"
      "$<TARGET_FILE:uni20_tensorcontraction_tests>"
      "--gtest_filter=TensorContractionEffectiveHamiltonianPlanTest.CompilesSyntheticSingleTermPlan"
  )
  add_test(
    NAME TensorContractionMPI.DataFilePlanCompile
    COMMAND
      "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "2"
      "bash" "-lc"
      "export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec \"$@\""
      "uni20-tensorcontraction-mpi"
      "$<TARGET_FILE:uni20_tensorcontraction_tests>"
      "--gtest_filter=TensorContractionDataFilePlanTest.CompilesOriginalFormatFixture"
  )
  add_test(
    NAME TensorContractionMPI.EffectiveHamiltonianPlanApply
    COMMAND
      "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "2"
      "bash" "-lc"
      "export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec \"$@\""
      "uni20-tensorcontraction-mpi"
      "$<TARGET_FILE:uni20_tensorcontraction_tests>"
      "--gtest_filter=TensorContractionEffectiveHamiltonianPlanTest.AppliesSingleTermPlan"
  )
  add_test(
    NAME TensorContractionMPI.EffectiveHamiltonianOperatorApply
    COMMAND
      "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "2"
      "bash" "-lc"
      "export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec \"$@\""
      "uni20-tensorcontraction-mpi"
      "$<TARGET_FILE:uni20_tensorcontraction_tests>"
      "--gtest_filter=TensorContractionEffectiveHamiltonianOperatorTest.AppliesSingleTermMatvec"
  )
endif()

add_executable(uni20_tensorcontraction_no_mpi_tests
  test_matrix_family_no_mpi.cpp
)

target_link_libraries(uni20_tensorcontraction_no_mpi_tests
  PRIVATE
    uni20_common
    uni20_core
    uni20_tensorcontraction_runtime
    GTest::gtest_main
)

target_compile_definitions(uni20_tensorcontraction_no_mpi_tests PRIVATE UNIT_TEST)

gtest_discover_tests(uni20_tensorcontraction_no_mpi_tests)
