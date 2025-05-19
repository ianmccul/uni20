# tests/kernel/CMakeLists.txt

#add_test_module(kernel
#  SOURCES test_contract.cpp
#  LIBS uni20_common uni20_kernel
#)


if(UNI20_BACKEND_CPU)
  add_test_module(test_contract_cpu
    SOURCES test_contract_cpu.cpp
    LIBS uni20_kernel uni20_common
  )
endif()

if(UNI20_BACKEND_BLAS)
  add_test_module(test_contract_blas
    SOURCES test_contract_blas.cpp
    LIBS uni20_kernel uni20_common
  )
endif()

