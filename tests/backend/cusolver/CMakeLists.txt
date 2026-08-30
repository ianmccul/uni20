add_test_module(cusolver
  SOURCES test_cusolver_svd.cpp
  LIBS uni20_async uni20_backend_cusolver uni20_linalg uni20_symmetry TBB::tbb
)
