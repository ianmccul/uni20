#include <gtest/gtest.h>

#include <cstdlib>

int main(int argc, char** argv)
{
  // MPS unit tests exercise algorithm semantics on tiny systems.  Keep them on
  // the host effective-Hamiltonian backend by default so the ordinary unit-test
  // binary does not reserve hundreds of GB of CUDA virtual address space.
  setenv("UNI20_TENSORCONTRACTION_BACKEND", "host", 0);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
