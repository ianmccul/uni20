# The matrix exponential function was originally adapted from EXPOKIT:

- R. B. Sidje, "EXPOKIT: A Software Package for Computing Matrix Exponentials", *ACM
  Transactions on Mathematical Software* 24(1):130-156, 1998.

The implementation incorporate results from the adaptive scaling-and-squaring
algorithm of Nicholas J. Higham and Awad H. Al-Mohy. The implementation in
`src/expokit/adaptive_exponential.cpp` follows the analysis presented in:

- N. J. Higham, "The Scaling and Squaring Method for the Matrix Exponential Revisited",
  *SIAM Journal on Matrix Analysis and Applications* 26(4):1179-1193, 2005.
- A. H. Al-Mohy and N. J. Higham, "A New Scaling and Squaring Algorithm for the Matrix
  Exponential", *SIAM Journal on Matrix Analysis and Applications* 31(3):970-989, 2011.

These references supersede the fixed-degree Padé scheme originally distributed with
EXPOKIT.
