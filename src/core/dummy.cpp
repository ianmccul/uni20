#include "dummy.hpp"

namespace uni20 {

int add(int a, int b) {
    return a + b;
}

double multiply(double a, double b) {
    return a * b;
}

double compute_heavy_operation(double a, double b) {
    double result = 0.0;
    // A dummy heavy operation: sum over a million iterations.
    for (int i = 1; i <= 1000000; ++i) {
        result += (a * b) / static_cast<double>(i);
    }
    return result;
}

} // namespace uni20
