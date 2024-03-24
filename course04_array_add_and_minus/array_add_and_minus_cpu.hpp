#pragma once

#include "common.hpp"

void array_add_and_minus_cpu(const double* a, const double* b, const double* c, double* d, const int ARRAY_LEN) {
    for (unsigned long i = 0; i < ARRAY_LEN; ++i) {
        d[i] = a[i] + b[i] - c[i];
    }
}

