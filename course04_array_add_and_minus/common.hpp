#pragma once

#include <cmath>

const unsigned long ARRAY_LEN = 1e8 + 10;                       // 数组长度为 1e8e8 + 10
const unsigned long  ARRAY_SIZE = sizeof(double) * ARRAY_LEN;   // 数组内存大小(字节数) = 8字节 * (1e8 + 10)，约为0.8GB
const double EPSILON = 1.0e-10;
const double VALUE_A = 1.11;
const double VALUE_B = 3.33;
const double VALUE_C = 2.22;
const double VALUE_D = 2.22;

bool check_array_add_and_minus(const double* d, const int ARRAY_LEN) {
    bool is_ok = true;
    for (unsigned long  i = 0; i < ARRAY_LEN; ++i) {
        if (fabs(d[i] - VALUE_D) < EPSILON) {
            continue;
        }
        is_ok = false;
    }
    return is_ok;
}