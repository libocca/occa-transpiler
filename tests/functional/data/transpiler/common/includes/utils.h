#pragma once

struct Data {
    float* data @ restrict;
    int* idxs @ restrict;
};

const int SIZE = 128;

float add(float a, float b);

float add2(float a, float b) {
    return a + b;
}

#define __my_attr__ inline
__my_attr__ float fastSqrt(float);
