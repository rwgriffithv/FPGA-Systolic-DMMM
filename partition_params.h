#ifndef PARTITION_PARAMS_H
#define PARTITION_PARAMS_H

#include <cmath>

#include "common_header_U1.h"


const size_t N = 128;
const size_t C = 128;
const size_t F = 256;
const size_t N_P = U1_I_T;    // dim of systolic array
const size_t C_P = U1_K_T;    // dim of systolic array
const size_t F_P = U1_J_T;    // dim of systolic array

const size_t SIZE_H = N * C;
const size_t SIZE_W = C * F;
const size_t SIZE_HW = N * F;

const size_t SIZE_H_R = N_P * C;
const size_t SIZE_HW_R = N_P * F;

const size_t SIZE_H_P = N_P * C_P;
const size_t SIZE_W_P = C_P * F_P;
const size_t SIZE_HW_P = N_P * F_P;


const size_t NUM_N_P = ceil(((float)N) / N_P);
const size_t NUM_C_P = ceil(((float)C) / C_P);
const size_t NUM_F_P = ceil(((float)F) / F_P);

const size_t NUM_H_P = NUM_N_P * NUM_C_P;
const size_t NUM_W_P = NUM_C_P * NUM_F_P;





#endif
