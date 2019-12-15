#include <stdlib.h>
#include <iostream>
#include <string>

#include "partition_params.h"
#include "dmmm.hpp"

int main() {
  std::cout << "N, C, F: " << N << ", " << C << ", " << F << "\n";
  std::cout << "N_P, C_P, F_P: " << N_P << ", " << C_P << ", " << F_P << "\n";
  std::cout << "NUM_N_P, NUM_C_P, NUM_F_P: " << NUM_N_P << ", " << NUM_C_P << ", " << NUM_F_P << "\n";
  float w[SIZE_W], h[SIZE_H], hw[SIZE_HW];
  for (size_t i = 0; i < SIZE_W; ++i) {
    w[i] = 2;//(float)rand();
  }
  for (size_t i = 0; i < SIZE_H; ++i) {
    h[i] = 5;//(float)rand();
  }

  const std::string k_name = "systolic_array_kernel";
  DMMM D(w, k_name);
  std::cout << "DMMM created\n";

  const float* hw_r;
  size_t i_hw = 0;
  for (size_t n = 0; n < NUM_N_P; ++n) {
    hw_r = D.operate_row(&(h[n * SIZE_H_R]), n);
    std::cout << "calculated output rows " << n * N_P << " to " << (n + 1) * N_P - 1 << "\n";
    for (size_t i = 0; i < SIZE_H_R && i_hw < SIZE_HW; ++i, ++i_hw) {
      hw[i_hw] = hw_r[i];
    }
    std::cout << "copied output rows to full hw matrix\n";
  }

  size_t error_cnt = 0;
  bool fin = false;
  for (size_t i = 0; i < N; ++i) {
    for (size_t k = 0; k < F; ++k) {
      float res = 0;
      for (size_t j = 0; j < C; ++j) {
      	res += h[i * C + j] * w[j * F + k];
      }
      if (abs(res - hw[i * F + k]) > 0.002) {
        std::cout << "ERROR: incorrect result @ ("<< i << ", " << k << ")\n";
        std::cout << "sw result: " << res << ", hw result: " << hw[i * F + k] << "\n\n";
        if (++error_cnt > 10) {
      	  std::cout << "Too many errors, quitting comparison\n";
	        fin = true;
	        break;
	      }
      }
      if (fin) break;
    }
    if (fin) break;
  }

  if (fin) {
    std::cout << "due to many errors, checking for edge value at (" << N-1 << "," << F-1 << "): " << hw[(N - 1) * F + F-1] << "\n";
    return 0;
  }

  std::cout << "SUCCESS? made it to the end!\n";
  return 0;
}
