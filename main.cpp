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
    w[i] = ((float)rand()) / RAND_MAX;
  }
  for (size_t i = 0; i < SIZE_H; ++i) {
    h[i] = ((float)rand()) / RAND_MAX;
  }

  const std::string k_name = "systolic_array_kernel";
  DMMM D(w, k_name);
  std::cout << "DMMM created\n";

  for (size_t n = 0; n < NUM_N_P; ++n) {
    D.operate_row(&(h[n * SIZE_H_R]), &(hw[n * SIZE_HW_R]), n);

    size_t end_row_id = (n == NUM_N_P - 1) ? N - 1 : (n + 1) * N_P - 1;
    std::cout << "calculated output rows " << n * N_P << " to " << end_row_id << "\n";
  }

  size_t error_cnt = 0;
  bool fin = false;
  size_t error_cnt2 = 0;
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
    }
    if (fin) {
      fin = false;
      error_cnt = 0;
      if (++error_cnt2 > 3) {
	fin= true;
	break;
      }
    }
  }

  if (fin) {
    std::cout << "due to many errors, checking for edge value at (" << N-1 << "," << F-1 << "): " << hw[(N - 1) * F + F-1] << "\n";
    std::cout << "due to many errors, checking for edge value at (" << 0 << "," << F-1 << "): " << hw[F-1] << "\n";
    std::cout << "due to many errors, checking for edge value at (" << N-3 << "," << 3 << "): " << hw[(N - 3) * F + F-3] << "\n";
    std::cout << "due to many errors, checking for edge value at (" << N-2 << "," << 2 << "): " << hw[(N - 2) * F + F-2] << "\n";
    std::cout << "due to many errors, checking for edge value at (" << N-1 << "," << 1 << "): " << hw[(N - 1) * F + F-1] << "\n";
  } else {
    std::cout << "SUCCESS!\n";
  }

  return 0;
}
