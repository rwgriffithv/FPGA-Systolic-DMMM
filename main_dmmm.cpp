#include <stdlib.h>
#include <iostream>
#include <string>

#include "partition_params.h"
#include "dmmm.hpp"

int main() {
  float w[C*F], h[N*C], o_hw[N*F];
  for (size_t i = 0; i < C*F; ++i) {
    w[i] = 2;//(float)rand();
  }
  for (size_t i = 0; i < N*C; ++i) {
    h[i] = 5;//(float)rand();
  }

  float wt[F*C];
  for (size_t c = 0; c < C; ++c) {
    for (size_t f = 0; f < F; ++f) {
      wt[f * C + c] = w[c * F + f];
    }
  }

  const std::string k_name = "systolic_array_kernel";
  DMMM D(w, k_name);
  std::cout << "DMMM created\n";

  const float* o_rows;
  size_t o_hw_j = 0;
  for (size_t i = 0; i < N*C; i+=N_P*C) {
    o_rows = D.operate(&(h[i]));
    std::cout << "calculated output rows " << i/C << " to " << i/C + N_P - 1 << "\n";
    for (size_t j = 0; j < N_P*F; ++j, ++o_hw_j) {
      o_hw[o_hw_j] = o_rows[j];
    }
    std::cout << "copied output rows to full o matrix\n";
  }


  size_t error_cnt = 0;
  bool fin = false;
  for (size_t i = 0; i < N; ++i) {
    for (size_t k = 0; k < F; ++k) {
      float res = 0;
      for (size_t j = 0; j < C; ++j) {
	res += h[i * C + j] * w[j * F + k];
      }
      if (res != o_hw[i * F + k]) {
	std::cout << "ERROR: incorrect multiplication result\n";
	std::cout << "i: " << i << ", k: " << k << "\n";
	std::cout << "software result: " << res << ", hardware results: " << o_hw[i * F + k] << "\n\n";
	if (++error_cnt > 10) {
	  std::cout << "Too many errors\n";
	  fin = true;
	  break;
	}
      }
      if (fin) break;
    }
    if (fin) break;
  }

  if (fin) {
    std::cout << "checking for edge value at (" << N-1 << "," << F-1 << "): " << o_hw[(N - 1) * F + F-1] << "\n";
    return 0;
  }

  std::cout << "SUCCESS? made it to the end!\n";
  return 0;
}
