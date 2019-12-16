#ifndef DMMMM_HPP
#define DMMMM_HPP

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <string>
#include "xcl2.hpp"  //no idea what this is, used in the class lab host code

#include "partition_params.h"
#include "common_header_U1.h"

/*  one DMMM object per layer
    controls all access to kernel
    upon construction, weight matrix w is permanently copied to device global memory
    when used to perform dense matrix matrix multiplication, only need to pass slice
    of layer matrix h
*/

class DMMM {
  
public:
  
  // @param input w:  C * F weight matrix for the layer
  // @param input kernel_name:  filename of the kernel binary and the top level function of the kernel
  DMMM(float* w, const std::string kernel_name) {
    // OpenCL host setup start
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    m_context = new cl::Context(device);
    m_queue = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string binaryFile = xcl::find_binary_file(device_name, kernel_name.c_str());
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(*m_context, devices, bins);
    cl::Kernel kernel(program, kernel_name.c_str());
    m_krnl_dmmm = new cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&, bool, unsigned int>(kernel);

    std::vector<cl::Memory> wt_p_bufs_vec;
    m_wt_ps = (float**)malloc(NUM_W_P * sizeof(float*));
    if (!m_wt_ps) {
      std::cerr << "ERROR: failed to allocate memory for weight transpose partition pointers in DMMM constructor\n";
      exit(1);
    }
    m_wt_p_bufs = (cl::Buffer**)malloc(NUM_W_P * sizeof(cl::Buffer*));
    if (!m_wt_p_bufs) {
      std::cerr << "ERROR: failed to allocate memory for weight transpose partition buffer pointers in DMMM constructor\n";
      exit(1);
    }

    // create transpose partitions of weight matrix w and associated OpenCL buffers
    for (size_t c = 0; c < NUM_C_P; ++c) {
      for (size_t f = 0; f < NUM_F_P; ++f) {
        
        // create F_P * C_P transpose partition wt_p
        float* wt_p = (float*)aligned_alloc(sizeof(float), SIZE_W_P * sizeof(float));
	if (!wt_p) {
	  std::cerr << "ERROR: failed to allocate memory for weight transpose partition in DMMM constructor\n";
	  exit(1);
	}
        
        for (size_t i = 0, w_i = c * C_P; i < C_P; ++i, ++w_i) {
          for (size_t j = 0, w_j = f * F_P; j < F_P; ++j, ++w_j) {
            if ((w_i >= C) || (w_j >= F)) {
              // zero padding to fit in systolic array size
              wt_p[j * C_P + i] = 0.0;
            } else {
              wt_p[j * C_P + i] = w[w_i * F + w_j];
            }
          }
        }

	m_wt_ps[c * NUM_F_P + f] = wt_p; // keep track for deallocation upon destruction
        cl::Buffer* w_p_buf = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                            SIZE_W_P * sizeof(float), wt_p);
        m_wt_p_bufs[c * NUM_F_P + f] = w_p_buf; // keep track to use in kernel calls, indexed normally

        wt_p_bufs_vec.push_back(*w_p_buf); // used for enqueing buffers to global memory
      }
    }

    // copy weight input data to device global memory
    m_queue->enqueueMigrateMemObjects(wt_p_bufs_vec, 0); // 0 means from host
    m_queue->finish();

    // allocate memory for layer partition, values will change depending on operate_row input
    m_h_p = (float*)aligned_alloc(sizeof(float), SIZE_H_P * sizeof(float));
    if (!m_h_p) {
      std::cerr << "ERROR: failed to allocate memory for layer partition in DMMM constructor\n";
      exit(1);
    }
    m_h_p_buf = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
			       SIZE_H_P * sizeof(float), m_h_p);
    m_h_p_mem.push_back(*m_h_p_buf);

    // allocate memory for product of layer partition and weight partition
    m_hw_p = (float*)aligned_alloc(sizeof(float), SIZE_HW_P * sizeof(float));
    if (!m_hw_p) {
      std::cerr << "ERROR: failed to allocate memory for layer-weight partition product in DMMM constructor\n";
      exit(1);
    }
    m_hw_p_buf = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
				SIZE_HW_P * sizeof(float), m_hw_p);
    m_hw_p_mem.push_back(*m_hw_p_buf);
  }

  ~DMMM() {
    // OpenCL setup
    delete m_context;
    delete m_queue;
    delete m_krnl_dmmm;
    // weights
    for (size_t i = 0; i < NUM_W_P; ++i) {
      free(m_wt_ps[i]);
      delete m_wt_p_bufs[i];
    }
    free(m_wt_ps);
    free(m_wt_p_bufs);
    // layer
    free(m_h_p);
    delete m_h_p_buf;
    // product
    free(m_hw_p);
    delete m_hw_p_buf;
  }

  // @param input   h_r:      K * C slice of layer matrix, K <= N_P
  // @param input   hw_r:     N_P * F array to store the product of h_r and weight matrix w
  // @param input   H_R_ID:   index of N_P * C_P slice in total layer matrix, < NUM_N_P
  void operate_row(float* h_r, float* hw_r, const size_t H_R_ID) {
    // zero initialize K * F output in which partial N_P * F_P are accumulated
    for (size_t i = 0, hw_r_i = H_R_ID * N_P; i < N_P; ++i, ++hw_r_i) {
      for (size_t j = 0; j < F; ++j) {
        if (hw_r_i < N) {
          hw_r[i * F + j] = 0.0;
        }
      }
    }
    
    // operate on each N_P * C_P partition
    for (size_t c = 0; c < NUM_C_P; ++c) {
      // partition h into m_h_p, already allocated space on heap
      for (size_t i = 0, h_i = H_R_ID * N_P; i < N_P; ++i, ++h_i) {
        for (size_t j = 0, h_j = c * C_P; j < C_P; ++j, ++h_j) {
          if ((h_i >= N) || (h_j >= C)) {
            // zero padding to fit in systolic array size
            m_h_p[i*C_P + j] = 0.0;
          } else {
            m_h_p[i*C_P + j] = h_r[i * C + h_j];
          }
        }
      }

      //Copy layer partition data to device global memory
      m_queue->enqueueMigrateMemObjects(m_h_p_mem, 0); // 0 means from host

      // operate on each C_P * F_P corresponding to the same c-dimension indices
      for (size_t f = 0; f < NUM_F_P; ++f) {
        cl::Buffer wt_p_buf = *(m_wt_p_bufs[c * NUM_F_P + f]);

        // Launch the systolic array dmmm Kernel, last 2 args always 1
        (*m_krnl_dmmm)(cl::EnqueueArgs(*m_queue, cl::NDRange(1, 1, 1), cl::NDRange(1, 1, 1)),
                                      *m_h_p_buf, wt_p_buf, *m_hw_p_buf, 1, 1);

        // Copy Result from Device Global Memory to Host Local Memory
        m_queue->enqueueMigrateMemObjects(m_hw_p_mem, CL_MIGRATE_MEM_OBJECT_HOST);
        m_queue->finish(); // synchronization barrier

        accumulate_hw_partition(m_hw_p, hw_r, H_R_ID, f);
      }
    }
  }

private:

  // @param input hw_p:       array of size N_P * F_P
  // @param input hw_r:       array of size N_P * F to which hw_p is added to
  // @param input HW_R_ID:    index of N_P * F slice of hw that hw_r is (for bounds checks)
  // @param input HW_P_ID:    index of N_P * F_P slice of hw_r that hw_p is added to (for bounds checks)
  static void accumulate_hw_partition(float* hw_p, float* hw_r, const size_t HW_R_ID, const size_t HW_P_ID) {
    for (size_t i = 0, hw_r_i = HW_R_ID * N_P; i < N_P; ++i, ++hw_r_i) {
      for (size_t j = 0, hw_r_j = HW_P_ID * F_P; j < F_P; ++j, ++hw_r_j) {
        if ((hw_r_i < N) && (hw_r_j < F)) {
          hw_r[(i * F) + hw_r_j] += hw_p[i * F_P + j];
        }
      }
    }
  }

  cl::Context* m_context;
  cl::CommandQueue* m_queue;
  cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&, bool, unsigned int>* m_krnl_dmmm;
  float** m_wt_ps;
  cl::Buffer** m_wt_p_bufs;
  float* m_h_p;   // N_P * C_P
  cl::Buffer* m_h_p_buf;
  float* m_hw_p;  // N_P * F_P
  cl::Buffer* m_hw_p_buf;

  std::vector<cl::Memory> m_h_p_mem; // for loading altered layer partitions to kernel
  std::vector<cl::Memory> m_hw_p_mem; // for recieving layer-weight partition product from kernel
};

#endif
