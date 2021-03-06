/**
 *  This file is automatically generated by PolySA CodeGen.
 *  Version: 1.0
 *  Authos: Jie Wang
 */

#include "common_header_U1.h"

void U1_DataFeed0Head(
  U1_bus_t0* A,
  bool init,
  unsigned int FILTER_S,
  stream<U1_Data0TransferChannelType> &fifo_transfer_out0
){
#pragma HLS INLINE off
#pragma HLS DATA_PACK variable=fifo_transfer_out0

  // loader buffer
  U1_bus_t0 A_buf[U1_DATA0_HEAD_BUF_SIZE / U1_DATA0_PACK_FACTOR];
#pragma HLS ARRAY_PARTITION variable=A_buf dim=1 block factor=1

  for (ap_uint<8> i_t = 0; i_t < U1_I; i_t += U1_I_T){
    unsigned int chunk_offset = i_t * U1_K;
    memcpy((void*)A_buf, (void*)(A + chunk_offset / U1_DATA0_PACK_FACTOR), sizeof(U1_data_t0) * U1_I_T * U1_K);
    for (ap_uint<8> j_t = 0; j_t < U1_J; j_t += U1_J_T){
      for (ap_uint<8> k_t = 0; k_t < U1_K / U1_SIMD_FACTOR; k_t += U1_K_T / U1_SIMD_FACTOR){
        bool init_internal = (k_t == 0);
        bool init_final = init && init_internal;
        bool last = (k_t == (U1_K - U1_K_T) / U1_SIMD_FACTOR);
        // write to SA
        ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> sel_tmp0[U1_DATA0_PACK_FACTOR / U1_DATA0_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp0 complete dim=1
        for (ap_uint<2> t0 = 0; t0 < U1_I_T / U1_COL_IL_FACTOR / U1_DATA0_FC_SPLIT_FACTOR; t0++){
          for (ap_uint<7> t1 = 0; t1 < U1_COL_IL_FACTOR; t1++){
            for (ap_uint<5> t2 = 0; t2 < U1_K_T / U1_DATA0_FC_SIMD_FACTOR; t2++){
            #pragma HLS PIPELINE II=1
              unsigned int local_i = t0 * U1_COL_IL_FACTOR + t1;
              unsigned int local_k = k_t * U1_SIMD_FACTOR + t2 * U1_DATA0_FC_SIMD_FACTOR;
              unsigned int feeder_id = t0 / U1_DATA0_FC_GROUP_FACTOR;
              unsigned int A_index = local_i * U1_K + local_k;
              unsigned int A_bus_index = A_index / U1_DATA0_PACK_FACTOR;
              unsigned int A_bus_offset = A_index % U1_DATA0_PACK_FACTOR;

              U1_bus_t0 bus_data0 = A_buf[0 * 1024 + A_bus_index];
              ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> wide_data0;
              for (ap_uint<2> s = 0; s < U1_DATA0_PACK_FACTOR / U1_DATA0_FC_SIMD_FACTOR; s++){
#pragma HLS UNROLL
                sel_tmp0[s] = bus_data0(U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR - 1, 0);
                bus_data0 = bus_data0 >> (U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR);
              }
              wide_data0 = sel_tmp0[A_bus_offset / U1_DATA0_FC_SIMD_FACTOR];
              fifo_transfer_out0.write(U1_Data0TransferChannelType(
                wide_data0,
                (unsigned int)feeder_id, init_final, last, FILTER_S));

            }
          }
        }
      }
    }
  }
}

void U1_DataFeed1Head(
  U1_bus_t1* B,
  bool init,
  unsigned int FILTER_S,
  stream<U1_Data1TransferChannelType> &fifo_transfer_out0
){
#pragma HLS INLINE off
#pragma HLS DATA_PACK variable=fifo_transfer_out0

  // loader buffer
  U1_bus_t1 B_buf[U1_DATA1_HEAD_BUF_SIZE / U1_DATA1_PACK_FACTOR];
#pragma HLS ARRAY_PARTITION variable=B_buf dim=1 block factor=1

  for (ap_uint<8> i_t = 0; i_t < U1_I; i_t += U1_I_T){
    for (ap_uint<8> j_t = 0; j_t < U1_J; j_t += U1_J_T){
      unsigned int chunk_offset = j_t *U1_K;
      memcpy((void*)B_buf, (void*)(B + chunk_offset / U1_DATA1_PACK_FACTOR), sizeof(U1_data_t1) * U1_J_T * U1_K);
      for (ap_uint<5> k_t = 0; k_t < U1_K / U1_SIMD_FACTOR; k_t += U1_K_T / U1_SIMD_FACTOR){
        bool init_internal = (k_t == 0);
        bool init_final = init && init_internal;
        bool last = (k_t == (U1_K - U1_K_T) / U1_SIMD_FACTOR);
        // write to SA
        ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> sel_tmp0[U1_DATA1_PACK_FACTOR / U1_DATA1_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp0 complete dim=1
        for (ap_uint<2> t0 = 0; t0 < U1_J_T / U1_ROW_IL_FACTOR / U1_DATA1_FC_SPLIT_FACTOR; t0++){
          for (ap_uint<7> t1 = 0; t1 < U1_ROW_IL_FACTOR; t1++){
            for (ap_uint<5> t2 = 0; t2 < U1_K_T / U1_DATA1_FC_SIMD_FACTOR; t2++){
            #pragma HLS PIPELINE II=1
              unsigned int local_j = t0 * U1_ROW_IL_FACTOR + t1;
              unsigned int local_k = k_t * U1_SIMD_FACTOR + t2 * U1_DATA1_FC_SIMD_FACTOR;
              unsigned int feeder_id = t0 / U1_DATA1_FC_GROUP_FACTOR;
              unsigned int B_index = local_j * U1_K + local_k;
              unsigned int B_bus_index = B_index / U1_DATA1_PACK_FACTOR;
              unsigned int B_bus_offset = B_index % U1_DATA1_PACK_FACTOR;

              U1_bus_t1 bus_data0 = B_buf[0 * 1024 + B_bus_index];
              ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> wide_data0;
              for (ap_uint<2> s = 0; s < U1_DATA1_PACK_FACTOR / U1_DATA1_FC_SIMD_FACTOR; s++){
#pragma HLS UNROLL
                sel_tmp0[s] = bus_data0(U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR - 1, 0);
                bus_data0 = bus_data0 >> (U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR);
              }
              wide_data0 = sel_tmp0[B_bus_offset / U1_DATA1_FC_SIMD_FACTOR];
              fifo_transfer_out0.write(U1_Data1TransferChannelType(
                wide_data0,
                (unsigned int)feeder_id, init_final, last, FILTER_S));

            }
          }
        }
      }
    }
  }
}

void U1_DataCollect2Head(
  U1_bus_t2* C,
  stream <U1_Data2TransferChannelType> &fifo_transfer_in0
){
#pragma HLS INLINE off
#pragma HLS DATA_PACK variable=fifo_transfer_in0

  // loader buffer
  U1_bus_t2 C_buf[U1_DATA2_HEAD_BUF_SIZE / U1_DATA2_PACK_FACTOR];
#pragma HLS ARRAY_PARTITION variable=C_buf dim=1 block factor=1

  ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> sel_tmp0[U1_DATA2_PACK_FACTOR / U1_DATA2_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp0 complete dim=1

  for (ap_uint<8> i_t = 0; i_t < U1_I; i_t += U1_I_T){
    for (ap_uint<8> j_t = 0; j_t < U1_J; j_t += U1_J_T){
      for (ap_int<3> t0 = U1_I_T / U1_COL_IL_FACTOR / U1_DATA2_FC_SPLIT_FACTOR - 1; t0 >= 0; t0--){
        for (ap_uint<7> t1 = 0; t1 < U1_COL_IL_FACTOR; t1++){
          for (ap_uint<5> t2 = 0; t2 < U1_J_T / U1_DATA2_FC_SIMD_FACTOR; t2++){
            #pragma HLS PIPELINE II=1
            unsigned int local_i = t0 * U1_COL_IL_FACTOR + t1;
            unsigned int local_j = t2 * U1_DATA2_FC_SIMD_FACTOR;
            unsigned int C_index = local_i * U1_J_T + local_j;

            unsigned int C_bus_index = C_index / U1_DATA2_PACK_FACTOR;
            unsigned int C_bus_offset = C_index % U1_DATA2_PACK_FACTOR;

            U1_Data2TransferChannelType fifo_data0 = fifo_transfer_in0.read();
            ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> wide_data0 = fifo_data0.data;
            U1_bus_t2 bus_data0;
            sel_tmp0[C_bus_offset / U1_DATA2_FC_SIMD_FACTOR] = wide_data0;
            if (C_bus_offset == U1_DATA2_PACK_FACTOR - U1_DATA2_FC_SIMD_FACTOR){
              bus_data0 = (
                sel_tmp0[1],
                sel_tmp0[0]
              );
              C_buf[0 * 1024 + C_bus_index] = bus_data0;
            }

          }
        }
      }
      // write to DRAM
      unsigned int chunk_offset = ((i_t / U1_I_T) * (U1_J / U1_J_T) + (j_t / U1_J_T)) * (U1_I_T * U1_J_T);
      memcpy((void*)(C + chunk_offset / U1_DATA2_PACK_FACTOR), (void*)C_buf, sizeof(U1_data_t2) * U1_I_T * U1_J_T);
    }
  }
}

