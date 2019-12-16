/**
 *  This file is automatically generated by PolySA CodeGen.
 *  Version: 1.0
 *  Authos: Jie Wang
 */

#include "common_header_U1.h"

void U1_PE_MAC(
  U1_Data0SIMDType op0,
  U1_Data1SIMDType op1,
  U1_data_t2* op2,
  bool init
){
#pragma HLS INLINE
#pragma HLS DATA_PACK variable=op0
#pragma HLS DATA_PACK variable=op1
  ap_uint<256> op0_data = op0;
  ap_uint<256> op1_data = op1;

  float op0_u[U1_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=op0_u complete
  float op1_u[U1_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=op1_u complete

  for (int i = 0; i < U1_SIMD_FACTOR; i++){
#pragma HLS UNROLL
    ap_uint<U1_DATA0_WIDTH> sel0 = op0_data(U1_DATA0_WIDTH-1, 0);
    op0_u[i] = Reinterpret<U1_data_t0>(sel0);
    op0_data = op0_data >> U1_DATA0_WIDTH;
    ap_uint<U1_DATA1_WIDTH> sel1 = op1_data(U1_DATA1_WIDTH-1, 0);
    op1_u[i] = Reinterpret<U1_data_t1>(sel1);
    op1_data = op1_data >> U1_DATA1_WIDTH;
  }

  U1_data_t2 sum = (init == 1)? (U1_data_t2) 0: *op2;
  for (int i = 0; i < U1_SIMD_FACTOR; i++){
#pragma HLS UNROLL
    sum += op0_u[i] * op1_u[i];
  }

  *op2 = sum;
}

void U1_op0_transfer(
  stream<U1_Data0PEChannelType> &fifo0_in,
  stream<U1_Data0PEChannelType> &fifo0_out,
  stream<U1_Data0PEChannelType> &fifo0_local
){
#pragma HLS DATA_PACK variable=fifo0_in
#pragma HLS DATA_PACK variable=fifo0_out
#pragma HLS DATA_PACK variable=fifo0_local
#pragma HLS INLINE off

  for (ap_uint<6> i_t = 0; i_t < 32; i_t += 32)
    for (ap_uint<6> j_t = 0; j_t < 32; j_t += 32)
      for (ap_uint<6> k_t = 0; k_t < 32; k_t += 32)
      {
        for (int la_counter = 0; la_counter < U1_LOCAL_ACCUM_NUM; la_counter++)
          for (int local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
            U1_Data0PEChannelType fifo0_in_data;
            fifo0_in_data = fifo0_in.read();
            fifo0_out.write(fifo0_in_data);
            fifo0_local.write(fifo0_in_data);
          }
      }
}

void U1_op0_transfer_wrapper(
  stream<U1_Data0PEChannelType> &fifo0_in,
  stream<U1_Data0PEChannelType> &fifo0_out,
  stream<U1_Data0PEChannelType> &fifo0_local
){
  U1_op0_transfer(
    fifo0_in,
    fifo0_out,
    fifo0_local);
}

void U1_op0_transfer_last(
  stream<U1_Data0PEChannelType> &fifo0_in,
  stream<U1_Data0PEChannelType> &fifo0_local
){
#pragma HLS DATA_PACK variable=fifo0_in
#pragma HLS DATA_PACK variable=fifo0_local
#pragma HLS INLINE off

  for (ap_uint<6> i_t = 0; i_t < 32; i_t += 32)
    for (ap_uint<6> j_t = 0; j_t < 32; j_t += 32)
      for (ap_uint<6> k_t = 0; k_t < 32; k_t += 32)
      {
        for (int la_counter = 0; la_counter < U1_LOCAL_ACCUM_NUM; la_counter++)
          for (int local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
            U1_Data0PEChannelType fifo0_in_data;
            fifo0_in_data = fifo0_in.read();
            fifo0_local.write(fifo0_in_data);
          }
      }
}

void U1_op0_transfer_last_wrapper(
  stream<U1_Data0PEChannelType> &fifo0_in,
  stream<U1_Data0PEChannelType> &fifo0_local
){
  U1_op0_transfer_last(
    fifo0_in,
    fifo0_local);
}

void U1_op1_transfer(
  stream<U1_Data1PEChannelType> &fifo1_in,
  stream<U1_Data1PEChannelType> &fifo1_out,
  stream<U1_Data1PEChannelType> &fifo1_local
){
#pragma HLS DATA_PACK variable=fifo1_in
#pragma HLS DATA_PACK variable=fifo1_out
#pragma HLS DATA_PACK variable=fifo1_local
#pragma HLS INLINE off

  for (ap_uint<6> i_t = 0; i_t < 32; i_t += 32)
    for (ap_uint<6> j_t = 0; j_t < 32; j_t += 32)
      for (ap_uint<6> k_t = 0; k_t < 32; k_t += 32)
      {
        for (int la_counter = 0; la_counter < U1_LOCAL_ACCUM_NUM; la_counter++)
          for (int local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
            U1_Data1PEChannelType fifo1_in_data;
            fifo1_in_data = fifo1_in.read();
            fifo1_out.write(fifo1_in_data);
            fifo1_local.write(fifo1_in_data);
          }
      }
}

void U1_op1_transfer_wrapper(
  stream<U1_Data1PEChannelType> &fifo1_in,
  stream<U1_Data1PEChannelType> &fifo1_out,
  stream<U1_Data1PEChannelType> &fifo1_local
){
  U1_op1_transfer(
    fifo1_in,
    fifo1_out,
    fifo1_local);
}

void U1_op1_transfer_last(
  stream<U1_Data1PEChannelType> &fifo1_in,
  stream<U1_Data1PEChannelType> &fifo1_local
){
#pragma HLS DATA_PACK variable=fifo1_in
#pragma HLS DATA_PACK variable=fifo1_local
#pragma HLS INLINE off

  for (ap_uint<6> i_t = 0; i_t < 32; i_t += 32)
    for (ap_uint<6> j_t = 0; j_t < 32; j_t += 32)
      for (ap_uint<6> k_t = 0; k_t < 32; k_t += 32)
      {
        for (int la_counter = 0; la_counter < U1_LOCAL_ACCUM_NUM; la_counter++)
          for (int local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
            U1_Data1PEChannelType fifo1_in_data;
            fifo1_in_data = fifo1_in.read();
            fifo1_local.write(fifo1_in_data);
          }
      }
}

void U1_op1_transfer_last_wrapper(
  stream<U1_Data1PEChannelType> &fifo1_in,
  stream<U1_Data1PEChannelType> &fifo1_local
){
  U1_op1_transfer_last(
    fifo1_in,
    fifo1_local);
}

void U1_compute(
  stream<U1_Data0PEChannelType> &fifo0_local,
  stream<U1_Data1PEChannelType> &fifo1_local,
  stream<U1_Data2PEChannelType> &fifo2_local
){
#pragma HLS DATA_PACK variable=fifo0_local
#pragma HLS DATA_PACK variable=fifo1_local
#pragma HLS INLINE off

  U1_data_t2 local_buffer[U1_LOCAL_REG_NUM];

  for (ap_uint<6> i_t = 0; i_t < 32; i_t += 32)
    for (ap_uint<6> j_t = 0; j_t < 32; j_t += 32)
      for (ap_uint<6> k_t = 0; k_t < 32; k_t += 32)
      {
        for (int la_counter = 0; la_counter < U1_LOCAL_ACCUM_NUM; la_counter++)
          for (int local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE inter false variable=local_buffer
            U1_Data0PEChannelType fifo0_in_data;
            fifo0_in_data = fifo0_local.read();
            U1_Data1PEChannelType fifo1_in_data;
            fifo1_in_data = fifo1_local.read();
            bool init = fifo0_in_data.new_pair;
            bool last = fifo0_in_data.last_pair;
            U1_PE_MAC(fifo0_in_data.data, fifo1_in_data.data, &local_buffer[local_reg_id], (init == 1 && la_counter == 0)? 1:0);
            if (la_counter == U1_LOCAL_ACCUM_NUM - 1 && last){
              fifo2_local.write(U1_Data2PEChannelType(local_buffer[local_reg_id]));
            }
          }
      }
}

void U1_compute_wrapper(
  stream<U1_Data0PEChannelType> &fifo0_local,
  stream<U1_Data1PEChannelType> &fifo1_local,
  stream<U1_Data2PEChannelType> &fifo2_local
){
  U1_compute(
    fifo0_local,
    fifo1_local,
    fifo2_local
  );
}

void U1_res_transfer(
  stream<U1_Data2PEChannelType> &fifo2_local,
  stream<U1_Data2PEChannelType> &fifo2_in,
  stream<U1_Data2PEChannelType> &fifo2_out,
  ap_uint<2> pe_row_id,
  ap_uint<2> pe_col_id
){
#pragma HLS DATA_PACK variable=fifo2_in
#pragma HLS DATA_PACK variable=fifo2_out
#pragma HLS INLINE off

  U1_data_t2 local_buffer[U1_LOCAL_REG_NUM];

  for (ap_uint<6> i_t = 0; i_t < 32; i_t += 32)
    for (ap_uint<6> j_t = 0; j_t < 32; j_t += 32)
      for (ap_uint<6> k_t = 0; k_t < 32; k_t += 32)
      {
        if (k_t == U1_K - U1_K_T){
          for (int local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
            U1_Data2PEChannelType fifo2_local_data = fifo2_local.read();
            local_buffer[local_reg_id] = fifo2_local_data.data;
          }

          int transfer_num = U1_LOCAL_REG_NUM * (pe_row_id + 1);
          for (int inter_counter = 0; inter_counter < transfer_num; inter_counter++){
#pragma HLS PIPELINE II=1
            fifo2_out.write(U1_Data2PEChannelType(local_buffer[inter_counter % U1_LOCAL_REG_NUM]));
            if (inter_counter < transfer_num - U1_LOCAL_REG_NUM){
              U1_Data2PEChannelType fifo2_in_data = fifo2_in.read();
              local_buffer[inter_counter % U1_LOCAL_REG_NUM] = fifo2_in_data.data;
            }
          }
        }
      }
}

void U1_res_transfer_wrapper(
  stream<U1_Data2PEChannelType> &fifo2_local,
  stream<U1_Data2PEChannelType> &fifo2_in,
  stream<U1_Data2PEChannelType> &fifo2_out,
  ap_uint<2> pe_row_id,
  ap_uint<2> pe_col_id
){
  U1_res_transfer(
    fifo2_local,
    fifo2_in,
    fifo2_out,
    pe_row_id,
    pe_col_id);
}

void U1_res_transfer_first(
  stream<U1_Data2PEChannelType> &fifo2_local,
  stream<U1_Data2PEChannelType> &fifo2_out,
  ap_uint<2> pe_row_id,
  ap_uint<2> pe_col_id
){
#pragma HLS DATA_PACK variable=fifo2_out
#pragma HLS INLINE off

  U1_data_t2 local_buffer[U1_LOCAL_REG_NUM];

  for (ap_uint<6> i_t = 0; i_t < 32; i_t += 32)
    for (ap_uint<6> j_t = 0; j_t < 32; j_t += 32)
      for (ap_uint<6> k_t = 0; k_t < 32; k_t += 32)
      {
        if (k_t == U1_K - U1_K_T){
          for (int local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
            U1_Data2PEChannelType fifo2_local_data = fifo2_local.read();
            local_buffer[local_reg_id] = fifo2_local_data.data;
          }

          int transfer_num = U1_LOCAL_REG_NUM * (pe_row_id + 1);
          for (int inter_counter = 0; inter_counter < transfer_num; inter_counter++){
#pragma HLS PIPELINE II=1
            fifo2_out.write(U1_Data2PEChannelType(local_buffer[inter_counter % U1_LOCAL_REG_NUM]));
          }
        }
      }
}

void U1_res_transfer_first_wrapper(
  stream<U1_Data2PEChannelType> &fifo2_local,
  stream<U1_Data2PEChannelType> &fifo2_out,
  ap_uint<2> pe_row_id,
  ap_uint<2> pe_col_id
){
  U1_res_transfer_first(
    fifo2_local,
    fifo2_out,
    pe_row_id,
    pe_col_id);
}

void U1_kernel(
  U1_bus_t0* A,
  U1_bus_t1* B,
  U1_bus_t2* C,
  bool init,
  unsigned int FILTER_S
){
#pragma HLS DATAFLOW

  // FIFOs
  stream<U1_Data0PEChannelType> fifo0_feed0_0;
#pragma HLS STREAM variable=fifo0_feed0_0 depth=2
  stream<U1_Data0PEChannelType> fifo0_feed0_1;
#pragma HLS STREAM variable=fifo0_feed0_1 depth=2
  stream<U1_Data0PEChannelType> fifo0_feed0_2;
#pragma HLS STREAM variable=fifo0_feed0_2 depth=2
  stream<U1_Data0PEChannelType> fifo0_feed1_0;
#pragma HLS STREAM variable=fifo0_feed1_0 depth=2
  stream<U1_Data0PEChannelType> fifo0_feed1_1;
#pragma HLS STREAM variable=fifo0_feed1_1 depth=2
  stream<U1_Data0PEChannelType> fifo0_feed1_2;
#pragma HLS STREAM variable=fifo0_feed1_2 depth=2
  stream<U1_Data0PEChannelType> fifo0_feed2_0;
#pragma HLS STREAM variable=fifo0_feed2_0 depth=2
  stream<U1_Data0PEChannelType> fifo0_feed2_1;
#pragma HLS STREAM variable=fifo0_feed2_1 depth=2
  stream<U1_Data0PEChannelType> fifo0_feed2_2;
#pragma HLS STREAM variable=fifo0_feed2_2 depth=2
  stream<U1_Data1PEChannelType> fifo1_feed0_0;
#pragma HLS STREAM variable=fifo1_feed0_0 depth=2
  stream<U1_Data1PEChannelType> fifo1_feed0_1;
#pragma HLS STREAM variable=fifo1_feed0_1 depth=2
  stream<U1_Data1PEChannelType> fifo1_feed0_2;
#pragma HLS STREAM variable=fifo1_feed0_2 depth=2
  stream<U1_Data1PEChannelType> fifo1_feed1_0;
#pragma HLS STREAM variable=fifo1_feed1_0 depth=2
  stream<U1_Data1PEChannelType> fifo1_feed1_1;
#pragma HLS STREAM variable=fifo1_feed1_1 depth=2
  stream<U1_Data1PEChannelType> fifo1_feed1_2;
#pragma HLS STREAM variable=fifo1_feed1_2 depth=2
  stream<U1_Data1PEChannelType> fifo1_feed2_0;
#pragma HLS STREAM variable=fifo1_feed2_0 depth=2
  stream<U1_Data1PEChannelType> fifo1_feed2_1;
#pragma HLS STREAM variable=fifo1_feed2_1 depth=2
  stream<U1_Data1PEChannelType> fifo1_feed2_2;
#pragma HLS STREAM variable=fifo1_feed2_2 depth=2
  stream<U1_Data2PEChannelType> fifo2_collect0_0;
#pragma HLS STREAM variable=fifo2_collect0_0 depth=2
  stream<U1_Data2PEChannelType> fifo2_collect0_1;
#pragma HLS STREAM variable=fifo2_collect0_1 depth=2
  stream<U1_Data2PEChannelType> fifo2_collect0_2;
#pragma HLS STREAM variable=fifo2_collect0_2 depth=2
  stream<U1_Data2PEChannelType> fifo2_collect1_0;
#pragma HLS STREAM variable=fifo2_collect1_0 depth=2
  stream<U1_Data2PEChannelType> fifo2_collect1_1;
#pragma HLS STREAM variable=fifo2_collect1_1 depth=2
  stream<U1_Data2PEChannelType> fifo2_collect1_2;
#pragma HLS STREAM variable=fifo2_collect1_2 depth=2
  stream<U1_Data2PEChannelType> fifo2_collect2_0;
#pragma HLS STREAM variable=fifo2_collect2_0 depth=2
  stream<U1_Data2PEChannelType> fifo2_collect2_1;
#pragma HLS STREAM variable=fifo2_collect2_1 depth=2
  stream<U1_Data2PEChannelType> fifo2_collect2_2;
#pragma HLS STREAM variable=fifo2_collect2_2 depth=2
  stream<U1_Data0TransferChannelType> fifo0_transfer0;
#pragma HLS STREAM variable=fifo0_transfer0 depth=2
  stream<U1_Data0TransferChannelType> fifo0_transfer1;
#pragma HLS STREAM variable=fifo0_transfer1 depth=2
  stream<U1_Data0TransferChannelType> fifo0_transfer2;
#pragma HLS STREAM variable=fifo0_transfer2 depth=2
  stream<U1_Data1TransferChannelType> fifo1_transfer0;
#pragma HLS STREAM variable=fifo1_transfer0 depth=2
  stream<U1_Data1TransferChannelType> fifo1_transfer1;
#pragma HLS STREAM variable=fifo1_transfer1 depth=2
  stream<U1_Data1TransferChannelType> fifo1_transfer2;
#pragma HLS STREAM variable=fifo1_transfer2 depth=2
  stream<U1_Data2TransferChannelType> fifo2_transfer0;
#pragma HLS STREAM variable=fifo2_transfer0 depth=2
  stream<U1_Data2TransferChannelType> fifo2_transfer1;
#pragma HLS STREAM variable=fifo2_transfer1 depth=2
  stream<U1_Data2TransferChannelType> fifo2_transfer2;
#pragma HLS STREAM variable=fifo2_transfer2 depth=2
  stream<U1_Data0PEChannelType> PE0_0_fifo0_local;
#pragma HLS STREAM variable=PE0_0_fifo0_local depth=2
  stream<U1_Data1PEChannelType> PE0_0_fifo1_local;
#pragma HLS STREAM variable=PE0_0_fifo1_local depth=2
  stream<U1_Data2PEChannelType> PE0_0_fifo2_local;
#pragma HLS STREAM variable=PE0_0_fifo2_local depth=2
  stream<U1_Data0PEChannelType> PE0_1_fifo0_local;
#pragma HLS STREAM variable=PE0_1_fifo0_local depth=2
  stream<U1_Data1PEChannelType> PE0_1_fifo1_local;
#pragma HLS STREAM variable=PE0_1_fifo1_local depth=2
  stream<U1_Data2PEChannelType> PE0_1_fifo2_local;
#pragma HLS STREAM variable=PE0_1_fifo2_local depth=2
  stream<U1_Data0PEChannelType> PE1_0_fifo0_local;
#pragma HLS STREAM variable=PE1_0_fifo0_local depth=2
  stream<U1_Data1PEChannelType> PE1_0_fifo1_local;
#pragma HLS STREAM variable=PE1_0_fifo1_local depth=2
  stream<U1_Data2PEChannelType> PE1_0_fifo2_local;
#pragma HLS STREAM variable=PE1_0_fifo2_local depth=2
  stream<U1_Data0PEChannelType> PE1_1_fifo0_local;
#pragma HLS STREAM variable=PE1_1_fifo0_local depth=2
  stream<U1_Data1PEChannelType> PE1_1_fifo1_local;
#pragma HLS STREAM variable=PE1_1_fifo1_local depth=2
  stream<U1_Data2PEChannelType> PE1_1_fifo2_local;
#pragma HLS STREAM variable=PE1_1_fifo2_local depth=2

  // modules
  U1_DataFeed0Head(
    A, init, FILTER_S,
    fifo0_transfer0
  );

  U1_DataFeed0Engine0_wrapper(
    fifo0_transfer0,
    fifo0_transfer1,
    fifo0_feed0_0,
    0
  );

  U1_DataFeed0EngineLast(
    fifo0_transfer1,
    fifo0_feed0_1,
    1
  );

  U1_DataFeed1Head(
    B, init, FILTER_S,
    fifo1_transfer0
  );

  U1_DataFeed1Engine0_wrapper(
    fifo1_transfer0,
    fifo1_transfer1,
    fifo1_feed0_0,
    0
  );

  U1_DataFeed1EngineLast(
    fifo1_transfer1,
    fifo1_feed1_0,
    1
  );

  // PE modules
  U1_op0_transfer_wrapper(
    fifo0_feed0_0,
    fifo0_feed1_0,
    PE0_0_fifo0_local);

  U1_op1_transfer_wrapper(
    fifo1_feed0_0,
    fifo1_feed0_1,
    PE0_0_fifo1_local);

  U1_compute_wrapper(
    PE0_0_fifo0_local,
    PE0_0_fifo1_local,
    PE0_0_fifo2_local
  );

  U1_res_transfer_first_wrapper(
    PE0_0_fifo2_local,
    fifo2_collect0_0,
    0,
    0
  );

  U1_op0_transfer_wrapper(
    fifo0_feed0_1,
    fifo0_feed1_1,
    PE0_1_fifo0_local);

  U1_op1_transfer_last_wrapper(
    fifo1_feed0_1,
    PE0_1_fifo1_local);

  U1_compute_wrapper(
    PE0_1_fifo0_local,
    PE0_1_fifo1_local,
    PE0_1_fifo2_local
  );

  U1_res_transfer_first_wrapper(
    PE0_1_fifo2_local,
    fifo2_collect0_1,
    0,
    1
  );

  U1_op0_transfer_last_wrapper(
    fifo0_feed1_0,
    PE1_0_fifo0_local);

  U1_op1_transfer_wrapper(
    fifo1_feed1_0,
    fifo1_feed1_1,
    PE1_0_fifo1_local);

  U1_compute_wrapper(
    PE1_0_fifo0_local,
    PE1_0_fifo1_local,
    PE1_0_fifo2_local
  );

  U1_res_transfer_wrapper(
    PE1_0_fifo2_local,
    fifo2_collect0_0,
    fifo2_collect1_0,
    1,
    0
  );

  U1_op0_transfer_last_wrapper(
    fifo0_feed1_1,
    PE1_1_fifo0_local);

  U1_op1_transfer_last_wrapper(
    fifo1_feed1_1,
    PE1_1_fifo1_local);

  U1_compute_wrapper(
    PE1_1_fifo0_local,
    PE1_1_fifo1_local,
    PE1_1_fifo2_local
  );

  U1_res_transfer_wrapper(
    PE1_1_fifo2_local,
    fifo2_collect0_1,
    fifo2_collect1_1,
    1,
    1
  );

  U1_DataCollect2EngineLast(
    fifo2_transfer0,
    fifo2_collect1_0,
    1);

  U1_DataCollect2Engine0_wrapper(
    fifo2_transfer0,
    fifo2_transfer1,
    fifo2_collect1_1,
    0);

  U1_DataCollect2Head(
    C,
    fifo2_transfer1
  );

}
