/**
 *  This file is automatically generated by PolySA CodeGen.
 *  Version: 1.0
 *  Authos: Jie Wang
 */

#include "common_header_U1.h"

void U1_Data2WriteData0(
  U1_data_t2 buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE],
  stream<U1_Data2TransferChannelType> &fifo_transfer_in,
  stream<U1_Data2TransferChannelType> &fifo_transfer_out,
  unsigned int engine_id
){
#pragma HLS INLINE off

  bool LAST_ENGINE = (engine_id == 2 / U1_DATA2_FC_SPLIT_FACTOR - 1);

  ap_uint<15> transfer_counter = 0;
  ap_uint<15> LOCAL_TRANSFER_SIZE = U1_DATA2_BUF_SIZE * (2 / U1_DATA2_FC_SPLIT_FACTOR - engine_id) * U1_DATA2_FC_GROUP_FACTOR;

  bool more_to_read_from_buffer = true;
  bool more_to_collect_from_sys_arr = true;
  ap_uint<14> buffer_read_counter = 0;
  ap_uint<1> buffer_gs_id = 0;
  while(more_to_read_from_buffer){
#pragma HLS PIPELINE II=1
    bool data_is_from_local_buffer = LAST_ENGINE || (!LAST_ENGINE && (transfer_counter < U1_DATA2_BUF_SIZE * U1_DATA2_FC_GROUP_FACTOR / U1_DATA2_FC_SIMD_FACTOR));
    bool data_is_from_external_buffer = !LAST_ENGINE && (transfer_counter >= U1_DATA2_BUF_SIZE * U1_DATA2_FC_GROUP_FACTOR / U1_DATA2_FC_SIMD_FACTOR);

    U1_Data2TransferChannelType data_write_to_fifo;

    if (data_is_from_external_buffer){
      data_write_to_fifo = fifo_transfer_in.read();
    } else {
      U1_data_t2 data0 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 0];
      ap_uint<U1_DATA2_WIDTH> data0_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data0);
      U1_data_t2 data1 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 1];
      ap_uint<U1_DATA2_WIDTH> data1_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data1);
      U1_data_t2 data2 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 2];
      ap_uint<U1_DATA2_WIDTH> data2_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data2);
      U1_data_t2 data3 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 3];
      ap_uint<U1_DATA2_WIDTH> data3_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data3);
      U1_data_t2 data4 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 4];
      ap_uint<U1_DATA2_WIDTH> data4_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data4);
      U1_data_t2 data5 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 5];
      ap_uint<U1_DATA2_WIDTH> data5_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data5);
      U1_data_t2 data6 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 6];
      ap_uint<U1_DATA2_WIDTH> data6_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data6);
      U1_data_t2 data7 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 7];
      ap_uint<U1_DATA2_WIDTH> data7_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data7);
      ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> pack_data = (
        data7_cast,
        data6_cast,
        data5_cast,
        data4_cast,
        data3_cast,
        data2_cast,
        data1_cast,
        data0_cast
      );
      data_write_to_fifo.data = pack_data;
    }
    fifo_transfer_out.write(data_write_to_fifo);
    if (data_is_from_local_buffer){
      buffer_read_counter++;
      if (buffer_read_counter == U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR){
        buffer_read_counter = 0;
        buffer_gs_id++;
        if (buffer_gs_id == U1_DATA2_FC_GROUP_FACTOR){
          buffer_gs_id = 0;
        }
      }
    }

    transfer_counter++;
    if (transfer_counter == LOCAL_TRANSFER_SIZE / U1_DATA2_FC_SIMD_FACTOR){
      transfer_counter = 0;
      more_to_read_from_buffer = false;
    }
  }

}

void U1_Data2WriteDataLast(
  U1_data_t2 buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE],
  stream<U1_Data2TransferChannelType> &fifo_transfer_out,
  unsigned int engine_id
){
#pragma HLS INLINE off

  bool LAST_ENGINE = (engine_id == 2 / U1_DATA2_FC_SPLIT_FACTOR - 1);

  ap_uint<15> transfer_counter = 0;
  ap_uint<15> LOCAL_TRANSFER_SIZE = U1_DATA2_BUF_SIZE * (2 / U1_DATA2_FC_SPLIT_FACTOR - engine_id) * U1_DATA2_FC_GROUP_FACTOR;

  bool more_to_read_from_buffer = true;
  bool more_to_collect_from_sys_arr = true;
  ap_uint<14> buffer_read_counter = 0;
  ap_uint<1> buffer_gs_id = 0;
  while(more_to_read_from_buffer){
#pragma HLS PIPELINE II=1
    bool data_is_from_local_buffer = LAST_ENGINE || (!LAST_ENGINE && (transfer_counter < U1_DATA2_BUF_SIZE * U1_DATA2_FC_GROUP_FACTOR / U1_DATA2_FC_SIMD_FACTOR));
    bool data_is_from_external_buffer = !LAST_ENGINE && (transfer_counter >= U1_DATA2_BUF_SIZE * U1_DATA2_FC_GROUP_FACTOR / U1_DATA2_FC_SIMD_FACTOR);

    U1_Data2TransferChannelType data_write_to_fifo;

    if (data_is_from_external_buffer){
    } else {
      U1_data_t2 data0 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 0];
      ap_uint<U1_DATA2_WIDTH> data0_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data0);
      U1_data_t2 data1 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 1];
      ap_uint<U1_DATA2_WIDTH> data1_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data1);
      U1_data_t2 data2 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 2];
      ap_uint<U1_DATA2_WIDTH> data2_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data2);
      U1_data_t2 data3 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 3];
      ap_uint<U1_DATA2_WIDTH> data3_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data3);
      U1_data_t2 data4 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 4];
      ap_uint<U1_DATA2_WIDTH> data4_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data4);
      U1_data_t2 data5 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 5];
      ap_uint<U1_DATA2_WIDTH> data5_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data5);
      U1_data_t2 data6 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 6];
      ap_uint<U1_DATA2_WIDTH> data6_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data6);
      U1_data_t2 data7 = buffer[U1_DATA2_FC_GROUP_FACTOR - 1 - buffer_gs_id][buffer_read_counter * U1_DATA2_FC_SIMD_FACTOR + 7];
      ap_uint<U1_DATA2_WIDTH> data7_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data7);
      ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> pack_data = (
        data7_cast,
        data6_cast,
        data5_cast,
        data4_cast,
        data3_cast,
        data2_cast,
        data1_cast,
        data0_cast
      );
      data_write_to_fifo.data = pack_data;
    }
    fifo_transfer_out.write(data_write_to_fifo);
    if (data_is_from_local_buffer){
      buffer_read_counter++;
      if (buffer_read_counter == U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR){
        buffer_read_counter = 0;
        buffer_gs_id++;
        if (buffer_gs_id == U1_DATA2_FC_GROUP_FACTOR){
          buffer_gs_id = 0;
        }
      }
    }

    transfer_counter++;
    if (transfer_counter == LOCAL_TRANSFER_SIZE / U1_DATA2_FC_SIMD_FACTOR){
      transfer_counter = 0;
      more_to_read_from_buffer = false;
    }
  }

}

void U1_Data2ReadData0(
  U1_data_t2 buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE],
  stream<U1_Data2PEChannelType> &fifo_collect_0
){
#pragma HLS INLINE off

  bool more_to_collect_from_sys_arr = true;
  ap_uint<1> buffer_gs_id = 0;
  ap_uint<14> buffer_read_counter = 0;
  ap_uint<7> c0_counter = 0;
  ap_uint<7> c1_counter = 0;
  ap_uint<2> c2_counter = 0;

  while(more_to_collect_from_sys_arr){
#pragma HLS PIPELINE II=1
    ap_uint<14> buffer_ind_to_collect_from_sys_arr = c1_counter * U1_SA_ROWS * U1_ROW_IL_FACTOR + ((2 - 1 - c2_counter) * U1_ROW_IL_FACTOR + c0_counter);

    U1_Data2PEChannelType data_to_collect_0;
    data_to_collect_0 = fifo_collect_0.read();
    buffer[0][buffer_ind_to_collect_from_sys_arr] = data_to_collect_0.data;

    // counter logic
    c0_counter++;
    if (c0_counter == 64){
      c0_counter = 0;
      c1_counter++;
      if (c1_counter == 64){
        c1_counter = 0;
        c2_counter++;
        if (c2_counter == 2){
          c2_counter = 0;
          more_to_collect_from_sys_arr = false;
        }
      }
    }
  }
}

void U1_DataCollect2Engine0(
  stream<U1_Data2TransferChannelType> &fifo_transfer_in,
  stream<U1_Data2TransferChannelType> &fifo_transfer_out,
  stream<U1_Data2PEChannelType> &fifo_collect_0,
  unsigned int engine_id
){
#pragma HLS DATA_PACK variable=fifo_transfer_in
#pragma HLS DATA_PACK variable=fifo_transfer_out
#pragma HLS DATA_PACK variable=fifo_collect_0
#pragma HLS INLINE off

  U1_data_t2 ping_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE];
  U1_data_t2 pong_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE];
#pragma HLS RESOURCE variable=ping_buffer core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=pong_buffer core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=2 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=2 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=1 complete
#pragma HLS DATA_PACK variable=ping_buffer
#pragma HLS DATA_PACK variable=pong_buffer

  unsigned int initial_round = 0;
  for (ap_uint<8> i_t = 0; i_t < 128; i_t += 128)
    for (ap_uint<8> j_t = 0; j_t < 128; j_t += 128)
      for (ap_uint<8> k_t = 0; k_t < 128; k_t += 128)
      {
        if (k_t == U1_K - U1_K_T){
          if (initial_round == 0){
            U1_Data2ReadData0(
              ping_buffer,
              fifo_collect_0
            );
          } else {
            if (initial_round % 2 == 1){
              U1_Data2ReadData0(
                pong_buffer,
                fifo_collect_0
              );
              U1_Data2WriteData0(ping_buffer, fifo_transfer_in, fifo_transfer_out, engine_id);
            } else {
              U1_Data2ReadData0(
                ping_buffer,
                fifo_collect_0
              );
              U1_Data2WriteData0(pong_buffer, fifo_transfer_in, fifo_transfer_out, engine_id);
            }
          }
          initial_round++;
        }
      }
  if (initial_round % 2 == 1){
    U1_Data2WriteData0(ping_buffer, fifo_transfer_in, fifo_transfer_out, engine_id);
  } else {
    U1_Data2WriteData0(pong_buffer, fifo_transfer_in, fifo_transfer_out, engine_id);
  }
}

void U1_DataCollect2Engine0_wrapper(
  stream<U1_Data2TransferChannelType> &fifo_transfer_in,
  stream<U1_Data2TransferChannelType> &fifo_transfer_out,
  stream<U1_Data2PEChannelType> &fifo_collect_0,
  unsigned int engine_id
){
  U1_DataCollect2Engine0(
    fifo_transfer_in,
    fifo_transfer_out,
    fifo_collect_0,
    engine_id
  );
}

void U1_DataCollect2EngineLast(
  stream<U1_Data2TransferChannelType> &fifo_transfer_out,
  stream<U1_Data2PEChannelType> &fifo_collect_0,
  unsigned int engine_id
){
#pragma HLS DATA_PACK variable=fifo_transfer_out
#pragma HLS DATA_PACK variable=fifo_collect_0
#pragma HLS INLINE off

  U1_data_t2 ping_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE];
  U1_data_t2 pong_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE];
#pragma HLS RESOURCE variable=ping_buffer core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=pong_buffer core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=2 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=2 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=1 complete
#pragma HLS DATA_PACK variable=ping_buffer
#pragma HLS DATA_PACK variable=pong_buffer

  unsigned int initial_round = 0;
  for (ap_uint<8> i_t = 0; i_t < 128; i_t += 128)
    for (ap_uint<8> j_t = 0; j_t < 128; j_t += 128)
      for (ap_uint<8> k_t = 0; k_t < 128; k_t += 128)
      {
        if (k_t == U1_K - U1_K_T){
          if (initial_round == 0){
            U1_Data2ReadData0(
              ping_buffer,
              fifo_collect_0
            );
          } else {
            if (initial_round % 2 == 1){
              U1_Data2ReadData0(
                pong_buffer,
                fifo_collect_0
              );
              U1_Data2WriteDataLast(ping_buffer, fifo_transfer_out, engine_id);
            } else {
              U1_Data2ReadData0(
                ping_buffer,
                fifo_collect_0
              );
              U1_Data2WriteDataLast(pong_buffer, fifo_transfer_out, engine_id);
            }
          }
          initial_round++;
        }
      }
  if (initial_round % 2 == 1){
    U1_Data2WriteDataLast(ping_buffer, fifo_transfer_out, engine_id);
  } else {
    U1_Data2WriteDataLast(pong_buffer, fifo_transfer_out, engine_id);
  }
}
