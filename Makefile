#==========================================================================

# user must update top.cpp to be systolic_array_kernel.cpp and change the kernel function name to systolic_array_kernel

# Extract Vivado HLS include path
VHLS_PATH := $(dir $(shell which vivado_hls))/..
VHLS_INC ?= ${VHLS_PATH}/include

COMMON_REPO := ../../..

# wide Memory Access Application
include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk

# dmmm Host Application
main_SRCS=./main.cpp $(xcl2_SRCS)
main_HDRS=./gcn_layer.hpp $(xcl2_HDRS)
main_CXXFLAGS=-I./ $(xcl2_CXXFLAGS) $(opencl_CXXFLAGS) -I${VHLS_INC} -DK_CONST=3
main_LDFLAGS=$(opencl_LDFLAGS)

# dmmm Kernels
#systolic_array_kernel_HDRS=./common_header_U1.h
#systolic_array_kernel_SRCS=./systolic_array_kernel.cpp ./2DDataCollect_U1.cpp ./2DDataFeed_U1.cpp ./2DDataFeedCollect_U1.cpp ./2DPE_U1.cpp
systolic_array_kernel_SRCS=./systolic_array_kernel.cpp
systolic_array_kernel_CLFLAGS=-I./ -k systolic_array_kernel


EXES=main
XCLBINS=systolic_array_kernel

XOS=systolic_array_kernel

systolic_array_kernel_XOS=systolic_array_kernel

# check
check_EXE=main
check_XCLBINS=systolic_array_kernel

CHECKS=check

include $(COMMON_REPO)/utility/rules.mk

