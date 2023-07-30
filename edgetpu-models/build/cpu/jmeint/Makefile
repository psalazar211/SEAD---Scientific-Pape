CC			:= aarch64-linux-gnu-g++
LD			:= aarch64-linux-gnu-g++

#include ../config.mk

CFLAGS		:= -Wall #-lfann -lboost_regex -ltensorflow-lite -std=c++17
LFLAGS		:= -L. #-lfann -lboost_regex -ltensorflow-lite
HEADERS     := src
INCLUDE 	:= -I../../../tensorflow_pinned -I../../../tflite_pinned/flatbuffers/include -I.. -I. -I${HEADERS} 

LIB			:=  \
	-L../../../tflite_pinned -ltensorflowlite_c \
	-L../../../tflite_pinned/tensorflow-lite -ltensorflow-lite \
	-L../../../tflite_build/_deps/ruy-build/ruy -lruy_allocator -lruy_context_get_ctx -lruy_frontend -lruy_kernel_arm -lruy_pack_arm -lruy_apply_multiplier -lruy_context -lruy_cpuinfo -lruy_ctx -lruy_system_aligned_alloc -lruy_prepare_packed_matrices -lruy_trmul -lruy_prepacked_cache -lruy_thread_pool -lruy_tune -lruy_block_map -lruy_blocking_counter -lruy_wait -lruy_denormal \
	-L../../../tflite_pinned/_deps/xnnpack-build/ -lXNNPACK \
	-L../../../tflite_pinned/pthreadpool -lpthreadpool \
	-L../../../tflite_pinned/_deps/farmhash-build -lfarmhash \
        -L../../../tflite_pinned/_deps/fft2d-build -lfft2d_fftsg2d -lfft2d_fftsg \
	-L../../../tflite_pinned/_deps/flatbuffers-build/ -lflatbuffers \
	-L../../../tflite_build/_deps/cpuinfo-build -lcpuinfo \
	-ldl -lpthread \
        -L../.. -l:libusb-1.0.so.0 -l:libudev.so.1 \
	-L.. -l:libedgetpu.so.1

MODULE		:= jmeint.out

CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))


.PHONY: all clean
	
all: DIR $(MODULE)

DIR:
	@echo ${CPP_FILES}
	@echo ${OBJ_FILES}
	if [ ! -d "./bin" ];then 	\
		mkdir bin;				\
	fi
	if [ ! -d "./obj" ];then 	\
		mkdir obj;				\
	fi

$(MODULE): $(OBJ_FILES)
	$(LD) $^ $(LIB) $(LFLAGS) -o bin/$@

obj/%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

clean:
	@rm -rf *.o
	@rm -rf *.d
	@rm -rf *.out
	@rm -rf bin
	@rm -rf obj
