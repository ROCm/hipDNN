HIP_PATH ?= /opt/rocm/hip
HCC_PATH ?= /opt/rocm/hcc
HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIP_INCLUDE = -I${HIP_PATH}/include -I${HCC_PATH}/include
BUILD_DIR ?= build

HIPCC = ${HIP_PATH}/bin/hipcc

CUDNN_INCLUDE_DIR = /usr/local/cudnn-6.0/cuda/include/
CUDNN_LIBDIR= /usr/local/cudnn-6.0/cuda/lib64/

CPPFLAGS =
CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)
$(info ${CPPFLAGS})
LDFLAGS = -lm -lcudnn

ifeq (${HIP_PLATFORM}, nvcc)
   CPPFLAGS += -arch=compute_52 -Wno-deprecated-gpu-targets
endif

SOURCEDIR = src/nvcc_detail
SOURCESLIST := $(shell find $(SOURCEDIR) -name '*.cpp')
SOURCES = $(notdir $(SOURCESLIST))
 
OBJECTS = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(SOURCES)))
TEST_EXE = ${BUILD_DIR}/testexe

.PHONY: all clean run 

all: ${TEST_EXE}

${TEST_EXE}: ${OBJECTS}
	${HIPCC} -L${CUDNN_LIBDIR} -Wno-deprecated-gpu-targets -shared -o ${BUILD_DIR}/libhipDNN.so ${LDFLAGS} ${OBJECTS}

${BUILD_DIR}/%.o: src/nvcc_detail/%.cpp
	mkdir -p ${BUILD_DIR}
	${HIPCC} ${HIP_INCLUDE} -I./include -I${CUDNN_INCLUDE_DIR} ${CPPFLAGS} -c -o $@ $< -Xcompiler "-fPIC"

run: 
	HCC_LAZYINIT=ON ${TEST_EXE}

clean:
	rm -rf ${BUILD_DIR}
