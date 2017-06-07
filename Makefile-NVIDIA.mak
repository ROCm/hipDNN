HIP_PATH ?= ~/opt/rocm/hip
HCC_PATH ?= ~/opt/rocm/hcc
HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIP_INCLUDE = -I${HIP_PATH}/include -I${HCC_PATH}/include
BUILD_DIR ?= build

HIPCC = ${HIP_PATH}/bin/hipcc

CUDNN_INCLUDE_DIR = /usr/local/cuda-8.0/include
CUDNN_LIBDIR= /usr/local/cuda-8.0/lib64

CPPFLAGS =
CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)

LDFLAGS = -lm -lcudnn

ifeq (${HIP_PLATFORM}, nvcc)
    CPPFLAGS += -arch=compute_20
endif

SOURCES = $(wildcard *.cpp)
OBJECTS = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(SOURCES)))
TEST_EXE = ${BUILD_DIR}/testexe

.PHONY: all clean run 

all: ${TEST_EXE}

${TEST_EXE}: ${OBJECTS}
	${HIPCC} ${LDFLAGS} -o ${TEST_EXE} ${OBJECTS}

${BUILD_DIR}/%.o: %.cpp Makefile
	mkdir -p ${BUILD_DIR}
	${HIPCC} ${HIP_INCLUDE} -I. ${CPPFLAGS} -c -o $@ $<  

run: 
	HCC_LAZYINIT=ON ${TEST_EXE}

clean:
	rm -rf ${BUILD_DIR}