HIP_PATH ?= /opt/rocm/hip
HCC_PATH ?= /opt/rocm/hcc
HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIP_INCLUDE = -I${HIP_PATH}/include
BUILD_DIR ?= build

ifndef INSTALL_DIR
	INSTALL_DIR = /opt/rocm/hipDNN
endif

ifndef MIOPEN_PATH
       MIOPEN_PATH=/opt/rocm/miopen
endif


HIPCC = ${HIP_PATH}/bin/hipcc

CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)
ifeq (${HIP_PLATFORM}, nvcc)
INCLUDE_DIR = /usr/local/cudnn-7.0/cuda/include/

LIB_DIR= /usr/local/cudnn-7.0/cuda/lib64/

LDFLAGS = -lm -lcudnn

CPPFLAGS = -arch=compute_52 -Wno-deprecated-gpu-targets -Xcompiler

SOURCEDIR = src/nvcc_detail
endif

ifeq (${HIP_PLATFORM}, hcc)
INCLUDE_DIR=${MIOPEN_PATH}/include/

LIB_DIR=${MIOPEN_PATH}lib

LDFLAGS = -lm -lMIOpen

SOURCEDIR = src/hcc_detail

HIP_INCLUDE += -I${HCC_PATH}/include
endif

COMMONFLAGS = -fPIC

SOURCESLIST := $(shell find $(SOURCEDIR) -name '*.cpp')
SOURCES = $(notdir $(SOURCESLIST)) 
OBJECTS = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(SOURCES)))

.PHONY: all clean 

all: HIPDNN_SO

HIPDNN_SO: ${OBJECTS}
	sudo mkdir -p ${INSTALL_DIR}
	sudo cp -r ./include ${INSTALL_DIR}/
	mkdir -p lib
	${HIPCC} ${CPPFLAGS} -L${LIB_DIR} -shared -o lib/libhipDNN.so ${LDFLAGS} ${OBJECTS}
	sudo cp -r ./lib ${INSTALL_DIR}/

${BUILD_DIR}/%.o: ${SOURCEDIR}/%.cpp
	mkdir -p ${BUILD_DIR}
	${HIPCC} ${HIP_INCLUDE} -I./include -I${INCLUDE_DIR} ${CPPFLAGS} ${COMMONFLAGS} -c -o $@ $<

clean:
	rm -rf ${BUILD_DIR}
	sudo rm -rf ${INSTALL_DIR}
