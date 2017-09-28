HIP_PATH ?= /opt/rocm/hip
HCC_PATH ?= /opt/rocm/hcc
HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIP_INCLUDE = -I${HIP_PATH}/include -I${HCC_PATH}/include
BUILD_DIR ?= build

HIPCC = ${HIP_PATH}/bin/hipcc

MIOPEN_INCLUDE_DIR = /opt/rocm/miopen/include/ 
MIOPEN_INCLUDE_DIR2 = /home/hgaspara/github/MLOpen/build/include
MIOPEN_LIB_DIR = /opt/rocm/miopen/lib

CPPFLAGS = -DMIOPEN_BACKEND_HIP
CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)

LDFLAGS = -lm -lMIOpen

SOURCEDIR = src/hcc_detail
SOURCESLIST = $(shell find ${SOURCEDIR} -name '*.cpp')
SOURCES = $(notdir ${SOURCESLIST})
OBJECTS = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(SOURCES)))
TEST_EXE = ${BUILD_DIR}/testexe

.PHONY: all clean run 

all: ${TEST_EXE}

${TEST_EXE}: ${OBJECTS}
	${HIPCC} ${LDFLAGS} -L${MIOPEN_LIB_DIR} -shared -o ${BUILD_DIR}/libhipDNN.so ${OBJECTS}

${BUILD_DIR}/%.o: src/hcc_detail/%.cpp
	mkdir -p ${BUILD_DIR}
	${HIPCC} ${HIP_INCLUDE} -I./include -I${MIOPEN_INCLUDE_DIR} -I${MIOPEN_INCLUDE_DIR2} ${CPPFLAGS} -fPIC -c -o $@ $<  

run: 
	HCC_LAZYINIT=ON ${TEST_EXE}

clean:
	rm -rf ${BUILD_DIR}
