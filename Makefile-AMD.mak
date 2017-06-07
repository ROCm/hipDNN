HIP_PATH ?= /opt/rocm/hip
HCC_PATH ?= /opt/rocm/hcc
HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIP_INCLUDE = -I${HIP_PATH}/include -I${HCC_PATH}/include
BUILD_DIR ?= build

HIPCC = ${HIP_PATH}/bin/hipcc

MIOPEN_INCLUDE_DIR = /home/hgaspara/github/MLOpen/include 
MIOPEN_INCLUDE_DIR2 = /home/hgaspara/github/MLOpen/build/include
MIOPEN_LIB_DIR = /home/hgaspara/github/MLOpen/build/lib

CPPFLAGS = -DMIOPEN_BACKEND_HIP
CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)

LDFLAGS = -lm -lMIOpen


SOURCES = $(wildcard *.cpp)
OBJECTS = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(SOURCES)))
TEST_EXE = ${BUILD_DIR}/testexe

.PHONY: all clean run 

all: ${TEST_EXE}

${TEST_EXE}: ${OBJECTS}
	${HIPCC} ${LDFLAGS} -L${MIOPEN_LIB_DIR} -o ${TEST_EXE} ${OBJECTS}

${BUILD_DIR}/%.o: %.cpp Makefile
	mkdir -p ${BUILD_DIR}
	${HIPCC} ${HIP_INCLUDE} -I. -I${MIOPEN_INCLUDE_DIR} -I${MIOPEN_INCLUDE_DIR2} ${CPPFLAGS} -c -o $@ $<  

run: 
	HCC_LAZYINIT=ON ${TEST_EXE}

clean:
	rm -rf ${BUILD_DIR}