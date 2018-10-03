# Taken from facebook/TensorComprehensions

# - Try to find MIOpen
#
# The following variables are optionally searched for defaults
#  MIOPEN_ROOT_DIR:            Base directory where all MIOpen components are found
#
# The following are set after configuration is done:
#  MIOPEN_FOUND
#  MIOPEN_INCLUDE_DIRS
#  MIOPEN_LIBRARIES
#  MIOPEN_LIBRARY_DIRS


set(MIOPEN_ROOT_DIR "" CACHE PATH "Folder contains ROCM MIOpen")
set(MIOPEN_VERSION "" CACHE STRING "version number")

find_path(MIOPEN_INCLUDE_DIR miopen_kernels.h
    HINTS ${HIP_PATH} /opt/rocm/
    PATH_SUFFIXES miopen/include include)

find_path(MIOPEN_LIBRARY_DIR libMIOpen.so
HINTS ${HIP_PATH} /opt/rocm/
PATH_SUFFIXES miopen/lib lib lib64)

find_library(MIOPEN_LIBRARY MIOpen
    HINTS ${HIP_PATH} /opt/rocm/
    PATH_SUFFIXES lib lib64 miopen/lib)

if(MIOPEN_FOUND)
    # get MIOpen version
  file(READ ${MIOPEN_INCLUDE_DIR}/miopen/version.h MIOPEN_HEADER_CONTENTS)
    string(REGEX MATCH "define MIOPEN_VERSION_MAJOR * +([0-9]+)"
                 MIOPEN_VERSION_MAJOR "${MIOPEN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define MIOPEN_VERSION_MAJOR * +([0-9]+)" "\\1"
                 MIOPEN_VERSION_MAJOR "${MIOPEN_VERSION_MAJOR}")
    string(REGEX MATCH "define MIOPEN_VERSION_MINOR * +([0-9]+)"
                 MIOPEN_VERSION_MINOR "${MIOPEN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define MIOPEN_VERSION_MINOR * +([0-9]+)" "\\1"
                 MIOPEN_VERSION_MINOR "${MIOPEN_VERSION_MINOR}")
    string(REGEX MATCH "define MIOPEN_VERSION_PATCH * +([0-9]+)"
                 MIOPEN_VERSION_PATCH "${MIOPEN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define MIOPEN_VERSION_PATCH * +([0-9]+)" "\\1"
                 MIOPEN_VERSION_PATCH "${MIOPEN_VERSION_PATCH}")
  # Assemble MIOpen version
  if(MIOPEN_VERSION_MAJOR)
    set(MIOPEN_VERSION "?")
  else()
    set(MIOPEN_VERSION "${MIOPEN_VERSION_MAJOR}.${MIOPEN_VERSION_MINOR}.${MIOPEN_VERSION_PATCH}")
  endif()

  set(MIOPEN_INCLUDE_DIRS ${MIOPEN_INCLUDE_DIR})
  set(MIOPEN_LIBRARIES ${MIOPEN_LIBRARY})
  message(STATUS "Found MIOpen: v${MIOPEN_VERSION}  (include: ${MIOPEN_INCLUDE_DIR}, library: ${MIOPEN_LIBRARY})")
  mark_as_advanced(MIOPEN_ROOT_DIR MIOPEN_LIBRARY MIOPEN_INCLUDE_DIR)
endif()