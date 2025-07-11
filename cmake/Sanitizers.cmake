# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Enable Address Sanitizer and set linker flags for security
message(STATUS "Building with Sanitizers: ${BUILD_ADDRESS_SANITIZER}")

if (BUILD_ADDRESS_SANITIZER)
    link_directories(${ROCM_LLVM_LIB_DIR}/clang/19/lib/linux)

    # Define sanitizer flags as variables for reuse
    set(SANITIZER_COMPILE_FLAGS 
        -fsanitize=address
        -fno-omit-frame-pointer
    )
    
    set(SANITIZER_LINK_FLAGS 
        -fsanitize=address
        -fno-omit-frame-pointer
        -shared-libasan
    )

    # Apply sanitizer flags globally (can be overridden per target)
    add_compile_options(${SANITIZER_COMPILE_FLAGS})
    add_link_options(${SANITIZER_LINK_FLAGS})
    
    # Add compile definition for conditional compilation
    add_compile_definitions(ADDRESS_SANITIZER)

    # Set environment variables for Address Sanitizer
    set(TEST_ENVIRONMENT 
        "ASAN_SYMBOLIZER_PATH=${CMAKE_SYMBOLIZER}" 
        "HSA_XNACK=1"
        #"ASAN_OPTIONS=halt_on_error=1:abort_on_error=1"
    )
    
    # Disable ROCFFT kernel cache for Address Sanitizer
    set(ROCFFT_KERNEL_CACHE_ENABLE off)
endif()
