cmake_minimum_required(VERSION 3.18)
project(campusgpt_native LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# optional CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(HAVE_CUDA TRUE)
else()
    set(HAVE_CUDA FALSE)
    message(STATUS "CUDA not found - quantization kernels will not be built")
endif()

# tokenizer library
add_library(fast_tokenizer SHARED
    src/tokenizer/fast_tokenizer.c
)

target_include_directories(fast_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer
)

# quantization library (if CUDA available)
if(HAVE_CUDA)
    add_library(nf4_quant SHARED
        src/quantization/nf4_kernels.cu
        src/quantization/quantize_wrapper.cpp
    )

    target_include_directories(nf4_quant PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization
    )

    set_target_properties(nf4_quant PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# install targets
install(TARGETS fast_tokenizer
    LIBRARY DESTINATION lib
)

if(HAVE_CUDA)
    install(TARGETS nf4_quant
        LIBRARY DESTINATION lib
    )
endif()
