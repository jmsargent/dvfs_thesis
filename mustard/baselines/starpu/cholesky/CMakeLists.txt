find_path(STARPU_INCLUDE_DIR starpu.h
    PATHS "$ENV{HOME}/.local/starpu/include/starpu/1.3"
    NO_DEFAULT_PATH)
find_library(STARPU_LIBRARY starpu-1.3
    PATHS "$ENV{HOME}/.local/starpu/lib"
    NO_DEFAULT_PATH)

if(NOT STARPU_INCLUDE_DIR OR NOT STARPU_LIBRARY)
    message(WARNING "StarPU not found. StarPU Cholesky baseline will be skipped.\n"
                    "  Expected install at $ENV{HOME}/.local/starpu (run build_starpu.sh first).")
    return()
endif()

find_package(BLAS REQUIRED)

add_executable(starpu_cholesky_tile_tag
    cholesky_tile_tag.c
    cholesky_models.c
    cholesky_kernels.c
    ../common/blas.c)

target_compile_definitions(starpu_cholesky_tile_tag PRIVATE STARPU_OPENBLAS)
target_include_directories(starpu_cholesky_tile_tag PRIVATE
    ${STARPU_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(starpu_cholesky_tile_tag PRIVATE
    ${STARPU_LIBRARY}
    ${BLAS_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::nvml
    pthread m)
