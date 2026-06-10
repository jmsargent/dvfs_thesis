add_executable(test_timer test_timer.cu)
set_target_properties(test_timer PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(test_timer CUDA::cudart)

add_executable(test_freq_controller test_freq_controller.cu)
set_target_properties(test_freq_controller PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(test_freq_controller CUDA::cudart CUDA::nvml MPI::MPI_CXX)

add_executable(test_spanned_dag test_spanned_dag.cpp)
set_target_properties(test_spanned_dag PROPERTIES LINKER_LANGUAGE CXX)

add_executable(cholesky_mustard cholesky_mustard.cu sync_kernels.cu
        StridedPanel.h)
add_executable(pingpong_mustard pingpong_mustard.cu)
add_executable(latency_signal latency_signal.cu)
add_executable(latency_tile_pull latency_tile_pull.cu)
add_executable(retune_latency_hiding retune_latency_hiding.cu)
add_executable(lu_mustard lu_mustard.cu sync_kernels.cu
        StridedPanel.h)
add_executable(p_lu_mustard lu_mustard_p.cu)
add_executable(cg_mustard cg_mustard.cu)
add_executable(cuda_scheduler_ordering cuda_scheduler_ordering.cu sync_kernels.cu)


find_package(NVSHMEM REQUIRED)
find_package(MPI REQUIRED)
find_package(METIS)

if(METIS_FOUND)
    target_link_libraries(cholesky_mustard PUBLIC METIS::METIS)
    target_link_libraries(lu_mustard PUBLIC METIS::METIS)
    target_link_libraries(p_lu_mustard PUBLIC METIS::METIS)
    target_link_libraries(cg_mustard PUBLIC METIS::METIS)
    target_link_libraries(pingpong PUBLIC METIS::METIS)
else()
    message(WARNING "METIS library not found. METIS support will be disabled.")
endif()

set(MUSTARD_COMMON_LIBS
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::nvml
    -lnvshmem_host
    -lnvshmem_device
    MPI::MPI_CXX)

set(MUSTARD_NVSHMEM_TARGETS
    cholesky_mustard lu_mustard p_lu_mustard pingpong_mustard
    latency_signal latency_tile_pull retune_latency_hiding cg_mustard cuda_scheduler_ordering)

target_link_libraries(cholesky_mustard ${MUSTARD_COMMON_LIBS})
target_link_libraries(lu_mustard ${MUSTARD_COMMON_LIBS})
target_link_libraries(p_lu_mustard ${MUSTARD_COMMON_LIBS} CUDA::curand)
target_link_libraries(pingpong_mustard ${MUSTARD_COMMON_LIBS})
target_link_libraries(latency_signal ${MUSTARD_COMMON_LIBS})
target_link_libraries(latency_tile_pull ${MUSTARD_COMMON_LIBS})
target_link_libraries(retune_latency_hiding ${MUSTARD_COMMON_LIBS})
target_link_libraries(cg_mustard ${MUSTARD_COMMON_LIBS} ${NVSHMEM_LIBRARIES})
target_link_libraries(cuda_scheduler_ordering ${MUSTARD_COMMON_LIBS})

foreach(tgt IN LISTS MUSTARD_NVSHMEM_TARGETS)
    target_link_options(${tgt} PRIVATE "$<DEVICE_LINK:-lnvshmem_device>")
endforeach()
