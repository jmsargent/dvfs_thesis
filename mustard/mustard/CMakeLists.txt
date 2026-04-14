add_executable(test_timer test_timer.cu)
set_target_properties(test_timer PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(test_timer CUDA::cudart)

add_executable(cholesky_mustard cholesky_mustard.cu)
add_executable(pingpong_mustard pingpong_mustard.cu)
add_executable(lu_mustard lu_mustard.cu)
add_executable(p_lu_mustard lu_mustard_p.cu)
add_executable(cg_mustard cg_mustard.cu)

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
    -lnvshmem_host -lnvshmem_device
    MPI::MPI_CXX)

target_link_libraries(cholesky_mustard ${MUSTARD_COMMON_LIBS})
target_link_libraries(lu_mustard ${MUSTARD_COMMON_LIBS})
target_link_libraries(p_lu_mustard ${MUSTARD_COMMON_LIBS} CUDA::curand)
target_link_libraries(pingpong_mustard ${MUSTARD_COMMON_LIBS})
target_link_libraries(cg_mustard ${MUSTARD_COMMON_LIBS} ${NVSHMEM_LIBRARIES})
