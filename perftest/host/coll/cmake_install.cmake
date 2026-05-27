# Install script for directory: /data/users/sargent/.local/nvshmem/perftest/host/coll

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/alltoall")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall_on_stream")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall_on_stream"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall_on_stream")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/alltoall_on_stream")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall_on_stream")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall_on_stream"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/alltoall_on_stream")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/barrier")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/barrier_all")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all_on_stream")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all_on_stream"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all_on_stream")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/barrier_all_on_stream")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all_on_stream")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all_on_stream"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_all_on_stream")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_on_stream")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_on_stream"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_on_stream")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/barrier_on_stream")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_on_stream")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_on_stream"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/barrier_on_stream")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/broadcast")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast_on_stream")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast_on_stream"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast_on_stream")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/broadcast_on_stream")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast_on_stream")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast_on_stream"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/broadcast_on_stream")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/fcollect")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect_on_stream")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect_on_stream"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect_on_stream")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/fcollect_on_stream")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect_on_stream")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect_on_stream"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/fcollect_on_stream")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/reduction")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction_on_stream")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction_on_stream"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction_on_stream")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/reduction_on_stream")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction_on_stream")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction_on_stream"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/reduction_on_stream")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/sync")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/sync_all")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all_on_stream")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all_on_stream"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all_on_stream")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/sync_all_on_stream")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all_on_stream")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all_on_stream"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_all_on_stream")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_on_stream")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_on_stream"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_on_stream")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/coll/sync_on_stream")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_on_stream" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_on_stream")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_on_stream"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/coll/sync_on_stream")
    endif()
  endif()
endif()

