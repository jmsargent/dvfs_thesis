# Install script for directory: /data/users/sargent/.local/nvshmem/perftest/host/init

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
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/init/malloc" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/init/malloc")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/init/malloc"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/init/malloc")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/init" TYPE EXECUTABLE FILES "/data/users/sargent/.local/nvshmem/perftest_build/host/init/malloc")
  if(EXISTS "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/init/malloc" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/init/malloc")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/init/malloc"
         OLD_RPATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64/stubs:/data/users/sargent/.local/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/users/sargent/.local/nvshmem/perftest/perftest_install/host/init/malloc")
    endif()
  endif()
endif()

