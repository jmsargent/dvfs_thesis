#!/bin/bash
# Update all your wrappers to include the -B for your workspace
/usr/bin/apptainer exec --nv \
  -B /tmp \
  -B /data/users/sargent/dvfs_thesis \
  /data/users/sargent/dvfs_thesis/containers/container.sif \
  g++ "$@" # (or g++/gcc)