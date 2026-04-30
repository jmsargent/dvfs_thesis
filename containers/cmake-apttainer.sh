#!/bin/bash
# Note: Using the specific CMake version path you mentioned earlier
apptainer exec --nv -B /tmp /data/users/sargent/dvfs_thesis/containers/container.sif /data/users/sargent/mustard/cmake-3.28.1-linux-x86_64/bin/cmake "$@"