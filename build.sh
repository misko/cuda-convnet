#!/bin/sh

# Fill in these environment variables.

# This file and others have been updated to work with CUDA 5.0.

# Only use Fermi-generation cards. Older cards won't work.

# If you're not sure what these paths should be, 
# you can use the find command to try to locate them.
# For example, NUMPY_INCLUDE_PATH contains the file
# arrayobject.h. So you can search for it like this:
# 
# find /usr -name arrayobject.h
# 
# (it'll almost certainly be under /usr)

# CUDA toolkit installation directory.
#export CUDA_INSTALL_PATH=/usr/local/cuda-5.0
export CUDA_INSTALL_PATH=/usr/local/cuda

# CUDA SDK installation directory.
export CUDA_SDK_PATH=$CUDA_INSTALL_PATH

# Python include directory. This should contain the file Python.h, among others.
#export PYTHON_INCLUDE_PATH=/usr/include/python2.6
export PYTHON_INCLUDE_PATH=/usr/include/python2.7

# Numpy include directory. This should contain the file arrayobject.h, among others.
#export NUMPY_INCLUDE_PATH=/usr/local/lib64/python2.6/site-packages/numpy-1.8.0-py2.6-linux-x86_64.egg/numpy/core/include/numpy/
export NUMPY_INCLUDE_PATH=/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/

# ATLAS library directory. This should contain the file libcblas.so, among others.
export ATLAS_LIB_PATH=/usr/lib64/atlas

make $*

