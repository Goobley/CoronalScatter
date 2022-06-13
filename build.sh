#!/bin/bash

ISPC_EXE=ispc-v1.18.0-linux/bin/ispc
CXX=c++

${ISPC_EXE} -O3 --target=avx2-i64x4 ispc_advance.ispc -o advance.o
${CXX} -O3 -std=c++17 -march=native -mtune=native -c tasksys.cpp Scatter.cpp
${CXX} -pthread -flto Scatter.o advance.o -o scatter
# ${CXX} -std=c++17 -O3 -pthread -flto Scatter.cpp -o scatter
