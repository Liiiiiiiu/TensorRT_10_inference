#!/bin/bash

# Remove existing build directory if it exists
if [ -d "build" ]; then
  rm -r build
fi

# Create a new build directory
mkdir build && cd build

# Run CMake to configure the project
cmake ..

# Compile the project using 2 threads
make -j$(nproc)

