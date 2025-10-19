#!/bin/bash

# build folder path
BUILD_DIR="./build"

# make sure build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Creating..."
    mkdir -p "$BUILD_DIR"
fi

# run cmake if needed
cd "$BUILD_DIR" || exit
cmake .. # generates Makefiles (safe even if already done)
make     # builds the target

# run the executable
./bin/mnist
