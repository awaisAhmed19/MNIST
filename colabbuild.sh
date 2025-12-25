#!/bin/bash
set -e  # Exit on any error

echo "Starting MNIST build script for Google Colab..."
echo "Detected environment: Google Colab (Ubuntu Linux)"

# Update package list and install essential build tools (quietly)
echo "Installing/ensuring build essentials and dependencies..."
apt-get update -qq
apt-get install -y -qq g++ cmake make git unzip libglfw3-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxrandr-dev libxi-dev

# Install raylib from source (system-wide) – reliable fallback for Colab
if ! pkg-config --exists raylib; then
    echo "raylib not found. Building and installing from source..."
    cd /tmp
    git clone https://github.com/raysan5/raylib.git --depth 1
    cd raylib/src
    make PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=SHARED -j$(nproc)
    make install PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=SHARED
    ldconfig  # Update shared library cache
    cd /content/MNIST  # Go back to your project
else
    echo "raylib already installed."
fi

# Unzip dataset if needed
DATA_DIR="data"
if [ ! -d "$DATA_DIR" ] && [ -f "data.zip" ]; then
    echo "Unzipping dataset..."
    unzip -q data.zip
fi

# Create build directory
BUILD_DIR="build"
echo "Creating build directory: $BUILD_DIR"
rm -rf "$BUILD_DIR"  # Clean previous build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run CMake (CPU-only, as requested)
echo "Configuring with CMake (CPU-only mode)..."
cmake .. -DUSE_CUDA=OFF

# Build the project
echo "Building the project..."
make -j$(nproc)

# Run the binary (adjust path if your executable is named differently)
if [ -f "mnist" ]; then
    echo "Build successful! Running MNIST application..."
    ./mnist
elif [ -f "bin/mnist" ]; then
    echo "Build successful! Running MNIST application..."
    ./bin/mnist
else
    echo "Error: Executable not found! Check CMake output for target name."
    ls -la
    exit 1
fi
