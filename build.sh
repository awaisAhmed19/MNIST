#!/bin/bash
set -e

echo "Starting MNIST build script..."

OS_TYPE="$(uname)"
echo "Detected OS: $OS_TYPE"

install_deps_linux() {
  echo "Checking dependencies on Linux..."
  for pkg in g++ cmake make libglfw3-dev libraylib-dev; do
    if ! dpkg -s $pkg &>/dev/null; then
      echo "Installing $pkg..."
      sudo apt-get update
      sudo apt-get install -y $pkg
    fi
  done
}

install_deps_macos() {
  echo "Checking dependencies on macOS..."
  # Check for Homebrew
  if ! command -v brew &>/dev/null; then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi

  for pkg in cmake raylib; do
    if ! brew list $pkg &>/dev/null; then
      echo "Installing $pkg..."
      brew install $pkg
    fi
  done
}

case "$OS_TYPE" in
Linux*) install_deps_linux ;;
Darwin*) install_deps_macos ;;
*)
  echo "Unsupported OS: $OS_TYPE. Please install dependencies manually."
  exit 1
  ;;
esac

BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
  echo "Build directory not found. Creating..."
  mkdir -p "$BUILD_DIR"
fi

echo "Running CMake..."
cmake -S . -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"

if [ -f "./bin/mnist" ]; then
  echo "Running MNIST program..."
  ./bin/mnist
else
  echo "Error: Binary not found at ./bin/mnist"
  exit 1
fi
