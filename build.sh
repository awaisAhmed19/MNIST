#!/bin/bash
set -e

echo "Starting MNIST build script..."

OS_TYPE="$(uname)"
echo "Detected OS: $OS_TYPE"


install_deps_linux() {
  echo "Checking dependencies on Linux..."

  # Detect distro
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
  else
    echo "Can't detect distro. /etc/os-release not found."
    exit 1
  fi

  # Define dependencies
  DEPS=(g++ cmake make libglfw3-dev libraylib-dev)

  # Pick installer based on distro
  case "$DISTRO" in
    ubuntu|debian)
      PKG_CHECK="dpkg -s"
      INSTALL_CMD="sudo apt-get install -y"
      UPDATE_CMD="sudo apt-get update"
      ;;
    arch|manjaro)
      PKG_CHECK="pacman -Qi"
      INSTALL_CMD="sudo pacman -S --noconfirm"
      UPDATE_CMD="sudo pacman -Sy"
      # Replace package names for Arch equivalents
      DEPS=(gcc cmake make glfw raylib)
      ;;
    fedora)
      PKG_CHECK="rpm -q"
      INSTALL_CMD="sudo dnf install -y"
      UPDATE_CMD="sudo dnf check-update"
      DEPS=(gcc-c++ cmake make glfw-devel raylib-devel)
      ;;
    *)
      echo "Unsupported distro: $DISTRO"
      return 1
      ;;
  esac

  # Update package list once
  echo "Updating package database..."
  $UPDATE_CMD

  # Install missing packages
  for pkg in "${DEPS[@]}"; do
    if ! $PKG_CHECK "$pkg" &>/dev/null; then
      echo "Installing $pkg..."
      $INSTALL_CMD "$pkg"
    else
      echo "$pkg already installed."
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

DATA_DIR="data"
if [ ! -d "$DATA_DIR" ] && [ -f "data.zip" ]; then
  echo "Unzipping the dataset..."
  unzip -q data.zip -d .
fi
: << 'END'
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
