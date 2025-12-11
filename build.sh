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
  DEPS=(g++ cmake make )
  UPDATE_CMD=""
  INSTALL_CMD=""
  PKG_CHECK=""

  # Pick installer based on distro
  case "$DISTRO" in
  ubuntu | debian)
    PKG_CHECK="dpkg -s"
    INSTALL_CMD="sudo apt-get install -y"
    UPDATE_CMD="sudo apt-get update"
    ;;
  arch | manjaro)
    PKG_CHECK="pacman -Qi"
    INSTALL_CMD="sudo pacman -S --noconfirm"
    UPDATE_CMD="sudo pacman -Sy"
    DEPS=(gcc cmake make glfw raylib)
    ;;
  fedora)
    PKG_CHECK="rpm -q"
    INSTALL_CMD="sudo dnf install -y"
    UPDATE_CMD="sudo dnf check-update || true"
    DEPS=(gcc-c++ cmake make glfw-devel raylib-devel)
    ;;
  *)
    echo "Unsupported distro: $DISTRO"
    exit 1
    ;;
  esac

  # Check missing packages
  MISSING_PKGS=()
  for pkg in "${DEPS[@]}"; do
    if ! $PKG_CHECK "$pkg" &>/dev/null; then
      MISSING_PKGS+=("$pkg")
    fi
  done

  if [ ${#MISSING_PKGS[@]} -eq 0 ]; then
    echo "All dependencies already installed."
  else
    echo "Missing packages: ${MISSING_PKGS[*]}"
    echo "Updating package database..."
    $UPDATE_CMD
    echo "Installing missing dependencies..."
    $INSTALL_CMD "${MISSING_PKGS[@]}"
  fi
}

install_deps_macos() {
  echo "Checking dependencies on macOS..."
  if ! command -v brew &>/dev/null; then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi

  DEPS=(cmake raylib)
  MISSING_PKGS=()

  for pkg in "${DEPS[@]}"; do
    if ! brew list "$pkg" &>/dev/null; then
      MISSING_PKGS+=("$pkg")
    fi
  done

  if [ ${#MISSING_PKGS[@]} -eq 0 ]; then
    echo "All dependencies already installed."
  else
    echo "Installing missing packages: ${MISSING_PKGS[*]}"
    brew install "${MISSING_PKGS[@]}"
  fi
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
  echo "Unzipping dataset..."
  unzip -q data.zip -d .
fi

BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
  echo "Creating build directory..."
  mkdir -p "$BUILD_DIR"
fi

echo "Running CMake..."
cmake -S . -B "$BUILD_DIR" -DUSE_CUDA=OFF
cmake --build "$BUILD_DIR"

if [ -f "./bin/mnist" ]; then
  echo "Running MNIST binary..."
  ./bin/mnist
else
  echo "Error: Binary not found at ./bin/mnist"
  exit 1
fi
