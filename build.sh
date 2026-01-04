
#!/bin/bash
set -e

echo "🚀 MNIST Build Script"

# ASK FOR MODE
echo ""
echo "Select build mode:"
echo "  1) CPU   (Raylib UI, no CUDA)"
echo "  2) Colab (CUDA ON, no Raylib)"
echo ""

read -p "Enter choice (1/2): " MODE_CHOICE

if [ "$MODE_CHOICE" == "1" ]; then
    MODE="cpu"
elif [ "$MODE_CHOICE" == "2" ]; then
    MODE="colab"
else
    echo "❌ Invalid choice. Use 1 or 2."
    exit 1
fi

echo "✔ Selected mode: $MODE"
echo ""

OS_TYPE="$(uname)"
echo "Detected OS: $OS_TYPE"


# DEPENDENCIES
install_deps_linux() {

  echo "Checking dependencies on Linux..."

  if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
  else
    echo "Can't detect distro. Exiting."
    exit 1
  fi


DEPS=(cmake make)

  UPDATE_CMD=""
  INSTALL_CMD=""
  PKG_CHECK=""

  case "$DISTRO" in
    ubuntu|debian)
      PKG_CHECK="dpkg -s"
      INSTALL_CMD="sudo apt-get install -y"
      UPDATE_CMD="sudo apt-get update"
      DEPS+=(g++)
      ;;

arch|manjaro)
    PKG_CHECK="pacman -Qi"
    INSTALL_CMD="sudo pacman -S --noconfirm"
    UPDATE_CMD="sudo pacman -Sy"
    DEPS+=(gcc)
    ;;

    fedora)
      PKG_CHECK="rpm -q"
      INSTALL_CMD="sudo dnf install -y"
      UPDATE_CMD="sudo dnf check-update || true"
       DEPS+=(gcc-c++)
      ;;
    *)
      echo "Unsupported distro: $DISTRO"
      exit 1
      ;;
  esac

  # Raylib only for CPU mode
  if [ "$MODE" == "cpu" ]; then
    case "$DISTRO" in
      ubuntu|debian) DEPS+=(libraylib-dev) ;;
      arch|manjaro)  DEPS+=(glfw raylib) ;;
      fedora)        DEPS+=(glfw-devel raylib-devel) ;;
    esac
  fi

  MISSING=()
  for pkg in "${DEPS[@]}"; do
    if ! $PKG_CHECK "$pkg" &>/dev/null; then
      MISSING+=("$pkg")
    fi
  done

  if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Installing missing packages: ${MISSING[*]}"
    $UPDATE_CMD
    $INSTALL_CMD "${MISSING[@]}"
  else
    echo "All required packages installed."
  fi
}


install_deps_macos() {

  echo "Checking dependencies on macOS..."

  if ! command -v brew &>/dev/null; then
    echo "Homebrew missing — installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi

  DEPS=(cmake)

  if [ "$MODE" == "cpu" ]; then
    DEPS+=(raylib)
  fi

  for pkg in "${DEPS[@]}"; do
    if ! brew list "$pkg" &>/dev/null; then
      brew install "$pkg"
    fi
  done
}


case "$OS_TYPE" in
  Linux*) install_deps_linux ;;
  Darwin*) install_deps_macos ;;
  *) echo "Unsupported OS."; exit 1 ;;
esac


# DATA
if [ ! -d "data" ] && [ -f "data.zip" ]; then
  echo "Unzipping dataset..."
  unzip -q data.zip -d .
fi


# BUILD
BUILD_DIR="build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Configuring CMake (MODE=$MODE)..."

if [ "$MODE" == "cpu" ]; then
    cmake -S . -B "$BUILD_DIR" -DMODE=cpu
else
    cmake -S . -B "$BUILD_DIR" -DMODE=colab
fi

cmake --build "$BUILD_DIR"


# RUN
if [ -f "./bin/mnist" ]; then
  echo "Running MNIST..."
  ./bin/mnist
else
  echo "❌ Binary not found at ./bin/mnist"
  exit 1
fi
