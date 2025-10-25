
cmake_minimum_required(VERSION 3.16)
project(MNIST LANGUAGES CXX)

# =====================
# Compiler Settings
# =====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g -O0)

# =====================
# Output Directories
# =====================
# Put the compiled binary in the project root /bin instead of /build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# =====================
# Source Files
# =====================
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    src/*.cpp
    src/*.h
    src/Matrix/*.cpp
    src/Matrix/*.h
    src/NN/*.cpp
    src/NN/*.h
)

add_executable(mnist ${SRC_FILES})
target_include_directories(mnist PRIVATE ${PROJECT_SOURCE_DIR}/src)

# =====================
# Global Definitions
# =====================
# Let the program know where the project root is (for loading files like data/train.csv)
target_compile_definitions(mnist PRIVATE PROJECT_ROOT=\"${PROJECT_SOURCE_DIR}\")


# =====================
# CUDA Mode
# =====================
option(USE_CUDA "Enable CUDA support" OFF)

if (USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    message(STATUS "CUDA support enabled")

    target_compile_definitions(mnist PRIVATE USE_CUDA)

    file(GLOB_RECURSE CUDA_SRC src/*.cu)
    target_sources(mnist PRIVATE ${CUDA_SRC})

    target_link_libraries(mnist PRIVATE CUDA::cudart)
endif()

# =====================
# Raylib Integration
# =====================
find_package(raylib REQUIRED)

# Link Raylib + system deps
target_link_libraries(mnist PRIVATE raylib m pthread dl rt X11)

# =====================
# Run target (for dev convenience)
# =====================
add_custom_target(run
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mnist
    DEPENDS mnist
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

