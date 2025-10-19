
cmake_minimum_required(VERSION 3.16)
project(MNIST LANGUAGES CXX)

# =====================
# Compiler Settings
# =====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# =====================
# Source Files
# =====================
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp src/*.h src/Matrix/*.cpp src/Matrix/*.h)

add_executable(mnist ${SRC_FILES})

target_include_directories(mnist PRIVATE ${PROJECT_SOURCE_DIR}/src)

# =====================
# CUDA Mode
# =====================
option(USE_CUDA "Enable CUDA support" OFF)

if (USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    message(STATUS " CUDA support enabled")

    # Define CUDA-specific flag for conditional compilation
    target_compile_definitions(mnist PRIVATE USE_CUDA)
    
    # Add CUDA source files if you have them
    file(GLOB_RECURSE CUDA_SRC src/*.cu)
    target_sources(mnist PRIVATE ${CUDA_SRC})

    # Link CUDA runtime libraries
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
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

