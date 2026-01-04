cmake_minimum_required(VERSION 3.16)
project(MNIST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# MODE SELECTION
set(MODE "cpu" CACHE STRING "Build mode: cpu | colab")

message(STATUS "Build mode: ${MODE}")

# SOURCE FILES
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    src/*.cpp
    src/*.h
    src/Tensor/*.cpp
    src/Tensor/*.h
    src/NN/*.cpp
    src/NN/*.h
)

add_executable(mnist ${SRC_FILES})
target_include_directories(mnist PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_compile_definitions(mnist PRIVATE PROJECT_ROOT=\"${PROJECT_SOURCE_DIR}\")

# MODE: CPU (Raylib UI)
if(MODE STREQUAL "cpu")

    message(STATUS "CPU mode → enabling Raylib, disabling CUDA")

    set(USE_CUDA OFF)
    target_compile_definitions(mnist PRIVATE CPU_MODE)

    find_package(raylib REQUIRED)

    if(APPLE)
        target_link_libraries(mnist PRIVATE raylib m pthread)
    elseif(UNIX)
        target_link_libraries(mnist PRIVATE raylib m pthread dl rt X11)
    endif()

# MODE: COLAB (CUDA ONLY)
elseif(MODE STREQUAL "colab")

    message(STATUS "Google Colab mode → CUDA ON, Raylib OFF")

    set(USE_CUDA ON)
    target_compile_definitions(mnist PRIVATE USE_CUDA COLAB_MODE)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    file(GLOB_RECURSE CUDA_SRC
        src/*.cu
        src/Tensor/*.cu
        src/NN/*.cu
    )

    target_sources(mnist PRIVATE ${CUDA_SRC})
    target_include_directories(mnist PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(mnist PRIVATE CUDA::cudart)

else()
    message(FATAL_ERROR "Unknown MODE='${MODE}'. Use MODE=cpu or MODE=colab")
endif()
