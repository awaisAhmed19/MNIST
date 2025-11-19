cmake_minimum_required(VERSION 3.16)
project(MNIST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g -O0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

#==================================================================================
# SOURCE FILES
#==================================================================================

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

target_compile_definitions(mnist PRIVATE PROJECT_ROOT=\"${PROJECT_SOURCE_DIR}\")

#==================================================================================
# OPTIONAL CUDA
#==================================================================================

option(USE_CUDA "Enable CUDA GPU acceleration" ON)

if(USE_CUDA)
    find_package(CUDAToolkit)

    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA toolkit found → enabling GPU features")

        enable_language(CUDA)
        target_compile_definitions(mnist PRIVATE USE_CUDA)

        file(GLOB_RECURSE CUDA_SRC
            src/*.cu
            src/Matrix/*.cu
            src/NN/*.cu
        )

        target_sources(mnist PRIVATE ${CUDA_SRC})
        target_include_directories(mnist PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(mnist PRIVATE CUDA::cudart)
    else()
        message(WARNING "CUDA requested but NOT found → falling back to CPU mode")
        set(USE_CUDA OFF)
    endif()
else()
    message(STATUS "CUDA disabled → building CPU-only mode")
endif()

#==================================================================================
# RAYLIB
#==================================================================================

find_package(raylib REQUIRED)

if(APPLE)
    target_link_libraries(mnist PRIVATE raylib m pthread)
elseif(UNIX)
    target_link_libraries(mnist PRIVATE raylib m pthread dl rt X11)
endif()

#==================================================================================
# RUN TARGET
#==================================================================================

add_custom_target(run
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mnist
    DEPENDS mnist
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
