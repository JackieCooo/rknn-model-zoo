cmake_minimum_required(VERSION 3.16)

project(rknn-model-zoo)

option(RGA_ENABLE "Enable RGA support" ON)
option(NEON_ENABLE "Enable NEON support" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall)

if(RGA_ENABLE)
    add_compile_definitions(WITH_RGA)

    # rga
    set(RGA_PREFIX 3rd-party/librga)
    include_directories(${RGA_PREFIX}/include)
    link_directories(${RGA_PREFIX}/libs/Linux/gcc-aarch64)

    # rga-wrapper
    set(RGA_WRAPPER_PREFIX 3rd-party/rga-wrapper/src)
    include(${RGA_WRAPPER_PREFIX}/rga-wrapper-config.cmake)
    include_directories(${RGA_WRAPPER_INC})
endif(RGA_ENABLE)

if(NEON_ENABLE)
    add_compile_definitions(WITH_NEON)
endif(NEON_ENABLE)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/install/bin)

# rknpu2
set(RKNPU_PREFIX 3rd-party/rknpu2/Linux/librknn_api)
include_directories(${RKNPU_PREFIX}/include)
link_directories(${RKNPU_PREFIX}/aarch64)

# mpi
set(MPI_PREFIX 3rd-party/mpi)
include_directories(
    ${MPI_PREFIX}/include
    ${MPI_PREFIX}/lib/lib64
)
link_directories(${MPI_PREFIX}/lib/lib64)

# mpi-wrapper
set(MPI_WRAPPER_PREFIX 3rd-party/mpi-wrapper/src)
include(${MPI_WRAPPER_PREFIX}/mpi-wrapper.cmake)

# opencv
set(OpenCV_DIR 3rd-party/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(
    src
    src/utils
    src/task
    ${OpenCV_INCLUDE_DIRS}
)
file(GLOB PROJ_SRC src/utils/*.cpp src/task/engine.cpp)

set(CLASSIFY_TARGET classify-example)
file(GLOB CLS_SRC src/task/classify.cpp)
add_executable(${CLASSIFY_TARGET} ${PROJ_SRC} ${CLS_SRC} ${RGA_WRAPPER_SRC} example/classify_example.cpp)
target_link_libraries(${CLASSIFY_TARGET} PRIVATE rknnrt rga ${OpenCV_LIBS})
