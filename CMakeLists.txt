cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

project(NoFuncaJenny VERSION 1.0 LANGUAGES CXX CUDA)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${SDL2_IMAGE_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${GLM_INCLUDE_DIRS})

file(GLOB_RECURSE LIB_HEADERS lib/*.h)
foreach(FILE ${LIB_HEADERS})
    get_filename_component(DIR_PATH ${FILE} PATH)
    include_directories(${DIR_PATH})
endforeach()

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu lib/*.cpp lib/*.cu)
add_executable(NoFuncaJenny ${SRC_FILES})
target_link_libraries(NoFuncaJenny ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARIES} ${OpenCV_LIBS} SDL2_image)

set_property(TARGET NoFuncaJenny 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Define a macro for CUDA compilation
target_compile_definitions(NoFuncaJenny PRIVATE CUDA_COMPILE)

file(COPY ${CMAKE_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})