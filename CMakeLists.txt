cmake_minimum_required(VERSION 3.16)
project(kernelNE)

set(CMAKE_CXX_STANDARD 14)

include_directories(lib)

#add_subdirectory(ext/Graph)

#set(CMAKE_MODULE_PATH "./ext/Graph;${CMAKE_MODULE_PATH}")
set(GML_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/Graph/src)
set(GML_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/Graph/lib)
#include_directories(${GML_SRC_DIR})
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/Graph)
include_directories(${GML_LIB_DIR})

set(SOURCE_FILES src/Model.cpp src/Unigram.cpp src/Vocabulary.cpp src/Node.cpp)
set(LIB_FILES lib/Model.h lib/Unigram.h lib/Vocabulary.h lib/Node.h lib/Utilities.h)

add_executable(${CMAKE_PROJECT_NAME} main.cpp ${SOURCE_FILES} ${LIB_FILES})


target_link_libraries(kernelNE ${GML_LIB_DIR})