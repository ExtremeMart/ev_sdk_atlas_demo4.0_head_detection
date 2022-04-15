cmake_minimum_required(VERSION 3.5)
project(ji)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_BUILD_TYPE "Release")

# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
find_package(PkgConfig REQUIRED)

#jsoncpp相关头文件路径和源文件
include_directories(${CMAKE_SOURCE_DIR}/3rd/jsoncpp_simple/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd/jsoncpp_simple)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd/wkt_parser)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:/usr/local/ev_sdk/lib")


add_definitions(-DENABLE_DVPP_INTERFACE)
set(EDGE_INC_PATH $ENV{CANN_INC_PATH})
set(EDGE_LIB_PATH $ENV{CANN_LIB_PATH})

# 编译libji.so
list(APPEND libji_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ji.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SampleDetector.cpp        
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SampleAlgorithm.cpp        
        )
add_library (${PROJECT_NAME} SHARED ${libji_SOURCES})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3rd/wkt_parser/include       
        PUBLIC ${EDGE_INC_PATH}
        )       
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        glog
        WKTParser
        pthread
        jsoncpp
        ${EDGE_LIB_PATH}/libascendcl.so
        ${EDGE_LIB_PATH}/libacl_dvpp.so
        ${EDGE_LIB_PATH}/libascend_protobuf.so
       
        )

# Install libji.so
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rd/fonts DESTINATION lib)
# ---  END EV_SDK DEPENDENCIES  ---