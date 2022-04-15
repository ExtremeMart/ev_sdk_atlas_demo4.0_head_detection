cmake_minimum_required(VERSION 3.5)
project(test-ji-api)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")

# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
set(ENV{PKG_CONFIG_PATH} "ENV{PKG_CONFIG_PATH}")
find_package(PkgConfig REQUIRED)

add_definitions(-DENABLE_DVPP_INTERFACE)
set(EDGE_INC_PATH $ENV{CANN_INC_PATH})
set(EDGE_LIB_PATH $ENV{CANN_LIB_PATH})

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:/usr/local/ev_sdk/lib")

add_executable(${PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Algo.cpp
        )

target_include_directories(${PROJECT_NAME}
        PUBLIC ${EDGE_INC_PATH}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../3rd/wkt_parser/include
        PUBLIC "/usr/local/ev_sdk/include"
        )

target_link_libraries(${PROJECT_NAME}
        /usr/local/ev_sdk/lib/libji.so
        ${OpenCV_LIBS}
        glog
        pthread
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libWKTParser.so
        ${EDGE_LIB_PATH}/libascendcl.so
        ${EDGE_LIB_PATH}/libacl_dvpp.so
        ${EDGE_LIB_PATH}/libascend_protobuf.so
        )

install(TARGETS ${PROJECT_NAME} DESTINATION bin)