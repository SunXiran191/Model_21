cmake_minimum_required(VERSION 3.10)

project(icar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -mcpu=native -flto -pthread -Wno-sign-compare -Wno-unused-variable -Wno-unused-but-set-variable")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

pkg_search_module(GLIB REQUIRED glib-2.0)
pkg_search_module(SERIAL REQUIRED libserial)
#pkg_search_module(PPNC REQUIRED ppnc)
#pkg_search_module(ONNX REQUIRED onnx)

include(cmake/Sources.cmake)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/res/include
    ${CMAKE_CURRENT_SOURCE_DIR}/res/configs
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/ppseg
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/ocr
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/ppyoloe
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imgprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/src/special
    ${CMAKE_CURRENT_SOURCE_DIR}/src/standard
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thread
    ${OpenCV_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${SERIAL_INCLUDE_DIRS}
    #${PPNC_INCLUDE_DIRS}
    #${ONNX_INCLUDE_DIRS}
)

link_directories(
    ${SERIAL_LIBRARY_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/YOLO11_RK3588_object_detect-main/rknn_lib
    #${PPNC_LIBRARY_DIRS}
    #${ONNX_LIBRARY_DIRS}
)

# Always use main.cpp as the entry for the main executable.
add_executable(${PROJECT_NAME} main.cpp ${ICAR_CORE_SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${SERIAL_LIBRARIES}
    rknnrt
    #${PPNC_LIBRARIES}
    #${ONNX_LIBRARIES}
    pthread
)


