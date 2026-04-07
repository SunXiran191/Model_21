cmake_minimum_required(VERSION 3.10)

project(icar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(
        -O3								#最高优化级别
        -Wall							#开启所有警告
        -Wno-sign-compare				#屏蔽有符号和无符号比较警告
        -Wno-unused-variable			#屏蔽未使用变量警告
        -Wno-unused-but-set-variable 	#屏蔽未使用但已设置变量警告	
    )
endif()

find_package(OpenCV REQUIRED) #查找cv库

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)									#收集src目录下的所有cpp

#若有main则加入编译列表
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    list(APPEND PROJECT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
endif()

if(PROJECT_SOURCES STREQUAL "")
    message(FATAL_ERROR "No C++ source files were found. Check project structure.")
endif()

set(MAIN_HAS_ENTRY OFF)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" MAIN_FILE_CONTENT)
    string(FIND "${MAIN_FILE_CONTENT}" "main(" MAIN_POS)
    if(NOT MAIN_POS EQUAL -1)
        set(MAIN_HAS_ENTRY ON)		#是否存在main
    endif()
endif()

#有入口就编译成可执行文件，没有就编译成静态库
if(MAIN_HAS_ENTRY)
    add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
else()
    add_library(${PROJECT_NAME} STATIC ${PROJECT_SOURCES})
    message(WARNING "main.cpp does not contain main(). Building STATIC library target 'icar' only.")
endif()

#头文件搜索路径
target_include_directories(${PROJECT_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/res/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/res/configs"
)

#链接cv库
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${OpenCV_LIBRARIES}
)

if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PUBLIC pthread)	#链接pthread库
endif()

