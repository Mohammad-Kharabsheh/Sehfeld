cmake_minimum_required(VERSION 3.16)
project(Sehfeld VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)

set(OpenCV_DIR "D:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

set(ORT_DIR "D:/onnxruntime/onnxruntime-win-x64-1.20.1")
set(VCPKG_DIR "D:/vcpkg/installed/x64-windows")

set(CMAKE_AUTOUIC_SEARCH_PATHS app)

add_executable(Sehfeld
    app/main.cpp
    app/mainwindow.cpp
    app/mainwindow.h
    app/mainwindow.ui
    app/src/AnomalyDetector.cpp
    app/src/AnomalyDetector.hpp
    app/src/Reporter.cpp
    app/src/Reporter.hpp
)

target_include_directories(Sehfeld PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_DIR}/include
    ${VCPKG_DIR}/include
    app
    app/src
)

target_link_libraries(Sehfeld PRIVATE
    Qt6::Widgets
    Qt6::Concurrent
    ${OpenCV_LIBS}
    ${ORT_DIR}/lib/onnxruntime.lib
    ${VCPKG_DIR}/lib/faiss.lib
    ${VCPKG_DIR}/lib/openblas.lib
)

if(WIN32)
    set_target_properties(Sehfeld PROPERTIES WIN32_EXECUTABLE TRUE)
endif()

qt_finalize_executable(Sehfeld)