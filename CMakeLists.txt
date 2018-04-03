cmake_minimum_required(VERSION 2.8)
project(cam_cal)

set(CMAKE_CXX_STANDARD 11)

set(PROJECT_INCLUDE_DIR include)

set(PROJECT_SOURCE_DIR src)

find_package(OpenCV REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIR}
        ${PROJECT_INCLUDE_DIR}
)

set(CAM_CAL_SRC src/camCalExample.cpp src/camCal.cpp include/camCal.h)
set(ROI_DETECT_SRC src/roiDetectorExample.cpp src/roiDetector.cpp include/roiDetector.h )

add_executable(camCal ${CAM_CAL_SRC})
add_executable(roiDETECT ${ROI_DETECT_SRC})

target_link_libraries(camCal ${OpenCV_LIBS})
target_link_libraries(roiDETECT ${OpenCV_LIBS})
