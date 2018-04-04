cmake_minimum_required(VERSION 2.8)
project(cam_cal)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
set(Boost_LIBRARY_DIRS /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
set(BOOST_LIBRARYDIR /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)

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

add_executable(trainer
        src/trainer.cpp)
target_link_libraries(trainer ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(classifier src/classifier.cpp)
target_link_libraries(classifier ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(sampler src/sampler.cpp)
target_link_libraries(sampler  ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})


