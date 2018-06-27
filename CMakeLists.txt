cmake_minimum_required(VERSION 2.8)
project(vision)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
find_package(catkin REQUIRED COMPONENTS
		tf
#		genmsg
#		actionlib_msgs
#		actionlib
#        message_generation
#        std_msgs
        roscpp
		roslib
		cv_bridge
#        visualization_msgs
        )
find_package(OpenCV REQUIRED)
set(BOOST_INCLUDEDIR /usr/include)
set(BOOST_LIBRARYDIR /usr/lib/x86_64-linux-gnu)
find_package(Boost REQUIRED COMPONENTS system thread)
set(PROJECT_INCLUDE_DIR include)

set(PROJECT_SOURCE_DIR src)

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIR}
        ${PROJECT_INCLUDE_DIR}
        ${catkin_INCLUDE_DIRS}
	/usr/include/
)

catkin_package(
		DEPENDS
		roscpp
		tf
		roslib
		cv_bridge
#		message_runtime
		std_msgs
		)

set(CAM_CAL_SRC src/camCalExample.cpp src/camCal.cpp include/camCal.h)
set(ROI_DETECT_SRC src/roiDetectorExample.cpp src/roiDetector.cpp include/roiDetector.h )

add_executable(camCal ${CAM_CAL_SRC})
add_executable(roiDETECT ${ROI_DETECT_SRC})

target_link_libraries(camCal ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(roiDETECT ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(trainer
        src/trainer.cpp)
target_link_libraries(trainer ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(classifier src/classifier.cpp)
target_link_libraries(classifier ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(sampler src/sampler.cpp)
target_link_libraries(sampler  ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(recorder src/recorder.cpp)
target_link_libraries(recorder ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(working src/working.cpp)
target_link_libraries(working ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(stereo_cam_cal src/stereoCalExample.cpp src/camCal.cpp)
target_link_libraries(stereo_cam_cal ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
