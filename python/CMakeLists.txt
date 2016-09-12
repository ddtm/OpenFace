# Python stuff.
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
include_directories(../lib/local/LandmarkDetector/include)
include_directories(../lib/local/FaceAnalyser/include)	
			
add_library(pyopenface pyopenface.cpp detector.cpp)
target_link_libraries(pyopenface LandmarkDetector)
target_link_libraries(pyopenface FaceAnalyser)
target_link_libraries(pyopenface dlib)
target_link_libraries(pyopenface ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

set_property(TARGET pyopenface PROPERTY CXX_STANDARD 11)
set_property(TARGET pyopenface PROPERTY SUFFIX .so)
set_property(TARGET pyopenface PROPERTY PREFIX "")

install(TARGETS pyopenface DESTINATION python)
