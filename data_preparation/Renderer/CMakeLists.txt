cmake_minimum_required(VERSION 3.5) # Update the version
project(Renderer)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PNG REQUIRED)

# Set the C++ standard and other compile options
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -w") # Use -std=c++2b for C++23 support

# Set OpenMesh include and library directories
set(OpenMesh_INCLUDE_DIR "/usr/local/include/OpenMesh")
set(OpenMesh_LIB_DIR "/home/steve/Documents/ml3d/Project/Installs/OpenMesh-10.0.0/build/Build/lib")

# Set GLEW include and library directories
set(GLEW_INCLUDE_DIR "/usr/include/GL")
set(GLEW_LIBRARY "/usr/lib/x86_64-linux-gnu/libGLEW.so.2.1.0")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIR} # Include GLEW include directory
    ${GLUT_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${OpenMesh_INCLUDE_DIR} # Include OpenMesh include directory
)

# Add source files
set(SOURCES
    main.cpp
    tinyply.cpp
    # Add other source files you need to compile
)

# Add executable
add_executable(Renderer ${SOURCES})

# Link libraries
target_link_libraries(Renderer
    ${OpenCV_LIBS}
    ${GFlags_LIBS}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARY} # Link GLEW library directly
    ${GLUT_LIBRARIES}
    ${PNG_LIBRARY}
    gflags
    "${OpenMesh_LIB_DIR}/libOpenMeshCore.so.10.0"  # Link OpenMeshCore library
    "${OpenMesh_LIB_DIR}/libOpenMeshTools.so.10.0"  # Link OpenMeshTools library
)
