# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build

# Utility rule file for doxygen.

# Include the progress variables for this target.
include doc/CMakeFiles/doxygen.dir/progress.make

doc/CMakeFiles/doxygen: doc/Doxyfile
doc/CMakeFiles/doxygen: doc/root.markdown
doc/CMakeFiles/doxygen: ../doc/opencv.bib
doc/CMakeFiles/doxygen: ../modules/core/include
doc/CMakeFiles/doxygen: ../modules/core/doc
doc/CMakeFiles/doxygen: ../modules/imgproc/include
doc/CMakeFiles/doxygen: ../modules/imgproc/doc
doc/CMakeFiles/doxygen: ../modules/imgcodecs/include
doc/CMakeFiles/doxygen: ../modules/videoio/include
doc/CMakeFiles/doxygen: ../modules/videoio/doc
doc/CMakeFiles/doxygen: ../modules/highgui/include
doc/CMakeFiles/doxygen: ../modules/highgui/doc
doc/CMakeFiles/doxygen: ../modules/video/include
doc/CMakeFiles/doxygen: ../modules/calib3d/include
doc/CMakeFiles/doxygen: ../modules/calib3d/doc
doc/CMakeFiles/doxygen: ../modules/calib3d/doc/calib3d.bib
doc/CMakeFiles/doxygen: ../modules/features2d/include
doc/CMakeFiles/doxygen: ../modules/features2d/doc
doc/CMakeFiles/doxygen: ../modules/objdetect/include
doc/CMakeFiles/doxygen: ../modules/objdetect/doc
doc/CMakeFiles/doxygen: ../modules/ml/include
doc/CMakeFiles/doxygen: ../modules/ml/doc
doc/CMakeFiles/doxygen: ../modules/flann/include
doc/CMakeFiles/doxygen: ../modules/photo/include
doc/CMakeFiles/doxygen: ../modules/stitching/include
doc/CMakeFiles/doxygen: ../modules/stitching/doc
doc/CMakeFiles/doxygen: ../modules/cudaarithm/include
doc/CMakeFiles/doxygen: ../modules/cudabgsegm/include
doc/CMakeFiles/doxygen: ../modules/cudacodec/include
doc/CMakeFiles/doxygen: ../modules/cudafeatures2d/include
doc/CMakeFiles/doxygen: ../modules/cudafilters/include
doc/CMakeFiles/doxygen: ../modules/cudaimgproc/include
doc/CMakeFiles/doxygen: ../modules/cudalegacy/include
doc/CMakeFiles/doxygen: ../modules/cudaobjdetect/include
doc/CMakeFiles/doxygen: ../modules/cudaoptflow/include
doc/CMakeFiles/doxygen: ../modules/cudastereo/include
doc/CMakeFiles/doxygen: ../modules/cudawarping/include
doc/CMakeFiles/doxygen: ../modules/cudev/include
doc/CMakeFiles/doxygen: ../modules/shape/include
doc/CMakeFiles/doxygen: ../modules/superres/include
doc/CMakeFiles/doxygen: ../modules/videostab/include
doc/CMakeFiles/doxygen: ../modules/viz/include
doc/CMakeFiles/doxygen: ../modules/viz/doc
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/doc && /usr/bin/doxygen /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/doc/Doxyfile

doxygen: doc/CMakeFiles/doxygen
doxygen: doc/CMakeFiles/doxygen.dir/build.make

.PHONY : doxygen

# Rule to build all files generated by this target.
doc/CMakeFiles/doxygen.dir/build: doxygen

.PHONY : doc/CMakeFiles/doxygen.dir/build

doc/CMakeFiles/doxygen.dir/clean:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/doc && $(CMAKE_COMMAND) -P CMakeFiles/doxygen.dir/cmake_clean.cmake
.PHONY : doc/CMakeFiles/doxygen.dir/clean

doc/CMakeFiles/doxygen.dir/depend:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/doc /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/doc /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/doc/CMakeFiles/doxygen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/CMakeFiles/doxygen.dir/depend

