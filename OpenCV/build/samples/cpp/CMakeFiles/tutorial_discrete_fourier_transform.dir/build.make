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

# Include any dependencies generated for this target.
include samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/depend.make

# Include the progress variables for this target.
include samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/progress.make

# Include the compile flags for this target's objects.
include samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/flags.make

samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o: samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/flags.make
samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o: ../samples/cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o -c /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp

samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.i"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp > CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.i

samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.s"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp -o CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.s

samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o.requires:

.PHONY : samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o.requires

samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o.provides: samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o.requires
	$(MAKE) -f samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/build.make samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o.provides.build
.PHONY : samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o.provides

samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o.provides.build: samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o


# Object files for target tutorial_discrete_fourier_transform
tutorial_discrete_fourier_transform_OBJECTS = \
"CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o"

# External object files for target tutorial_discrete_fourier_transform
tutorial_discrete_fourier_transform_EXTERNAL_OBJECTS =

bin/cpp-tutorial-discrete_fourier_transform: samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o
bin/cpp-tutorial-discrete_fourier_transform: samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/build.make
bin/cpp-tutorial-discrete_fourier_transform: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/cpp-tutorial-discrete_fourier_transform: /usr/lib/x86_64-linux-gnu/libGL.so
bin/cpp-tutorial-discrete_fourier_transform: /usr/lib/x86_64-linux-gnu/libtbb.so
bin/cpp-tutorial-discrete_fourier_transform: 3rdparty/ippicv/ippicv_lnx/lib/intel64/libippicv.a
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_shape.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_stitching.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_superres.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_videostab.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_viz.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_objdetect.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_photo.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_calib3d.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_features2d.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_flann.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_highgui.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_ml.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_videoio.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_imgcodecs.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_video.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_imgproc.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: lib/libopencv_core.so.3.2.0
bin/cpp-tutorial-discrete_fourier_transform: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/cpp-tutorial-discrete_fourier_transform: /usr/lib/x86_64-linux-gnu/libGL.so
bin/cpp-tutorial-discrete_fourier_transform: samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/cpp-tutorial-discrete_fourier_transform"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tutorial_discrete_fourier_transform.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/build: bin/cpp-tutorial-discrete_fourier_transform

.PHONY : samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/build

samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/requires: samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp.o.requires

.PHONY : samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/requires

samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/clean:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && $(CMAKE_COMMAND) -P CMakeFiles/tutorial_discrete_fourier_transform.dir/cmake_clean.cmake
.PHONY : samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/clean

samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/depend:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : samples/cpp/CMakeFiles/tutorial_discrete_fourier_transform.dir/depend

